#!/usr/bin/env python
"""PHASE 3 - Central monitoring website.

    python website/app.py            # http://127.0.0.1:5000

Shows everything the detectors collected: who violated the rule, their captured
photo, how many members broke the rule on a given day, and the month history.
"""

from __future__ import annotations

import csv
import io
import os
import sys
import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from flask import (
    Flask,
    abort,
    Response,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from safety import database as db  # noqa: E402
from safety.config import (  # noqa: E402
    DATA_DIR,
    VIOLATION_COOLDOWN_SEC,
    VIOLATION_DIR,
    ensure_dirs,
)
from safety.detector import SafetyDetector  # noqa: E402
from safety.pipeline import SafetyMonitor  # noqa: E402

ensure_dirs()
db.init_db()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "construction-safety-monitor")
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024

ALLOWED_VIDEO = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
ALLOWED_IMAGE = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

JOBS: dict = {}
_MODEL = None
_MODEL_LOCK = threading.Lock()


def get_model():
    global _MODEL
    with _MODEL_LOCK:
        if _MODEL is None:
            _MODEL = SafetyDetector()
        return _MODEL


# --------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------- #
@app.template_filter("datestamp")
def datestamp(value: str) -> str:
    if not value:
        return "-"
    try:
        return datetime.fromisoformat(value).strftime("%d %b %Y %H:%M")
    except ValueError:
        return value


@app.template_filter("prettydate")
def prettydate(day: str) -> str:
    try:
        return datetime.strptime(day, "%Y-%m-%d").strftime("%a, %d %b %Y")
    except ValueError:
        return day


@app.template_filter("monthname")
def monthname(month: str) -> str:
    try:
        return datetime.strptime(month, "%Y-%m").strftime("%B %Y")
    except ValueError:
        return month


@app.context_processor
def inject_globals():
    months = db.month_list()
    return {
        "now": datetime.now(),
        "all_months": months,
        "VIOLATION_COOLDOWN_SEC": VIOLATION_COOLDOWN_SEC,
    }


def photo_path(row) -> Path | None:
    if not row or not row.get("image_path"):
        return None
    p = Path(row["image_path"])
    if p.exists():
        return p
    alt = VIOLATION_DIR / Path(row["image_path"]).name
    return alt if alt.exists() else None


# --------------------------------------------------------------------- #
# dashboard
# --------------------------------------------------------------------- #
@app.route("/")
def dashboard():
    stats = db.dashboard()
    return render_template("dashboard.html", stats=stats, active="dashboard")


@app.route("/api/dashboard")
def api_dashboard():
    return jsonify(db.dashboard())


@app.route("/api/series")
def api_series():
    limit = request.args.get("limit", 30, type=int)
    return jsonify(db.daily_series(limit))


# --------------------------------------------------------------------- #
# violations
# --------------------------------------------------------------------- #
@app.route("/violations")
def violations():
    day = request.args.get("day") or None
    month = request.args.get("month") or None
    kind = request.args.get("kind") or None
    source = request.args.get("source") or None
    page = max(1, request.args.get("page", 1, type=int))
    per_page = 24

    rows = db.list_violations(day=day, month=month, kind=kind, source=source,
                              limit=per_page, offset=(page - 1) * per_page)
    total = len(db.list_violations(day=day, month=month, kind=kind, source=source,
                                   limit=100000))
    title = "Violation records"
    if day:
        title = f"Violations on {day}"
    elif month:
        title = f"Violations in {month}"

    return render_template(
        "violations.html",
        rows=rows,
        total=total,
        page=page,
        per_page=per_page,
        title=title,
        filters={"day": day or "", "month": month or "", "kind": kind or "",
                 "source": source or ""},
        sources=[s["source"] for s in db.dashboard()["sources"]],
        active="violations",
    )


@app.route("/violations/<int:vid>")
def violation_detail(vid: int):
    row = db.get_violation(vid)
    if not row:
        abort(404)
    img = photo_path(row)
    return render_template("detail.html", row=row, img=img, active="violations")


@app.route("/photo/<int:vid>")
def photo(vid: int):
    row = db.get_violation(vid)
    if not row:
        abort(404)
    img = photo_path(row)
    if not img:
        abort(404)
    return send_file(img, mimetype="image/jpeg")


@app.route("/api/violations")
def api_violations():
    return jsonify(db.list_violations(
        day=request.args.get("day"),
        month=request.args.get("month"),
        kind=request.args.get("kind"),
        source=request.args.get("source"),
        limit=request.args.get("limit", 200, type=int),
        offset=request.args.get("offset", 0, type=int),
    ))


@app.route("/export.csv")
def export_csv():
    rows = db.list_violations(
        day=request.args.get("day"),
        month=request.args.get("month"),
        kind=request.args.get("kind"),
        limit=100000,
    )
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "day", "time", "kind", "helmet", "vest", "source",
                "track_id", "person_conf", "helmet_conf", "vest_conf",
                "captured_at", "image_path"])
    for r in rows:
        w.writerow([r["id"], r["day"], r["time"], r["kind"], r["helmet"], r["vest"],
                    r["source"], r["track_id"], r["person_conf"], r["helmet_conf"],
                    r["vest_conf"], r["captured_at"], r["image_path"]])
    data = buf.getvalue().encode()
    stamp = datetime.now().strftime("%Y%m%d")
    return Response(data, mimetype="text/csv", headers={
        "Content-Disposition": f"attachment; filename=violations_{stamp}.csv"})


# --------------------------------------------------------------------- #
# day report - "how many members violated the rules today"
# --------------------------------------------------------------------- #
@app.route("/day")
@app.route("/day/<day>")
def day_report(day: str = None):
    day = day or datetime.now().strftime("%Y-%m-%d")
    stats = db.day_stats(day)
    rows = db.list_violations(day=day, limit=500)

    members = {}
    for r in rows:
        key = r["track_id"] if r["track_id"] >= 0 else f"anon-{r['id']}"
        m = members.setdefault(key, {
            "member": f"Worker #{key}" if r["track_id"] >= 0 else f"Record #{r['id']}",
            "track_id": r["track_id"],
            "violations": 0,
            "helmet": 0,
            "vest": 0,
            "first": r["time"],
            "last": r["time"],
            "source": r["source"],
            "photo": r["id"],
            "ids": [],
        })
        m["violations"] += 1
        m["helmet"] += 1 if r["helmet"] == "missing" else 0
        m["vest"] += 1 if r["vest"] == "missing" else 0
        m["last"] = r["time"]
        m["ids"].append(r["id"])

    ordered = sorted(members.values(), key=lambda m: -m["violations"])

    yesterday = (datetime.strptime(day, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    tomorrow = (datetime.strptime(day, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

    return render_template(
        "day.html",
        day=day,
        stats=stats,
        rows=rows,
        members=ordered,
        yesterday=yesterday,
        tomorrow=tomorrow,
        active="day",
    )


# --------------------------------------------------------------------- #
# month history
# --------------------------------------------------------------------- #
@app.route("/history")
def history():
    return render_template("history.html", months=db.month_list(), active="history")


@app.route("/history/<month>")
def month_report(month: str):
    try:
        datetime.strptime(month, "%Y-%m")
    except ValueError:
        abort(404)
    data = db.month_summary(month)
    return render_template("month.html", month=month, data=data, active="history")


# --------------------------------------------------------------------- #
# repeat offenders
# --------------------------------------------------------------------- #
@app.route("/members")
def members():
    return render_template("members.html", members=db.unique_members(), active="members")


# --------------------------------------------------------------------- #
# upload & run phase 2
# --------------------------------------------------------------------- #
@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "GET":
        return render_template("upload.html", active="upload", job=None)

    file = request.files.get("file")
    if not file or not file.filename:
        flash("Please choose a video or image file.", "error")
        return redirect(url_for("upload"))

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_VIDEO | ALLOWED_IMAGE:
        flash(f"Unsupported file type: {suffix}", "error")
        return redirect(url_for("upload"))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(c for c in Path(file.filename).stem if c.isalnum() or c in "-_") or "upload"
    dest = DATA_DIR / "uploads" / f"{stamp}_{safe}{suffix}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    file.save(dest)

    strict = request.form.get("strict") == "on"
    cooldown = float(request.form.get("cooldown") or VIOLATION_COOLDOWN_SEC)
    label = request.form.get("label") or file.filename

    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"status": "running", "done": 0, "total": 0, "violations": 0,
                    "persons": 0, "error": None, "file": dest.name, "label": label}

    t = threading.Thread(target=_run_job, args=(job_id, dest, strict, cooldown, label),
                         daemon=True)
    t.start()
    return render_template("upload.html", active="upload", job=JOBS[job_id], job_id=job_id)


def _run_job(job_id: str, path: Path, strict: bool, cooldown: float, label: str):
    job = JOBS[job_id]
    try:
        import cv2

        monitor = SafetyMonitor(
            detector=get_model(),
            source=path.name,
            cooldown=cooldown,
            save_images=True,
            label=label,
        )
        is_image = path.suffix.lower() in ALLOWED_IMAGE
        cap = None
        if not is_image:
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise RuntimeError("cannot open video")
            job["total"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        frames, persons, viol = 0, 0, 0
        while True:
            if is_image:
                if frames >= 1:
                    break
                frame = cv2.imread(str(path))
                if frame is None:
                    raise RuntimeError("cannot read image")
            else:
                ok, frame = cap.read()
                if not ok:
                    break
            frames += 1
            analysis, new_rows = monitor.step(frame)
            persons += analysis.persons
            viol += len(new_rows)
            job.update(done=frames, persons=persons, violations=viol)

        if cap is not None:
            cap.release()
        monitor.close()
        job.update(status="done", done=frames, persons=persons, violations=viol)
    except Exception as exc:  # noqa: BLE001
        job.update(status="error", error=str(exc))


@app.route("/api/job/<job_id>")
def api_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        abort(404)
    return jsonify(job)


# --------------------------------------------------------------------- #
@app.route("/api/stats")
def api_stats():
    today = datetime.now().strftime("%Y-%m-%d")
    return jsonify(db.day_stats(today))


def main():
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 5000))
    print(f"[web] central safety monitor -> http://{host}:{port}")
    app.run(host=host, port=port, debug=os.environ.get("FLASK_DEBUG") == "1",
            threaded=True)


if __name__ == "__main__":
    main()
