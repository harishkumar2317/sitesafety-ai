"""SQLite storage for violation records, sessions and reporting queries."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .config import DB_PATH, ensure_dirs

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source      TEXT    NOT NULL,
    started_at  TEXT    NOT NULL,
    ended_at    TEXT,
    frames      INTEGER DEFAULT 0,
    persons     INTEGER DEFAULT 0,
    violations  INTEGER DEFAULT 0,
    label       TEXT    DEFAULT ''
);

CREATE TABLE IF NOT EXISTS violations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER,
    track_id    INTEGER NOT NULL DEFAULT -1,
    kind        TEXT    NOT NULL,
    helmet      TEXT    NOT NULL,
    vest        TEXT    NOT NULL,
    helmet_conf REAL    DEFAULT 0,
    vest_conf   REAL    DEFAULT 0,
    person_conf REAL    DEFAULT 0,
    source      TEXT    NOT NULL,
    captured_at TEXT    NOT NULL,
    day         TEXT    NOT NULL,
    month       TEXT    NOT NULL,
    time        TEXT    NOT NULL,
    image_path  TEXT,
    box_x1      INTEGER DEFAULT 0,
    box_y1      INTEGER DEFAULT 0,
    box_x2      INTEGER DEFAULT 0,
    box_y2      INTEGER DEFAULT 0,
    note        TEXT    DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_violations_day  ON violations(day);
CREATE INDEX IF NOT EXISTS idx_violations_month ON violations(month);
CREATE INDEX IF NOT EXISTS idx_violations_kind ON violations(kind);
"""


@contextmanager
def connect(path: Path = DB_PATH):
    ensure_dirs()
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(path: Path = DB_PATH) -> None:
    with connect(path) as conn:
        conn.executescript(SCHEMA)


# --------------------------------------------------------------------- #
# writes
# --------------------------------------------------------------------- #
def start_session(source: str, label: str = "", path: Path = DB_PATH) -> int:
    now = datetime.now()
    with connect(path) as conn:
        cur = conn.execute(
            "INSERT INTO sessions(source, started_at, label) VALUES (?,?,?)",
            (source, now.isoformat(timespec="seconds"), label),
        )
        return int(cur.lastrowid)


def finish_session(session_id: int, frames: int, persons: int, violations: int,
                   path: Path = DB_PATH) -> None:
    with connect(path) as conn:
        conn.execute(
            "UPDATE sessions SET ended_at=?, frames=?, persons=?, violations=? WHERE id=?",
            (datetime.now().isoformat(timespec="seconds"), frames, persons, violations, session_id),
        )


def add_violation(
    kind: str,
    helmet: str,
    vest: str,
    source: str,
    image_path: Optional[str] = None,
    track_id: int = -1,
    helmet_conf: float = 0.0,
    vest_conf: float = 0.0,
    person_conf: float = 0.0,
    box=None,
    session_id: Optional[int] = None,
    note: str = "",
    when: Optional[datetime] = None,
    path: Path = DB_PATH,
) -> int:
    when = when or datetime.now()
    with connect(path) as conn:
        cur = conn.execute(
            """INSERT INTO violations
               (session_id, track_id, kind, helmet, vest, helmet_conf, vest_conf,
                person_conf, source, captured_at, day, month, time, image_path,
                box_x1, box_y1, box_x2, box_y2, note)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                session_id, track_id, kind, helmet, vest,
                round(helmet_conf, 4), round(vest_conf, 4), round(person_conf, 4),
                source, when.isoformat(timespec="seconds"),
                when.strftime("%Y-%m-%d"), when.strftime("%Y-%m"),
                when.strftime("%H:%M:%S"), image_path,
                int(box[0]) if box else 0, int(box[1]) if box else 0,
                int(box[2]) if box else 0, int(box[3]) if box else 0,
                note,
            ),
        )
        return int(cur.lastrowid)


# --------------------------------------------------------------------- #
# reads / reporting
# --------------------------------------------------------------------- #
def _rows(sql: str, params: Iterable = (), path: Path = DB_PATH) -> List[Dict]:
    with connect(path) as conn:
        return [dict(r) for r in conn.execute(sql, tuple(params)).fetchall()]


def day_stats(day: str, path: Path = DB_PATH) -> Dict:
    total = _rows("SELECT COUNT(*) c FROM violations WHERE day=?", (day,), path)[0]["c"]
    helmet = _rows(
        "SELECT COUNT(*) c FROM violations WHERE day=? AND helmet='missing'", (day,), path
    )[0]["c"]
    vest = _rows(
        "SELECT COUNT(*) c FROM violations WHERE day=? AND vest='missing'", (day,), path
    )[0]["c"]
    members = _rows(
        "SELECT COUNT(DISTINCT source || ':' || track_id) c FROM violations "
        "WHERE day=? AND track_id>=0",
        (day,), path,
    )[0]["c"]
    if members == 0:
        members = _rows(
            "SELECT COUNT(DISTINCT id) c FROM violations WHERE day=?", (day,), path
        )[0]["c"]
    sources = _rows("SELECT COUNT(DISTINCT source) c FROM violations WHERE day=?", (day,), path)[0]["c"]
    return {
        "day": day,
        "violations": total,
        "helmet_missing": helmet,
        "vest_missing": vest,
        "members": members,
        "sources": sources,
    }


def daily_series(limit: int = 30, path: Path = DB_PATH) -> List[Dict]:
    return _rows(
        """SELECT day,
                  COUNT(*) AS violations,
                  SUM(CASE WHEN helmet='missing' THEN 1 ELSE 0 END) AS helmet_missing,
                  SUM(CASE WHEN vest='missing'   THEN 1 ELSE 0 END) AS vest_missing,
                  COUNT(DISTINCT CASE WHEN track_id>=0 THEN source||':'||track_id END) AS members
           FROM violations GROUP BY day ORDER BY day DESC LIMIT ?""",
        (limit,), path,
    )


def month_summary(month: str, path: Path = DB_PATH) -> Dict:
    stats = _rows(
        """SELECT COUNT(*) violations,
                  SUM(CASE WHEN helmet='missing' THEN 1 ELSE 0 END) helmet_missing,
                  SUM(CASE WHEN vest='missing'   THEN 1 ELSE 0 END) vest_missing,
                  COUNT(DISTINCT CASE WHEN track_id>=0 THEN source||':'||track_id END) members,
                  COUNT(DISTINCT day) active_days
           FROM violations WHERE month=?""",
        (month,), path,
    )[0]
    per_day = _rows(
        """SELECT day, COUNT(*) violations,
                  COUNT(DISTINCT CASE WHEN track_id>=0 THEN source||':'||track_id END) members
           FROM violations WHERE month=? GROUP BY day ORDER BY day""",
        (month,), path,
    )
    by_kind = _rows(
        "SELECT kind, COUNT(*) c FROM violations WHERE month=? GROUP BY kind ORDER BY c DESC",
        (month,), path,
    )
    return {
        "month": month,
        "violations": stats["violations"] or 0,
        "helmet_missing": stats["helmet_missing"] or 0,
        "vest_missing": stats["vest_missing"] or 0,
        "members": stats["members"] or 0,
        "active_days": stats["active_days"] or 0,
        "days": per_day,
        "kinds": by_kind,
    }


def month_list(path: Path = DB_PATH) -> List[Dict]:
    return _rows(
        """SELECT month, COUNT(*) violations,
                  COUNT(DISTINCT CASE WHEN track_id>=0 THEN source||':'||track_id END) members,
                  COUNT(DISTINCT day) active_days,
                  MIN(day) first_day, MAX(day) last_day
           FROM violations GROUP BY month ORDER BY month DESC""",
        (), path,
    )


def list_violations(
    day: Optional[str] = None,
    month: Optional[str] = None,
    kind: Optional[str] = None,
    source: Optional[str] = None,
    limit: int = 200,
    offset: int = 0,
    path: Path = DB_PATH,
) -> List[Dict]:
    sql = "SELECT * FROM violations WHERE 1=1"
    params: List = []
    if day:
        sql += " AND day=?"
        params.append(day)
    if month:
        sql += " AND month=?"
        params.append(month)
    if kind:
        if kind == "helmet":
            sql += " AND helmet='missing'"
        elif kind == "vest":
            sql += " AND vest='missing'"
        else:
            sql += " AND kind=?"
            params.append(kind)
    if source:
        sql += " AND source=?"
        params.append(source)
    sql += " ORDER BY captured_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    return _rows(sql, params, path)


def get_violation(vid: int, path: Path = DB_PATH) -> Optional[Dict]:
    rows = _rows("SELECT * FROM violations WHERE id=?", (vid,), path)
    return rows[0] if rows else None


def dashboard(path: Path = DB_PATH) -> Dict:
    today = datetime.now().strftime("%Y-%m-%d")
    totals = _rows(
        """SELECT COUNT(*) violations,
                  COUNT(DISTINCT CASE WHEN track_id>=0 THEN source||':'||track_id END) members,
                  COUNT(DISTINCT day) active_days,
                  MIN(day) first_day, MAX(day) last_day
           FROM violations""", (), path
    )[0]
    kinds = _rows(
        """SELECT CASE WHEN helmet='missing' AND vest='missing' THEN 'helmet+vest'
                       WHEN helmet='missing' THEN 'helmet'
                       ELSE 'vest' END AS kind,
                  COUNT(*) c
           FROM violations GROUP BY kind ORDER BY c DESC""", (), path
    )
    sessions = _rows(
        "SELECT id, source, started_at, frames, persons, violations, label "
        "FROM sessions ORDER BY id DESC LIMIT 10", (), path
    )
    sources = _rows(
        "SELECT source, COUNT(*) c FROM violations GROUP BY source ORDER BY c DESC", (), path
    )
    recent = _rows("SELECT * FROM violations ORDER BY captured_at DESC LIMIT 8", (), path)
    return {
        "today": day_stats(today, path),
        "totals": totals,
        "kinds": kinds,
        "series": daily_series(30, path),
        "sessions": sessions,
        "sources": sources,
        "months": month_list(path),
        "recent": recent,
    }


def unique_members(path: Path = DB_PATH) -> List[Dict]:
    return _rows(
        """SELECT source||':'||track_id AS member,
                  COUNT(*) violations,
                  MIN(day) first_day, MAX(day) last_day,
                  SUM(CASE WHEN helmet='missing' THEN 1 ELSE 0 END) helmet_hits,
                  SUM(CASE WHEN vest='missing' THEN 1 ELSE 0 END) vest_hits
           FROM violations WHERE track_id>=0
           GROUP BY member ORDER BY violations DESC""", (), path
    )


def purge(days: int = 90, path: Path = DB_PATH) -> int:
    from datetime import timedelta

    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    with connect(path) as conn:
        cur = conn.execute("DELETE FROM violations WHERE day < ?", (cutoff,))
        return cur.rowcount
