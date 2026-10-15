# SiteGuard — Construction Site Safety Monitoring

**OpenCV + YOLOv8** watches a construction site, finds every worker, checks if
they are wearing a **helmet** and a **safety vest**, photographs anyone who is
not, and shows everything on a **central website**: who broke the rules, their
photo, how many people broke the rules today, and the whole month history.

```
PHASE 1  →  detect people              (phase1_detect.py)
PHASE 2  →  detect helmet / vest       (phase2_monitor.py)
PHASE 3  →  central website            (website/app.py)
```

---
---

# 🚀 Quick Start — clone and run

Requirements: a Windows PC, internet, and about 20 minutes the first time.
You do **not** need a graphics card or a webcam.

---

## 1. Install these two programs

| Program | Link | What to do |
|---------|------|------------|
| **Python 3.12** | <https://www.python.org/downloads/> | Download and run the installer. ⚠️ **On the FIRST screen, tick the box that says "Add python.exe to PATH"** before clicking Install. |
| **VS Code** | <https://code.visualstudio.com/> | Download and install. Then restart your computer once. |

---

## 2. Get the code onto your computer

1. Open **VS Code**.
2. Press `` Ctrl + ` `` (the key above `Tab`) to open the terminal at the
   bottom. This is where you will type all the commands.
3. Copy-paste these lines **one at a time** and press `Enter` after each:

```powershell
cd $HOME\Documents
git clone https://github.com/harishkumar2317/sitesafety-ai.git
cd sitesafety-ai
```

4. Check it worked — type this and press Enter:

```powershell
dir
```

You should see files like `phase1_detect.py`, `phase2_monitor.py`, `website`.
✅ If you see them, you are in the right place. ❌ If a command is not
recognized, copy the **exact** error message.

---

## 3. Make the virtual environment

Copy-paste these two lines, one at a time:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

⚠️ **If the second line gives an error** about scripts being disabled, run this
line **once**, then close the terminal, open a new one (`` Ctrl+` ``), and do
the second line again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

👉 **How to know it worked:** the start of your terminal line now shows `(.venv)`.
You must always keep that `(.venv)` there. If you open a new terminal later and
it is missing, just retype `.venv\Scripts\Activate.ps1`.

---

## 4. Install the libraries (this takes 5–15 minutes)

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

⚠️ It looks like nothing is happening for a long time. That is normal — it is
downloading about 1 GB. Wait until the terminal gives you the line back.

---

## 5. Download the AI models

```powershell
python tools/fetch_weights.py
```

Wait for the last two lines:

```
[weights] ppe_yolov8s.pt: 13 classes
[weights] yolov8n.pt: 80 classes
```

**You only ever do Steps 1–5 once.** From now on you only need Steps 6 and 7.

---
---

# ▶ Step 6 — Run the detection (Phase 1 & 2)

Open a **new terminal** (`` Ctrl+` ``, then the `+` button → *PowerShell*).

Always activate the environment first, in every new terminal:

```powershell
cd $HOME\Documents\sitesafety-ai
.venv\Scripts\Activate.ps1
```

### Option A — you have a webcam

```powershell
python phase2_monitor.py --source 0
```

A window opens showing the site. Green box = ok, **red box = no helmet or no
vest**. Press **`q`** to stop.

### Option B — no webcam (use the demo video)

First time only, make the demo video:

```powershell
python tools/make_demo_video.py
```

Then run it:

```powershell
python phase2_monitor.py --source data\uploads\demo_site_feed.mp4
```

Press **`q`** to stop.

> Phase 1 (people only, no helmet check) works the same way:
> `python phase1_detect.py --source 0`

Every time someone breaks a rule you will see a line like:

```
[phase2] VIOLATION track#4 kind=vest -> id=12 image=...\data\violations\20260927_225317_....jpg
```

The photo is saved in `data\violations\` and the record is saved in the database.

---
---

# 🌐 Step 7 — Run the website (Phase 3)

In the **same** activated terminal:

```powershell
python website/app.py
```

Then open your browser and go to:

### <http://127.0.0.1:5000>

| Page | What you will see |
|------|-------------------|
| **Dashboard** | Today's violations, a 30-day chart, latest photos |
| **Violations** | A grid of photos. **Tick the boxes** on the workers you want, then click *Open selected* to compare their photos and data side by side |
| **Today's Report** | How many members broke the rules today, one card per worker with their photo |
| **Month History** | Every month, then a day-by-day table and chart |
| **Repeat Offenders** | Workers ranked by how often they break rules |
| **Analyse Footage** | Upload a video/photo from the browser and it runs Phase 2 |

⚠️ **Important:** the website only shows data **after** you have run Step 6 at
least once. If it looks empty, that is why.

💡 Want the charts to look full straight away? Run this once, then refresh the
website:

```powershell
python tools/seed_demo_data.py
```

---
---

# 🛑 How to stop everything

| What | How |
|------|-----|
| Detection window (Step 6) | press **`q`** |
| Website (Step 7) | click the terminal and press **`Ctrl + C`** |
| Turn off the environment | type `deactivate` |

**The website stops the moment you press `Ctrl + C` or close VS Code.** Keep
that terminal open while you are looking at the website.

To start again another day, just open VS Code and run:

```powershell
cd $HOME\Documents\sitesafety-ai
.venv\Scripts\Activate.ps1
python website/app.py
```

---
---

# 🚀 How to let other people open your website

Only needed if you want your friends on the same Wi-Fi to see it.

1. Find your IP address. In the terminal type `ipconfig` and look for
   **IPv4 Address**, e.g. `192.168.1.20`.
2. Start the website like this instead:

```powershell
$env:HOST="0.0.0.0"
python website/app.py
```

3. Your friend opens `http://192.168.1.20:5000` in their browser.

⚠️ Windows Firewall may pop up — click **Allow** for Python on *Private networks*.

> For a normal demo or presentation you do **not** need this. Just running
> `python website/app.py` on your own laptop is enough.

---
---

# What the 3 phases do

| Phase | File | What it does |
|-------|------|--------------|
| **1** | `phase1_detect.py` | OpenCV reads the camera, YOLOv8 draws a box around every person |
| **2** | `phase2_monitor.py` | Checks each worker for helmet + vest. Photographs violators and saves them to the database |
| **3** | `website/app.py` | The central website — dashboard, violators + photos, day report, month history |

Per worker the system decides:

```
helmet = worn / missing / unknown
vest   = worn / missing / unknown
```

If either is `missing` → photo taken + record saved.

---
---

# 🧠 Tech stack & the AI — what we used and why

## The tech stack

| Layer | Technology | Why this one |
|-------|------------|--------------|
| **Language** | **Python 3.12** | The standard language for computer vision and AI. Every library we need works with it, and it is easy to read. |
| **Video / images / drawing** | **OpenCV (`cv2`)** | Grabs frames from the camera, draws the coloured boxes, saves the photos. It is the "eyes" of the system. |
| **AI detection** | **Ultralytics YOLOv8** | Finds the workers, helmets and vests in each frame. |
| **Deep-learning framework** | **PyTorch** | The framework YOLOv8 runs on (installed automatically). |
| **Web server** | **Flask** | A tiny Python web server — no Node.js, no build step. One language for the whole project. |
| **Web pages** | **HTML + Jinja + JavaScript** | Pages are rendered by Flask. JavaScript only draws the charts and handles the tick-to-select. |
| **Charts** | **Chart.js** | Bundled inside the project so the website works **offline**. |
| **Database** | **SQLite** (`data/safety.db`) | A real SQL database that is just one file — nothing to install, easy to back up or archive. |
| **Model download** | **Hugging Face Hub** | Where the pre-trained PPE weights come from. |

### Who does what

```
   camera / video
        │
        ▼
   ┌─────────┐  frames   ┌──────────┐
   │ OpenCV  │ ────────▶ │ YOLOv8 AI│   "person? helmet? vest?"
   └─────────┘           └──────────┘
        │                      │ boxes
        ▼                      ▼
   ┌────────────────────────────────┐
   │   Python logic (safety/)       │  track each worker, decide OK or VIOLATION
   └────────────────────────────────┘
        │ photo + record
        ▼
   ┌─────────┐   SQL    ┌───────┐
   │ SQLite  │ ───────▶ │ Flask │  your browser
   └─────────┘          └───────┘
```

OpenCV can move pixels but **cannot understand a scene**. The AI can recognise
objects but **cannot serve a web page**. Each tool does what it is good at.

---

## What AI is used?

We use **object detection** — a neural network that draws a box around every
object it finds and labels it. **Two** YOLOv8 models are used:

### 1. `yolov8n.pt` — the person detector (Phase 1)

| | |
|---|---|
| **What** | YOLOv8-nano, pre-trained on the **COCO** dataset (80 everyday objects) |
| **Size** | ~6 MB, 3.2 million parameters |
| **Job** | Draw a reliable full-body box around every human |

It is the *nano* (smallest) model on purpose — finding a person is an easy job,
so we save CPU time for the harder PPE model.

### 2. `ppe_yolov8s.pt` — the PPE detector (Phase 2)

| | |
|---|---|
| **What** | YOLOv8-small, fine-tuned on ~80,000 construction-site images |
| **Size** | ~22 MB, 11 million parameters |
| **Accuracy** | mAP@50 ≈ 0.77 |
| **Source** | `ayushgupta7777/safetyvision-yolov8` on Hugging Face |

It understands **13 classes**:

| Class | Meaning | Class | Meaning |
|-------|---------|-------|---------|
| `Person` | a human | `NO-Mask` | no face mask |
| **`Hardhat`** | **wearing a helmet** | **`NO-Hardhat`** | **no helmet** |
| **`Safety Vest`** | **wearing a vest** | **`NO-Safety Vest`** | **no vest** |
| `Gloves` / `NO-Gloves` | hand protection | `Goggles` / `NO-Goggles` | eye protection |
| `Mask` | face mask | `No_Harness` | no fall harness |
| `Fall-Detected` | a person has fallen | | |

**Bold** = the classes this project uses. The others are already inside the
model, so adding gloves / goggles / harness / fall detection later is only a few
lines in `safety/config.py` — no retraining needed.

### How the two models work together

```
one video frame
   │
   ├─▶ yolov8n      → person boxes (full body)
   │
   └─▶ ppe_yolov8s  → helmet / no-helmet / vest / no-vest boxes
                            │
                            ▼
          each PPE box is attached to the person whose
          HEAD area (top 34%) or TORSO area (22–80%) it sits inside
                            │
                            ▼
          helmet = worn | missing | unknown
          vest   = worn | missing | unknown
                            │
                            ▼
             if missing → photograph it + save to SQLite
```

---

## Why did we use AI at all?

### The problem with doing it by hand

| A human safety officer | This system |
|------------------------|-------------|
| Must stand on site all day | One camera, runs 24/7 unattended |
| Can only watch one place at a time | Every camera at once, on one dashboard |
| Hard to **prove** it happened | Automatic timestamped **photo evidence** |
| Counts written on paper, often lost | Instant records, charts and CSV export |
| Gets tired, takes breaks, gets distracted | Never gets tired |
| Does not **scale** — more sites need more officers | Add a camera, costs almost nothing |
| Only notices after the incident | Flags the worker **while they are on site** |

### Why not "just" normal OpenCV (no AI)?

| Classic method | Why it fails on a real construction site |
|----------------|------------------------------------------|
| **Haar cascades** | Only work on near-frontal faces. A helmet, a turned head or a face mask destroys it. |
| **HOG + SVM people detector** | Finds a person-shaped blob but says **nothing** about what they are wearing. |
| **Colour thresholding** ("find yellow helmet") | Breaks on shadows, dusk and dust. And construction sites are **full** of yellow and orange — cranes, barriers, machines. |
| **Contour analysis** | A worker behind a scaffold pole becomes three separate shapes. |
| **Skin-colour detection** | Fails on different skin tones, and on long sleeves. |

**In short:** classic CV needs hand-written rules for every condition, and a
construction site has endless conditions. A neural network **learns** what a
helmet looks like from tens of thousands of examples, so it still works with a
white or scratched orange helmet, a worker looking down, backlighting, dust,
rain, or a face mask covering half the head.

### Why YOLOv8 and not another AI model?

| Model | Speed | Accuracy | Why not chosen |
|-------|-------|----------|----------------|
| **YOLOv8** ✅ | very fast, one pass | very good | **chosen** — real-time on a CPU, easy to use |
| YOLOv5 | fast | good | replaced by YOLOv8 |
| Faster R-CNN | slow | excellent | ~10× slower — cannot do live video on a CPU |
| SSD MobileNet | fast | mediocre | much worse on small things like a distant helmet |
| DETR / transformers | slow | excellent | needs a GPU, overkill here |

YOLO stands for **"You Only Look Once"** — it looks at a frame **one single
time** and draws every box at once. That is what makes live video possible.

### Why this PPE model specifically?

Because the project is about spotting the **absence** of equipment, we needed a
model that has explicit **`NO-Hardhat`** and **`NO-Safety Vest`** classes — not
just "helmet present". This one is trained on ~80,000 real site images, reports
mAP@50 ≈ 0.77, and ships standard `best.pt` weights that load directly with no
conversion.

### Why two models instead of one?

The PPE model's `Person` class is unreliable — it often does not fire. The
`yolov8n` model is purpose-built for finding people and gives a clean
full-body box. Splitting the job gives better person boxes **and** better
helmet/vest decisions, and it keeps **Phase 1 a real, separate part of the
project** as the brief asked.

The cost is one extra forward pass: about **2 FPS on a normal laptop CPU with
no GPU**. For a safety monitor that is perfectly fine — it does not need 30 FPS.

### Where the AI stops

The AI only **detects and reports**. It never acts by itself:

- it does **not** sound an alarm or lock a gate or punish anyone
- every flagged photo goes to the **website for a human to review**
- it can be wrong — bad light, heavy occlusion, unusual clothing
- photos are evidence for a **safety conversation**, not a punishment system

---
---

# 🔧 If something goes wrong

| Problem | Fix |
|---------|-----|
| `'python' is not recognized` | You did not tick "Add to PATH" during the Python install. Reinstall Python and tick it. |
| `Running scripts is disabled` | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, close the terminal, open a new one. |
| `ModuleNotFoundError: No module named 'cv2'` | The environment is not active. Re-run `.venv\Scripts\Activate.ps1`, then `pip install -r requirements.txt` again. |
| No preview window opens | Make sure you typed `pip install -r requirements.txt` (not a headless version). |
| `cannot open camera` | You have no webcam, or another app is using it. Use Option B (demo video) instead. |
| Model file not found | `python tools/fetch_weights.py` |
| Website shows nothing | Run Phase 2 first, or `python tools/seed_demo_data.py`, then refresh. |
| `Address already in use` | Close the other terminal running the website. |
| Very slow detection | Normal on a CPU (~2 FPS). Use `--imgsz 384` to make it faster. |
| The `(.venv)` disappeared from the terminal | Re-run `.venv\Scripts\Activate.ps1` |
| `git` is not recognized | Install from <https://git-scm.com/download/win>, then restart your computer. |

**Still stuck?** Open a [GitHub issue](https://github.com/harishkumar2317/sitesafety-ai/issues)
and paste the **exact** error message plus which step you were on.

---
---

# 🗂️ Project structure

```
sitesafety-ai/
├── phase1_detect.py           # PHASE 1  person detection
├── phase2_monitor.py          # PHASE 2  helmet / vest + data collection
├── website/app.py             # PHASE 3  central website
├── requirements.txt           # list of libraries
├── weights/                   # the two AI models (downloaded)
├── assets/                    # sample site photos for testing
├── safety/                    # all the detection logic
├── website/templates/         # the web pages
├── website/static/            # CSS + Chart.js
├── tools/
│   ├── fetch_weights.py       # download the AI models
│   ├── make_demo_video.py     # make the demo video (no webcam needed)
│   ├── seed_demo_data.py      # optional fake history for the charts
│   └── smoke_test.py          # checks every website page works
├── .vscode/                   # ready-made Run buttons in VS Code
└── data/
    ├── safety.db              # the database
    └── violations/            # the evidence photos
```

### In VS Code you can also just press F5

Open the **Run and Debug** panel (`Ctrl+Shift+D`), pick one of these from the
dropdown, and press **F5** — no typing needed:

- `Phase 2 - Safety Monitor`
- `Phase 2 - Demo Video`
- `Phase 3 - Website`

---
---

# 📝 Extra notes

- **Track ids** only stay the same while one program run is going. That is what
  makes "unique people who broke the rules today" accurate. They reset when you
  restart the program.
- **Speed** — about 2 FPS on CPU. Use `--imgsz 416` for speed, `640` for better
  accuracy on far-away workers.
- **Reset everything** (delete all records and photos): `python tools/seed_demo_data.py --reset`
  then run Phase 2 again.
- The PPE model also understands gloves, goggles, masks, harnesses and falls —
  see `safety/detector.py` to extend it.

---

# 📄 Credits

- Detection: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (AGPL-3.0)
- PPE weights: `ayushgupta7777/safetyvision-yolov8` on Hugging Face
- Charts: Chart.js (MIT)

**Detection is an aid for human review, not an automatic punishment system.**
