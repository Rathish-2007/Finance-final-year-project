# Running RASTA-QF in VS Code — including "live" mode

Everything below assumes the project folder (`rasta_qf/`) is open in VS Code.

---

## 0 · One-time setup (5 minutes)

### 0.1 Prerequisites
- **Python 3.10+** (this project is tested on 3.13). Check: `python3 --version` (macOS/Linux) or `python --version` (Windows).
- **VS Code** with the **Python extension** (`ms-python.python`).
- Optional but nice: `batisteo.vscode-django` (template syntax highlighting) and `qwtel.sqlite-viewer` (browse `db.sqlite3`).

Install extensions from the integrated terminal:

```bash
code --install-extension ms-python.python
code --install-extension batisteo.vscode-django
code --install-extension qwtel.sqlite-viewer
```

### 0.2 Create and activate a virtual environment

**macOS / Linux**
```bash
cd rasta_qf
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
cd rasta_qf
py -m venv .venv
.\.venv\Scripts\Activate.ps1
# if blocked: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

In VS Code: **Ctrl/Cmd+Shift+P → "Python: Select Interpreter" → pick `.venv`**. The status bar should now show `.venv`.

### 0.3 Install dependencies and prepare the database

```bash
pip install -r requirements.txt
python manage.py migrate
python manage.py collectstatic --noinput
```

`pip install` pulls: django, gunicorn, whitenoise, numpy, pandas, scipy, scikit-learn, statsmodels, h5py.
On Windows skip nothing here — gunicorn installs fine, it just can't *run* there (see §2).

### 0.4 (Optional but recommended) pre-compute the evidence once

```bash
python manage.py run_pipeline            # all 6 modules, ~1 minute
```

This fills `db.sqlite3` so every page has real numbers on first load. You can also do this later from the **Mission Control** page with the **▶ Run Full Pipeline** button.

---

## 1 · Run it — the three modes

### Mode A · Dev server with auto-reload (best for editing)

```bash
python manage.py runserver 0.0.0.0:8000
```

Open **http://localhost:8000**. Every code/template save hot-reloads.

> With `DEBUG=0` (the default) WhiteNoise serves the collected static files — that's why step 0.3 runs `collectstatic`. If you edit CSS/JS, re-run `collectstatic` or start with `DEBUG=1`.

### Mode B · Production-style, one command (best for "this is a real website")

**macOS / Linux**
```bash
./start.sh                 # deps → migrate → collectstatic → gunicorn :8000
PORT=9000 ./start.sh       # custom port
DEBUG=1 ./start.sh         # same script, falls back to the dev server
```

**Windows** (gunicorn has no Windows support — use Waitress, a production WSGI server)
```powershell
pip install waitress
python manage.py migrate
python manage.py collectstatic --noinput
waitress-serve --listen=0.0.0.0:8000 rastaqf.wsgi:application
```
Or simply `python manage.py runserver 0.0.0.0:8000` — on Windows that is the practical equivalent.

### Mode C · Debug from inside VS Code (F5, breakpoints in views/services)

Create `.vscode/launch.json` in the project (VS Code can generate it: Run & Debug → *create a launch.json file* → Python → Django), then use:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "RASTA-QF: Django server",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/manage.py",
      "args": ["runserver", "0.0.0.0:8000"],
      "django": true,
      "justMyCode": false,
      "env": { "DJANGO_DEBUG": "1" }
    },
    {
      "name": "RASTA-QF: run pipeline headless",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/manage.py",
      "args": ["run_pipeline"],
      "django": true,
      "justMyCode": false
    }
  ]
}
```

Press **F5** → server starts under the debugger. Breakpoints in `core/views.py`, `core/services/*.py` and the Django template context all work.

---

## 2 · Making it *look live*

The site already behaves like a live product; here's how to lean into it:

1. **Leave the server running in the VS Code terminal** (Mode A or B). Keep that terminal open — it streams every request in real time (the access log *is* the "live" feel). Use the **Split Terminal** (`Ctrl/Cmd+Shift+5`) if you want a second one for commands.

2. **Watch the pipeline run live.** Open **Mission Control** (`/console`) and click **▶ Run Full Pipeline**. You'll see per-module progress bars fill, the status dots flip to *done*, and the console log stream in — all polled from `/api/status`. Takes ~15–60 s.

3. **The Live Monitor auto-refreshes.** The homepage now shows a **pulsing green LIVE dot with a ticking `updated HH:MM:SS` clock** and re-fetches all six module result sets **every 30 seconds**, re-rendering whichever tab you're on. Hit **⟳ auto-refresh: on/off** in the header to toggle it. Run the pipeline in another tab and watch the homepage numbers change on their own.

4. **Port-forward if you want it off your machine.** Add to the VS Code **Ports** panel (or use the CLI) to expose `8000` publicly:
   ```bash
   # VS Code will prompt "Open in Browser / Forward" for :8000 automatically
   ```
   On a headless box: `ssh -L 8000:localhost:8000 user@host` then browse `http://localhost:8000`.

5. **Make it *actually* update daily (true live).** Schedule the pipeline and the pages will always show today's numbers:

   **Linux/macOS (cron, 18:30 IST)**
   ```bash
   crontab -e
   # add:
   30 18 * * * cd /path/to/rasta_qf && .venv/bin/python manage.py run_pipeline >> logs/pipeline.log 2>&1
   ```

   **Windows (Task Scheduler)**
   ```powershell
   schtasks /create /tn "RASTA-QF nightly" /sc daily /st 18:30 ^
     /tr "cmd /c cd /d C:\path\to\rasta_qf && .venv\Scripts\python.exe manage.py run_pipeline"
   ```

   For genuinely streaming tick data you'd swap `data_lake.build_dataset(use_yfinance=True)` and run the fetch step intraday — the dashboard, API and studio all consume whatever the modules write.

6. **Demo script for a supervisor/viva.** Screens are already live: ① homepage KPI strip + signal feed → ② Connectedness tab → ③ Mission Control, hit **Run Full Pipeline**, narrate the progress bars → ④ Backtesting Studio, drag the friction slider from 0 → 60 bps and let the ranking flip itself.

---

## 3 · Useful URLs while it runs

| URL | What it is |
|---|---|
| `http://localhost:8000/` | Live Monitor (8 tabs, auto-refresh) |
| `/console` | Mission Control — pipeline runner, logs |
| `/docs`, `/report`, `/blog`, `/apidocs` | Paper, live report, regime log, API reference |
| `/health` | JSON liveness probe |
| `/api/results/drl` | Raw JSON the charts read (great for debugging) |
| `/api/benchmark/download` | `RASTA_QF_Benchmark.h5` |

---

## 4 · Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'django'` | The venv isn't active — re-run activation, and confirm the VS Code interpreter is `.venv`. |
| Pages render but numbers say "no results yet" | Run `python manage.py run_pipeline` (or the UI button). |
| CSS/JS missing after editing | `python manage.py collectstatic --noinput` (or run with `DEBUG=1`). |
| `Permission denied: ./start.sh` | `chmod +x start.sh` (macOS/Linux), then `./start.sh`. |
| Port 8000 busy | `PORT=9000 ./start.sh`, or `python manage.py runserver 9000`. |
| Windows: `gunicorn` errors | Expected — gunicorn is POSIX-only. Use `runserver` or Waitress (§1 Mode B). |
| Edits to a running gunicorn don't apply | Gunicorn never auto-reloads: stop it (Ctrl+C) and re-run `./start.sh`. The dev server (`runserver`) does auto-reload. |
| Want a clean slate | Delete `db.sqlite3`, then `python manage.py migrate && python manage.py run_pipeline`. |
EOF
