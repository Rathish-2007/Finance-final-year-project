#!/usr/bin/env bash
# ============================================================
#  RASTA-QF framework — one-command website launcher
#  Usage:  ./start.sh            (production, gunicorn, :8000)
#          PORT=9000 ./start.sh  (custom port)
#          DEBUG=1 ./start.sh    (Django dev server + reload)
# ============================================================
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
PORT=${PORT:-8000}

echo "▸ [1/4] Checking dependencies…"
if ! $PY -c "import django, gunicorn, whitenoise, numpy, pandas, scipy, sklearn, statsmodels, h5py" 2>/dev/null; then
    echo "  installing missing packages (requirements.txt)…"
    $PY -m pip install --quiet -r requirements.txt
fi

echo "▸ [2/4] Preparing database…"
$PY manage.py migrate --noinput --verbosity 0

echo "▸ [3/4] Collecting static assets…"
$PY manage.py collectstatic --noinput --verbosity 0

echo "▸ [4/4] Starting RASTA-QF website on http://0.0.0.0:${PORT} …"
if [ "${DEBUG:-0}" = "1" ]; then
    exec $PY manage.py runserver 0.0.0.0:"$PORT"
else
    exec $PY -m gunicorn rastaqf.wsgi:application \
        --bind 0.0.0.0:"$PORT" \
        --workers 2 \
        --threads 4 \
        --timeout 300 \
        --access-logfile - --error-logfile -
fi
