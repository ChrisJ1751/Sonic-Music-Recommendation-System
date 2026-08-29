#!/bin/sh
# Boot sequence for the Sonic API container.
#
#   1. Make the mounted volume writable. Fly attaches volumes owned by root, so
#      an unprivileged process cannot write to one without this step.
#   2. Drop privileges to the `app` user via setpriv, which execs rather than
#      forking (so signals still reach the server directly).
#   3. Fetch + verify the 514 MiB EASE matrix. No-op when EASE_B_URL is unset,
#      or when the volume already holds a verified copy from an earlier boot.
#   4. exec uvicorn so it becomes PID 1 and receives SIGTERM directly, giving a
#      clean shutdown when Fly stops or suspends the machine.
set -eu

APP_USER=app
PORT="${PORT:-8000}"
CACHE_DIR="${EASE_B_CACHE_DIR:-/artifacts}"

if [ "$(id -u)" = "0" ]; then
    mkdir -p "$CACHE_DIR" /app/outputs/logs
    chown -R "$APP_USER:$APP_USER" "$CACHE_DIR" /app/outputs 2>/dev/null || true
    if command -v setpriv >/dev/null 2>&1; then
        exec setpriv --reuid="$APP_USER" --regid="$APP_USER" --init-groups "$0" "$@"
    fi
    echo "entrypoint | WARNING: setpriv unavailable; continuing as root" >&2
fi

python scripts/fetch_ease_b.py

# --no-access-log: the RequestLoggingMiddleware already emits one structured
# JSON line per request with more context than uvicorn's access line.
exec uvicorn api.main:app --host 0.0.0.0 --port "$PORT" --no-access-log
