#!/usr/bin/env bash
# Run the dashboard: FastAPI on :8000 + Vite on :5173.
#
#   ./dev.sh              start both, wait, print URLs (Ctrl-C stops both)
#   ./dev.sh start        start both in the background
#   ./dev.sh stop         stop both
#   ./dev.sh status       what is running
#
# The Gradio debug app is separate and managed by ./run.sh (port 7860).
# All three can run at once; they share the model only in the sense that each
# process loads its own copy, so expect ~1 GB extra RSS per Python process.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Local secrets (GROQ_API_KEY and friends) live in a git-ignored .env. Exported
# so the API process inherits them; absent file, nothing happens.
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$PROJECT_DIR/.env"
    set +a
fi
PYTHON="${PYTHON:-python3}"
API_PORT=8000
WEB_PORT=5173
API_PID="$PROJECT_DIR/.api.pid"
WEB_PID="$PROJECT_DIR/.web.pid"
API_LOG="$PROJECT_DIR/api.log"
WEB_LOG="$PROJECT_DIR/web.log"

alive() { [[ -f "$1" ]] && kill -0 "$(cat "$1")" 2>/dev/null; }

# Only LISTEN sockets count. A bare `lsof -ti` also matches the CLOSED client
# sockets editors leave behind after probing a port.
port_holder() { lsof -ti:"$1" -sTCP:LISTEN 2>/dev/null || true; }

start_api() {
    if alive "$API_PID"; then
        echo "API already running (PID $(cat "$API_PID"))"
        return 0
    fi
    rm -f "$API_PID"
    if [[ -n "$(port_holder "$API_PORT")" ]]; then
        echo "Port $API_PORT is busy (PID $(port_holder "$API_PORT")). Run './dev.sh stop'." >&2
        return 1
    fi
    echo "Starting API on :$API_PORT (model load takes ~20s)…"
    # Launched bare, not inside ( … & ): in `( cmd & echo $! )` the recorded PID
    # is the subshell's, not uvicorn's, which made stop/status track the wrong
    # process. </dev/null + disown fully detach it so this script can exit while
    # the server keeps running.
    #
    # No --reload either: it would reload a 79 MB checkpoint on every file save.
    cd "$PROJECT_DIR"
    nohup "$PYTHON" -u -m uvicorn api:app \
        --host 127.0.0.1 --port "$API_PORT" > "$API_LOG" 2>&1 < /dev/null &
    echo $! > "$API_PID"
    disown

    for _ in $(seq 1 45); do
        if curl -sf --max-time 2 -o /dev/null "http://127.0.0.1:$API_PORT/api/health"; then
            echo "  API ready → http://127.0.0.1:$API_PORT/api/health"
            return 0
        fi
        alive "$API_PID" || { echo "API failed to start:" >&2; tail -20 "$API_LOG" >&2; return 1; }
        sleep 1
    done
    echo "API started but not answering yet — check api.log" >&2
}

start_web() {
    if alive "$WEB_PID"; then
        echo "Web already running (PID $(cat "$WEB_PID"))"
        return 0
    fi
    rm -f "$WEB_PID"
    if [[ ! -d "$PROJECT_DIR/bird-frontend/node_modules" ]]; then
        echo "Installing frontend dependencies…"
        ( cd "$PROJECT_DIR/bird-frontend" && npm install )
    fi
    # Braced: bash 3.2 (the macOS system bash) mis-parses a multibyte character
    # sitting directly against a variable name, and under `set -u` that aborts
    # the script with "unbound variable".
    echo "Starting web on :${WEB_PORT}…"
    cd "$PROJECT_DIR/bird-frontend"
    nohup npm run dev > "$WEB_LOG" 2>&1 < /dev/null &
    echo $! > "$WEB_PID"
    disown
    cd "$PROJECT_DIR"

    for _ in $(seq 1 20); do
        # Vite binds localhost, which resolves to ::1 first on macOS — probe the
        # hostname rather than 127.0.0.1 or this never sees it come up.
        curl -sf --max-time 2 -o /dev/null "http://localhost:$WEB_PORT/" && break
        sleep 0.5
    done
    echo "  Dashboard → http://localhost:$WEB_PORT"
}

# Stop one service by PID, then make sure its port is actually free.
#
# Deliberately does NOT kill the process group. Both services are started from a
# single `dev.sh start`, so they share a group — killing it meant stopping the
# web server also silently killed the API. Instead: signal the recorded PID, and
# if something still holds the port (npm spawns vite as a child, so the listener
# is not the PID we recorded), signal that listener too.
stop_one() {
    local pidfile="$1" name="$2" port="$3"
    local pid=""
    [[ -f "$pidfile" ]] && pid="$(cat "$pidfile")"

    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        for _ in $(seq 1 10); do
            kill -0 "$pid" 2>/dev/null || break
            sleep 0.4
        done
        kill -9 "$pid" 2>/dev/null || true
        echo "$name stopped (PID $pid)."
    else
        echo "$name not running."
    fi
    rm -f "$pidfile"

    # Clean up an orphaned listener (a child that outlived its parent).
    local holder
    holder="$(port_holder "$port")"
    if [[ -n "$holder" ]]; then
        echo "  freeing port $port (PID $(echo "$holder" | tr '\n' ' '))"
        echo "$holder" | xargs kill 2>/dev/null || true
        sleep 1
        holder="$(port_holder "$port")"
        [[ -n "$holder" ]] && echo "$holder" | xargs kill -9 2>/dev/null || true
    fi
}

case "${1:-fg}" in
    start)
        start_api && start_web
        echo
        echo "Dashboard  http://localhost:$WEB_PORT"
        echo "API docs   http://127.0.0.1:$API_PORT/docs"
        echo "Logs       api.log · web.log"
        ;;
    stop)
        stop_one "$WEB_PID" "Web" "$WEB_PORT"
        stop_one "$API_PID" "API" "$API_PORT"
        ;;
    restart)
        "$0" stop
        "$0" start
        ;;
    status)
        alive "$API_PID" && echo "API running (PID $(cat "$API_PID")) → :$API_PORT" || echo "API not running."
        alive "$WEB_PID" && echo "Web running (PID $(cat "$WEB_PID")) → :$WEB_PORT" || echo "Web not running."
        ;;
    fg)
        start_api && start_web
        echo
        echo "Dashboard  http://localhost:$WEB_PORT"
        echo "Ctrl-C to stop both."
        trap '"$0" stop; exit 0' INT TERM
        # Idle until interrupted, surfacing either log if a process dies.
        while alive "$API_PID" && alive "$WEB_PID"; do sleep 2; done
        echo "A process exited — see api.log / web.log" >&2
        "$0" stop
        ;;
    *)
        echo "Usage: $0 {fg|start|stop|restart|status}" >&2
        exit 1
        ;;
esac
