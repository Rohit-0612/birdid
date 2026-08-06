#!/usr/bin/env bash
# Start/stop the Bird Identifier app.
#
#   ./run.sh start     launch in the background (logs to server.log)
#   ./run.sh stop      stop it
#   ./run.sh restart   stop then start
#   ./run.sh status    is it running?
#   ./run.sh fg        run in the foreground (Ctrl-C to quit)
#
# The old version was a bare `nohup ... &` with no PID tracking, so
# re-running it orphaned the previous process still holding port 7860.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP="$PROJECT_DIR/bird_complete_local.py"
PIDFILE="$PROJECT_DIR/.birdapp.pid"
LOGFILE="$PROJECT_DIR/server.log"
PORT=7860
PYTHON="${PYTHON:-python3}"

is_running() {
    [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null
}

start() {
    if is_running; then
        echo "Already running (PID $(cat "$PIDFILE")) → http://127.0.0.1:$PORT"
        return 0
    fi
    # Clear a stale pidfile, and warn if something else holds the port.
    rm -f "$PIDFILE"
    if lsof -ti:"$PORT" >/dev/null 2>&1; then
        echo "Port $PORT is already in use by PID $(lsof -ti:"$PORT" | tr '\n' ' ')" >&2
        echo "Run './run.sh stop' or free the port first." >&2
        return 1
    fi

    echo "Starting… (model load takes ~30s)"
    # -u keeps stdout unbuffered so server.log is useful while running
    nohup "$PYTHON" -u "$APP" > "$LOGFILE" 2>&1 &
    echo $! > "$PIDFILE"

    for _ in $(seq 1 60); do
        if curl -sf -o /dev/null "http://127.0.0.1:$PORT/"; then
            echo "Ready → http://127.0.0.1:$PORT  (PID $(cat "$PIDFILE"), logs: server.log)"
            return 0
        fi
        if ! is_running; then
            echo "Failed to start. Last lines of $LOGFILE:" >&2
            tail -20 "$LOGFILE" >&2
            rm -f "$PIDFILE"
            return 1
        fi
        sleep 2
    done
    echo "Started (PID $(cat "$PIDFILE")) but not responding yet — check server.log" >&2
}

stop() {
    if ! is_running; then
        echo "Not running."
        rm -f "$PIDFILE"
        return 0
    fi
    local pid
    pid="$(cat "$PIDFILE")"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 10); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.5
    done
    kill -9 "$pid" 2>/dev/null || true
    rm -f "$PIDFILE"
    echo "Stopped (PID $pid)."
}

case "${1:-start}" in
    start)   start ;;
    stop)    stop ;;
    restart) stop; start ;;
    status)
        if is_running; then
            echo "Running (PID $(cat "$PIDFILE")) → http://127.0.0.1:$PORT"
        else
            echo "Not running."
        fi ;;
    fg)      exec "$PYTHON" "$APP" ;;
    *)       echo "Usage: $0 {start|stop|restart|status|fg}" >&2; exit 1 ;;
esac
