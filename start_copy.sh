#!/bin/bash
# DeepLearning Labs — startup script (optimized for main.py)
# Usage: ./start.sh [start|stop|restart|status|help] [port]
#   Default command: start
#   Default port: 5000 (as hardcoded in main.py, see note below)
# Usage: ./start.sh               (start on port 5000)
#        ./start.sh 8000          (start on port 8000)
#        ./start.sh stop
#        ./start.sh restart 8000  (restart on port 8000)
#        ./start.sh status

CMD="${1:-start}"
PORT="${2:-5000}"

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$APP_DIR/.server.pid"
LOG_FILE="$APP_DIR/server.log"
PYTHON="/data/wj/envs/deeplearning_labs/bin/python"

start() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "Server is already running (PID $(cat "$PID_FILE"))."
        exit 1
    fi

    echo "Starting DeepLearning Labs on port $PORT..."
    cd "$APP_DIR" || exit 1
    nohup "$PYTHON" main.py > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    echo "Started (PID $!). Logs: $LOG_FILE"
}

stop() {
    if [ ! -f "$PID_FILE" ]; then
        echo "No PID file found. Server not running."
        return 1
    fi

    PID=$(cat "$PID_FILE")
    echo "Stopping server (PID $PID)..."
    kill "$PID" 2>/dev/null && rm -f "$PID_FILE" && echo "Stopped." || echo "Failed to stop."
}

status() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "Server is running (PID $(cat "$PID_FILE"))."
    else
        echo "Server is not running."
    fi
}

case "$CMD" in
    start|restart)
        stop 2>/dev/null || true
        [ "$CMD" = "restart" ] && sleep 1
        start
        ;;
    stop)
        stop
        ;;
    status)
        status
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status} [port]"
        exit 1
        ;;
esac