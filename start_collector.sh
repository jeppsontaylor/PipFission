#!/bin/bash

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
INTERPRETER="$SCRIPT_DIR/venv/bin/python"
COLLECTOR="$SCRIPT_DIR/oanda_collector.py"
LOG_FILE="$SCRIPT_DIR/collector.log"
PID_FILE="$SCRIPT_DIR/collector.pid"

# Default instruments if none provided
INSTRUMENTS=${1:-"EUR_USD"}

echo "Starting OANDA collector for $INSTRUMENTS in background..."

# Kill existing if running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p $OLD_PID > /dev/null; then
        echo "Stopping existing collector (PID $OLD_PID)..."
        kill $OLD_PID
        sleep 2
    fi
fi

# Run with nohup
# --v20-books enables the richest data
# --gzip can be added if you want smaller files
# Run in a loop for robustness (Supervisor Mode)
nohup bash -c "while true; do 
    echo \"\$(date -u) Starting Collector...\"
    $INTERPRETER $COLLECTOR --instruments \"$INSTRUMENTS\" --env practice --v20-books --v20-poll 3.0 --flush-every 1
    EXIT_CODE=\$?
    echo \"\$(date -u) Collector exited with code \$EXIT_CODE. Restarting in 5s...\"
    sleep 5
done" >> "$LOG_FILE" 2>&1 &

NEW_PID=$!
echo $NEW_PID > "$PID_FILE"

echo "Collector started with PID $NEW_PID"
echo "Log file: $LOG_FILE"
echo "To view logs, run: tail -f collector.log"
