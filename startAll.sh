#!/bin/bash

# This script starts the server and workers.
# Usage:
#   ./startAll.sh      # Runs normal server and worker
#   ./startAll.sh -c  # Runs compressed versions

alias python=python3

python --version > /dev/null 2>&1

if [ $? -ne 0 ]; then
    echo "python not found"
    exit 1
fi

# Install required Python packages
if [ -f "requirements.txt" ]; then
    echo "Installing required packages..."
    pip install -r requirements.txt
else
    echo "requirements.txt not found. Skipping package installation."
fi

# Determine whether to use compressed or Galore versions
if [[ "$1" == "-c" ]]; then
    SERVER_SCRIPT="server_compressed.py"
    WORKER_SCRIPT="worker_compressed.py"
    SERVER_LOG="./logs/server_compressed_log.txt"
    WORKER0_LOG="./logs/worker_compressed_log0.txt"
    WORKER1_LOG="./logs/worker_compressed_log1.txt"
    WORKER2_LOG="./logs/worker_compressed_log2.txt"
elif [[ "$1" == "-d" ]]; then
    SERVER_SCRIPT="server.py"
    WORKER_SCRIPT="dynamic_bound_loss/worker_trainer.py"
    SERVER_LOG=".logs/server_dynamic_bound_loss_log.txt"
    WORKER0_LOG="./logs/worker_dynamic_bound_loss_log0.txt"
    WORKER1_LOG="./logs/worker_dynamic_bound_loss_log1.txt"
    WORKER2_LOG="./logs/worker_dynamic_bound_loss_log2.txt"
else
    SERVER_SCRIPT="server.py"
    WORKER_SCRIPT="worker.py"
    SERVER_LOG="./logs/server_log.txt"
    WORKER0_LOG="./logs/worker_log0.txt"
    WORKER1_LOG="./logs/worker_log1.txt"
    WORKER2_LOG="./logs/worker_log2.txt"
fi

# Check if any .pkl files exist
if ls *.pkl > /dev/null 2>&1; then
    echo ".pkl files found."
else
    echo "No .pkl files found. Generating necessary pre-train data"
    python ./prepare_data.py
    echo "Pre-training data generated. Please re-run the script."
    exit 1
fi

# Create the logs directory if it doesn't exist
mkdir -p ./logs

# Start the server and redirect output to SERVER_LOG
python -u ./$SERVER_SCRIPT > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Function to kill the server process on exit 
cleanup() {
    echo "Cleaning up server process (PID: $SERVER_PID)..."
    kill -9 "$SERVER_PID" 2>/dev/null
}

# Set up trap to call cleanup on script exit (non-zero status)
trap cleanup EXIT

# Wait for the server to start
TIMEOUT=100
START_TIME=$(date +%s)
FOUND_MESSAGE=false

while true; do
    # Check if the server process is still running
    if ! ps -p $SERVER_PID > /dev/null; then
        echo "Server process failed or exited unexpectedly."
        exit 1
    fi

    # Check the log file for the expected message
    if grep -q "Server listening ..." "$SERVER_LOG"; then
        # Count the number of occurrences of the message
        COUNT=$(grep -c "Server listening ..." "$SERVER_LOG")
        if [ $COUNT -eq 1 ]; then
            FOUND_MESSAGE=true
            break
        else
            echo "Unexpected number of 'Server listening ...' messages found."
            exit 1
        fi
    fi

    # Check if the timeout has been reached
    CURRENT_TIME=$(date +%s)
    ELAPSED_TIME=$((CURRENT_TIME - START_TIME))
    if [ $ELAPSED_TIME -ge $TIMEOUT ]; then
        echo "Timeout reached. Server did not start successfully."
        exit 1
    fi

    # Wait for a short period before checking again
    sleep 1
done

sleep 10

# If the message was found, proceed
if $FOUND_MESSAGE; then
    echo "Server started successfully. Proceeding..."
    python -u ./$WORKER_SCRIPT 0 > "$WORKER0_LOG" 2>&1 &
    python -u ./$WORKER_SCRIPT 1 > "$WORKER1_LOG" 2>&1 &
    python -u ./$WORKER_SCRIPT 2 > "$WORKER2_LOG" 2>&1 &
else
    echo "Server did not start successfully."
    exit 1
fi


# Monitor logs for completion messages
WORKERS_DONE=0
WORKERS_FINISHED=()
while [ $WORKERS_DONE -lt 3 ]; do
    for WORKER_LOG in "$WORKER0_LOG" "$WORKER1_LOG" "$WORKER2_LOG"; do
        if [[ ! " ${WORKERS_FINISHED[@]} " =~ " $WORKER_LOG " ]] && tail -n 3 "$WORKER_LOG" | grep -q "Worker .* finished training."; then
            # echo "Training completion detected in $WORKER_LOG:"
            tail -n 1 "$WORKER_LOG"
            WORKERS_FINISHED+=("$WORKER_LOG")
            WORKERS_DONE=$((WORKERS_DONE + 1))
        fi
    done
    sleep 5
done

echo "All workers have finished training. Exiting."
exit 0