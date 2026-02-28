#!/bin/bash
# Watch for GPU to become free, then launch symbiogenesis evolution.
# Usage: nohup bash gpu_watcher.sh &

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$SCRIPT_DIR/symbiogenesis_run.log"

echo "[$(date)] GPU watcher started. Waiting for GPU to be free..." | tee "$LOG"

while true; do
    # Check if any process is using the GPU
    GPU_PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)

    if [ "$GPU_PROCS" -eq 0 ] || [ "$GPU_MEM" -lt 500 ]; then
        echo "[$(date)] GPU is free! (${GPU_MEM}MiB used, ${GPU_PROCS} processes)" | tee -a "$LOG"
        echo "[$(date)] Launching symbiogenesis evolution..." | tee -a "$LOG"
        cd "$SCRIPT_DIR"
        python3 run_symbiogenesis_gemma.py >> "$LOG" 2>&1
        echo "[$(date)] Run complete. Exit code: $?" | tee -a "$LOG"
        exit 0
    fi

    # Check every 60 seconds
    sleep 60
done
