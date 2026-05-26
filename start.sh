#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${OPENHOST_APP_DATA_DIR:-/data/app_data/code-completion}"
MODELS_DIR="${DATA_DIR}/models"
STATE_FILE="${DATA_DIR}/state.json"
mkdir -p "$MODELS_DIR"

# Initialize state file if it doesn't exist
if [ ! -f "$STATE_FILE" ]; then
    echo '{}' > "$STATE_FILE"
fi

export MODELS_DIR
export STATE_FILE

# Number of CPU threads for inference (default: all available minus 2)
N_THREADS="${LLM_THREADS:-$(( $(nproc) - 2 ))}"
if [ "$N_THREADS" -lt 1 ]; then
    N_THREADS=1
fi
export N_THREADS

# Context size
export CTX_SIZE="${LLM_CTX_SIZE:-4096}"

# Number of slots for concurrent requests
export N_SLOTS="${LLM_SLOTS:-2}"

echo "Starting code-completion server..."
echo "  Models dir: ${MODELS_DIR}"
echo "  Threads: ${N_THREADS}"
echo "  Context: ${CTX_SIZE}"
echo "  Slots: ${N_SLOTS}"

exec python3 /app/server.py
