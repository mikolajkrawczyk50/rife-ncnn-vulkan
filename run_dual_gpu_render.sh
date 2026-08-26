#!/usr/bin/env bash
set -e

REPO_ROOT="/home/user/repos/rife-ncnn-vulkan"
GLVK_LIB="/home/user/repos/glvk/build"
OUTPUT_DIR="$REPO_ROOT/outputs/render_full_inputs"
INPUT_DIR="$REPO_ROOT/inputs"
MODEL_DIR="$REPO_ROOT/models/rife-v4.6"
PORT=19200

export LD_LIBRARY_PATH="$GLVK_LIB:$LD_LIBRARY_PATH"

mkdir -p "$OUTPUT_DIR"

echo "========================================================"
echo " Starting Dual-GPU Distributed RIFE v4.6 Render on Inputs"
echo " Master Port : $PORT"
echo " Model       : $MODEL_DIR"
echo " Inputs      : $INPUT_DIR (302 frames @ 1080p)"
echo " Outputs     : $OUTPUT_DIR"
echo " Workers     : GT 730 (GLVK_DEVICE=0) + HD 4600 (GLVK_DEVICE=1)"
echo "========================================================"

# Launch Master
"$REPO_ROOT/build/rife-ncnn-vulkan-master" \
    -p "$PORT" \
    -m "$MODEL_DIR" \
    -i "$INPUT_DIR" \
    -o "$OUTPUT_DIR" \
    -v &
MASTER_PID=$!

sleep 1

# Launch Worker 1 on GT 730
GLVK_DEVICE=0 "$REPO_ROOT/build/rife-ncnn-vulkan-worker" \
    -c "127.0.0.1:$PORT" \
    -g 0 &
WORKER1_PID=$!

# Launch Worker 2 on HD 4600
GLVK_DEVICE=1 "$REPO_ROOT/build/rife-ncnn-vulkan-worker" \
    -c "127.0.0.1:$PORT" \
    -g 0 &
WORKER2_PID=$!

cleanup() {
    echo "Stopping workers and master..."
    kill -TERM $WORKER1_PID 2>/dev/null || true
    kill -TERM $WORKER2_PID 2>/dev/null || true
    kill -TERM $MASTER_PID 2>/dev/null || true
    wait $WORKER1_PID 2>/dev/null || true
    wait $WORKER2_PID 2>/dev/null || true
    wait $MASTER_PID 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Wait for master to finish
wait $MASTER_PID

echo "Master completed! Stopping worker daemons..."
kill -TERM $WORKER1_PID 2>/dev/null || true
kill -TERM $WORKER2_PID 2>/dev/null || true
wait $WORKER1_PID 2>/dev/null || true
wait $WORKER2_PID 2>/dev/null || true

echo "Dual-GPU render finished successfully!"
