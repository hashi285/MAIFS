#!/bin/bash
#
# MAIFS - vLLM Server Startup Script
# Qwen 30B with 4 GPU Tensor Parallel
#

# Configuration
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-32B-Instruct}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"

echo "============================================"
echo "MAIFS vLLM Server"
echo "============================================"
echo "Model: $MODEL_NAME"
echo "Tensor Parallel: $TENSOR_PARALLEL_SIZE GPUs"
echo "Port: $PORT"
echo "Max Context Length: $MAX_MODEL_LEN"
echo "============================================"

# Check CUDA availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. CUDA not available."
    exit 1
fi

echo "Available GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "============================================"

# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --trust-remote-code \
    --enable-chunked-prefill \
    --max-num-batched-tokens 32768 \
    --guided-decoding-backend outlines \
    --disable-log-requests \
    "$@"
