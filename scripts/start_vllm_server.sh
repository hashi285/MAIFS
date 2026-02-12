#!/bin/bash
#
# MAIFS - vLLM Server Startup Script
# Qwen 30B-A3B-Thinking with configurable tensor parallel
# Default GPU allocation avoids GPU 0 (primary GPU)
#
# Note: TP size must be compatible with the selected model architecture.
# Set TENSOR_PARALLEL_SIZE explicitly for your GPU layout.

# Configuration
MODEL_NAME="${MODEL_NAME:-$HOME/models/qwen3-30b-a3b-thinking-2507}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"

# GPU allocation: default to GPU 0 for single-GPU setups
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# V100 (compute capability 7.0) cannot use FlashAttention v2.
# Force a compatible attention backend unless the user overrides it.
if [ -z "${ATTENTION_BACKEND:-}" ]; then
    FIRST_GPU="${CUDA_VISIBLE_DEVICES%%,*}"
    FIRST_CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i "$FIRST_GPU" 2>/dev/null | head -1)"
    CC_MAJOR="${FIRST_CC%%.*}"
    if [ -n "$CC_MAJOR" ] && [ "$CC_MAJOR" -lt 8 ]; then
        ATTENTION_BACKEND="TRITON_ATTN"
    fi
fi
ATTENTION_ARGS=()
if [ -n "${ATTENTION_BACKEND:-}" ]; then
    ATTENTION_ARGS=(--attention-backend "$ATTENTION_BACKEND")
fi

echo "============================================"
echo "MAIFS vLLM Server"
echo "============================================"
echo "Model: $MODEL_NAME"
echo "Tensor Parallel: $TENSOR_PARALLEL_SIZE GPUs"
echo "GPU Allocation: $CUDA_VISIBLE_DEVICES (LLM)"
if [ -n "${ATTENTION_BACKEND:-}" ]; then
    echo "Attention Backend: $ATTENTION_BACKEND"
fi
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
# Note: vLLM 0.14.0 uses client-side JSON schema validation via extra_body
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --trust-remote-code \
    "${ATTENTION_ARGS[@]}" \
    --enable-chunked-prefill \
    --max-num-batched-tokens 32768 \
    --disable-log-requests \
    "$@"
