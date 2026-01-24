#!/usr/bin/env bash
set -euo pipefail

export HF_HUB_ENABLE_HF_TRANSFER=1

python - <<'PY'
from huggingface_hub import snapshot_download

models = [
    "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
]

for model_id in models:
    print(f"\nDownloading {model_id} ...", flush=True)
    path = snapshot_download(repo_id=model_id, token=True)
    print(f"Downloaded to: {path}", flush=True)

print("All downloads complete.")
PY
