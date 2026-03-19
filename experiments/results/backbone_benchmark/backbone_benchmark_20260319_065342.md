# Backbone Benchmark

- Timestamp: `20260319_065342`
- Hardware: CPU `AMD EPYC 9354 32-Core Processor`, GPU `NVIDIA H200 NVL`
- Torch: `2.10.0+cu128`
- Batch: `1`
- Scope: forward-only latency (preprocess / disk I/O / host-device copy excluded)

## Size

| Model | Params (M) | Trainable (M) | State Dict (MiB) | Checkpoint (MiB) | Note |
|---|---:|---:|---:|---:|---|
| mobilenetv2_dualstream | 5.7656 | 5.7656 | 22.255 | 22.486 | Track1 MobileNetV2 dual-stream (RGB + SRM residual) |

## Latency

| Model | GPU mean / p50 / p95 (ms) | GPU FPS | GPU peak MiB | CPU mean / p50 / p95 (ms) | CPU FPS | Input |
|---|---:|---:|---:|---:|---:|---|
| mobilenetv2_dualstream | 19.359 / 17.015 / 24.952 | 51.654 | 66.435 | 37.302 / 37.18 / 38.275 | 26.808 | `[1, 3, 224, 224]` |

## Notes

- `mobilenetv2_dualstream` is the actual Track-1 noise backbone used by `run_phase3_tracks.py`.
- `mobilenetv2_proxy` is a legacy single-stream reference kept only for historical comparison.
- `checkpoint_mib` reflects the on-disk checkpoint file currently used in the repo.
- `state_dict_mib` reflects raw tensor storage in memory and is more comparable across checkpoint formats.
