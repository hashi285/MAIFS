# Backbone Benchmark

- Timestamp: `20260319_065356`
- Hardware: CPU `AMD EPYC 9354 32-Core Processor`, GPU `NVIDIA H200 NVL`
- Torch: `2.10.0+cu128`
- Batch: `1`
- Scope: forward-only latency (preprocess / disk I/O / host-device copy excluded)

## Size

| Model | Params (M) | Trainable (M) | State Dict (MiB) | Checkpoint (MiB) | Note |
|---|---:|---:|---:|---:|---|
| forma | 37.2553 | 36.6981 | 142.132 | 422.614 | ForMa 37.3M, VMamba splicing detector, cuda_mamba_backend=15 |
| mobileclip_ft4 | 99.3765 | 33.2804 | 379.568 | 380.314 | MobileCLIP-S2 forensics ft4 |
| tiny_ladeda | 0.0013 | 0.0013 | 0.005 | 0.011 | Tiny-LaDeDa WildRF binary screener |
| mobilenetv2_dualstream | 5.7656 | 5.7656 | 22.255 | 22.486 | Track1 MobileNetV2 dual-stream (RGB + SRM residual) |

## Latency

| Model | GPU mean / p50 / p95 (ms) | GPU FPS | GPU peak MiB | CPU mean / p50 / p95 (ms) | CPU FPS | Input |
|---|---:|---:|---:|---:|---:|---|
| forma | 16.796 / 17.009 / 17.221 | 59.537 | 289.821 | 1612.582 / 1572.285 / 1695.525 | 0.62 | `[1, 3, 512, 512]` |
| mobileclip_ft4 | 15.46 / 14.916 / 18.349 | 64.683 | 436.364 | 123.846 / 124.588 / 128.17 | 8.075 | `[1, 3, 256, 256]` |
| tiny_ladeda | 5.791 / 6.048 / 6.166 | 172.675 | 38.129 | 2.469 / 2.403 / 2.954 | 405.034 | `[1, 3, 224, 224]` |
| mobilenetv2_dualstream | 18.934 / 16.981 / 24.949 | 52.814 | 67.435 | 35.852 / 30.905 / 57.24 | 27.892 | `[1, 3, 224, 224]` |

## Notes

- `mobilenetv2_dualstream` is the actual Track-1 noise backbone used by `run_phase3_tracks.py`.
- `mobilenetv2_proxy` is a legacy single-stream reference kept only for historical comparison.
- `checkpoint_mib` reflects the on-disk checkpoint file currently used in the repo.
- `state_dict_mib` reflects raw tensor storage in memory and is more comparable across checkpoint formats.
