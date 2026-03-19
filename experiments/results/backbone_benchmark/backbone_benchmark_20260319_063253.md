# Backbone Benchmark

- Timestamp: `20260319_063253`
- Hardware: CPU `AMD EPYC 9354 32-Core Processor`, GPU `NVIDIA H200 NVL`
- Torch: `2.10.0+cu128`
- Batch: `1`
- Scope: forward-only latency (preprocess / disk I/O / host-device copy excluded)

## Size

| Model | Params (M) | Trainable (M) | State Dict (MiB) | Checkpoint (MiB) | Note |
|---|---:|---:|---:|---:|---|
| forma | 37.2553 | 36.6981 | 142.132 | 422.614 | ForMa 37.3M, VMamba splicing detector |
| mobileclip_ft4 | 99.3765 | 33.2804 | 379.568 | 380.314 | MobileCLIP-S2 forensics ft4 |
| tiny_ladeda | 0.0013 | 0.0013 | 0.005 | 0.011 | Tiny-LaDeDa WildRF binary screener |
| mobilenetv2_proxy | 2.2277 | 2.2277 | 8.629 | N/A | Proxy only: torchvision MobileNetV2 single-stream, dual-stream Track1 not implemented |

## Latency

| Model | GPU mean / p50 / p95 (ms) | GPU FPS | GPU peak MiB | CPU mean / p50 / p95 (ms) | CPU FPS | Input |
|---|---:|---:|---:|---:|---:|---|
| forma | N/A | N/A | N/A | 1615.511 / 1597.638 / 1669.889 | 0.619 | `[1, 3, 512, 512]` |
| mobileclip_ft4 | N/A | N/A | N/A | 112.864 / 112.874 / 114.06 | 8.86 | `[1, 3, 256, 256]` |
| tiny_ladeda | N/A | N/A | N/A | 2.628 / 2.634 / 2.704 | 380.484 | `[1, 3, 224, 224]` |
| mobilenetv2_proxy | N/A | N/A | N/A | 19.535 / 14.163 / 43.307 | 51.189 | `[1, 3, 224, 224]` |

## Notes

- `mobilenetv2_proxy` is not the final Track-1 dual-stream implementation; it is a single-stream torchvision proxy.
- `checkpoint_mib` reflects the on-disk checkpoint file currently used in the repo.
- `state_dict_mib` reflects raw tensor storage in memory and is more comparable across checkpoint formats.
