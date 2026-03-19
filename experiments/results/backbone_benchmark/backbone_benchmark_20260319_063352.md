# Backbone Benchmark

- Timestamp: `20260319_063352`
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
| mobilenetv2_proxy | 2.2277 | 2.2277 | 8.629 | N/A | Proxy only: torchvision MobileNetV2 single-stream, dual-stream Track1 not implemented |

## Latency

| Model | GPU mean / p50 / p95 (ms) | GPU FPS | GPU peak MiB | CPU mean / p50 / p95 (ms) | CPU FPS | Input |
|---|---:|---:|---:|---:|---:|---|
| forma | 16.773 / 17.024 / 17.211 | 59.62 | 289.821 | 1630.842 / 1591.518 / 1709.757 | 0.613 | `[1, 3, 512, 512]` |
| mobileclip_ft4 | 16.502 / 16.236 / 18.365 | 60.6 | 436.364 | 111.146 / 110.495 / 113.352 | 8.997 | `[1, 3, 256, 256]` |
| tiny_ladeda | 5.556 / 6.002 / 6.147 | 180.001 | 38.129 | 2.432 / 2.423 / 2.547 | 411.195 | `[1, 3, 224, 224]` |
| mobilenetv2_proxy | 3.226 / 3.251 / 3.268 | 309.954 | 52.234 | 14.865 / 14.879 / 14.983 | 67.273 | `[1, 3, 224, 224]` |

## Notes

- `mobilenetv2_proxy` is not the final Track-1 dual-stream implementation; it is a single-stream torchvision proxy.
- `checkpoint_mib` reflects the on-disk checkpoint file currently used in the repo.
- `state_dict_mib` reflects raw tensor storage in memory and is more comparable across checkpoint formats.
