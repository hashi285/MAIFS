# Backbone Benchmark

- Timestamp: `20260319_063243`
- Hardware: CPU `AMD EPYC 9354 32-Core Processor`, GPU `NVIDIA H200 NVL`
- Torch: `2.10.0+cu128`
- Batch: `1`
- Scope: forward-only latency (preprocess / disk I/O / host-device copy excluded)

## Size

| Model | Params (M) | Trainable (M) | State Dict (MiB) | Checkpoint (MiB) | Note |
|---|---:|---:|---:|---:|---|
| tiny_ladeda | 0.0013 | 0.0013 | 0.005 | 0.011 | Tiny-LaDeDa WildRF binary screener |

## Latency

| Model | GPU mean / p50 / p95 (ms) | GPU FPS | GPU peak MiB | CPU mean / p50 / p95 (ms) | CPU FPS | Input |
|---|---:|---:|---:|---:|---:|---|
| tiny_ladeda | N/A | N/A | N/A | 6.241 / 6.253 / 6.404 | 160.23 | `[1, 3, 224, 224]` |

## Notes

- `mobilenetv2_proxy` is not the final Track-1 dual-stream implementation; it is a single-stream torchvision proxy.
- `checkpoint_mib` reflects the on-disk checkpoint file currently used in the repo.
- `state_dict_mib` reflects raw tensor storage in memory and is more comparable across checkpoint formats.
