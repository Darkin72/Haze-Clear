# P1 Baseline Report

## Gate G1

Status: REVIEW

| Check | Observed | Criterion | Result |
|---|---:|---:|---|
| PSNR | 31.3601 | 32.1500 +/- 0.5000 | REVIEW |
| SSIM | 0.971214 | 0.972300 +/- 0.005000 | PASS |
| DEN staged latency share | 71.71% | >= 50% | PASS |

## Quality and efficiency

| Metric | Value |
|---|---:|
| GPU latency / image | 34.7718 ms |
| GPU FPS | 28.7589 |
| Peak allocated VRAM | 22.9758 GiB |
| Estimated FLOPs / image | 288.3000 G |

## Native stage profile

| Stage | ms/image | Share |
|---|---:|---:|
| DWT | 0.0327 | 0.09% |
| LFRN | 9.0026 | 25.89% |
| IWT | 0.7997 | 2.30% |
| DEN | 24.9351 | 71.71% |

## Reproducibility

- Repository commit: e5d7c78e2e1a500bc7be4741d0494caafd144416
- Checkpoint: /content/wdmamba_phase1/checkpoints/unzipped/WDMamba_ckpts/reside6k_32.15.pth
- GPU: NVIDIA A100-SXM4-80GB
- PyTorch/CUDA: 2.8.0+cu128 / 12.8
- Batch size: 100
- Max side: 512

## Decision

Proceed to Phase 2 only after every mismatch is explained and all CSV/JSON artifacts are retained.
