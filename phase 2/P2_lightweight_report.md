# P2 Lightweight RGB Report

## Gate G2

Status: REVIEW

| Check | Observed | Criterion | Result |
|---|---:|---:|---|
| Baseline available | True | Full Phase 1 summary.csv | PASS |
| Protocol match | batch=100, max_side=512 | inherited from Phase 1 | PASS |
| Phase 1 mismatch accepted | False | mismatch explained before Phase 2 | REVIEW |
| Delta PSNR | -10.95699520281979 | >= -0.300 dB | REVIEW |
| Speedup | 29.327804483768542x | >= 1.25x | PASS |
| Artifact review | False | Manual review complete | REVIEW |

## Selected B1

- Development PSNR/SSIM: 20.3730 / 0.828265
- Test PSNR: 20.4031 dB
- Test SSIM: 0.853927
- Latency: 1.1856 ms/image
- FPS: 843.4357
- Parameters: 0.0043 M
- Peak VRAM: 6.8864 GiB
- Training iterations: recorded in metadata

## Required artifacts

- `search_results.csv`
- `dev_summary.csv`, `dev_image_metrics.csv`, `test_summary.csv`, and `test_image_metrics.csv`
- `pareto_quality_speed.png`
- `failure_cases/`
- `metadata.json`

## Decision

Proceed to Phase 3 only if Gate G2 is PASS. Otherwise adjust channels/block depth or retain part of DEN; do not add ASM yet.
