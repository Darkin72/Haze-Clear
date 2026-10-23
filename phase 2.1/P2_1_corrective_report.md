# P2.1 Corrective RGB Report

## Gate G2.1

Status: REVIEW

| Check | Observed | Criterion | Result |
|---|---:|---:|---|
| Protocol match | True | Phase 1 batch/max_side | PASS |
| Delta PSNR vs Phase 1 | -4.9330 dB | >= -0.3 dB | REVIEW |
| Speedup vs Phase 1 | 2.0977x | >= 1.25x | PASS |
| PSNR gain vs failed Phase 2 | 6.0240 dB | >= 3.0 dB | PASS |
| Artifact review | False | complete | REVIEW |
| Phase 1 baseline accepted | False | mismatch explained | REVIEW |

## Selected model

- Dev PSNR/SSIM: 29.0004 / 0.942619
- Test PSNR/SSIM: 26.4271 / 0.953298
- Latency/FPS: 16.5763 ms / 60.33
- Parameters: 3.2153 M
- Peak VRAM: 30.9316 GiB

## Decision

Proceed to Phase 3 only if G2.1 is PASS. If quality remains below the threshold, retain/fine-tune part of the original DEN instead of adding wavelet or ASM branches.
