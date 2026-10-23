# LightWaveDehaze — tổng hợp số liệu đã báo cáo

**Lượt chạy nguồn:** `wavelite_all_20261021_152321_2d9efd`.

Đây là bản lưu của lượt đã báo cáo ngày 21/10/2026, tách riêng với lượt chạy lại.

| Dataset | PSNR | SSIM | FPS | FPS/MSE | GFLOPs 256* | Trạng thái |
|---|---:|---:|---:|---:|---:|---|
| SOTS ITS | 29.727 | 0.9691 | 23.1 | 17,370 | 28.57 | REVIEW |
| SOTS OTS | 31.550 | 0.9749 | 25.0 | 24,968 | 28.57 | REVIEW |
| O-HAZY | 27.860 | 0.9113 | 17.0 | 9,729 | 28.57 | REVIEW |
| I-HAZE | 20.003 | 0.8137 | 28.4 | 2,479 | 18.30 | PASS |

## Giao thức

FP32, TF32 tắt, batch 1, max-side 768, crop border 4, không TTA/tiling. PSNR, SSIM và MSE lấy trung bình theo ảnh RGB [0,1]. FPS đo forward có pad/crop, không gồm I/O, transfer và metric. FPS/MSE = FPS / MSE trung bình theo ảnh. GFLOPs gồm counted và ước tính scan/FFT/Haar, theo 1 MAC = 2 FLOPs; chưa đếm đầy đủ normalization/elementwise. Không gộp trung bình PSNR/FPS giữa các dataset.

## Phạm vi bản lưu

Số liệu lấy nguyên từ log nguồn. Không bao gồm trọng số, per-image metrics, timing samples, training history hay metadata gốc của lượt cũ. Không tái tạo các file chưa có bằng chứng nguồn. Chi tiết nguồn và phạm vi nằm trong snapshot_manifest.json.
