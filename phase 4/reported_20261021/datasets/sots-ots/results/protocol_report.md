# LightWaveDehaze — SOTS OTS

**Lượt chạy nguồn:** `wavelite_all_20261021_152321_2d9efd`. Trạng thái trong log nguồn: **REVIEW**.

| Model | PSNR | SSIM | MSE | FPS | FPS/MSE | FPS 256 | GFLOPs 256* | Params M |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Reference | 33.2994 | 0.98158 | 0.00071089 | 11.30 | 15,892.52 | 41.91 | 68.892 | 11.254 |
| LightWaveDehaze eager | 31.5496 | 0.97487 | 0.00100009 | 25.32 | 25,321.71 | 65.27 | 28.570 | 7.458 |
| LightWaveDehaze deploy | 31.5496 | 0.97487 | 0.00100009 | 24.97 | 24,967.76 | 65.22 | 28.570 | 7.458 |

ΔPSNR -1.750 dB; tăng tốc 2.21×; FPS/MSE 1.57×.

Bản lưu từ log nguồn, không phải kết quả của lượt chạy lại. File CSV giữ nguyên nhãn model và toàn bộ số đo trong log; HTML/Markdown dùng tên trình bày LightWaveDehaze.

## Giao thức

FP32, TF32 tắt, batch 1, max-side 768, crop border 4, không TTA/tiling. PSNR, SSIM và MSE lấy trung bình theo ảnh RGB [0,1]. FPS đo forward có pad/crop, không gồm I/O, transfer và metric. FPS/MSE = FPS / MSE trung bình theo ảnh. GFLOPs gồm counted và ước tính scan/FFT/Haar, theo 1 MAC = 2 FLOPs; chưa đếm đầy đủ normalization/elementwise. Không gộp trung bình PSNR/FPS giữa các dataset.

Test: 500 ảnh.
