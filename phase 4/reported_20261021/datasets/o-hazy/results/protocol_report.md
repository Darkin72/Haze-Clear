# LightWaveDehaze — O-HAZY

**Lượt chạy nguồn:** `wavelite_all_20261021_152321_2d9efd`. Trạng thái trong log nguồn: **REVIEW**.

| Model | PSNR | SSIM | MSE | FPS | FPS/MSE | FPS 256 | GFLOPs 256* | Params M |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Reference | 27.9067 | 0.91089 | 0.00169826 | 10.05 | 5,919.87 | 50.60 | 48.357 | 9.353 |
| LightWaveDehaze eager | 27.8598 | 0.91126 | 0.00174385 | 16.94 | 9,713.65 | 62.43 | 28.570 | 7.458 |
| LightWaveDehaze deploy | 27.8598 | 0.91126 | 0.00174385 | 16.97 | 9,728.75 | 63.26 | 28.570 | 7.458 |

ΔPSNR -0.047 dB; tăng tốc 1.69×; FPS/MSE 1.64×.

Bản lưu từ log nguồn, không phải kết quả của lượt chạy lại. File CSV giữ nguyên nhãn model và toàn bộ số đo trong log; HTML/Markdown dùng tên trình bày LightWaveDehaze.

## Giao thức

FP32, TF32 tắt, batch 1, max-side 768, crop border 4, không TTA/tiling. PSNR, SSIM và MSE lấy trung bình theo ảnh RGB [0,1]. FPS đo forward có pad/crop, không gồm I/O, transfer và metric. FPS/MSE = FPS / MSE trung bình theo ảnh. GFLOPs gồm counted và ước tính scan/FFT/Haar, theo 1 MAC = 2 FLOPs; chưa đếm đầy đủ normalization/elementwise. Không gộp trung bình PSNR/FPS giữa các dataset.

Test: 5 ảnh.
