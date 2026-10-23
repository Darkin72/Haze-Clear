# LightWaveDehaze — SOTS ITS

**Lượt chạy nguồn:** `wavelite_all_20261021_152321_2d9efd`. Trạng thái trong log nguồn: **REVIEW**.

| Model | PSNR | SSIM | MSE | FPS | FPS/MSE | FPS 256 | GFLOPs 256* | Params M |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Reference | 31.4809 | 0.97325 | 0.00093331 | 10.12 | 10,846.41 | 41.32 | 68.892 | 11.254 |
| LightWaveDehaze eager | 29.7275 | 0.96907 | 0.00132889 | 23.06 | 17,355.13 | 64.23 | 28.570 | 7.458 |
| LightWaveDehaze deploy | 29.7275 | 0.96907 | 0.00132889 | 23.08 | 17,370.19 | 64.89 | 28.570 | 7.458 |

ΔPSNR -1.753 dB; tăng tốc 2.28×; FPS/MSE 1.60×.

Bản lưu từ log nguồn, không phải kết quả của lượt chạy lại. File CSV giữ nguyên nhãn model và toàn bộ số đo trong log; HTML/Markdown dùng tên trình bày LightWaveDehaze.
