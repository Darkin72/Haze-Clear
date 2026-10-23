# LightWaveDehaze — I-HAZE

**Lượt chạy nguồn:** `wavelite_all_20261021_152321_2d9efd`. Trạng thái trong log nguồn: **PASS**.

| Model | PSNR | SSIM | MSE | FPS | FPS/MSE | FPS 256 | GFLOPs 256* | Params M |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Reference | 18.9435 | 0.80633 | 0.01364698 | 11.28 | 826.49 | 50.19 | 48.357 | 9.353 |
| LightWaveDehaze eager | 20.0026 | 0.81371 | 0.01144874 | 28.55 | 2,493.32 | 74.53 | 18.302 | 6.507 |
| LightWaveDehaze deploy | 20.0026 | 0.81371 | 0.01144874 | 28.39 | 2,479.45 | 73.03 | 18.302 | 6.507 |

ΔPSNR +1.059 dB; tăng tốc 2.52×; FPS/MSE 3.00×.

Bản lưu từ log nguồn, không phải kết quả của lượt chạy lại. File CSV giữ nguyên nhãn model và toàn bộ số đo trong log; HTML/Markdown dùng tên trình bày LightWaveDehaze.
