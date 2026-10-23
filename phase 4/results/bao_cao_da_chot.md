# LightWaveDehaze — bản số liệu đã báo cáo

**Lượt chạy được chốt:** `wavelite_all_20261021_152321_2d9efd` (21/10/2026).
**Ngày lập bản lưu báo cáo:** 23/10/2026.

Các bảng trong phần 1–3 giữ nguyên số đo của lượt đã báo cáo trước đó. Đây là bản lưu của lượt cũ, không phải số đo của lượt chạy lại. Phần 4 đối chiếu với lượt mới để theo dõi thay đổi.

## 1. Kết quả đã chốt

| Dataset | PSNR (dB) | SSIM | FPS | FPS/MSE | GFLOPs 256×256* | Trạng thái lượt cũ |
|---|---:|---:|---:|---:|---:|---|
| SOTS ITS | 29.727 | 0.9691 | 23.1 | 17,370 | 28.57 | REVIEW |
| SOTS OTS | 31.550 | 0.9749 | 25.0 | 24,968 | 28.57 | REVIEW |
| O-HAZY | 27.860 | 0.9113 | 17.0 | 9,729 | 28.57 | REVIEW |
| I-HAZE | 20.003 | 0.8137 | 28.4 | 2,479 | 18.30 | PASS |

## 2. So sánh với model tham chiếu trong lượt đã chốt

| Dataset | PSNR tham chiếu | PSNR LightWaveDehaze | ΔPSNR (dB) | FPS tham chiếu | FPS LightWaveDehaze | Tăng tốc |
|---|---:|---:|---:|---:|---:|---:|
| SOTS ITS | 31.481 | 29.727 | -1.753 | 10.12 | 23.08 | 2.28× |
| SOTS OTS | 33.299 | 31.550 | -1.750 | 11.30 | 24.97 | 2.21× |
| O-HAZY | 27.907 | 27.860 | -0.047 | 10.05 | 16.97 | 1.69× |
| I-HAZE | 18.943 | 20.003 | +1.059 | 11.28 | 28.39 | 2.52× |

## 3. Sai số và độ phức tạp của lượt đã chốt

| Dataset | MSE RGB [0,1] | Params (M) | GFLOPs 256×256* | FPS/MSE tham chiếu | FPS/MSE LightWaveDehaze | Tăng FPS/MSE |
|---|---:|---:|---:|---:|---:|---:|
| SOTS ITS | 0.00132889 | 7.457650 | 28.57 | 10,846 | 17,370 | 1.60× |
| SOTS OTS | 0.00100009 | 7.457650 | 28.57 | 15,893 | 24,968 | 1.57× |
| O-HAZY | 0.00174385 | 7.457650 | 28.57 | 5,920 | 9,729 | 1.64× |
| I-HAZE | 0.01144874 | 6.506962 | 18.30 | 826 | 2,479 | 3.00× |

## 4. Đối chiếu lượt chạy lại

**Lượt chạy lại:** `wavelite_all_20261022_140137_86b927`. Nguồn: [summary_all.csv](summary_all.csv). Chênh lệch tính bằng mới trừ cũ.

| Dataset | PSNR đã báo cáo | PSNR lượt mới | ΔPSNR (dB) | SSIM đã báo cáo | SSIM lượt mới |
|---|---:|---:|---:|---:|---:|
| SOTS ITS | 29.727497 | 29.727246 | -0.000251 | 0.969069 | 0.969073 |
| SOTS OTS | 31.549627 | 31.549731 | +0.000103 | 0.974874 | 0.974872 |
| O-HAZY | 27.859820 | 27.859692 | -0.000127 | 0.911261 | 0.911270 |
| I-HAZE | 20.002576 | 18.411195 | -1.591381 | 0.813714 | 0.784605 |

| Dataset | FPS đã báo cáo | FPS lượt mới | ΔFPS (%) | FPS/MSE đã báo cáo | FPS/MSE lượt mới |
|---|---:|---:|---:|---:|---:|
| SOTS ITS | 23.083 | 22.921 | -0.70% | 17,370 | 17,247 |
| SOTS OTS | 24.970 | 25.085 | +0.46% | 24,968 | 25,083 |
| O-HAZY | 16.966 | 17.015 | +0.29% | 9,729 | 9,757 |
| I-HAZE | 28.387 | 28.597 | +0.74% | 2,479 | 1,780 |

SOTS/O-HAZY có chênh lệch PSNR dưới 0,001 dB giữa hai lượt. I-HAZE thay đổi từ 20,002576 xuống 18,411195 dB, giảm 1,591381 dB; đây không phải khác biệt do làm tròn. Chất lượng model tham chiếu I-HAZE cũng thay đổi từ 18,943454 xuống 16,798732 dB.

I-HAZE ở lượt mới vẫn có trạng thái PASS vì các điều kiện được so với model tham chiếu của chính lượt mới; PASS không khẳng định chất lượng bằng lượt cũ. Metadata lượt mới ghi final chạy 8.600 bước và best tại bước 3.800; chưa đủ đối chứng để kết luận nguyên nhân chênh lệch.

Các CSV, log, metadata, báo cáo tự sinh và trọng số lượt mới được giữ nguyên. Số liệu lượt cũ chỉ mô tả artifact của lượt cũ, không gán cho trọng số của lượt chạy lại.

## 5. Nguồn và giao thức

- Số liệu đã chốt: log người dùng cung cấp trong “Văn bản đã dán.txt”, đính kèm `48a740b5-9383-4d9b-9d2f-b2e4a1f51461`, lượt `wavelite_all_20261021_152321_2d9efd`.
- Dòng LightWaveDehaze trong tài liệu lấy đúng bản deploy của log; model tham chiếu lấy dòng Reference. Chỉ đổi tên trình bày, không thay số đo.
- Lượt chạy lại: `results/summary_all.csv`; bảng từng dataset nằm dưới `datasets/<dataset>/results/summary.csv`. Thông tin I-HAZE: [metadata](../datasets/i-haze/results/metadata.json).
- FP32, batch 1, TF32 tắt, max-side 768, crop border 4; không TTA/tiling. SOTS có 500 ảnh test mỗi miền, O-HAZY/I-HAZE có 5 ảnh mỗi bộ.
- PSNR/SSIM/MSE là trung bình theo ảnh RGB [0,1]. FPS đo forward có pad/crop, không gồm I/O, transfer và tính metric.
- FPS/MSE = FPS / MSE trung bình theo ảnh; không suy MSE từ PSNR trung bình.
- *GFLOPs gồm counted và ước tính scan/FFT/Haar, 1 MAC = 2 FLOPs, chưa đếm đầy đủ normalization/elementwise.
- Không gộp trung bình PSNR/FPS giữa các dataset.
