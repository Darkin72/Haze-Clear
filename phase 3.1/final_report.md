# Fast-WDMamba trên O-HAZY

Trạng thái: **REVIEW**

Mô hình cuối: LFRN nguyên bản + DEN rút gọn còn 1 U-block/group; giữ FFT, channel attention và residual.

Train/val/test: 36/4/5; test scene 41–45.
eval_max_side=768; crop border=4; RGB FP32, batch 1.

| Model | PSNR | SSIM | ms/ảnh | FPS | FPS 256 |
|---|---:|---:|---:|---:|---:|
| WDMamba_O_HAZY | 27.9067 | 0.91089 | 97.448 | 10.26 | 54.64 |
| Fast_WDMamba_O_HAZY | 27.3989 | 0.90049 | 36.640 | 27.29 | 80.92 |

ΔPSNR=-0.5077 dB; speedup=2.66x; speedup 256=1.48x.
Bootstrap CI95 cho mean ΔPSNR trên 5 scene: [-0.5859, -0.4418] dB.

Mục tiêu: ΔPSNR ≥ -0.30 dB và speedup ≥ 2.00x.
REVIEW nghĩa là chưa đạt mục tiêu, không phải đã đảm bảo PSNR/FPS.

Đây là compression/fine-tuning dùng checkpoint O-HAZY và loss GT + distillation, không phải ablation cùng loss/from-scratch.
Teacher đã học 1–40, gồm cả ảnh validation; validation ở đây chỉ chọn checkpoint compression, không phải validation độc lập cho pretraining.
Số điểm dưới max_side=768 không được đồng nhất với protocol paper resize 1600x1200 hoặc full resolution.
Không dùng test để chọn lại kiến trúc sau khi xem báo cáo. Chỉ 5 scene nên kết luận còn hạn chế.
