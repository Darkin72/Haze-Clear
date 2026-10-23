# Phase 3.2 — SWDNet train từ đầu

| Dataset | Model | PSNR | SSIM | FPS | FPS 256 | GFLOPs counted | Haar bổ sung | GFLOPs tổng ước tính | Params M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| o-hazy | SWDNet | 23.5330 | 0.85786 | 284.25 | 467.87 | 1.778 | 0.008 | 1.786 | 0.345 |

Input hazy: PSNR=15.0586, SSIM=0.65633.
Checkpoint: step 10250; validation PSNR=21.7042; nguồn train=o-hazy.
Thiết bị: NVIDIA RTX PRO 6000 Blackwell Server Edition; train BF16, đánh giá FP32 eager.
GFLOPs là số phép tính mỗi ảnh; normalization/gating/activation/pooling/elementwise không nằm trong số đếm.
Không suy ra FPS từ GFLOPs; cần đo các baseline trên cùng GPU và evaluator trước khi kết luận tăng tốc.
Khởi tạo ngẫu nhiên, không teacher/KD/pretrained; validation chọn EMA checkpoint, test chỉ lập báo cáo.
O-HAZY chỉ có 36 train/4 val/5 test trong protocol này; test các phase trước đã được xem.
MEASURED không có nghĩa là đã đạt PSNR/FPS của WDMamba pretrained.
