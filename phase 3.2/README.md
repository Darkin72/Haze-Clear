# Phase 3.2 — SWDNet train từ đầu

Mở [phase3_2_swdnet_scratch.ipynb](phase3_2_swdnet_scratch.ipynb) trên Colab/Linux GPU.
Phase này là một nhánh độc lập: **không nạp WDMamba, Phase 3 hoặc Phase 3.1 để khởi tạo**, không dùng
teacher/KD và không dùng pretrained weights. SWDNet được khởi tạo ngẫu nhiên mỗi run.

Kiến trúc dùng Haar down/up ở các mức ảnh, gated depthwise convolution, channel normalization và một
nhánh detail rẻ ở độ phân giải gốc. Chiều rộng mặc định 32, depths `[2, 2, 4, 2, 2]`, khoảng 0,345 M
tham số và khoảng 1,786 GFLOPs/ảnh ở 256×256 theo bộ đếm của runner. Đây là kiến trúc thử nghiệm mới
trong repo, không phải tuyên bố novelty học thuật.

Cell cấu hình:

```python
DATASET_NAME = 'o-hazy'  # reside6k | sots-its | sots-ots | o-hazy | i-haze
DATA_ROOT_OVERRIDE = ''
TRAIN_ROOT = ''          # bắt buộc nếu train ITS/OTS riêng để đánh giá SOTS
EVAL_ONLY = None
RUN_NAME = 'swdnet_v1'
RUN_MODE = 'train'       # train | resume | benchmark
```

O-HAZY mặc định lấy từ `dataset/O-HAZY` trong checkout, không tải dataset. I-HAZE giữ split train/val/test
trong repo. RESIDE-6K có thể chứa nhiều ảnh hazy dùng chung một clear; loader ghép theo scene và giữ chúng
trong cùng split. SOTS Indoor/Outdoor mặc định là evaluation-only; train từ đầu trên SOTS cần đặt
`TRAIN_ROOT` tách biệt chứa ITS/OTS train, còn thư mục SOTS vẫn là test.

Schedule mặc định có ba giai đoạn: 10.000 bước crop 256 batch 4, 4.000 bước crop 384 batch 2 và
2.000 bước ảnh đầy đủ batch 1. Có EMA, cosine decay sau warmup, loss MSE + L1 + edge,
gradient clipping và checkpoint atomic. BF16 chỉ dùng trong train khi GPU hỗ trợ; đánh giá luôn FP32.
`resume` kiểm tra architecture hash, manifest hash, cấu hình, precision và trạng thái RNG trước khi tiếp tục.
Nếu output đã có checkpoint, `train` dừng để tránh ghi đè; đổi `RUN_NAME` cho run ngẫu nhiên mới.

Validation chọn EMA `best.pth`; test chỉ được mở trong bước benchmark cuối. Manifest lưu SHA256 từng ảnh,
scene và split. `summary.csv` có:

```text
dataset, model, PSNR, SSIM, FPS, FPS_256,
GFLOPs_256_counted, GFLOPs_256_supplemental, GFLOPs_256_estimated_total, params_M
```

Counted gồm Conv/Linear với 1 MAC = 2 FLOPs. Haar là supplemental; normalization, gating, activation,
pooling và phép cộng elementwise không nằm trong số đếm. FPS là batch 1, eager, FP32, đã warm-up, không
gồm I/O/transfer. Cần đo WDMamba/Phase 3 trong cùng GPU và evaluator trước khi gọi SWDNet nhanh hơn.

Kết quả nằm ở `/content/wdmamba_v3_2/<dataset>/<run_name>/results`, gồm `best.pth`, `last.pth`,
`history.csv`, manifest, test metrics/timing/GFLOPs, metadata và report. Nếu benchmark lỗi, các file
`quality_partial.json` và `test_metrics.csv` vẫn giữ PSNR trước khi đo latency.

Kiểm tra cục bộ không cần GPU:

```powershell
python "solution/Haze Clear/phase 3.2/test_cpu.py"
python "solution/Haze Clear/phase 3.2/build_notebook.py"
```

Chưa có kết quả GPU Phase 3.2; chỉ kết luận sau khi notebook hoàn tất train và benchmark.
O-HAZY chỉ có 36 scene train trong protocol này: train từ đầu không bảo đảm đạt PSNR của teacher
pretrained. Kết quả kiểm thử CPU chỉ xác nhận kiến trúc/luồng chạy và khả năng học trên dữ liệu giả.
