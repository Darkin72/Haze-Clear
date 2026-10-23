# Fast-WDMamba V2 — chọn dataset, PSNR/FPS/GFLOPs

Mở [phase3_1_psnr_fps_gflops.ipynb](phase3_1_psnr_fps_gflops.ipynb) trên Colab GPU.
Đổi `DATASET_NAME` ở cell cấu hình; **O-HAZY là mặc định**.

| `DATASET_NAME` | Dữ liệu trong checkout | Chế độ mặc định | Teacher |
|---|---|---|---|
| `o-hazy` | `dataset/O-HAZY` | Tiếp tục Phase 3, split 36/4/5; test 41–45 | `Ohaze_1200_1600_27.22.pth` |
| `i-haze` | `dataset/I-HAZE` | Fine-tune; giữ split repo 25/5/5 | O-HAZY, khởi tạo chuyển miền |
| `reside6k` | `dataset/RESIDE-6K` | Fine-tune trên train, đánh giá test | `reside6k_32.15.pth` |
| `sots-its` | `dataset/Synthetic Objective Testing Set (SOTS) [RESIDE]/indoor` | Chỉ đánh giá SOTS Indoor | RESIDE-6K |
| `sots-ots` | `dataset/Synthetic Objective Testing Set (SOTS) [RESIDE]/outdoor` | Chỉ đánh giá SOTS Outdoor | RESIDE-6K |

SOTS Indoor/Outdoor là benchmark test; ITS/OTS là bộ train tương ứng.
Checkout hiện tại có O-HAZY và I-HAZE. Các bộ còn lại cần có trong checkout chạy notebook hoặc đặt
`DATA_ROOT_OVERRIDE` tới thư mục đã có. Notebook không tải ZIP dataset; nếu chưa có checkout, chỉ clone
repo VDT_GD2 và lấy thư mục đã chọn. Teacher archive vẫn được tải nếu chưa có checkpoint.

```python
DATASET_NAME = 'o-hazy'   # reside6k | sots-its | sots-ots | o-hazy | i-haze
DATA_ROOT_OVERRIDE = ''  # Ghi đè thư mục dữ liệu
EVAL_ONLY = None         # Tự nhận diện; True chỉ đánh giá; False yêu cầu train/val
TRAIN_ROOT = ''          # Bộ train ITS/OTS riêng khi chọn SOTS
CHECKPOINT = ''          # Teacher tùy chọn
PHASE3_CHECKPOINT = ''   # Student đúng dataset, nếu đã train
```

O-HAZY tự tìm `phase 3/fast_wdmamba_o_hazy_best.pth`; nếu thiếu, cell student cho phép upload.
Dataset khác không tự nạp student O-HAZY: để trống sẽ nén teacher còn một block DEN/group trước fine-tune.
SOTS chỉ đánh giá có thể dùng student đã train RESIDE-6K; nếu chưa có sẽ đo student vừa nén teacher,
ghi rõ `Compressed_init`. Đây chưa phải checkpoint đã phục hồi PSNR. Kết quả chỉ đánh giá có trạng thái
`EVALUATION_ONLY`, không được coi là đạt mục tiêu train tăng PSNR.

Loader nhận `{train,val,test}/{clear,hazy}`, `GT/hazy`, hậu tố `_hazy`/`_GT`, và tên kiểu
`0001_0.8_0.2` cùng clear `0001`. Nhiều ảnh hazy chung clear được nhóm theo scene. I-HAZE/RESIDE-6K
giữ split thư mục; nếu thiếu validation thì tách 10% scene **từ train**, seed 2026.
Dataset train dạng phẳng cần tổ chức train/test rõ ràng. SOTS dạng phẳng hoặc được chia lại trong repo
đều được xem là benchmark test. `TRAIN_ROOT` phải là thư mục ITS/OTS riêng, không chứa test.
Manifest ghi dataset, split, ID, scene, đường dẫn và SHA256; kiểm tra trùng scene/hash giữa các split.
Với SOTS Indoor gốc, loader cắt khung 10 pixel ở GT khi GT lớn hơn hazy đúng 20 pixel mỗi chiều;
những cặp lệch kích thước khác sẽ báo lỗi.

V2 giữ một block DEN/group, fine-tune crop lớn rồi ảnh đầy đủ để phục hồi PSNR.
Conv–BN fusion/CUDA Graph được thử trên validation để tăng FPS triển khai ở FP32.
Mặc định `TRY_CUDA_GRAPH=False` sau lỗi capture trên GPU; vẫn dùng Conv–BN fusion eager.
Fusion thực hiện ngoài `inference_mode` để trọng số mới có version counter, tránh lỗi
`Inference tensors do not track version counter` khi chạy convolution/cuDNN.
Nếu không có validation, dùng tensor giả 256×256 để chọn runtime trước test.
Cache LRU giới hạn bởi `MAX_CACHE_IMAGES=40`, vẫn lấy mẫu từ toàn bộ train RESIDE-6K.
Với bộ lớn, cache miss cần chạy teacher/coarse nên train sẽ lâu hơn. Crop tự giới hạn theo cạnh ngắn ảnh,
làm tròn xuống bội 32. Bảng cuối đo lại mọi model trong cùng phiên GPU.

GFLOPs được báo cáo ở 256×256 và từng kích thước test sau padding:

- Counted: Conv/Linear và phép chiếu SS2D, 1 MAC = 2 FLOPs.
- Supplemental: công thức ước tính selective scan, FFT và Haar.
- Không đếm normalization, activation, pooling, bias và phép elementwise; đây là
  ước tính độ phức tạp, không phải instruction count đầy đủ hoặc GFLOP/s.

Kết quả lưu riêng ở `/content/wdmamba_v2/<dataset>/results/<dataset>/`, gồm checkpoint nếu có train, lịch sử
validation, CSV GFLOPs/timing, ảnh dự đoán và `final_report.md`. `PASS` yêu cầu tăng
PSNR ít nhất 0,10 dB so baseline (Phase 3 hoặc student khởi tạo), FPS triển khai tăng 3%, chênh PSNR so teacher
không thấp hơn -0,30 dB và deployment lệch eager không quá 0,01 dB.

Chưa có kết quả GPU V2. CUDA Graph không làm giảm GFLOPs của kiến trúc. Các số
PSNR/FPS mới chỉ được xác nhận sau khi chạy notebook, không dùng lại số từ báo cáo cũ.

`summary.csv` có `dataset`, `model`, `PSNR`, `SSIM`, `FPS`, `FPS_256`, `params_M`,
`GFLOPs_256_counted`, `GFLOPs_256_supplemental`, `GFLOPs_256_estimated_total`.
O-HAZY vẫn giữ hạn chế: test Phase 3 đã được xem; teacher đã học scene 1–40 gồm validation compression.
Với teacher pretrained khác, cần kiểm tra giao scene với benchmark trước khi claim tổng quát hóa.

Để dựng lại notebook và kiểm tra CPU:

```powershell
python "solution/Haze Clear/phase 3.1/build_notebook_v2.py"
python "solution/Haze Clear/phase 3.1/test_cpu.py"
```

Nếu training đã xong nhưng đánh giá deploy bị lỗi, copy cell code trong
[recover_v2_benchmark.ipynb](recover_v2_benchmark.ipynb) vào notebook Colab còn giữ checkpoint rồi chạy.
Cell đọc config/manifest cũ, cập nhật runner, tắt CUDA Graph và chỉ benchmark trong process mới;
không tải lại dữ liệu, không chạy lại training, không cần xóa/restart runtime. Có xuất ZIP kết quả.
Checkpoint mặc định: `/content/wdmamba_v2/o-hazy/results/o-hazy/fast_wdmamba_v2_best.pth`.
Nếu runtime đã mất cần khôi phục checkpoint V2, baseline, teacher, config/manifest và dữ liệu trước.

Notebook chính v2.1 đã nhúng bản sửa trực tiếp. Mặc định `RUN_MODE='auto'`: có checkpoint V2
trong output thì chỉ benchmark lại; chưa có thì train/đánh giá bình thường. Chọn `RUN_MODE='run'`
để chủ động train lại, hoặc `RUN_MODE='benchmark'` để bắt buộc dùng checkpoint đã có.
`V2_CHECKPOINT` cho phép ghi đè đường dẫn checkpoint cần đo.
Runner kiểm tra dataset/protocol và SHA256 teacher/baseline trước khi nạp checkpoint; không chọn lại
trọng số bằng test. `summary_partial.csv` lưu những model đã đánh giá xong nếu model tiếp theo bị lỗi.
Lệnh CLI dùng config của lượt training: `python refine_o_hazy.py <config.json> benchmark`.
