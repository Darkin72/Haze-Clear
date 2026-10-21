# Phase 4 — WaveLite-Final

Mở [phase4_wavelite_final.ipynb](phase4_wavelite_final.ipynb) trên Colab/Linux GPU.
Notebook tự chứa runner và dùng dữ liệu trong checkout; không tải lại ZIP dataset.

Repo mã nguồn của Haze Clear là [`Darkin72/WDMamba`](https://github.com/Darkin72/WDMamba).
Notebook tự nhận diện `solution/Haze Clear` nếu đã checkout chung với dataset; khi không có,
notebook clone repo `WDMamba` vào `/content/WDMamba` và dùng thư mục đó cho kiến trúc.
Dataset vẫn lấy từ `VDT_GD2` hoặc các đường dẫn `DATA_ROOT_OVERRIDE(S)` đã điền.
Nếu đã clone mã nguồn ở vị trí khác, đặt `ARCH_ROOT_OVERRIDE` ở cell cấu hình.

`DATASET_NAME='all'` (mặc định) chạy **SOTS Indoor → SOTS Outdoor → O-HAZY → I-HAZE**,
không chạy RESIDE-6K thành một bộ riêng. Mỗi job dùng process và thư mục riêng dưới
`all/<run_name>/datasets/<dataset>/`; chỉ cài môi trường một lần. Điền đường dẫn từng bộ
vào `DATA_ROOT_OVERRIDES`; `TRAIN_ROOTS` cho ITS/OTS riêng khi dùng standard. Scalar
`DATA_ROOT_OVERRIDE`/`TRAIN_ROOT` chỉ dành cho chạy một dataset. `RESIDE_CHECKPOINT` dùng
chung cho hai bộ SOTS; `STUDENT_CHECKPOINT` trong all chỉ áp dụng O-HAZY.

Lỗi một bộ được ghi vào `suite_status.json` rồi tiếp tục bộ sau. Bảng tổng hợp là
`results/summary_all.csv`, `final_report_all.md` và `report_all.html`; không trộn trung bình
metric giữa các dataset, không đưa kết quả cũ của lượt lỗi vào bảng mới. ZIP all chứa cả
report tổng và toàn bộ artifact riêng từng dataset.

| PRESET | O-HAZY | I-HAZE mỗi model | Lượt đo latency |
|---|---:|---:|---:|
| `quick` | tối đa 900 bước | tối đa 5.000 bước | 15 |
| `standard` | tối đa 4.000 bước | tối đa 27.000 bước | 40 |

I-HAZE có hai model độc lập nên tổng ngân sách gấp đôi số bước mỗi model; early stopping có
thể dừng sớm. Cả hai preset đều đánh giá toàn bộ test, không bảo đảm lịch dài hơn sẽ tốt hơn.

Kiến trúc cuối được mô tả trong [ARCHITECTURE.md](ARCHITECTURE.md): giữ toàn bộ nhánh
wavelet/Mamba khôi phục LL, giảm DEN còn **một U-block + một residual depthwise block/group**,
giữ ba nhánh FFT/attention, thêm residual detail adapter ở đầu ra. Các adapter mới có output
bằng 0 lúc khởi tạo. Đây là cải tiến kế thừa WDMamba, không phải tuyên bố kiến trúc nghiên cứu mới.

| Dataset selector | Luồng chạy |
|---|---|
| `o-hazy` | Tối ưu checkpoint cùng miền. Ưu tiên teacher đầy đủ; tự dùng student Phase 3 nếu SHA256 teacher khớp. Nếu chỉ có Phase 3 thì nó là tham chiếu thực tế. |
| `reside6k` | Dùng checkpoint RESIDE-6K và train/val riêng. Giữ test sẵn có. |
| `sots-its` | Checkpoint Indoor hoặc RESIDE-6K; có `TRAIN_ROOT` ITS riêng thì tối ưu kiến trúc cuối. Thiếu train thì giữ cấu trúc đầy đủ và chỉ chọn runtime fusion. |
| `sots-ots` | Tương tự Indoor, dùng checkpoint Outdoor và `TRAIN_ROOT` OTS riêng. |
| `i-haze` | Hai model khởi tạo ngẫu nhiên độc lập: tham chiếu DEN bốn U-block/group và model cuối. Không kế thừa trọng số hoặc dùng KD giữa hai model này. |

Các thiết lập chính nằm ở cell đầu:

```python
DATASET_NAME = 'o-hazy'
DATA_ROOT_OVERRIDE = ''
TRAIN_ROOT = ''
CHECKPOINTS = {'o-hazy': '', 'sots-its': '', 'sots-ots': '', 'reside6k': ''}
STUDENT_CHECKPOINT = ''
CHECKPOINT_ZIP = ''
CHECKPOINT_SEARCH_ROOTS = [...]
ALLOW_WEIGHT_DOWNLOAD = False
RUN_MODE = 'auto'  # auto | train | resume | benchmark
```

Preset mặc định hiện là `PRESET='quick'`, với `RUN_NAME='wavelite_quick_v2'`.
O-HAZY chạy 600 bước DEN/detail + 300 bước joint; I-HAZE dùng lịch ngắn 4.000 + 1.000 bước
cho mỗi model. Đổi sang `standard` để dùng lịch dài bên dưới.

Để chạy nhanh cho SOTS, dùng `PRESET='quick'` và đặt một checkpoint RESIDE-6K vào
`RESIDE_CHECKPOINT` (hoặc để runner tự tìm `reside6k_32.15.pth`). Indoor/Outdoor sẽ dùng
chung checkpoint đó, bỏ qua fine-tune trên SOTS và chỉ benchmark runtime. Preset nhanh giảm
warm-up/repeats và có thể dùng chung latency/GFLOPs cho các ảnh cùng shape; chất lượng vẫn
được tính trên toàn bộ test. `protocol_report.md` ghi giao thức đầy đủ.
`report.html` trình bày bốn thẻ chỉ số, bảng so sánh và các tỉ lệ cải thiện thực đo;
`final_report.md` là bản ngắn, `protocol_report.md` giữ chi tiết tái lập.

Trong checkout cục bộ khi tạo phase này có O-HAZY, I-HAZE và checkpoint Fast-WDMamba O-HAZY
của Phase 3/3.1. Chưa tìm thấy dataset SOTS/RESIDE hoặc checkpoint teacher đầy đủ trên máy này.
Những file có trên Colab/Drive cần được trỏ bằng các biến trên. Runner không tự dùng O-HAZY cho SOTS.
Nếu không có checkpoint Indoor/Outdoor riêng, tên mặc định được tìm là `reside6k_32.15.pth`;
nguồn chính xác và hash được giữ trong metadata. Muốn so với WDMamba đầy đủ trên O-HAZY, cần
trỏ tới `Ohaze_1200_1600_27.22.pth`; không lấy số teacher cũ để tính speedup của phiên mới.

O-HAZY giữ split 36/4/5, test 41–45; I-HAZE giữ split 25/5/5 trong repo. Train/val/test được
kiểm tra trùng scene, path và nội dung SHA256. SOTS luôn là test, kể cả bản repack có thư mục
`train`; ITS/OTS train cần đường dẫn riêng. Nhiều ảnh hazy chung GT được giữ cùng split scene.

Mặc định cập nhật DEN/detail trước (3.000 bước crop 384), sau đó joint trên ảnh đầy đủ
(1.000 bước). Loss MSE + 0,02 L1 + 0,01 edge + 0,10 teacher MSE. BN inherited được cố định
running stats. Nhánh I-HAZE không có teacher loss, lịch mặc định mỗi model 20.000 bước crop256,
5.000 crop384 và 2.000 ảnh đầy đủ, với early stopping theo validation ở mỗi stage.
Lịch có thể điều chỉnh trước khi tạo run; 25 scene train không bảo đảm chất lượng tương đương
mô hình đã học nhiều dữ liệu. Không thể bảo đảm đạt PSNR/FPS chỉ từ thiết kế.

`auto` benchmark artifact đã hoàn tất hoặc tiếp tục từ last. `train` yêu cầu output mới;
`resume` yêu cầu trọng số đã lưu; `benchmark` chỉ khôi phục best. Resume kiểm tra hash mã,
manifest, nguồn, cấu hình, RNG và optimizer. Mỗi stage quay về best EMA trước stage kế tiếp.
Các file `.pth` được ghi qua file tạm rồi replace. Thay source/dataset/config sau khi train
phải dùng `RUN_NAME` khác. Report lỗi benchmark không làm mất file metric đã tính xong.

Mọi model được đo lại trong cùng process, FP32, TF32 tắt, batch1, không TTA/tiling.
Cùng max-side 768 mặc định (0 để giữ ảnh gốc), crop border4. Runtime chỉ thử Conv–BN fusion,
chọn trên validation hoặc tensor giả trước khi mở test. Không dùng CUDA Graph sau các lỗi
capture trước đó. Fusion tạo tensor bình thường kể cả caller bật inference mode.

Metric mới:

```text
MSE = trung bình đều MSE RGB mỗi ảnh, trên thang [0,1], sau crop border4
FPS = 1000 / latency trung bình theo ảnh (ms)
FPS/MSE = FPS / MSE
```

Không suy MSE từ PSNR trung bình; không lấy trung bình FPS/MSE từng ảnh để thay số tổng.
Nếu MSE=0, tỉ số là vô cực theo định nghĩa. Giá trị phụ thuộc GPU, kích thước và protocol.
Vẫn bắt buộc kiểm tra PSNR riêng để không đổi chất lượng lấy tốc độ tùy ý.

GFLOPs dùng 1 MAC=2 FLOPs, counted Conv/Linear + SS2D projections; supplemental là ước tính
scan/FFT/Haar. Activation/normalization/pooling/elementwise không đếm đầy đủ. Bảng giữ
GFLOPs256 và GFLOPs trung bình kích thước test; FPS256 không dùng làm tử số cho MSE ảnh test.

`PASS` yêu cầu đồng thời ΔPSNR ≥ −0,30 dB, FPS ≥2×, FPS256 ≥1,10×, FPS/MSE không giảm và
PSNR deploy lệch eager ≤0,01 dB. `REVIEW` giữ nguyên số đo chưa đạt. Không chọn lại model
hoặc kiến trúc bằng test. O-HAZY test đã được xem ở phase trước, validation cũng có thể
trùng dữ liệu huấn luyện teacher; không coi đây là holdout nghiên cứu độc lập.

Artifact ở `/content/wdmamba_phase4/<dataset>/<run_name>/results`:

- `summary.csv`, `*_metrics.csv`, `*_timing.csv`, `*_gflops.csv`.
- `final_best.pth`, `final_last.pth`, `reference_best.pth`/`reference_last.pth` khi có train tham chiếu.
- `model_final.pth`: eager export tự chứa, spec + nguồn gốc, nạp strict=True qua `phase4_arch.skeleton`.
- `metadata.json`, `config.json`, `inputs.json`, manifest, source và `final_report.md`.

Report chính mô tả kiến trúc và kết quả; metadata ghi chính xác nguồn khởi tạo/trọng số.
Không mô tả việc kế thừa trọng số thành huấn luyện ngẫu nhiên từ đầu.

Kiểm thử cục bộ:

```powershell
python -X utf8 "solution/Haze Clear/phase 4/build_notebook.py"
python -X utf8 "solution/Haze Clear/phase 4/test_cpu.py"
```

Các kiểm thử dùng mô hình nhỏ để kiểm tra luồng train/resume/fusion/metric; không thay cho
forward/backward CUDA của WDMamba. Máy cục bộ hiện chỉ có PyTorch CPU, nên chưa có kết quả
GPU Phase 4. Không có bảng PSNR/FPS giả hoặc giá trị mượn từ phase cũ.

Kiểm thử hiện tại: 22/22 đạt, gồm resume sau gián đoạn cho kết quả khớp từng trọng số với
lượt liên tục, phục hồi benchmark không gọi train, xuất/nạp eager, và SOTS runtime-only
không gọi train hoặc cắt block. Notebook đã kiểm tra schema, cú pháp và source nhúng.
