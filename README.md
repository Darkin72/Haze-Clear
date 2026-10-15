# Haze Clear

## Viettel HazeClear: giải pháp khử sương ảnh

Đề tài nghiên cứu và tối ưu mô hình khử sương nhằm phục hồi tương phản, màu sắc
và chi tiết cho hệ thống camera, giám sát hạ tầng và thị giác máy tính.
Repository cung cấp mã mô hình, cấu hình huấn luyện, công cụ suy luận và
benchmark để phát triển giải pháp ứng dụng **Viettel HazeClear**.

Giải pháp kế thừa kiến trúc WDMamba. Tên lớp mạng `WaveMamba`, tên checkpoint
và notebook được giữ để tương thích với trọng số hiện có; đổi tên dự án không
làm thay đổi kiến trúc hoặc tạo ra một checkpoint mới.

## Mục tiêu đề tài

- Xây dựng và hoàn thiện mô hình khử sương ảnh hướng tới dữ liệu thực tế.
- Đánh giá chất lượng bằng PSNR/SSIM và chi phí bằng latency, throughput, VRAM, FLOPs.
- Tối ưu thành phần gây nghẽn mà vẫn duy trì chất lượng phục hồi.
- Chuẩn bị mô-đun tiền xử lý cho camera và các tác vụ nhận dạng.

## Giải pháp kỹ thuật

```text
Ảnh có sương -> DWT -> LFRN dùng Mamba -> IWT -> DEN dùng CNN -> Ảnh RGB
```

1. **Wavelet:** tách các dải tần; thành phần thấp tần chứa cấu trúc lớn,
   độ sáng và tương phản chịu ảnh hưởng mạnh của sương.
2. **LFRN:** phục hồi cấu trúc toàn cục bằng Mamba, selective scan hai chiều
   và thiết kế đa tỉ lệ với Haar Wavelet Downsampling.
3. **IWT:** tái dựng ảnh khử sương thô từ biểu diễn đã phục hồi.
4. **DEN:** CNN tăng cường chi tiết, kết hợp Frequency Domain Enhancement
   xử lý biên độ và pha trong miền Fourier để tạo ảnh cuối cùng.

Đây là luồng tuần tự, không phải hai nhánh LFRN/DEN song song. Mamba nằm trong
LFRN, còn DEN dùng CNN. Trong huấn luyện, mã nguồn kết hợp L1, SSIM, ràng buộc
tần số thấp, FFT và contrastive regularization; contrastive không chạy như
một bước riêng khi suy luận.

## Kết quả thực nghiệm

### Benchmark tổng hợp

| Tập dữ liệu | PSNR (dB) | SSIM | Thời gian (ms/ảnh) |
| --- | ---: | ---: | ---: |
| SOTS-Indoor | 30,31 | 0,9686 | 29,22 |
| SOTS-Outdoor | **32,35** | **0,9739** | 35,66 |
| O-HAZE | 27,25 | 0,9078 | 103,23 |
| I-HAZE | 15,48 | 0,8078 | 207,35 |

Nguồn: workbook `report/VDT2.xlsx`, sheet Benchmark, hàng WDMamba trong workspace
đề tài. Workbook và Final Report không nằm trong repository con này. Các tập
có kích thước và điều kiện đo riêng; không dùng thời gian giữa các hàng để
kết luận tốc độ tương đối nếu chưa chuẩn hóa protocol.

### Phép đo kiểm soát RESIDE-6K

1.000 ảnh test, NVIDIA A100-SXM4-80GB, FP32, TF32 tắt, batch 100, cạnh dài tối
đa 512 px, checkpoint `reside6k_32.15.pth`:

| Chỉ số | Giá trị |
| --- | ---: |
| PSNR | 31,3601 dB |
| SSIM | 0,971214 |
| Thông lượng model GPU theo batch | 28,7589 ảnh/giây |
| Forward quy đổi theo ảnh | 34,7718 ms/ảnh |
| Peak allocated VRAM | 22,9758 GiB |
| FLOPs ước tính | 288,3 GFLOPs/ảnh |

Nguồn đo cục bộ: `phase 1/summary.csv` và `phase 1/metadata.json` từ gói report.
Các file kết quả người chạy mang về không mặc định được theo dõi trên GitHub.
Metric dùng RGB uint8, Gaussian SSIM sigma 1,5, không crop biên.
FLOPs là ước tính, chưa bao phủ đầy đủ phép toán elementwise.

Throughput theo batch không phải FPS camera trực tiếp; thời gian quy đổi không
phải latency của một ảnh chạy riêng. PSNR hiện thấp hơn giá trị paper 32,15 dB,
nên còn cần rà soát khác biệt protocol trước khi kết luận tái lập tương đương.

## Repository đã có gì

| Thành phần | Nội dung |
| --- | --- |
| `basicsr/` | Kiến trúc, dataset, loss, công cụ huấn luyện |
| `options/` | Cấu hình RESIDE-6K, Haze4K, O-HAZE, NH-HAZE, Dense-Haze |
| `inference.py`, `psnr_ssim.py` | Script suy luận và đánh giá kế thừa, cần cấu hình theo dữ liệu |
| [Benchmark các tập chuẩn](benchmark_sots_wdmamba.ipynb) | Môi trường, dữ liệu/checkpoint, suy luận và xuất kết quả |
| [Benchmark và profiling](phase%201/phase1_baseline_wdmamba.ipynb) | Đo chất lượng, tài nguyên và chi phí từng khối; đóng gói report |
| [Thử nghiệm mô hình nhẹ](phase%202/phase2_lightweight_rgb_wdmamba.ipynb) | Tìm cấu hình, huấn luyện và đánh giá đánh đổi chất lượng/tốc độ |
| [Corrective RGB refinement](phase%202.1/phase2_1_corrective_rgb_wdmamba.ipynb) | Cấu hình cải thiện tiếp theo, cần chạy để xác nhận kết quả |
| [Kế hoạch cải tiến](Improve%20Plan.md) | Điều kiện đánh giá, báo cáo và quyết định tiếp tục |

Tên thư mục thí nghiệm được giữ để truy vết lịch sử. Checkpoint/dữ liệu tải về
và kết quả chưa theo dõi không được tự động đưa lên GitHub.

## Bắt đầu sử dụng

URL GitHub hiện được giữ nguyên để các notebook Colab tiếp tục clone đúng repo:

```bash
git clone https://github.com/Darkin72/WDMamba.git "Haze Clear"
cd "Haze Clear"
```

Đường dẫn có khoảng trắng cần đặt trong dấu nháy. Với Colab, mở notebook
benchmark phù hợp và chạy lần lượt từ cấu hình đến xuất ZIP. Notebook tự tạo
môi trường riêng, tải mã nguồn và ghi log; không cần dùng tên thư mục cục bộ
làm tên thư mục runtime trên Colab.

Môi trường đã đo RESIDE-6K dùng PyTorch 2.8.0+cu128 và CUDA 12.8 trên A100.
GPU khác, đặc biệt Blackwell, cần xác nhận wheel/kernel `mamba-ssm` và
`causal-conv1d` tương thích bằng cell kiểm tra môi trường trước khi benchmark.
Không coi cấu hình A100 là bằng chứng đã kiểm thử trên mọi GPU.

### Huấn luyện cục bộ

Trên Linux/CUDA, dùng môi trường đã cài PyTorch và các phụ thuộc tương thích
theo notebook. Các file YAML còn chứa đường dẫn dataset của môi trường gốc;
hãy sửa `dataroot_gt`, `dataroot_lq`, checkpoint, batch và số worker trước khi chạy:

```bash
python basicsr/train.py -opt options/train_WDMamba_RESIDE_6K.yml
```

Các script `train_*.sh` kế thừa cũng chứa đường dẫn tuyệt đối cũ. Lệnh trên dùng
đường dẫn tương đối, không phụ thuộc tên thư mục repository.

`inference.py` kế thừa hiện yêu cầu ground truth để tính PSNR và ghép tên theo
quy tắc Haze4K (`tên_trước_dấu_gạch_dưới.png`). Script còn gọi
`restoration_network`, không có trong lớp `WaveMamba` hiện tại, nên cần cập nhật
API trước khi sử dụng. Đây chưa phải API suy luận cho camera không có ảnh tham
chiếu. Ưu tiên notebook cho benchmark đã chuẩn hóa.

## Khả năng tái lập

Mỗi lượt đo cần lưu commit, checkpoint, GPU, phiên bản CUDA/PyTorch, precision,
batch, `max_side`, split, số ảnh, quy tắc ghép cặp và cách tính metric.
Gói report cần có summary, metric chi tiết, log, metadata, profiling và failure
case khi có lỗi. Không thay đổi metadata của lần chạy cũ khi đổi tên dự án.

## Hướng phát triển và mức độ hoàn thiện

Mô hình và benchmark đã hoạt động. Profiling xác định DEN chiếm **71,71%**
thời gian đo từng khối, LFRN chiếm **25,89%**. Trọng tâm tiếp theo là tối ưu DEN.

Mục tiêu, chưa phải kết quả đạt được: giảm ít nhất 20% thời gian suy luận, giữ
PSNR từ 31 dB và giảm không quá 0,3 dB so với cấu hình hiện tại trên cùng protocol.
Các thử nghiệm mô hình nhẹ chưa được coi là thay thế tương đương chất lượng.

Hướng ứng dụng là tận dụng ảnh RGB từ camera hiện có để tiền xử lý cho giám sát
và nhận dạng. Cần bổ sung API, kiểm thử video, đo trên phần cứng đích và đánh giá
mAP/recall của tác vụ sau khử sương. Lợi ích kinh tế và giảm công kiểm tra thủ
công cần lượng hóa bằng thử nghiệm nghiệp vụ; chưa có cam kết triển khai edge
hoặc thời gian thực.

## Nguồn kiến trúc và ghi nhận

- Jie Sun, Heng Liu, Yongzhen Wang, Xiao-Ping Zhang, Mingqiang Wei.
  *WDMamba: When Wavelet Degradation Prior Meets Vision Mamba for Image Dehazing*.
  IEEE TCSVT, 2025. [Paper](https://arxiv.org/abs/2505.04369),
  DOI: 10.1109/TCSVT.2025.3614173.
- Albert Gu và Tri Dao. *Mamba: Linear-Time Sequence Modeling with Selective
  State Spaces*. [Paper](https://arxiv.org/abs/2312.00752), 2024.
- Các thành phần kế thừa: [BasicSR](https://github.com/xinntao/BasicSR) và
  [Wave-Mamba](https://github.com/AlexZou14/Wave-Mamba).

Viettel HazeClear là tên giải pháp của đề tài, không phải tuyên bố phát minh
lại kiến trúc gốc. Giữ thông tin tác giả và tuân thủ điều khoản của mã nguồn,
dữ liệu, checkpoint khi phân phối hoặc tích hợp sản phẩm.
