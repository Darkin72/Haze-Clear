# IMPROVED RESEARCH PLAN

## Physics-Guided Frequency-Adaptive Image Dehazing
### Improving the Efficiency, Restoration Quality, and Interpretability of WDMamba

**Loại nghiên cứu:** Image Restoration, Efficient Deep Learning, Physics-Informed Learning

**Mô hình nền:** WDMamba - *When Wavelet Degradation Prior Meets Vision Mamba for Image Dehazing*

**Thời gian dự kiến:** 12 tuần

**Trạng thái:** Research proposal; chưa có kết quả thực nghiệm cho kiến trúc mới.

**Nguyên tắc quyết định:** Không chuyển sang phase tiếp theo nếu báo cáo bắt buộc của phase hiện tại chưa hoàn tất. Mọi kết luận phải dựa trên cùng evaluator, cùng split, cùng loss protocol và cùng điều kiện benchmark.

---

# I. MỤC TIÊU VÀ PHẠM VI

## 1.1. Mục tiêu

Nghiên cứu kiểm tra giả thuyết sau:

> Có thể thay thế phần DEN có chi phí cao của WDMamba bằng một refinement network nhẹ hơn, trong đó wavelet detail và thông tin ASM được dùng như guidance, để giảm latency mà vẫn giữ chất lượng phục hồi chấp nhận được.

Ba mục tiêu được tách riêng:

1. **Efficiency:** giảm latency và peak VRAM so với WDMamba gốc.
2. **Restoration:** giữ chất lượng gần baseline dưới cùng giao thức đánh giá.
3. **Guidance và interpretability:** kiểm tra liệu transmission proxy và các contribution map có thực sự ảnh hưởng đến output hay chỉ là tín hiệu phụ bị network bỏ qua.

Transmission map chỉ được gọi là **transmission vật lý** khi có ground truth hoặc có quy trình sinh dữ liệu biết rõ \(t,A\). Trong các trường hợp còn lại, dùng thuật ngữ **transmission proxy** hoặc **ASM-conditioned map**.

## 1.2. Câu hỏi nghiên cứu

- RQ1: Lightweight refinement có thể giảm latency bao nhiêu với mức giảm PSNR được kiểm soát?
- RQ2: Wavelet refinement có tốt hơn RGB refinement khi giữ cùng ngân sách tham số và FLOPs không?
- RQ3: Transmission-conditioned guidance có cải thiện model nhẹ hơn so với model nhẹ không có guidance không?
- RQ4: Physics consistency loss có cải thiện chất lượng hoặc tính ổn định trên dữ liệu có kiểm soát không?
- RQ5: Contribution map có vượt qua random/shuffled baseline trong phép can thiệp không?

## 1.3. Giả thuyết và cách bác bỏ

| ID | Giả thuyết | Bác bỏ khi |
|---|---|---|
| H1 | Refinement nhẹ giảm latency mà không giảm PSNR quá 0.3 dB so với baseline tái lập | Khoảng tin cậy của ΔPSNR thấp hơn -0.3 dB hoặc latency không giảm đáng kể |
| H2 | Wavelet refinement tốt hơn RGB refinement cùng ngân sách | Không có cải thiện có ý nghĩa hoặc chỉ tăng chi phí |
| H3 | Transmission guidance giúp refinement tốt hơn unguided refinement | B3 không tốt hơn B2 sau khi match capacity và loss |
| H4 | Physics loss giúp tăng consistency hoặc chất lượng trên dữ liệu có ground truth \(t,A\) | Không cải thiện hoặc gây nghiệm suy biến |
| H5 | Contribution map faithful hơn random/shuffled map | Deletion/intervention AUC không vượt baseline đối chứng |

---

# II. RÀNG BUỘC TỪ IMPLEMENTATION GỐC

## 2.1. Luồng WDMamba phải được cố định

Implementation hiện tại thực hiện:

1. Haar DWT với thứ tự tensor **LL, HL, LH, HH**.
2. Chỉ đưa LL vào LFRN/Mamba.
3. Giữ nguyên ba high-frequency subbands.
4. IWT để tạo coarse image.
5. Đưa coarse image vào DEN.

Do đó, mọi mô tả và code mới phải giữ đúng thứ tự `LL, HL, LH, HH`. Không được dùng thứ tự `LL, LH, HL, HH` nếu không có phép chuyển đổi rõ ràng.

## 2.2. Loss baseline

B0 và mọi model ablation đầu tiên phải dùng cùng loss với implementation gốc:

- L1 + SSIM cho output cuối.
- LL supervision cho output của LFRN.
- Contrastive loss gồm conventional CR và SGCR.
- FFT loss chỉ bật nếu cấu hình baseline đang bật.

Wavelet loss mới chỉ được thêm trong một ablation riêng. Không thay đồng thời loss, kiến trúc và evaluator.

## 2.3. Giao thức đánh giá

Evaluator chính phải cố định:

- PSNR/SSIM theo cùng crop border và Y-channel protocol của BasicSR.
- Một split cố định, có manifest tên file và checksum nếu có thể.
- Latency batch 1 ở resolution 256x256 để so sánh FLOPs/paper.
- Latency ở resolution tự nhiên của benchmark để kiểm tra triển khai thực tế.
- FP32 là kết quả chính; FP16/BF16 báo cáo riêng.
- Có GPU warm-up, CUDA synchronization, số lần lặp và độ lệch chuẩn.
- Không tính data loading vào model latency; báo cáo riêng nếu cần.

Script inference cũ không được dùng làm evaluator cuối nếu khác protocol trên.

---

# III. KIẾN TRÚC ĐỀ XUẤT

## 3.1. Stage 1 - Coarse restoration

Với ảnh đầu vào (I):

\[
\{LL,HL,LH,HH\}=DWT(I)
\]

\[
\widehat{LL}=f_{LFRN}(LL)
\]

\[
I_0=IWT(\widehat{LL},HL,LH,HH)
\]

Ở phiên bản đầu, LFRN được lấy từ WDMamba gốc và freeze sau khi baseline được xác nhận. Ba high-frequency subbands của ảnh hazy chỉ là tín hiệu đầu vào, không được gọi là detail đã phục hồi.

## 3.2. Stage 2 - Transmission proxy và atmospheric light

Estimator nhẹ dự đoán:

\[
(\hat t,\hat A)=f_{ASM}(I,I_0)
\]

Trong đó \(\hat t\) có một channel và \(\hat A\) là global RGB vector. Estimator có thể hoạt động ở resolution thấp rồi upsample để tránh làm mất lợi ích latency.

Không dùng phép chia cứng \(1/\hat t\) trong prototype đầu tiên. Transmission chỉ được dùng như soft guidance hoặc bounded residual scale.

Physics loss:

\[
\hat I=Y\hat t+\hat A(1-\hat t), \qquad
L_{phy}=\|\hat I-I\|_1
\]

Physics loss không được xem là bằng chứng đủ để kết luận \(\hat t\) là transmission vật lý.

## 3.3. Stage 3 - Lightweight refinement

DWT trên \(I_0\), sau đó dùng shared feature extractor cho ba nhánh detail:

\[
\widehat H_k=H_k+M_k\odot R_k, \qquad k\in\{HL,LH,HH\}
\]

\[
M_k=\sigma(g_k(H_k,\operatorname{downsample}(\hat t)))
\]

Thiết kế khởi đầu:

- 32 channels.
- 2 residual blocks.
- Depthwise-separable convolution.
- Shared backbone, prediction head riêng cho từng subband.
- IWT sau refinement.
- Không dùng U-Block hoặc FFT module trong phiên bản đầu.

Một RGB lightweight baseline phải được xây dựng trước để xác định lợi ích thật sự của wavelet guidance.

---

# IV. THIẾT KẾ ABLATION

| ID | Kiến trúc | ASM | Physics loss | Mục đích |
|---|---|---:|---:|---|
| B0 | WDMamba gốc | Không | Không | Baseline tái lập |
| B1 | LFRN + lightweight RGB refinement | Không | Không | Đo lợi ích của lightweight design |
| B2 | LFRN + lightweight wavelet refinement | Không | Không | Đo lợi ích riêng của wavelet |
| B3 | B2 + transmission-conditioned gating | Có | Không | Đo lợi ích của guidance |
| B4 | B3 + physics consistency loss | Có | Có | Đo lợi ích của physics loss |
| B5 | B3/B4 + supervision \(t,A\) | Có | Có | Kiểm tra ý nghĩa vật lý trên dữ liệu có nhãn |

B1-B4 phải được capacity-match theo parameter/FLOPs gần tương đương. B5 chỉ áp dụng trên synthetic subset có ground truth \(t,A\) hoặc dữ liệu được sinh lại với tham số biết trước.

Control bắt buộc:

- Transmission bị shuffle theo không gian.
- Transmission ngẫu nhiên cùng histogram.
- Tắt riêng từng nhánh HL/LH/HH.
- Bỏ physics loss nhưng giữ nguyên mọi thứ khác.
- So sánh cùng model ở FP32 và FP16/BF16.

---

# V. DATASET VÀ METRIC

## 5.1. Dataset protocol

- **RESIDE-6K:** dataset chính để tái lập baseline và chọn kiến trúc.
- **Haze4K:** benchmark synthetic độc lập.
- **HSTS-Synthetic:** giữ lại vì paper gốc báo cáo kết quả trên dataset này.
- **O-HAZE:** paired real-world benchmark; chỉ có 5 ảnh test theo protocol paper, không dùng để tuning.

Phải ghi rõ mỗi kết quả là:

1. Train và test cùng dataset; hoặc
2. Train trên RESIDE-6K rồi test cross-dataset.

Hai loại kết quả không được gộp vào một bảng.

## 5.2. Metrics

**Restoration:** PSNR, SSIM; LPIPS và color error là phụ trợ.

**Efficiency:** parameter count, counted FLOPs, estimated FLOPs, latency, FPS, peak VRAM.

**ASM validity:** MAE/RMSE của \(t,A\) chỉ khi có reference.

**Faithfulness:** deletion/insertion AUC, intervention sensitivity và chênh lệch so với random/shuffled baseline.

---

# VI. KẾ HOẠCH 12 TUẦN VÀ CỔNG QUYẾT ĐỊNH

## Phase 0 - Chuẩn hóa môi trường và protocol
### Tuần 1

### Việc phải làm

- Khóa Python, PyTorch, CUDA, Mamba và dependency versions.
- Xác định commit code, checkpoint, GPU, precision và seed.
- Chuẩn bị dataset manifest, split và quy tắc đặt tên file.
- Viết evaluator thống nhất cho PSNR/SSIM/LPIPS.
- Viết unit test DWT/IWT: shape, thứ tự subband, reconstruction error.
- Kiểm tra forward B0 trên một batch nhỏ.

### Báo cáo bắt buộc

`P0_protocol_report.md` phải có:

- Environment manifest và commit hash.
- Checkpoint hash hoặc cách tạo checkpoint.
- Dataset manifest và split.
- Output của unit tests.
- So sánh evaluator mới với evaluator cũ.
- Một log inference tái lập được.

### Cổng G0 - Có được đi tiếp không?

**Đi tiếp nếu:** B0 chạy được end-to-end, DWT/IWT đúng, split cố định và evaluator cho kết quả ổn định qua ít nhất 3 lần chạy.

**Phải sửa lại nếu:** thiếu checkpoint, mismatch shape/subband, không xác định được split hoặc chênh lệch metric chưa giải thích được. Không bắt đầu huấn luyện model mới khi G0 chưa đạt.

## Phase 1 - Tái lập và profiling baseline
### Tuần 2

### Việc phải làm

- Đo B0 trên RESIDE-6K theo evaluator chuẩn.
- Profile riêng DWT, LFRN, IWT và DEN.
- Đo parameter, FLOPs, latency, peak VRAM.
- Đo ở 256x256 batch 1 và resolution tự nhiên.
- Warm-up GPU và dùng CUDA synchronization.

### Báo cáo bắt buộc

`P1_baseline_report.md` phải có:

- PSNR/SSIM/LPIPS của B0.
- Mean, standard deviation, p50 và p95 latency.
- Bảng latency theo module.
- Counted FLOPs và supplemental estimates tách riêng.
- Peak VRAM và precision.
- Danh sách bottleneck có bằng chứng.

### Cổng G1

**Đi tiếp nếu:** tái lập được baseline trong sai số đã giải thích trước; DEN thực sự là bottleneck; điều kiện benchmark ổn định.

**Phải sửa lại nếu:** DEN không chiếm phần đáng kể latency, hoặc số FLOPs không tương quan với latency. Khi đó thay đổi mục tiêu từ “thay DEN để giảm 50% latency” sang bottleneck thực tế.

## Phase 2 - Lightweight RGB baseline
### Tuần 3-4

### Việc phải làm

- Xây B1 với cùng input/output và cùng loss protocol B0.
- Thử tối đa 3 cấu hình channels/block depth.
- Không thêm ASM, wavelet loss hoặc physics loss.
- Train trước trên subset phát triển, sau đó train cấu hình được chọn trên full train split.

### Báo cáo bắt buộc

`P2_lightweight_report.md` phải có:

- Cấu hình từng model và số parameter.
- PSNR/SSIM trên development split và test split.
- Latency/VRAM theo cùng protocol P1.
- Quality-speed Pareto plot.
- Failure cases và artifact examples.
- Chi phí huấn luyện và số iteration.

### Cổng G2

**Đi tiếp nếu:** B1 giảm ít nhất 20% latency, giữ ΔPSNR không thấp hơn -0.3 dB so với B0, và không tạo artifact nghiêm trọng.

**Phải sửa lại nếu:** giảm chất lượng quá ngưỡng hoặc speedup không đủ. Khi đó điều chỉnh channels/block depth hoặc giữ lại một phần DEN; chưa thêm ASM.

## Phase 3 - Wavelet refinement
### Tuần 5

### Việc phải làm

- Xây B2 từ B1 với shared feature extractor cho HL/LH/HH.
- Kiểm tra đúng thứ tự subband bằng unit test.
- So sánh B1 và B2 với capacity gần tương đương.
- Thử wavelet reconstruction loss trong một ablation riêng, không gộp vào kết quả chính.

### Báo cáo bắt buộc

`P3_wavelet_report.md` phải có:

- Sơ đồ tensor và thứ tự subband.
- Bảng B1 vs B2 về quality, latency, VRAM, parameters.
- Kết quả tắt từng nhánh HL/LH/HH.
- So sánh có/không wavelet loss.
- Visual artifacts và lỗi tại biên.

### Cổng G3

**Đi tiếp nếu:** B2 tốt hơn hoặc Pareto tốt hơn B1 trong cùng ngân sách; không có nhánh subband bị bỏ qua.

**Phải sửa lại nếu:** B2 không tốt hơn B1 hoặc overhead IWT/gating làm mất speedup. Khi đó giữ B1 làm baseline chính và không thêm physics guidance.

## Phase 4 - ASM estimator và kiểm tra nghiệm suy biến
### Tuần 6-7

### Việc phải làm

- Xác định synthetic subset có ground-truth \(t,A\), hoặc sinh dữ liệu kiểm soát từ ảnh sạch.
- Huấn luyện estimator ở resolution thấp trước.
- Đánh giá \(\hat t,\hat A\) độc lập với output dehazing.
- Kiểm tra các nghiệm suy biến: \(t\approx1\), \(t\approx0\), map hằng, map nhiễu.
- Đo overhead estimator trong inference.

### Báo cáo bắt buộc

`P4_asm_report.md` phải có:

- Định nghĩa nguồn ground-truth \(t,A\).
- MAE/RMSE, bias và calibration của \(t,A\).
- Visualizations trên vùng trời, vật thể xa, biên và texture.
- Physics reconstruction error.
- Ablation supervised estimator vs physics-only estimator.
- Latency/VRAM overhead.
- Phân tích nghiệm suy biến và failure cases.

### Cổng G4

**Đi tiếp nếu:** estimator có chất lượng đo được trên dữ liệu có nhãn, overhead nằm trong ngân sách, và không xuất hiện nghiệm suy biến phổ biến.

**Chuyển sang training-only nếu:** estimator giúp loss nhưng không có ý nghĩa ở inference hoặc overhead quá lớn.

**Dừng nhánh ASM nếu:** không có dữ liệu kiểm chứng, estimator không ổn định hoặc không tốt hơn baseline đơn giản.

## Phase 5 - Physics-guided refinement
### Tuần 8

### Việc phải làm

- Huấn luyện B3, B4 và B5 theo factorial ablation.
- Giữ nguyên B2 làm control.
- So sánh learned gating với shuffled/random transmission.
- Đánh giá cả synthetic có nhãn và benchmark dehazing chính.

### Báo cáo bắt buộc

`P5_physics_ablation_report.md` phải có:

- Bảng B2-B5 với cùng seed/split/loss protocol.
- Paired ΔPSNR, ΔSSIM, Δlatency và confidence interval.
- Kết quả random/shuffled transmission.
- Tỷ lệ gate saturation và sensitivity của output theo \(\hat t\).
- Physics error và image reconstruction error.
- Kết luận rõ H3/H4 được hỗ trợ hay bác bỏ.

### Cổng G5

**Chọn physics-guided model nếu:** model thắng B2 trong cùng ngân sách và cải thiện không chỉ xuất hiện ở một seed hoặc một subset.

**Chọn lightweight non-physics model nếu:** B3/B4 không thắng B2. Đây là kết quả hợp lệ; không thêm ASM chỉ để giữ câu chuyện nghiên cứu.

## Phase 6 - Generalization và explainability
### Tuần 9-10

### Việc phải làm

- Đánh giá cấu hình được chọn trên RESIDE-6K, Haze4K, HSTS-Synthetic và O-HAZE theo protocol đã định.
- Tách rõ same-dataset và cross-dataset evaluation.
- Tạo contribution map bằng intervention từng subband.
- Chạy deletion/insertion với random và shuffled baselines.
- Phân tích failure cases và ảnh hưởng của haze density.

### Báo cáo bắt buộc

`P6_generalization_explainability_report.md` phải có:

- Benchmark table theo từng training protocol.
- Mean và độ lệch chuẩn trên các seed được chọn.
- Deletion/insertion AUC.
- Intervention sensitivity theo HL/LH/HH.
- So sánh contribution map với random/shuffled map.
- Ít nhất 10 failure cases có phân loại nguyên nhân.
- Giới hạn của ASM trên ảnh không đồng nhất.

### Cổng G6

**Đi tiếp nếu:** model không chỉ tốt trên RESIDE-6K, contribution map vượt random baseline và failure cases có thể giải thích.

**Phải hạ claim nếu:** model chỉ tốt trên synthetic hoặc map không faithful. Khi đó báo cáo đây là lightweight restoration, không gọi là interpretable physical dehazing.

## Phase 7 - Repeated trials, reproducibility và viết bài
### Tuần 11-12

### Việc phải làm

- Chạy ít nhất 3 seed cho B0, model nhẹ được chọn và model cuối.
- Benchmark latency cuối cùng sau khi cố định graph, precision và warm-up.
- Lưu config, checkpoint hash, logs, metric CSV và figures.
- Viết paper với cả kết quả âm tính và limitation.

### Báo cáo bắt buộc

`P7_final_reproducibility_report.md` phải có:

- Mean ± standard deviation và confidence interval.
- Final quality-speed Pareto curve.
- Hardware/software manifest.
- Lệnh chạy lại từng bảng chính.
- Danh sách checkpoint và checksum.
- Bảng đối chiếu claim với bằng chứng thực nghiệm.

### Cổng G7

Chỉ hoàn tất paper khi mọi claim trong phần abstract/contribution đều trỏ được tới một bảng hoặc phép đo trong báo cáo cuối.

---

# VII. TIÊU CHÍ THÀNH CÔNG

## 7.1. Tiêu chí chính

Model được xem là đạt mục tiêu nếu đồng thời:

1. ΔPSNR so với B0 không thấp hơn -0.3 dB trên protocol chính.
2. Latency giảm ít nhất 50% trong benchmark batch 1, 256x256, FP32 đã khóa.
3. Peak VRAM giảm hoặc không tăng đáng kể.
4. Model thắng lightweight control ở cùng ngân sách nếu claim có physics/wavelet benefit.
5. Kết quả ổn định qua các seed được chọn.

## 7.2. Tiêu chí phụ

- Cải thiện trên ít nhất một benchmark ngoài dataset train.
- Giảm artifact ở vùng haze dày.
- Contribution map vượt random/shuffled baseline.
- Transmission proxy có metric vật lý chỉ khi có reference.

## 7.3. Không được kết luận

Không được kết luận rằng:

- transmission proxy là depth hoặc transmission thật nếu không có ground truth;
- physics loss tự nó chứng minh network học đúng ASM;
- FLOPs thấp đồng nghĩa latency thấp;
- attention/gating map là explanation faithful nếu chưa có intervention test.

---

# VIII. RỦI RO VÀ CÁCH ỨNG PHÓ

| Rủi ro | Tác động | Cách xử lý |
|---|---|---|
| Lightweight model mất detail | PSNR giảm | Tăng channels có kiểm soát hoặc giữ một phần DEN |
| Wavelet branch có overhead | Không đạt speedup | Shared backbone, giảm resolution estimator, benchmark thực tế |
| ASM nghiệm suy biến | Map không có ý nghĩa | Supervision \(t,A\), bounded guidance, hoặc bỏ claim vật lý |
| Physics loss làm sai màu | Artifact | Giảm trọng số, chỉ dùng training-only, kiểm tra color error |
| Chỉ tốt trên synthetic | Kém real-world | Cross-dataset test và báo cáo failure cases |
| Attribution map không faithful | Explainability yếu | Intervention, random/shuffle controls, hạ claim |
| Thiếu GPU/checkpoint/dependency | Không tái lập được | Khóa environment, checkpoint hash và report G0 |

---

# IX. ĐÓNG GÓP DỰ KIẾN

Chỉ tuyên bố đóng góp sau khi có bằng chứng:

1. **Efficient refinement:** refinement nhẹ hơn DEN với quality-speed trade-off được đo rõ.
2. **Frequency-guided restoration:** wavelet refinement thắng RGB control trong cùng ngân sách.
3. **ASM-conditioned guidance:** transmission proxy giúp cải thiện refinement hoặc tính ổn định trong điều kiện đã xác định.
4. **Intervention-based analysis:** contribution map được kiểm tra bằng can thiệp, không chỉ trực quan hóa.

Nếu H3 hoặc H4 bị bác bỏ, kết luận hợp lệ là lightweight wavelet restoration không cần ASM, kèm theo phân tích nguyên nhân.

