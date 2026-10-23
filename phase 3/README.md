# Fast-WDMamba trên O-HAZY

Mở [final_o_hazy_fast_wdmamba.ipynb](final_o_hazy_fast_wdmamba.ipynb) trên Colab GPU và chạy cell theo thứ tự. Notebook lấy trực tiếp `dataset/O-HAZY` trong checkout VDT_GD2; không tải ZIP dataset ngoài repo. Trên Colab chưa có checkout, notebook clone repo và sparse checkout `dataset/O-HAZY` cùng `solution/Haze Clear`.

Cải tiến cuối giữ DWT/LFRN/IWT và rút DEN xuống một U-block mỗi group, sau đó distillation/fine-tuning O-HAZY. Báo cáo `PASS` chỉ khi ΔPSNR ≥ -0,30 dB và speedup ≥ 2x ở cả kích thước đánh giá lẫn 256×256. `REVIEW` là kết quả chưa đạt mục tiêu, không phải kết luận chất lượng.

Kết quả chạy nằm trong `wdmamba_o_hazy_final/results`: checkpoint student, manifest SHA256, CSV metric, latency, ảnh dự đoán, log và `final_report.md`.

## Vòng cải tiến tiếp theo

Mở [phase3_1_psnr_fps_gflops.ipynb](../phase 3.1/phase3_1_psnr_fps_gflops.ipynb) để tiếp tục từ checkpoint Phase 3. Vòng này fine-tune lại với crop lớn và ảnh đầy đủ, thử Conv–BN fusion/CUDA Graph ở FP32, đồng thời báo cáo GFLOPs counted và supplemental. Kết quả được ghi riêng trong `phase 3.1` và không ghi đè run cũ.
