"""Build a standalone Colab notebook, embedding both runners for portability."""
from pathlib import Path
import nbformat as nbf

HERE = Path(__file__).resolve().parent
old = nbf.read(HERE.parent/'phase 3/final_o_hazy_fast_wdmamba.ipynb', as_version=4)
nb = nbf.v4.new_notebook(metadata=dict(old.metadata))
nb.metadata['colab']['name'] = 'phase3_1_psnr_fps_gflops.ipynb'
nb.metadata['wdmamba_runner_version'] = 'v2.1-fusion-fix-auto-resume'


def md(source):
    nb.cells.append(nbf.v4.new_markdown_cell(source))


def code(source):
    nb.cells.append(nbf.v4.new_code_cell(source))


md('''# Fast-WDMamba V2: PSNR, FPS và GFLOPs trên nhiều bộ dữ liệu

Mặc định chọn **O-HAZY** và tiếp tục từ checkpoint Phase 3. Có thể đổi `DATASET_NAME` thành `reside6k`, `sots-its`, `sots-ots` hoặc `i-haze` ở cell cấu hình.
Kết quả cũ được giữ nguyên. Notebook này không chạy lại search hoặc train từ đầu.

**Bản v2.1 đã sửa lỗi deploy** `Inference tensors do not track version counter`.
Fusion tạo trọng số ngoài inference mode, CUDA Graph mặc định tắt. `RUN_MODE='auto'` tự dùng
checkpoint V2 đã lưu để benchmark lại sau lỗi; chưa có checkpoint thì chạy training bình thường.

- **Phục hồi PSNR:** fine-tune DEN trên crop 384, sau đó fine-tune toàn mạng trên **ảnh đầy đủ** ở max-side 768; GT MSE + 0,02 L1 + 0,1 KD MSE, EMA và early stopping bằng validation. Không giảm KD về 0 đột ngột.
- **Tăng FPS:** fusion Conv–BN và thử CUDA Graph ở FP32, chọn runtime nhanh hơn trên validation trước test. Không giảm độ phân giải hoặc đổi FP16 để so với FP32 cũ.
- **GFLOPs:** đếm Conv/Linear và phép chiếu SS2D; tách ước tính selective scan, FFT, Haar. Báo cáo 256×256 và từng kích thước test sau padding. `1 MAC = 2 FLOPs`; GFLOPs/ảnh khác GFLOP/s.

V2 giữ số block nên GFLOPs gần như giữ nguyên so Phase 3. CUDA Graph giảm overhead thực thi; không gọi đó là giảm độ phức tạp kiến trúc. **Chưa có kết quả GPU cho V2**: chỉ kết luận tăng cả PSNR và FPS khi báo cáo mới xác nhận.''')
md('''## 1. Checkout và cấu hình

Dùng dataset đã có trong checkout VDT_GD2. Mặc định là `dataset/O-HAZY`; không tải lại dataset. Selector hợp lệ: `reside6k`, `sots-its` (SOTS Indoor/ITS), `sots-ots` (SOTS Outdoor/OTS), `o-hazy`, `i-haze`. SOTS chuẩn chỉ evaluation-only; muốn fine-tune phải đặt `TRAIN_ROOT` tới bộ train riêng và không được trỏ vào test.

Nếu chạy trong phiên Colab Phase 3, notebook tự tìm checkpoint cũ. Nếu mở một runtime mới, có thể upload checkpoint ở cell 3 hoặc đặt `PHASE3_CHECKPOINT` tới Drive. Chỉ checkpoint teacher được tự tải nếu chưa có.''')
import ast
core_source = (HERE/'phase3_core.py').read_text(encoding='utf-8')
spec_node = next(n for n in ast.parse(core_source).body
                 if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'DATASET_SPECS' for t in n.targets))
specs = ast.literal_eval(spec_node.value)
setup = (HERE/'notebook_setup.py').read_text(encoding='utf-8').replace('__DATASET_SPECS__', repr(specs))
code(setup)
md('''## 2. Môi trường GPU riêng

Tái sử dụng môi trường Python 3.11/CUDA từ Phase 3 nếu có. Cell kiểm tra CUDA/Mamba trước train. Tất cả kết quả chính là FP32 và TF32 tắt. Phần tính GFLOPs dùng hook phân tích trực tiếp, không cần cài THOP và không để lại hook/buffer profiling trên model.''')
code(old.cells[4].source)
md('''## 3. Checkpoint student

O-HAZY tiếp tục checkpoint Phase 3 đã có; nếu thiếu, cell cho phép upload. Các dataset khác để trống `PHASE3_CHECKPOINT` để khởi tạo bằng nén teacher, hoặc chỉ định student đã train đúng dataset. Với SOTS chỉ đánh giá, student tùy chọn phải thuộc RESIDE-6K. Teacher đặt ở `CHECKPOINT`, không đặt vào `PHASE3_CHECKPOINT`.''')
code('''if DATASET_NAME == 'o-hazy' and not CONFIG['phase3_checkpoint']:
    candidates = [
        PROJECT_ROOT/'solution/Haze Clear/phase 3/fast_wdmamba_o_hazy_best.pth',
        Path('/content/wdmamba_o_hazy_final/results/fast_wdmamba_o_hazy_best.pth'),
        Path('/content/fast_wdmamba_o_hazy_best.pth'),
    ]
    found = next((p for p in candidates if p.is_file()), None)
    if found is None:
        from google.colab import files
        uploaded = files.upload()
        names = [name for name in uploaded if name.endswith('.pth')]
        if len(names) != 1:
            raise ValueError('Chọn đúng một checkpoint Phase 3 .pth.')
        found = Path(names[0]).resolve()
    CONFIG['phase3_checkpoint'] = str(found)
if CONFIG['phase3_checkpoint'] and not Path(CONFIG['phase3_checkpoint']).is_file():
    raise FileNotFoundError(CONFIG['phase3_checkpoint'])
print('Student checkpoint:', CONFIG['phase3_checkpoint'] or 'nén teacher; không tự nạp checkpoint O-HAZY')''')
md('''## 4. Mã chạy tự chứa

Notebook chứa toàn bộ mã core Phase 3 và runner V2. Teacher, checkpoint Phase 3 và dữ liệu được kiểm tra SHA256 trước train. Chỉ train được dùng cho cache/KD; validation chọn checkpoint và runtime, test chỉ dùng lập bảng cuối.

Fine-tuning dùng ảnh đầy đủ trong bước joint để khớp ngữ cảnh Mamba/FFT lúc đánh giá. Checkpoint ban đầu cũng được chấm validation và lưu; checkpoint tệ hơn trên validation không ghi đè bản tốt nhất. Các checkpoint lưu là **trọng số eager chưa fusion** để nạp lại đúng kiến trúc.''')
core = (HERE/'phase3_core.py').read_text(encoding='utf-8')
runner = (HERE/'refine_o_hazy.py').read_text(encoding='utf-8')
code("CORE_SOURCE = r'''\n"+core+"\n'''\nRUNNER_SOURCE = r'''\n"+runner+"\n'''\n"+'''
import ast
ast.parse(CORE_SOURCE); ast.parse(RUNNER_SOURCE)
(WORK_ROOT/'phase3_core.py').write_text(CORE_SOURCE, encoding='utf-8')
RUNNER_PATH = WORK_ROOT/'refine_o_hazy.py'
RUNNER_PATH.write_text(RUNNER_SOURCE, encoding='utf-8')
def run_phase(phase):
    CONFIG_PATH.write_text(json.dumps(CONFIG, indent=2), encoding='utf-8')
    run_command([PYTHON, '-u', RUNNER_PATH, CONFIG_PATH, phase], cwd=REPO_ROOT, log_name=phase+'.log')
print('Runner v2.1: fusion fix + auto-resume; syntax PASS')''')
md('''## 5. Chuẩn bị manifest và train/đánh giá V2

Giữ `EVAL_MAX_SIDE=768` để so Phase 3. `EVAL_ONLY=None` tự train nếu có train/val; SOTS chuẩn chỉ đánh giá. `EVAL_ONLY=False` yêu cầu train/val riêng. `TRAIN_ROOT` dành cho ITS/OTS tách khỏi test SOTS. I-HAZE giữ split trong repo; RESIDE-6K giữ train/test có sẵn và tách validation theo scene từ train nếu thiếu val. Nhiều ảnh hazy chung clear luôn nằm cùng split.

Cache RAM có giới hạn `MAX_CACHE_IMAGES`, lấy mẫu từ toàn bộ train; không chọn một tập con cố định cho RESIDE-6K. Crop tự giới hạn theo ảnh nhỏ. Nếu thiếu VRAM, giảm số bước full-image hoặc điều chỉnh cấu hình rồi ghi rõ protocol; notebook không tự giảm độ phân giải.

Mặc định `TRY_CUDA_GRAPH=False`: dùng fusion eager, tránh lỗi capture của CUDA/Mamba. Có thể bật để thử trên validation; chỉ dùng khi output gần eager và latency thấp hơn. Capture/setup không tính vào latency; copy vào static input có tính trong replay latency.

Mặc định `RUN_MODE='auto'`: nếu checkpoint `fast_wdmamba_v2_best.pth` còn trong output, notebook chỉ benchmark lại và kiểm tra nguồn checkpoint, không train lại. Chưa có checkpoint thì train/đánh giá theo dataset. Chọn `run` để chủ động train lại hoặc `benchmark` để bắt buộc dùng checkpoint đã có. Không thay bằng `EVAL_ONLY=True`, vì đó là chế độ đánh giá cross-dataset, không phải phục hồi lượt fine-tune.''')
code("if RUN_MODE not in ('auto', 'run', 'benchmark'):\n    raise ValueError('RUN_MODE phải là auto, run hoặc benchmark')\nrun_phase('prepare')\nrun_phase(RUN_MODE)")
md('''## 6. Bảng so sánh PSNR, FPS và GFLOPs

Mọi model đều FP32, batch 1, cùng dataset và evaluator. Baseline là `Phase3` nếu có student checkpoint, hoặc `Compressed_init` trước fine-tune. SOTS chỉ đánh giá xuất `Student_eager`/`Student_deploy`, trạng thái `EVALUATION_ONLY`; không dùng gate tăng PSNR để claim đã train cải tiến.
`V2_eager` chỉ thay trọng số; `V2_deploy` thêm tối ưu runtime. Gate kiểm tra: tăng PSNR ít nhất 0,10 dB so Phase 3, FPS triển khai tăng ít nhất 3%, giảm PSNR không quá 0,30 dB so teacher và PSNR deployment lệch eager không quá 0,01 dB.

**Lưu ý GFLOPs:** số counted không bao gồm activation, normalization, pooling, bias và phép elementwise. Selective scan/FFT/Haar là ước tính theo công thức công khai trong runner; tổng estimated không phải instruction count đầy đủ. ConvTranspose dùng quy ước MAC dày gồm vùng padding.

Test Phase 3 đã được xem để định hướng cải tiến này; đây là một vòng phát triển tiếp, không phải bộ holdout mới. Năm scene chưa đủ để khẳng định tổng quát hóa.''')
code('''from IPython.display import Markdown, display
import csv
display(Markdown((OUTPUT_ROOT/'final_report.md').read_text(encoding='utf-8')))
with (OUTPUT_ROOT/'summary.csv').open(encoding='utf-8') as f:
    for row in csv.DictReader(f):
        print(row['model'], 'counted GFLOPs 256:', row['GFLOPs_256_counted'],
              'supplemental:', row['GFLOPs_256_supplemental'],
              'test estimated mean:', row['GFLOPs_test_estimated_mean'])''')
md('''## 7. Xuất checkpoint và toàn bộ số đo

`fast_wdmamba_v2_best.pth`, `history_v2.csv`, `summary.csv`, CSV GFLOPs/timing từng scene, ảnh dự đoán và metadata được lưu riêng, không ghi đè kết quả Phase 3.

Để dựng model: import `phase3_core`, gọi `teacher_and_student(config, teacher_path)` với `den_blocks=1`, rồi nạp `checkpoint_state(student_path)` vào student bằng `strict=True`. Inference FP32, pad bội 32 và crop lại. Fusion/CUDA Graph chỉ bật sau khi model ở eval; không nạp checkpoint eager vào model đã fusion.''')
code('''import shutil
for src in (WORK_ROOT/'phase3_core.py', RUNNER_PATH, WORK_ROOT/'inputs.json',
            WORK_ROOT/'split_manifest.csv', WORK_ROOT/'prepare.log', WORK_ROOT/(RUN_MODE+'.log')):
    if src.is_file(): shutil.copy2(src, OUTPUT_ROOT/src.name)
archive = shutil.make_archive(str(WORK_ROOT/('fast_wdmamba_v2_' + DATASET_NAME + '_results')), 'zip', OUTPUT_ROOT)
print('Kết quả:', archive)
try:
    from google.colab import files
    files.download(archive)
except ImportError:
    pass''')
nbf.validate(nb)
nbf.write(nb, HERE/'phase3_1_psnr_fps_gflops.ipynb')
print('Created', HERE/'phase3_1_psnr_fps_gflops.ipynb')

# A standalone recovery cell reads the existing runtime config/manifest. No setup,
# data download, prepare, optimizer or checkpoint selection is performed.
recovery = nbf.v4.new_notebook(metadata=dict(nb.metadata))
recovery.metadata['colab']['name'] = 'recover_v2_benchmark.ipynb'
recovery.cells = [nbf.v4.new_markdown_cell('''# Phục hồi benchmark V2 — không train lại

Copy cell code dưới đây vào **notebook Colab đang giữ checkpoint V2**, rồi chạy.
Cell đọc config/manifest đã có, cập nhật runner đã sửa lỗi inference tensor, tắt CUDA Graph và
khởi chạy benchmark trong process mới. Không cần restart/xóa runtime hoặc tải lại dữ liệu.
Mặc định dùng `/content/wdmamba_v2/o-hazy`; sửa `WORK_ROOT` nếu dataset khác.
Nếu runtime cũ đã mất, cần khôi phục checkpoint V2, baseline, teacher và dữ liệu trước.
'''), nbf.v4.new_code_cell(
    "CORE_SOURCE = " + repr(core) + "\nRUNNER_SOURCE = " + repr(runner) + "\n" + '''
from pathlib import Path
import json, subprocess, shutil, ast

WORK_ROOT = Path('/content/wdmamba_v2/o-hazy')
config_path = WORK_ROOT/'config.json'
if not config_path.is_file() or not (WORK_ROOT/'inputs.json').is_file():
    raise FileNotFoundError('Cần chạy trong runtime cũ còn config.json và inputs.json.')
cfg = json.loads(config_path.read_text(encoding='utf-8'))
output = Path(cfg['output_root'])
checkpoint = Path(cfg.get('v2_checkpoint') or output/'fast_wdmamba_v2_best.pth')
if not checkpoint.is_file():
    raise FileNotFoundError(f'Chưa tìm thấy checkpoint V2 đã train: {checkpoint}')
cfg['try_cuda_graph'] = False
python = Path('/content/wdmamba_o_hazy_env311/bin/python')
if not python.is_file():
    raise FileNotFoundError('Không có môi trường Python GPU cũ; cần phục hồi runtime trước.')
output.mkdir(parents=True, exist_ok=True)
for filename, source in [('phase3_core.py', CORE_SOURCE), ('refine_o_hazy.py', RUNNER_SOURCE)]:
    ast.parse(source)
    target = WORK_ROOT/filename
    backup = output/('before_recovery_' + filename)
    if target.is_file() and not backup.exists():
        shutil.copy2(target, backup)
    target.write_text(source, encoding='utf-8')
recovery_config = WORK_ROOT/'config_benchmark_recovery.json'
recovery_config.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding='utf-8')
args = [str(python), '-u', str(WORK_ROOT/'refine_o_hazy.py'), str(recovery_config), 'benchmark']
print('Chỉ benchmark checkpoint đã chọn:', checkpoint, flush=True)
with (WORK_ROOT/'benchmark_recovery.log').open('a', encoding='utf-8') as log:
    with subprocess.Popen(args, cwd=cfg['repo_root'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding='utf-8', errors='replace', bufsize=1) as proc:
        try:
            for line in proc.stdout:
                print(line, end='', flush=True)
                log.write(line); log.flush()
            rc = proc.wait()
        except BaseException:
            proc.kill(); proc.wait(); raise
if rc:
    raise subprocess.CalledProcessError(rc, args)
from IPython.display import Markdown, display
display(Markdown((output/'final_report.md').read_text(encoding='utf-8')))
print('Kết quả:', output)
for src in (WORK_ROOT/'phase3_core.py', WORK_ROOT/'refine_o_hazy.py', WORK_ROOT/'inputs.json',
            WORK_ROOT/'split_manifest.csv', recovery_config, WORK_ROOT/'benchmark_recovery.log'):
    if src.is_file() and src.resolve() != (output/src.name).resolve():
        shutil.copy2(src, output/src.name)
archive = shutil.make_archive(str(WORK_ROOT/'v2_benchmark_recovered'), 'zip', output)
print('Archive:', archive)
try:
    from google.colab import files
    files.download(archive)
except ImportError:
    pass
''')]
nbf.validate(recovery)
nbf.write(recovery, HERE/'recover_v2_benchmark.ipynb')
print('Created', HERE/'recover_v2_benchmark.ipynb')
