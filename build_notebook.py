from pathlib import Path
import nbformat as nbf
root=Path('solution/Haze Clear/phase 3')
nb=nbf.v4.new_notebook()
nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11'},'accelerator':'GPU','colab':{'name':'final_o_hazy_fast_wdmamba.ipynb','provenance':[]}}
def md(s):nb.cells.append(nbf.v4.new_markdown_cell(s))
def code(s):nb.cells.append(nbf.v4.new_code_cell(s))
md('''# Fast-WDMamba: cải tiến cuối, chỉ train và đánh giá O-HAZY

**Chạy trên Google Colab/Linux có GPU CUDA**, chọn Runtime → GPU rồi chạy lần lượt. Notebook dùng **`dataset/O-HAZY` có sẵn trong repo VDT_GD2**; trên Colab tự clone repo nếu chưa có. Sau đó lấy checkpoint O-HAZY, fine-tune một cấu hình và xuất báo cáo. Không cần tải ZIP dataset ngoài repo, upload báo cáo Phase 1/2 hoặc chạy RESIDE.

**Cấu hình cuối:** giữ nguyên `DWT → LFRN/Mamba → IWT`; rút DEN từ 4 xuống **1 U-block cho mỗi group**, giữ FFT, channel attention, convolution cuối group và mọi trọng số tương ứng. Đây là phương án giữ một phần DEN để tránh mức giảm gần 5 dB của refinement RGB trước đó. Không thay bằng một CNN nhỏ khởi tạo lại.

**Train nhanh hai bước:** cache ảnh coarse và teacher trên train một lần → fine-tune DEN bằng GT + distillation → fine-tune toàn mạng với learning rate nhỏ, EMA và giảm dần distillation. Teacher không có trong đường inference cuối.

**Mục tiêu kiểm chứng:** PSNR giảm không quá **0,30 dB** và FPS tăng ít nhất **2 lần**, cả ở 256×256 và kích thước đánh giá O-HAZY. Đây là mục tiêu; chưa phải kết quả đã đo. Model/độ phân giải/precision được cố định trước test, báo cáo `REVIEW` nếu chưa đạt.

Checkpoint pretrained O-HAZY đã học ảnh 1–40. Bốn scene validation dưới đây chỉ dùng để chọn checkpoint compression; test 41–45 luôn giữ riêng. Thí nghiệm này là **compression/fine-tuning** với loss mới, không phải ablation from-scratch hay thí nghiệm physics.''')
md('''## 1. Cấu hình

Mặc định lấy dataset từ `PROJECT_ROOT / 'dataset/O-HAZY'`. Nếu đang mở notebook trong checkout VDT_GD2, cell tự tìm checkout đó; trên Colab chưa có repo thì clone `Darkin72/VDT_GD2` với sparse checkout chỉ lấy O-HAZY và mã Haze Clear. Có thể đặt `LOCAL_PROJECT` để dùng một checkout đã có hoặc `DATA_ROOT` để chỉ định thư mục dữ liệu khác. Checkpoint vẫn được tải riêng nếu chưa cung cấp `CHECKPOINT`/`CHECKPOINT_ZIP`.

Notebook ghép theo scene ID và chia lại **trong manifest, không di chuyển file**: 36 train / 4 val (5, 15, 25, 35) / 5 test (41–45). Do đó không kế thừa split 32/6/7 hiện có trong workspace, vốn khác protocol paper và có thể đưa ảnh teacher đã học vào test.

`EVAL_MAX_SIDE=768` dành cho vòng chạy nhanh; đặt `0` trước khi chạy nếu cần ảnh gốc. Đây là hai protocol riêng. Điểm ở 768 không được so trực tiếp với PSNR paper ở 1600×1200. Giữ nguyên cấu hình test trong một thí nghiệm.''')
code(r'''from pathlib import Path
import json, os, platform, subprocess, sys, time

REPO_URL = 'https://github.com/Darkin72/WDMamba.git'
PROJECT_URL = 'https://github.com/Darkin72/VDT_GD2.git'
LOCAL_PROJECT = ''          # checkout VDT_GD2 đã có; trống = tự tìm hoặc clone trên Colab
LOCAL_REPO = ''             # checkout WDMamba/Haze Clear trên Linux nếu đã có
DATA_ROOT = ''              # trống = PROJECT_ROOT / 'dataset/O-HAZY', không tải ZIP dataset
CHECKPOINT = ''             # file Ohaze_1200_1600_27.22.pth; trống = tự tải archive
CHECKPOINT_ZIP = ''         # ZIP WDMamba_ckpts đã tải sẵn nếu Drive bị quota
PROJECT_ROOT = Path(LOCAL_PROJECT).expanduser().resolve() if LOCAL_PROJECT else next(
    (p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'dataset/O-HAZY').is_dir()),
    Path('/content/VDT_GD2'))
WORK_ROOT = Path('/content/wdmamba_o_hazy_final')
OUTPUT_ROOT = WORK_ROOT / 'results'
ENV_ROOT = Path('/content/wdmamba_o_hazy_env311')
PYTHON = ENV_ROOT / 'bin/python'
EVAL_MAX_SIDE = 768         # 0 = ảnh gốc; không tự resize khi OOM
TRAIN_MAX_SIDE = 768
CROP_SIZE = 256             # bội 32
BATCH_SIZE = 2
DEN_STEPS = 4000            # chỉ DEN, dùng coarse/teacher cache; không chạy Mamba mỗi step
JOINT_STEPS = 1000          # fine-tune toàn mạng; đặt 0 để chỉ train DEN

if platform.system() != 'Linux':
    raise RuntimeError('Dùng notebook trên Colab/Linux GPU. Máy Windows hiện tại chỉ có PyTorch CPU.')
WORK_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

def run_command(args, cwd=None, log_name=None):
    args = list(map(str, args))
    print('$', ' '.join(args), flush=True)
    log_path = WORK_ROOT / (log_name or 'setup.log')
    with log_path.open('a', encoding='utf-8') as log:
        with subprocess.Popen(args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
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

if not DATA_ROOT:
    if not PROJECT_ROOT.exists():
        run_command(['git', 'clone', '--depth', '1', '--filter=blob:none', '--sparse', PROJECT_URL, PROJECT_ROOT])
        run_command(['git', 'sparse-checkout', 'set', 'dataset/O-HAZY', 'solution/Haze Clear'], cwd=PROJECT_ROOT)
    DATA_ROOT = str((PROJECT_ROOT / 'dataset/O-HAZY').resolve())
if not Path(DATA_ROOT).expanduser().is_dir():
    raise FileNotFoundError(f'Không có dataset/O-HAZY trong checkout: {DATA_ROOT}. Kiểm tra LOCAL_PROJECT/DATA_ROOT.')
project_arch = PROJECT_ROOT / 'solution/Haze Clear'
REPO_ROOT = (Path(LOCAL_REPO).expanduser().resolve() if LOCAL_REPO else
             project_arch if (project_arch / 'basicsr/archs/wavemamba_arch.py').is_file() else
             Path('/content/WDMamba_o_hazy_final'))
if not REPO_ROOT.exists():
    run_command(['git', 'clone', REPO_URL, REPO_ROOT])
if not (REPO_ROOT / 'basicsr/archs/wavemamba_arch.py').is_file():
    raise FileNotFoundError('LOCAL_REPO phải chứa basicsr/archs/wavemamba_arch.py.')
run_command(['git', 'log', '-1', '--oneline'], cwd=REPO_ROOT)
CONFIG = {
    'repo_root': str(REPO_ROOT), 'project_root': str(PROJECT_ROOT),
    'work_root': str(WORK_ROOT), 'output_root': str(OUTPUT_ROOT),
    'data_root': DATA_ROOT, 'checkpoint': CHECKPOINT, 'checkpoint_zip': CHECKPOINT_ZIP,
    'checkpoint_name': 'Ohaze_1200_1600_27.22.pth',
    'checkpoint_drive_id': '1Vm-1mtMSc9YIUICq4nWKu-4AHCtYnPpF',
    'val_ids': [5, 15, 25, 35], 'seed': 2026, 'den_blocks': 1,
    'eval_max_side': EVAL_MAX_SIDE, 'train_max_side': TRAIN_MAX_SIDE,
    'crop_size': CROP_SIZE, 'batch_size': BATCH_SIZE,
    'den_steps': DEN_STEPS, 'joint_steps': JOINT_STEPS,
    'den_lr': 1e-4, 'joint_den_lr': 2e-5, 'joint_lfrn_lr': 5e-6,
    'weight_decay': 1e-4, 'ema_decay': 0.99, 'l1_weight': 0.1, 'kd_weight': 0.10,
    'val_every': 250, 'log_every': 100, 'crop_border': 4,
    'warmup': 5, 'timing_repeats': 30, 'psnr_tolerance_db': 0.30, 'target_speedup': 2.0,
}
CONFIG_PATH = WORK_ROOT / 'config.json'
CONFIG_PATH.write_text(json.dumps(CONFIG, indent=2), encoding='utf-8')
print(json.dumps(CONFIG, indent=2))''')
md('''## 2. Môi trường riêng và kiểm tra CUDA/Mamba

Dùng Python 3.11, PyTorch 2.8/CUDA 12.8 như các lần chạy trước. Mamba/causal-conv1d có thể cần biên dịch nên lần cài đầu lâu hơn; Colab cần có CUDA toolkit (`nvcc`). Mọi train/eval chạy bằng Python riêng, không cần restart kernel. Phiên bản thực tế được ghi trong `environment.txt`.

Đo **FP32, tắt TF32**, không TTA/ensemble, không `torch.compile`. Không âm thầm đổi precision, batch hoặc độ phân giải khi thiếu VRAM.''')
code(r'''run_command(['nvidia-smi'])
run_command(['nvcc', '--version'])
run_command([sys.executable, '-m', 'pip', 'install', '-q', 'uv'])
run_command([sys.executable, '-m', 'uv', 'python', 'install', '3.11'])
if not PYTHON.is_file():
    run_command([sys.executable, '-m', 'uv', 'venv', '--python', '3.11', ENV_ROOT])
run_command([sys.executable, '-m', 'uv', 'pip', 'install', '--python', PYTHON,
             '--index-url', 'https://download.pytorch.org/whl/cu128', 'torch==2.8.0', 'torchvision==0.23.0'])
constraints = ENV_ROOT / 'constraints.txt'
constraints.write_text('torch==2.8.0\ntorchvision==0.23.0\nnumpy<2\nsetuptools<81\n', encoding='utf-8')
run_command([sys.executable, '-m', 'uv', 'pip', 'install', '--python', PYTHON, '--constraint', constraints,
             'numpy<2', 'opencv-python-headless<4.12', 'scipy', 'scikit-image', 'einops',
             'timm==1.0.15', 'pytorch-wavelets', 'PyWavelets', 'setuptools<81', 'packaging',
             'ninja', 'wheel', 'transformers', 'gdown', 'tqdm'])
run_command([sys.executable, '-m', 'uv', 'pip', 'install', '--python', PYTHON, '--constraint', constraints,
             '--no-build-isolation', 'causal-conv1d', 'mamba-ssm'])
freeze = subprocess.check_output([sys.executable, '-m', 'uv', 'pip', 'freeze', '--python', str(PYTHON)], text=True)
(OUTPUT_ROOT / 'environment.txt').write_text(freeze, encoding='utf-8')
check = r"""
import torch
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
assert torch.cuda.is_available()
u = torch.zeros(1,4,16,device='cuda')
y = selective_scan_fn(u, torch.ones_like(u), -torch.ones(4,1,device='cuda'),
                      torch.ones(1,1,1,16,device='cuda'), torch.ones(1,1,1,16,device='cuda'), delta_softplus=True)
assert torch.isfinite(y).all()
print('CUDA/Mamba PASS:', torch.__version__, torch.version.cuda, torch.cuda.get_device_name())
"""
run_command([PYTHON, '-c', check])''')
md('''## 3. Runner tự chứa trong notebook

Cell này ghi mã độc lập ra thư mục làm việc. Không sửa implementation WDMamba gốc. Checkpoint được load **strict**, tự nhận DEN 4/6 block; student lấy trực tiếp các module có trọng số từ teacher rồi bỏ block lặp. Đường DWT/IWT và tensor shape được kiểm tra trước train.

Loss GT là MSE + 0,1 L1, thêm KD L1 với trọng số 0,1 rồi giảm về 0 khi fine-tune toàn mạng. Teacher/coarse được cache trên **ảnh train đầy đủ sau resize**; lấy crop đồng bộ hazy/GT/coarse/teacher. Crop teacher chứa ngữ cảnh ảnh đầy đủ, không giả định bằng teacher chạy trên crop. BN pretrained được cố định vì tập train nhỏ.''')
runner=(root/'o_hazy_fast_runner.py').read_text(encoding='utf-8-sig')
code("RUNNER_SOURCE = r'''\n"+runner+"\n'''"+'''\nimport ast
ast.parse(RUNNER_SOURCE)
RUNNER_PATH = WORK_ROOT / 'o_hazy_fast_runner.py'
RUNNER_PATH.write_text(RUNNER_SOURCE, encoding='utf-8')
def run_phase(phase):
    CONFIG_PATH.write_text(json.dumps(CONFIG, indent=2), encoding='utf-8')
    run_command([PYTHON, '-u', RUNNER_PATH, CONFIG_PATH, phase], cwd=REPO_ROOT, log_name=phase+'.log')
print('Runner syntax: PASS')''')
md('''## 4. Kiểm tra O-HAZY trong repo và chuẩn bị checkpoint

Cell đọc trực tiếp `dataset/O-HAZY` từ checkout và ghi SHA256, không tải ZIP dataset. Cần đủ scene 1–45, không tự rơi về một split ngẫu nhiên khi thiếu ảnh. Tên gốc `01_outdoor_hazy.jpg`/`01_outdoor_GT.jpg` và tên đã chuẩn hóa `01_outdoor.jpg` đều được hỗ trợ. Archive WDMamba có nhiều checkpoint nhưng notebook chỉ load checkpoint O-HAZY.''')
code('''run_phase('prepare')
inputs = json.loads((WORK_ROOT / 'inputs.json').read_text(encoding='utf-8'))
print('Train IDs:', [x['id'] for x in inputs['splits']['train']])
print('Val IDs:', [x['id'] for x in inputs['splits']['val']])
print('Test IDs:', [x['id'] for x in inputs['splits']['test']])
print('Teacher:', inputs['checkpoint'])''')
md('''## 5. Train một cấu hình cuối và đánh giá

4.000 step DEN dùng cache, tiếp theo 1.000 step toàn mạng. Checkpoint tốt nhất được chọn bằng **PSNR EMA trên validation**, không bằng test. Giảm steps ở cell cấu hình chỉ dành cho smoke test; không coi kết quả train vài step là kết luận về kiến trúc.

PSNR RGB float/SSIM Gaussian 11×11, crop border 4. Resize hazy và GT **cùng nhau trước padding**, giữ tỉ lệ ảnh; padding lên bội 32 chỉ dùng cho mạng, cắt bỏ trước tính metric. CUDA events, warm-up 5, lặp 30; batch 1; data loading/metric nằm ngoài latency. FPS trên test là `1000 / mean latency` của 5 scene. CSV lưu mean/std/p50/p95 từng scene; đo thêm 256×256. Khi benchmark, mỗi lần chỉ một model ở GPU để VRAM so sánh được.

Cell này thực sự train và benchmark, không dùng lại số FPS/PSNR từ Phase 1/2 hay từ phần README.''')
code("run_phase('run')")
md('''## 6. Xem báo cáo và ảnh khó

`PASS` chỉ khi đạt cả ngưỡng PSNR và mục tiêu tốc độ tại hai kích thước. `REVIEW` giữ nguyên kết quả thực đo; không có cơ chế tự sửa điểm hoặc giảm độ phân giải để đạt ngưỡng. Năm scene test là một mẫu nhỏ; bootstrap CI được lưu để minh họa bất định và không thay thế chạy nhiều seed.

Ảnh dự đoán được lưu PNG. Cell dưới hiển thị ba scene có ΔPSNR thấp nhất để kiểm tra sai màu, ringing và mất chi tiết.''')
code(r'''from IPython.display import Markdown, display
import csv
import numpy as np
import matplotlib.pyplot as plt
report = (OUTPUT_ROOT / 'final_report.md').read_text(encoding='utf-8')
display(Markdown(report))
with (OUTPUT_ROOT / 'WDMamba_O_HAZY_image_metrics.csv').open(encoding='utf-8') as f:
    baseline = {int(v['id']): float(v['PSNR']) for v in csv.DictReader(f)}
with (OUTPUT_ROOT / 'Fast_WDMamba_O_HAZY_image_metrics.csv').open(encoding='utf-8') as f:
    improved = {int(v['id']): float(v['PSNR']) for v in csv.DictReader(f)}
worst = sorted(improved, key=lambda k: improved[k]-baseline[k])[:3]
# Use PIL in the notebook kernel; all training dependencies are isolated in PYTHON.
from PIL import Image
for sid in worst:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, name in zip(axes, ('WDMamba_O_HAZY', 'Fast_WDMamba_O_HAZY')):
        ax.imshow(Image.open(OUTPUT_ROOT/'predictions'/name/f'{sid:02d}_pred.png'))
        ax.set_title(f'{name} | scene {sid}'); ax.axis('off')
    fig.suptitle(f'ΔPSNR = {improved[sid]-baseline[sid]:+.3f} dB')
    plt.show()''')
md('''## 7. Đóng gói kết quả và dùng checkpoint

ZIP gồm checkpoint student, config, manifest/checksum dữ liệu, mã runner, môi trường, log, metric và ảnh dự đoán. **Không gồm dataset và checkpoint teacher** để tránh tải lại archive lớn.

Để nạp checkpoint student trong một process sạch: dùng hàm `teacher_and_student(config, teacher_path)` của runner để dựng đúng kiến trúc, sau đó `student.load_state_dict(checkpoint_state(student_path), strict=True)`; đưa teacher về CPU hoặc xóa teacher trước triển khai. DEN student có 1 block/group nên **không** load thẳng vào cấu hình WDMamba gốc 4 block/group. Inference dùng RGB [0,1], pad bội 32 rồi crop lại kích thước ban đầu.

Đây là một cấu hình cải tiến cuối có quy trình kiểm chứng. Chỉ khi bảng kết quả đạt ngưỡng mới kết luận giữ PSNR và tăng FPS. Nếu cần sửa cấu hình sau `REVIEW`, dùng validation cho lựa chọn và ghi rõ đó là một thí nghiệm mới; không chọn nhiều lần dựa trên 5 ảnh test.''')
code(r'''import shutil
for src in (WORK_ROOT/'split_manifest.csv', WORK_ROOT/'inputs.json', RUNNER_PATH,
            WORK_ROOT/'prepare.log', WORK_ROOT/'run.log'):
    if src.is_file():
        shutil.copy2(src, OUTPUT_ROOT/src.name)
archive = shutil.make_archive(str(WORK_ROOT/'fast_wdmamba_o_hazy_results'), 'zip', OUTPUT_ROOT)
print('Kết quả:', archive)
try:
    from google.colab import files
    files.download(archive)
except ImportError:
    pass''')
nbf.validate(nb)
nbf.write(nb,root/'final_o_hazy_fast_wdmamba.ipynb')
print('Notebook:',len(nb.cells),'cells;',len(runner),'runner chars')
