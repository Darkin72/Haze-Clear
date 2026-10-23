"""Configuration cell template; build_notebook_v2 injects DATASET_SPECS."""
from pathlib import Path
import json, os, platform, subprocess, sys, time

DATASET_NAME = 'o-hazy'  # @param ['reside6k', 'sots-its', 'sots-ots', 'o-hazy', 'i-haze']
DATA_ROOT_OVERRIDE = ''  # Thư mục bộ dữ liệu đã có; trống = đường dẫn trong checkout.
TRAIN_ROOT = ''          # Tập train ITS/OTS riêng nếu muốn fine-tune khi chọn SOTS.
EVAL_ONLY = None         # None = tự nhận diện; True = chỉ đánh giá; False = bắt buộc có train/val.
LOCAL_PROJECT = ''       # Checkout VDT_GD2 đã có trên Colab/Linux.
LOCAL_REPO = ''          # Thư mục chứa basicsr/archs/wavemamba_arch.py, nếu cần ghi đè.
CHECKPOINT = ''          # Teacher .pth riêng; trống = checkpoint mặc định bên dưới.
CHECKPOINT_ZIP = ''      # Archive teacher đã có nếu không muốn tải lại checkpoint.
PHASE3_CHECKPOINT = ''   # Student cùng dataset; O-HAZY tự tìm Phase 3, dataset khác không bắt buộc.
RUN_MODE = 'auto'       # @param ['auto', 'run', 'benchmark']
# auto: đã có checkpoint V2 thì chỉ benchmark; chưa có thì train rồi benchmark.
V2_CHECKPOINT = ''      # Khi benchmark: trống = checkpoint tốt nhất trong OUTPUT_ROOT.
TRY_CUDA_GRAPH = False # Tùy chọn thử nghiệm; mặc định dùng fusion eager để tránh lỗi capture.
EVAL_MAX_SIDE = 768      # 0 = ảnh gốc; cùng kích thước cho mọi model trong một lần chạy.
TRAIN_MAX_SIDE = 768
CROP_SIZE = 384          # Tự giới hạn theo cạnh ngắn của ảnh, làm tròn xuống bội 32.
MAX_CACHE_IMAGES = 40    # LRU trên RAM; vẫn lấy mẫu từ toàn bộ train của RESIDE-6K.

DATASET_SPECS = __DATASET_SPECS__
if DATASET_NAME not in DATASET_SPECS:
    raise ValueError(f'DATASET_NAME phải thuộc {list(DATASET_SPECS)}')
SPEC = DATASET_SPECS[DATASET_NAME]
REPO_URL = 'https://github.com/Darkin72/WDMamba.git'
PROJECT_URL = 'https://github.com/Darkin72/VDT_GD2.git'
PROJECT_ROOT = Path(LOCAL_PROJECT).expanduser().resolve() if LOCAL_PROJECT else next(
    (p for p in (Path.cwd(), *Path.cwd().parents) if (p/'dataset').is_dir()), Path('/content/VDT_GD2'))
WORK_ROOT = Path('/content/wdmamba_v2') / DATASET_NAME
OUTPUT_ROOT = WORK_ROOT / 'results' / DATASET_NAME
ENV_ROOT = Path('/content/wdmamba_o_hazy_env311')
PYTHON = ENV_ROOT / 'bin/python'
if platform.system() != 'Linux':
    raise RuntimeError('Chạy notebook trên Colab/Linux GPU. Kiểm thử CPU dùng test_cpu.py.')
WORK_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def run_command(args, cwd=None, log_name=None):
    args = list(map(str, args))
    print('$', ' '.join(args), flush=True)
    with (WORK_ROOT / (log_name or 'setup.log')).open('a', encoding='utf-8') as log:
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


if not DATA_ROOT_OVERRIDE and not PROJECT_ROOT.exists():
    run_command(['git', 'clone', '--depth', '1', '--filter=blob:none', '--sparse', PROJECT_URL, PROJECT_ROOT])
    # --skip-checks supports the SOTS directory name containing brackets.
    run_command(['git', 'sparse-checkout', 'set', '--skip-checks', SPEC['relative_root'],
                 'solution/Haze Clear'], cwd=PROJECT_ROOT)
DATA_ROOT = Path(DATA_ROOT_OVERRIDE).expanduser().resolve() if DATA_ROOT_OVERRIDE else PROJECT_ROOT / SPEC['relative_root']
if not DATA_ROOT.is_dir() and not DATA_ROOT_OVERRIDE and (PROJECT_ROOT/'.git/info/sparse-checkout').is_file():
    run_command(['git', 'sparse-checkout', 'add', '--skip-checks', SPEC['relative_root']], cwd=PROJECT_ROOT)
if not DATA_ROOT.is_dir():
    raise FileNotFoundError(f'Không có {SPEC["label"]} tại {DATA_ROOT}. Đặt DATA_ROOT_OVERRIDE tới dữ liệu có sẵn.')
project_arch = PROJECT_ROOT / 'solution/Haze Clear'
REPO_ROOT = (Path(LOCAL_REPO).expanduser().resolve() if LOCAL_REPO else
             project_arch if (project_arch/'basicsr/archs/wavemamba_arch.py').is_file() else Path('/content/WDMamba_v2'))
if not REPO_ROOT.exists():
    run_command(['git', 'clone', REPO_URL, REPO_ROOT])
if not (REPO_ROOT/'basicsr/archs/wavemamba_arch.py').is_file():
    raise FileNotFoundError('LOCAL_REPO phải chứa basicsr/archs/wavemamba_arch.py.')

CONFIG = {
    'dataset_name': DATASET_NAME, 'eval_only': EVAL_ONLY, 'train_root': TRAIN_ROOT,
    'repo_root': str(REPO_ROOT), 'project_root': str(PROJECT_ROOT),
    'work_root': str(WORK_ROOT), 'output_root': str(OUTPUT_ROOT), 'data_root': str(DATA_ROOT),
    'checkpoint': CHECKPOINT, 'checkpoint_zip': CHECKPOINT_ZIP, 'checkpoint_name': SPEC['checkpoint_name'],
    'checkpoint_drive_id': '1Vm-1mtMSc9YIUICq4nWKu-4AHCtYnPpF',
    'phase3_checkpoint': PHASE3_CHECKPOINT, 'v2_checkpoint': V2_CHECKPOINT, 'val_ids': SPEC['val_ids'], 'seed': 2026,
    'den_blocks': 1, 'eval_max_side': EVAL_MAX_SIDE, 'train_max_side': TRAIN_MAX_SIDE,
    'crop_size': CROP_SIZE, 'max_cache_images': MAX_CACHE_IMAGES,
    'crop_steps': 1000, 'full_steps': 600, 'accum_steps': 2,
    'crop_lr': 2e-5, 'full_den_lr': 5e-6, 'full_lfrn_lr': 1e-6,
    'l1_weight': 0.02, 'kd_mse_weight': 0.10, 'weight_decay': 1e-5, 'ema_decay': 0.99,
    'val_every': 100, 'log_every': 50, 'patience': 4, 'crop_border': 4,
    'warmup': 5, 'timing_repeats': 30, 'psnr_tolerance_db': 0.30,
    'try_cuda_graph': TRY_CUDA_GRAPH, 'fusion_max_error': 1e-4,
    'min_psnr_gain_db': 0.10, 'min_fps_ratio': 1.03, 'deploy_psnr_tolerance_db': 0.01,
}
if DATASET_NAME == 'o-hazy' and not CHECKPOINT and not CHECKPOINT_ZIP:
    prior_inputs = Path('/content/wdmamba_o_hazy_final/inputs.json')
    if prior_inputs.is_file():
        prior = json.loads(prior_inputs.read_text(encoding='utf-8'))
        if Path(prior['checkpoint']).is_file():
            CONFIG['checkpoint'] = prior['checkpoint']
CONFIG_PATH = WORK_ROOT / 'config.json'
CONFIG_PATH.write_text(json.dumps(CONFIG, indent=2, ensure_ascii=False), encoding='utf-8')
print('Dataset:', SPEC['label'], '| thư mục:', DATA_ROOT)
print('Chế độ:', 'SOTS chỉ đánh giá mặc định' if DATASET_NAME.startswith('sots-') and not TRAIN_ROOT else 'tự nhận diện train/val/test')
if DATASET_NAME == 'i-haze' and not CHECKPOINT:
    print('I-HAZE dùng teacher O-HAZY để chuyển miền; có thể đặt CHECKPOINT riêng.')
print(json.dumps(CONFIG, indent=2, ensure_ascii=False))
