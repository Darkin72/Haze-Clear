"""Generate a portable Colab notebook with all Phase 4 runner sources embedded."""
from pathlib import Path
import ast
import nbformat as nbf

HERE = Path(__file__).resolve().parent
modules = {p.name: p.read_text(encoding='utf-8-sig') for p in sorted(HERE.glob('phase4_*.py'))}
tree = ast.parse(modules['phase4_data.py'])
specs = ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign)
    and any(isinstance(t, ast.Name) and t.id == 'DATASET_SPECS' for t in n.targets)))
nb = nbf.v4.new_notebook(metadata={
    'kernelspec': {'display_name':'Python 3', 'language':'python', 'name':'python3'},
    'colab': {'name':'phase4_wavelite_final.ipynb'}, 'wdmamba_phase':'4-wavelite-uden1-dw-detail-v1'})
def md(s): nb.cells.append(nbf.v4.new_markdown_cell(s))
def code(s): nb.cells.append(nbf.v4.new_code_cell(s))

md('''# Phase 4 — WaveLite-Final

Kiến trúc đích: **Haar DWT → Mamba/UNet trên LL, giữ dải cao → IWT → DEN một U-block và một depthwise block/group
với ba nhánh FFT + attention → residual depthwise detail → RGB**.

Giữ năng lực khôi phục ngữ cảnh của WDMamba; giảm lặp ở DEN để tăng FPS; bổ sung detail adapter nhỏ.
Mục tiêu đo: **ΔPSNR ≥ −0,30 dB**, **FPS ≥ 2×**, FPS 256 ≥ 1,10× và FPS/MSE không giảm so tham chiếu.
Đây là điều kiện kiểm tra, chưa phải kết quả đạt được. Test không quyết định trọng số hoặc runtime.
Notebook chứa toàn bộ runner; dùng dataset đã có trong checkout và không tải ZIP dataset.''')
md('''## 1. Chọn dataset, đường dẫn và artifact

`PRESET` chọn lịch chạy: `quick` giảm thời gian train/benchmark, `standard` dùng lịch dài hơn.
| Thiết lập | quick | standard |
|---|---:|---:|
| O-HAZY, tối đa số bước | 900 | 4.000 |
| I-HAZE, mỗi model | 5.000 | 27.000 |
| Lặp đo tốc độ | 15 | 40 |

Cả hai đều tính chất lượng trên toàn bộ test. SOTS ở quick chỉ dùng checkpoint RESIDE-6K;
standard chỉ fine-tune khi có ITS/OTS train riêng. Lịch dài hơn không bảo đảm PSNR cao hơn.

`DATASET_NAME='all'` chạy lần lượt **SOTS Indoor → SOTS Outdoor → O-HAZY → I-HAZE**;
không bao gồm RESIDE-6K riêng. Mỗi dataset có thư mục riêng, cuối lượt có bảng tổng hợp.

Preset `quick` ưu tiên thời gian: RESIDE/SOTS nạp chung checkpoint RESIDE-6K, bỏ qua train,
đánh giá đầy đủ test và chỉ tối ưu runtime. O-HAZY dùng lịch phục hồi ngắn từ trọng số sẵn có.
I-HAZE vẫn khởi tạo ngẫu nhiên hai model độc lập để so sánh.
SOTS chuẩn là benchmark test. Không có `TRAIN_ROOT` ITS/OTS riêng thì chỉ thử fusion trên model đầy đủ,
không cắt block mà thiếu bước phục hồi chất lượng. `DATA_ROOT_OVERRIDE` và `CHECKPOINTS` cho phép dùng
các file đã có trên Drive hoặc runtime cũ. `RUN_MODE='auto'` tiếp tục lượt dở hoặc benchmark lượt đã xong.''')
code("DATASET_SPECS = " + repr(specs) + '\n' + r'''
from pathlib import Path
import ast, json, os, platform, subprocess, sys

DATASET_NAME = 'all'  # @param ['all','o-hazy','sots-its','sots-ots','i-haze','reside6k']
PRESET = 'quick'        # @param ['quick','standard']
LOCAL_PROJECT = ''
ARCH_ROOT_OVERRIDE = ''  # checkout WDMamba đã có; để trống sẽ tự clone
ARCH_REPO_URL = 'https://github.com/Darkin72/WDMamba.git'
DATA_ROOT_OVERRIDE = ''
DATA_ROOT_OVERRIDES = {'sots-its':'', 'sots-ots':'', 'o-hazy':'', 'i-haze':''}  # dùng khi all
TRAIN_ROOT = ''             # ITS/OTS train riêng, tách khỏi benchmark SOTS
TRAIN_ROOTS = {'sots-its':'', 'sots-ots':''}  # dùng khi all + standard
EVAL_ONLY = None            # None tự nhận diện; False bắt buộc train/val
CHECKPOINTS = {'o-hazy': '', 'sots-its': '', 'sots-ots': '', 'reside6k': ''}
RESIDE_CHECKPOINT = ''      # Điền một lần: dùng chung cho RESIDE-6K, SOTS Indoor và Outdoor
STUDENT_CHECKPOINT = ''     # tùy chọn: Phase 3 O-HAZY hoặc student cùng dataset
CHECKPOINT_ZIP = ''         # ZIP trọng số có sẵn, không phải dataset ZIP
CHECKPOINT_SEARCH_ROOTS = [
    '/content/wdmamba_o_hazy_final/weights', '/content/wdmamba_v2',
    '/content/wdmamba_sots', '/content/WDMamba_ckpts',
]
ALLOW_WEIGHT_DOWNLOAD = False  # chỉ tải trọng số nếu chủ động bật; không tải dataset
RUN_NAME = 'wavelite_quick_v2'  # run mới, không ghi đè phase 4 cũ
RUN_MODE = 'auto'           # auto | train | resume | benchmark
EVAL_MAX_SIDE = 768         # 0 = kích thước gốc; giữ 768 khi đối chiếu các phase cũ
PYTHON_OVERRIDE = ''        # Python CUDA/Mamba đã có, nếu cần

if platform.system() != 'Linux':
    raise RuntimeError('Notebook cần Colab/Linux GPU; kiểm thử CPU dùng test_cpu.py.')
PROJECT_ROOT = Path(LOCAL_PROJECT).expanduser().resolve() if LOCAL_PROJECT else next(
    (p for p in (Path.cwd(), *Path.cwd().parents) if (p/'dataset').is_dir()), Path('/content/VDT_GD2'))
ACTIVE_DATASETS = ['sots-its','sots-ots','o-hazy','i-haze'] if DATASET_NAME == 'all' else [DATASET_NAME]
if DATASET_NAME == 'all' and (DATA_ROOT_OVERRIDE or TRAIN_ROOT):
    raise ValueError('Khi all, điền DATA_ROOT_OVERRIDES / TRAIN_ROOTS cho từng dataset.')
SPEC = DATASET_SPECS[ACTIVE_DATASETS[0]]
RUN_ROOT = Path('/content/wdmamba_phase4') / DATASET_NAME / RUN_NAME
OUTPUT_ROOT = RUN_ROOT/'results'
RUN_ROOT.mkdir(parents=True, exist_ok=True); OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

def command(args, cwd=None, log='setup.log'):
    args = list(map(str, args)); print('$', ' '.join(args), flush=True)
    with (RUN_ROOT/log).open('a', encoding='utf-8') as out:
        with subprocess.Popen(args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding='utf-8', errors='replace', bufsize=1) as proc:
            try:
                for line in proc.stdout:
                    print(line, end='', flush=True); out.write(line); out.flush()
                rc = proc.wait()
            except BaseException:
                proc.kill(); proc.wait(); raise
    if rc: raise subprocess.CalledProcessError(rc, args)

if not PROJECT_ROOT.exists():
    command(['git','clone','--depth','1','--filter=blob:none','--sparse',
             'https://github.com/Darkin72/VDT_GD2.git',PROJECT_ROOT])
    command(['git','sparse-checkout','set','--skip-checks',
             *[DATASET_SPECS[name]['relative_root'] for name in ACTIVE_DATASETS],
             'solution/Haze Clear'], cwd=PROJECT_ROOT)
DATA_ROOT = Path(DATA_ROOT_OVERRIDE).expanduser().resolve() if DATA_ROOT_OVERRIDE else PROJECT_ROOT/SPEC['relative_root']
if (PROJECT_ROOT/'.git/info/sparse-checkout').is_file():
    missing = [DATASET_SPECS[name]['relative_root'] for name in ACTIVE_DATASETS
               if not (PROJECT_ROOT/DATASET_SPECS[name]['relative_root']).is_dir()
               and not (DATA_ROOT_OVERRIDES.get(name) if DATASET_NAME == 'all' else DATA_ROOT_OVERRIDE)]
    if missing: command(['git','sparse-checkout','add','--skip-checks',*missing], cwd=PROJECT_ROOT)
if not DATA_ROOT.is_dir() and DATASET_NAME != 'all':
    raise FileNotFoundError(f'Không có {DATASET_NAME} tại {DATA_ROOT}. Đặt DATA_ROOT_OVERRIDE tới dataset đã có.')
LOCAL_ARCH = Path(ARCH_ROOT_OVERRIDE).expanduser().resolve() if ARCH_ROOT_OVERRIDE else PROJECT_ROOT/'solution/Haze Clear'
REPO_ROOT = LOCAL_ARCH
if not (REPO_ROOT/'basicsr/archs/wavemamba_arch.py').is_file():
    # VDT_GD2 chứa dataset; WDMamba là repo mã nguồn độc lập để notebook clone được
    # trực tiếp và không phụ thuộc sparse-checkout của repo cha.
    REPO_ROOT = Path('/content/WDMamba')
    if not (REPO_ROOT/'basicsr/archs/wavemamba_arch.py').is_file():
        command(['git','clone','--depth','1',ARCH_REPO_URL,REPO_ROOT])
if not (REPO_ROOT/'basicsr/archs/wavemamba_arch.py').is_file():
    raise FileNotFoundError(f'Không tìm thấy WDMamba architecture tại {REPO_ROOT}.')

CONFIG = {
    'dataset_name': ACTIVE_DATASETS[0], 'project_root': str(PROJECT_ROOT), 'repo_root': str(REPO_ROOT),
    'data_root': str(DATA_ROOT), 'train_root': TRAIN_ROOT, 'eval_only': EVAL_ONLY,
    'work_root': str(RUN_ROOT), 'output_root': str(OUTPUT_ROOT),
    'checkpoint': CHECKPOINTS.get(DATASET_NAME, ''), 'reside_checkpoint': RESIDE_CHECKPOINT,
    'student_checkpoint': STUDENT_CHECKPOINT,
    'checkpoint_zip': CHECKPOINT_ZIP, 'checkpoint_search_roots': CHECKPOINT_SEARCH_ROOTS,
    'allow_weight_download': ALLOW_WEIGHT_DOWNLOAD,
    'seed': 2040, 'device': 'cuda', 'den_blocks': 1, 'detail_width': 16, 'reference_depth': 4,
    'train_max_side': EVAL_MAX_SIDE, 'eval_max_side': EVAL_MAX_SIDE, 'crop_border': 4,
    'cache_images': 40, 'warmup': 8, 'timing_repeats': 40, 'log_every': 50, 'val_every': 200,
    'patience': 8, 'l1_weight': 0.02, 'edge_weight': 0.01, 'kd_weight': 0.10,
    'ema_decay': 0.999, 'weight_decay': 1e-5,
    'psnr_tolerance_db': 0.30, 'target_speedup': 2.0, 'target_speedup_256': 1.10,
    'deploy_psnr_tolerance_db': 0.01, 'fusion_max_error': 1e-4,
    'adapt_schedule': [
        {'steps': 3000, 'crop': 384, 'accum': 2, 'lr': 1e-4, 'den_only': True},
        {'steps': 1000, 'crop': 0, 'accum': 1, 'lr': 1e-5, 'trunk_lr': 2e-6}],
    'scratch_schedule': [
        {'steps': 20000, 'crop': 256, 'accum': 2, 'lr': 2e-4},
        {'steps': 5000, 'crop': 384, 'accum': 2, 'lr': 5e-5},
        {'steps': 2000, 'crop': 0, 'accum': 1, 'lr': 1e-5}],
    'reference_schedule': [
        {'steps': 20000, 'crop': 256, 'accum': 2, 'lr': 2e-4},
        {'steps': 5000, 'crop': 384, 'accum': 2, 'lr': 5e-5},
        {'steps': 2000, 'crop': 0, 'accum': 1, 'lr': 1e-5}],
}
if PRESET not in ('quick', 'standard'): raise ValueError('PRESET không hợp lệ.')
CONFIG['preset'] = PRESET
if PRESET == 'quick':
    CONFIG.update(runtime_only_datasets=['reside6k','sots-its','sots-ots'],
                  cache_shape_benchmarks=True, skip_duplicate_eager=True,
                  warmup=5, timing_repeats=15, val_every=100, patience=4)
    CONFIG['adapt_schedule'] = [
        {'steps':600, 'crop':384, 'accum':1, 'lr':2e-5, 'den_only':True},
        {'steps':300, 'crop':0, 'accum':1, 'lr':8e-6, 'trunk_lr':1e-6}]
    # I-HAZE is the only branch that still needs random initialization.
    CONFIG['scratch_schedule'] = [
        {'steps':4000, 'crop':256, 'accum':1, 'lr':2e-4},
        {'steps':1000, 'crop':0, 'accum':1, 'lr':2e-5}]
    CONFIG['reference_schedule'] = list(CONFIG['scratch_schedule'])
    if DATASET_NAME in ('sots-its', 'sots-ots'):
        # SOTS is evaluated with the RESIDE-6K checkpoint; never fine-tune on
        # SOTS images in the quick path, even when TRAIN_ROOT is present.
        CONFIG['eval_only'] = True
CONFIG_PATH = RUN_ROOT/'config.json'
CONFIG_PATH.write_text(json.dumps(CONFIG, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(CONFIG, indent=2, ensure_ascii=False))
''')
md('''## 2. Môi trường CUDA/Mamba

Tái dùng Python GPU cũ nếu smoke test selective scan đạt. Nếu chưa có, tạo môi trường riêng phase 4;
không hạ phiên bản môi trường đang dùng. Nhánh I-HAZE cần thời gian huấn luyện hai model và không có
bảo đảm PSNR khi chỉ học từ 25 scene. Các lịch train là mặc định có thể sửa trước khi tạo run mới.''')
code(r'''
check = r"""
import torch
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
import cv2, skimage, timm, einops, pytorch_wavelets
assert torch.cuda.is_available()
u = torch.zeros(1,4,16,device='cuda')
y = selective_scan_fn(u, torch.ones_like(u), -torch.ones(4,1,device='cuda'),
                      torch.ones(1,1,1,16,device='cuda'), torch.ones(1,1,1,16,device='cuda'), delta_softplus=True)
assert torch.isfinite(y).all()
print('CUDA/Mamba PASS:', torch.__version__, torch.version.cuda, torch.cuda.get_device_name())
"""
candidates = [Path(PYTHON_OVERRIDE)] if PYTHON_OVERRIDE else [
    Path('/content/wdmamba_phase4_env311/bin/python'),
    Path('/content/wdmamba_o_hazy_env311/bin/python'), Path(sys.executable)]
PYTHON = None
for candidate in candidates:
    if candidate.is_file():
        probe = subprocess.run([str(candidate), '-c', check], capture_output=True, text=True)
        if probe.returncode == 0:
            PYTHON = candidate; print(probe.stdout); break
        print('Môi trường chưa phù hợp:', candidate, probe.stderr[-1200:])
if PYTHON is None:
    if PYTHON_OVERRIDE: raise RuntimeError('PYTHON_OVERRIDE không vượt qua kiểm tra CUDA/Mamba.')
    command(['nvidia-smi']); command(['nvcc','--version'])
    command([sys.executable,'-m','pip','install','-q','uv'])
    command([sys.executable,'-m','uv','python','install','3.11'])
    env_root = Path('/content/wdmamba_phase4_env311')
    PYTHON = env_root/'bin/python'
    if not PYTHON.is_file(): command([sys.executable,'-m','uv','venv','--python','3.11',env_root])
    command([sys.executable,'-m','uv','pip','install','--python',PYTHON,
             '--index-url','https://download.pytorch.org/whl/cu128','torch==2.8.0','torchvision==0.23.0'])
    constraints = env_root/'constraints.txt'
    constraints.write_text('torch==2.8.0\ntorchvision==0.23.0\nnumpy<2\nsetuptools<81\n', encoding='utf-8')
    command([sys.executable,'-m','uv','pip','install','--python',PYTHON,'--constraint',constraints,
             'numpy<2','opencv-python-headless<4.12','scipy','scikit-image','einops','timm==1.0.15',
             'pytorch-wavelets','PyWavelets','setuptools<81','packaging','ninja','wheel','transformers','gdown','tqdm'])
    command([sys.executable,'-m','uv','pip','install','--python',PYTHON,'--constraint',constraints,
             '--no-build-isolation','causal-conv1d','mamba-ssm'])
    command([PYTHON,'-c',check])
(OUTPUT_ROOT/'environment.txt').write_text(subprocess.check_output(
    [str(PYTHON),'-m','pip','freeze'], text=True) if subprocess.run(
    [str(PYTHON),'-m','pip','--version'], capture_output=True).returncode == 0 else subprocess.check_output(
    [sys.executable,'-m','uv','pip','freeze','--python',str(PYTHON)], text=True), encoding='utf-8')
''')
md('''## 3. Runner nhúng và manifest

Không cần phase 4 đã xuất hiện trên GitHub: notebook ghi trực tiếp các source bên dưới. `prepare` hash
mã, dữ liệu và nguồn trọng số; có khác biệt với một run đã train sẽ yêu cầu đổi `RUN_NAME`.
Checkpoint O-HAZY tự dò theo tên, sau đó mới dùng Phase 3 local nếu không có bản đầy đủ.
Khi có teacher đầy đủ, tự dùng student Phase 3 đã phục hồi chất lượng nếu SHA256 teacher khớp.
Nguồn tham chiếu thực tế luôn được giữ trong metadata; không tính speedup với một model vắng mặt.
Mặc định không tải trọng số; có thể đặt ZIP local hoặc chủ động bật `ALLOW_WEIGHT_DOWNLOAD`.''')
code('MODULE_SOURCES = ' + repr(modules) + '\n' + r'''
for name, source in MODULE_SOURCES.items():
    ast.parse(source)
    (RUN_ROOT/name).write_text(source, encoding='utf-8')
if str(RUN_ROOT) not in sys.path: sys.path.insert(0, str(RUN_ROOT))
from phase4_batch import make_configs, prepare_jobs, run_jobs, collect_reports

if DATASET_NAME == 'all':
    CONFIGS = make_configs(CONFIG, DATASET_SPECS, RUN_ROOT, DATA_ROOT_OVERRIDES, TRAIN_ROOTS,
                           CHECKPOINTS, {'o-hazy': STUDENT_CHECKPOINT})
else:
    CONFIGS = [CONFIG]

def invoke_job(cfg, action):
    work = Path(cfg['work_root']); work.mkdir(parents=True, exist_ok=True)
    Path(cfg['output_root']).mkdir(parents=True, exist_ok=True)
    for name, source in MODULE_SOURCES.items():
        (work/name).write_text(source, encoding='utf-8')
    config_path = work/'config.json'
    config_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding='utf-8')
    command([PYTHON,'-u',work/'phase4_runner.py',config_path,action], cwd=REPO_ROOT,
            log=cfg['dataset_name']+'_'+action+'.log')
def run_phase(action):
    CONFIG_PATH.write_text(json.dumps(CONFIG, indent=2, ensure_ascii=False), encoding='utf-8')
    command([PYTHON,'-u',RUN_ROOT/'phase4_runner.py',CONFIG_PATH,action], cwd=REPO_ROOT, log=action+'.log')
if RUN_MODE not in ('auto','train','resume','benchmark'): raise ValueError('RUN_MODE không hợp lệ.')
if DATASET_NAME == 'all':
    SUITE_STATES = prepare_jobs(CONFIGS, invoke_job, OUTPUT_ROOT)
    print('Chuẩn bị:', SUITE_STATES)
else:
    run_phase('prepare')
    inputs = json.loads((RUN_ROOT/'inputs.json').read_text(encoding='utf-8'))
    print('Split:', {k:len(v) for k,v in inputs['splits'].items()})
    print('Nguồn tham chiếu:', inputs['source'] or 'khởi tạo ngẫu nhiên I-HAZE')
''')
md('''## 4. Tối ưu và benchmark

O-HAZY/RESIDE/ITS/OTS có train: tối ưu DEN/detail trước, joint trên ảnh đầy đủ sau; MSE + L1 + edge
và teacher loss nhỏ; EMA/early stopping theo validation. Student Phase 3 một block được mở rộng thành
hai block với nhánh depthwise residual mới khởi tạo bằng 0, giữ output ban đầu. I-HAZE train độc lập model tham chiếu bốn U-block
và model cuối một U-block cộng một depthwise block/group, không chuyển trọng số hay KD giữa hai model.

Conv–BN fusion chỉ được nhận khi output khớp và latency tốt hơn trên validation hoặc tensor giả.
Không dùng CUDA Graph để tránh lỗi capture đã gặp. `auto` phục hồi từ last; `benchmark` phục hồi best.
Mỗi stage mới quay lại best EMA để tránh tiếp tục từ một iterate bị overfit.''')
code("if DATASET_NAME == 'all':\n    run_jobs(CONFIGS, SUITE_STATES, invoke_job, RUN_MODE, OUTPUT_ROOT)\nelse:\n    run_phase(RUN_MODE)")
md('''## 5. Báo cáo và metric

**FPS/MSE = FPS / MSE trung bình theo ảnh**, RGB [0,1], cùng crop border với PSNR. Không suy MSE từ
PSNR trung bình; không lấy trung bình các tỉ số. FPS đo batch 1, FP32, TF32 tắt, gồm pad/crop và loại
I/O/transfer/metrics. Đây là metric phụ thuộc GPU, không thay thế ngưỡng giảm PSNR riêng.

GFLOPs tại 256×256 và kích thước test được tách counted/supplemental. Model tham chiếu, eager và deploy
đều đo lại trong cùng runtime. `PASS` chỉ xuất khi đủ mọi điều kiện; số đo chưa đạt vẫn giữ nguyên.''')
code(r'''
from IPython.display import Markdown, display
import csv
report_name = 'final_report_all.md' if DATASET_NAME == 'all' else 'final_report.md'
summary_name = 'summary_all.csv' if DATASET_NAME == 'all' else 'summary.csv'
display(Markdown((OUTPUT_ROOT/report_name).read_text(encoding='utf-8')))
print('Report trình bày:', OUTPUT_ROOT/('report_all.html' if DATASET_NAME == 'all' else 'report.html'))
with (OUTPUT_ROOT/summary_name).open(encoding='utf-8') as f:
    for row in csv.DictReader(f):
        print(row['dataset'], row['model'], 'MSE=',row['MSE'], 'FPS/MSE=',row['FPS_over_MSE'])
''')
md('''## 6. Xuất artifact

`model_final.pth` là bản eager tự chứa, kèm spec và nguồn gốc. Dựng lại bằng
`phase4_arch.skeleton(repo_root, saved['model_spec'])`, nạp `saved['params_ema']` strict=True rồi eval.
Không cần nguồn checkpoint ban đầu để dựng model đã xuất. Fusion chỉ thực hiện sau khi nạp eager.
Metadata/config/manifest lưu nguồn khởi tạo thực tế; report chính tập trung kiến trúc và kết quả.
O-HAZY giữ test 41–45, đã được xem ở các phase trước; không phải holdout mới.''')
code(r'''
import shutil
for cfg in CONFIGS:
    work, destination = Path(cfg['work_root']), Path(cfg['output_root'])
    destination.mkdir(parents=True, exist_ok=True)
    for src in [work/name for name in MODULE_SOURCES] + [work/'config.json',work/'inputs.json',work/'split_manifest.csv']:
        if src.is_file(): shutil.copy2(src, destination/src.name)
    if (OUTPUT_ROOT/'environment.txt').is_file() and destination != OUTPUT_ROOT:
        shutil.copy2(OUTPUT_ROOT/'environment.txt', destination/'environment.txt')
archive_root = RUN_ROOT if DATASET_NAME == 'all' else OUTPUT_ROOT
archive = shutil.make_archive(str(RUN_ROOT.parent/(RUN_NAME+'_phase4_'+DATASET_NAME+'_results')), 'zip', archive_root)
print('Kết quả:', archive)
try:
    from google.colab import files
    files.download(archive)
except ImportError:
    pass
''')
for cell in nb.cells:
    if cell.cell_type == 'code': ast.parse(cell.source)
nbf.validate(nb)
nbf.write(nb, HERE/'phase4_wavelite_final.ipynb')
print('Created', HERE/'phase4_wavelite_final.ipynb')
