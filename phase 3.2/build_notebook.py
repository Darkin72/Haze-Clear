"""Build the standalone Phase 3.2 scratch-training notebook."""
from pathlib import Path
import ast
import pprint
import nbformat as nbf

HERE = Path(__file__).resolve().parent
data_source = (HERE/'data_utils.py').read_text(encoding='utf-8')
arch_source = (HERE/'swdnet.py').read_text(encoding='utf-8')
runner_source = (HERE/'train_scratch.py').read_text(encoding='utf-8')
tree = ast.parse(data_source)
node = next(n for n in tree.body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == 'DATASET_SPECS' for t in n.targets))
specs = ast.literal_eval(node.value)

nb = nbf.v4.new_notebook()
nb.metadata.update({'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                    'colab': {'name': 'phase3_2_swdnet_scratch.ipynb'},
                    'wdmamba_phase': '3.2-swdnet-random-init'})


def md(text):
    nb.cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    nb.cells.append(nbf.v4.new_code_cell(text))


md('''# Phase 3.2 — SWDNet train từ đầu

Phase này bỏ checkpoint WDMamba/Phase 3 và khởi tạo **SWDNet bằng trọng số ngẫu nhiên**.
SWDNet là encoder–decoder Haar nhẹ với gated depthwise blocks, được viết mới cho thử nghiệm này;
không dùng teacher, KD hay pretrained weights. Mục tiêu là có một baseline kiến trúc mới ổn định,
đo được PSNR, SSIM, FPS và GFLOPs trên cùng protocol.

Notebook không tải ZIP dataset. `O-HAZY` là mặc định và lấy từ checkout VDT_GD2. SOTS chuẩn chỉ
đánh giá; muốn train từ đầu phải đặt `TRAIN_ROOT` riêng chứa ITS/OTS train và giữ SOTS test làm test.
Không dùng test để chọn checkpoint.''')
md('''## 1. Chọn dataset và thiết lập runtime

Các giá trị hợp lệ: `reside6k`, `sots-its`, `sots-ots`, `o-hazy`, `i-haze`.
Kết quả nằm trong `/content/wdmamba_v3_2/<dataset>/<run_name>`.''')
setup = f'''from pathlib import Path
import ast, json, os, platform, subprocess, sys

DATASET_NAME = 'o-hazy'  # @param ['reside6k', 'sots-its', 'sots-ots', 'o-hazy', 'i-haze']
DATA_ROOT_OVERRIDE = ''  # Dùng thư mục dataset đã có; trống = PROJECT_ROOT/dataset/...
TRAIN_ROOT = ''          # Chỉ cần cho SOTS nếu muốn train ITS/OTS tách biệt.
EVAL_ONLY = None         # None tự nhận diện; False yêu cầu train/val; True chỉ test.
LOCAL_PROJECT = ''       # Checkout VDT_GD2 đã có trên Colab/Linux.
RUN_NAME = 'swdnet_v1'   # Đổi tên nếu muốn một lần train từ đầu khác.
RUN_MODE = 'train'       # train | resume | benchmark
EVAL_MAX_SIDE = 768
TRAIN_MAX_SIDE = 768
RUN_ROOT = Path('/content/wdmamba_v3_2') / DATASET_NAME / RUN_NAME
OUTPUT_ROOT = RUN_ROOT / 'results'
ENV_ROOT = Path('/content/wdmamba_o_hazy_env311')
PYTHON = str(ENV_ROOT/'bin/python') if (ENV_ROOT/'bin/python').is_file() else sys.executable
EVALUATION_CHECKPOINT = '' # SWDNet đã train nếu chỉ benchmark/cross-dataset SOTS.
PROJECT_URL = 'https://github.com/Darkin72/VDT_GD2.git'
SPEC = {repr(specs)}[DATASET_NAME]
PROJECT_ROOT = Path(LOCAL_PROJECT).expanduser().resolve() if LOCAL_PROJECT else next(
    (p for p in (Path.cwd(), *Path.cwd().parents) if (p/'dataset').is_dir()), Path('/content/VDT_GD2'))
if platform.system() != 'Linux':
    raise RuntimeError('Notebook cần Colab/Linux GPU; test CPU chạy test_cpu.py.')
RUN_ROOT.mkdir(parents=True, exist_ok=True); OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

def command(args, cwd=None, log='setup.log'):
    with (RUN_ROOT/log).open('a', encoding='utf-8') as out:
        proc = subprocess.Popen(list(map(str,args)), cwd=cwd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
        try:
            for line in proc.stdout:
                print(line, end='', flush=True); out.write(line); out.flush()
            rc = proc.wait()
        except BaseException:
            proc.kill(); proc.wait(); raise
    if rc: raise subprocess.CalledProcessError(rc, args)

if not DATA_ROOT_OVERRIDE and not PROJECT_ROOT.exists():
    command(['git','clone','--depth','1','--filter=blob:none','--sparse',PROJECT_URL,PROJECT_ROOT])
    command(['git','sparse-checkout','set','--skip-checks',SPEC['relative_root'],'solution/Haze Clear'], cwd=PROJECT_ROOT)
DATA_ROOT = Path(DATA_ROOT_OVERRIDE).expanduser().resolve() if DATA_ROOT_OVERRIDE else PROJECT_ROOT/SPEC['relative_root']
if not DATA_ROOT.is_dir() and not DATA_ROOT_OVERRIDE and (PROJECT_ROOT/'.git/info/sparse-checkout').is_file():
    command(['git','sparse-checkout','add','--skip-checks',SPEC['relative_root']], cwd=PROJECT_ROOT)
if not DATA_ROOT.is_dir():
    raise FileNotFoundError(f'Không có dataset {{DATASET_NAME}} tại {{DATA_ROOT}}. Đặt DATA_ROOT_OVERRIDE.')

CONFIG = {{
    'dataset_name': DATASET_NAME, 'data_root': str(DATA_ROOT), 'project_root': str(PROJECT_ROOT),
    'train_root': TRAIN_ROOT, 'eval_only': EVAL_ONLY, 'work_root': str(RUN_ROOT),
    'output_root': str(OUTPUT_ROOT), 'seed': 2032, 'device': 'cuda',
    'model': {{'width': 32, 'depths': [2,2,4,2,2]}},
    'schedule': [{{'name':'crop256','steps':10000,'crop':256,'batch_size':4}},
                 {{'name':'crop384','steps':4000,'crop':384,'batch_size':2}},
                 {{'name':'full','steps':2000,'crop':0,'batch_size':1}}],
    'lr': 3e-4, 'min_lr': 1e-6, 'warmup_steps': 500, 'weight_decay': 1e-4,
    'ema_decay': .999, 'l1_weight': .05, 'edge_weight': .02, 'train_precision': 'bf16',
    'train_max_side': TRAIN_MAX_SIDE, 'eval_max_side': EVAL_MAX_SIDE, 'crop_border': 4,
    'cache_images': 40, 'val_every': 250, 'log_every': 100,
    'early_stop_patience': 0, 'early_stop_min_steps': 5000,
    'warmup': 10, 'timing_repeats': 50, 'evaluation_checkpoint': EVALUATION_CHECKPOINT,
}}
(RUN_ROOT/'config.json').write_text(json.dumps(CONFIG, indent=2, ensure_ascii=False), encoding='utf-8')
print('Dataset:', SPEC['label'], '| root:', DATA_ROOT, '| mode:', RUN_MODE)
print(json.dumps(CONFIG, indent=2, ensure_ascii=False))
'''
code(setup)
md('''## 2. Mã tự chứa và kiểm tra CUDA

Các nguồn được ghi vào runtime trước khi chạy. Kiến trúc chỉ dùng PyTorch, không cần mamba-ssm.
Training BF16 chỉ bật khi GPU hỗ trợ; đánh giá và benchmark luôn FP32.''')
code('''# Reuse the old Python environment if present; otherwise use the Colab kernel Python.
dependency_probe = "import importlib.util,json; print(json.dumps([m for m in ['cv2','numpy','skimage'] if importlib.util.find_spec(m) is None]))"
missing = json.loads(subprocess.check_output([PYTHON, '-c', dependency_probe], text=True))
if missing:
    command([PYTHON, '-m', 'pip', 'install', 'numpy<2', 'opencv-python-headless<4.12', 'scikit-image'])
check = "import torch,cv2,numpy,skimage; assert torch.cuda.is_available(), 'Cần Colab GPU'; print('CUDA:',torch.cuda.get_device_name(),torch.__version__,torch.version.cuda)"
command([PYTHON, '-c', check])
freeze = subprocess.check_output([PYTHON, '-m', 'pip', 'freeze'], text=True)
(OUTPUT_ROOT/'environment.txt').write_text(freeze, encoding='utf-8')
''')
code("DATA_SOURCE = " + repr(data_source) + "\nARCH_SOURCE = " + repr(arch_source) + "\nRUNNER_SOURCE = " + repr(runner_source) + "\n" + r'''
import ast
for source in (DATA_SOURCE, ARCH_SOURCE, RUNNER_SOURCE): ast.parse(source)
(RUN_ROOT/'data_utils.py').write_text(DATA_SOURCE, encoding='utf-8')
(RUN_ROOT/'swdnet.py').write_text(ARCH_SOURCE, encoding='utf-8')
(RUN_ROOT/'train_scratch.py').write_text(RUNNER_SOURCE, encoding='utf-8')
sys.path.insert(0, str(RUN_ROOT))
print('Embedded Phase 3.2 sources: PASS')
smoke = "import torch; from swdnet import SWDNet; m=SWDNet().cuda(); x=torch.rand(1,3,65,71,device='cuda'); y=m(x); y.square().mean().backward(); assert y.shape==x.shape and torch.isfinite(y).all(); print('GPU forward/backward: PASS')"
command([PYTHON, '-c', smoke], cwd=RUN_ROOT)
''')
md('''## 3. Manifest, train từ đầu và đánh giá

`train` luôn yêu cầu output run mới; `resume` tiếp tục `last.pth` sau gián đoạn; `benchmark` chỉ nạp
`best.pth`. Mỗi checkpoint lưu architecture hash, manifest hash, seed, precision và lịch train.''')
code("""def run_phase(mode):
    CONFIG_PATH = RUN_ROOT/'config.json'
    command([PYTHON, '-u', RUN_ROOT/'train_scratch.py', CONFIG_PATH, mode], log=mode+'.log')
run_phase('prepare')
if RUN_MODE not in ('train', 'resume', 'benchmark'):
    raise ValueError('RUN_MODE phải là train, resume hoặc benchmark')
run_phase(RUN_MODE)
""")
md('''## 4. Đọc báo cáo và đóng gói

`summary.csv` ghi PSNR, SSIM, FPS, FPS 256, GFLOPs counted, GFLOPs Haar bổ sung, GFLOPs tổng và số tham số.
GFLOPs là phép tính mỗi ảnh, không phải GFLOP/s; normalization, activation, pooling và cộng elementwise
được nêu riêng và không tính vào counted.''')
code("""from IPython.display import Markdown, display
display(Markdown((OUTPUT_ROOT/'final_report.md').read_text(encoding='utf-8')))
""")
code("""import shutil
for name in ('swdnet.py', 'data_utils.py', 'train_scratch.py', 'inputs.json', 'split_manifest.csv',
             'prepare.log', RUN_MODE+'.log'):
    source = RUN_ROOT/name
    if source.is_file(): shutil.copy2(source, OUTPUT_ROOT/name)
archive = shutil.make_archive(str(RUN_ROOT/'phase3_2_swdnet_results'), 'zip', OUTPUT_ROOT)
print('Kết quả:', archive)
try:
    from google.colab import files
    files.download(archive)
except ImportError:
    pass
""")
nbf.validate(nb)
nbf.write(nb, HERE/'phase3_2_swdnet_scratch.ipynb')
print('Created', HERE/'phase3_2_swdnet_scratch.ipynb')
