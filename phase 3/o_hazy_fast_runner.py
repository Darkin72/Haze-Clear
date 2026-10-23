"""O-HAZY-only compression/fine-tuning runner embedded in the final notebook."""
from pathlib import Path
import copy
import csv
import hashlib
import importlib.machinery
import importlib.util
import json
import math
import random
import re
import subprocess
import sys
import time
import zipfile

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from skimage.metrics import structural_similarity

EXT = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def safe_extract(archive, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError(f'ZIP lỗi: {archive}')
        for item in z.infolist():
            if not (destination / item.filename).resolve().is_relative_to(destination):
                raise ValueError('Đường dẫn ZIP nằm ngoài thư mục giải nén.')
        z.extractall(destination)


def scene_id(path):
    match = re.match(r'^(\d{1,2})(?:\D|$)', Path(path).stem)
    if not match or not 1 <= int(match[1]) <= 45:
        raise ValueError(f'Tên ảnh không có ID O-HAZY 01–45: {path}')
    return int(match[1])


def collect_pairs(root):
    # Scan all copies, including the repository's existing train/val/test layout.
    # Reassign by scene ID to keep official test 41–45 outside all tuning.
    groups = {'clear': {}, 'hazy': {}}
    for p in sorted(Path(root).rglob('*')):
        if not p.is_file() or p.suffix.lower() not in EXT:
            continue
        parents = {a.name.lower() for a in p.parents if a != Path(root).parent}
        stem = p.stem.lower()
        if parents & {'gt', 'clear', 'clean', 'groundtruth', 'ground_truth'} or stem.endswith('_gt'):
            kind = 'clear'
        elif parents & {'hazy', 'haze', 'input', 'inputs'} or stem.endswith('_hazy'):
            kind = 'hazy'
        else:
            continue
        sid = scene_id(p)
        if sid in groups[kind]:
            raise ValueError(f'Trùng scene {sid}/{kind}: {groups[kind][sid]} và {p}')
        groups[kind][sid] = p.resolve()
    expected = set(range(1, 46))
    for kind, entries in groups.items():
        if set(entries) != expected:
            raise ValueError(f'O-HAZY cần đúng 45 scene/{kind}; thiếu {sorted(expected-set(entries))}, '
                             f'thừa {sorted(set(entries)-expected)}. Đặt data_root đúng thư mục O-HAZY.')
    return [{'id': i, 'hazy': str(groups['hazy'][i]), 'clear': str(groups['clear'][i])}
            for i in sorted(expected)]


def partition(pairs, cfg):
    val = set(cfg['val_ids'])
    if not val or not val <= set(range(1, 41)):
        raise ValueError('val_ids phải thuộc 1–40 và không rỗng.')
    return {'train': [p for p in pairs if p['id'] <= 40 and p['id'] not in val],
            'val': [p for p in pairs if p['id'] in val],
            'test': [p for p in pairs if 41 <= p['id'] <= 45]}


def prepare_inputs(cfg):
    work = Path(cfg['work_root'])
    work.mkdir(parents=True, exist_ok=True)
    if cfg.get('data_root'):
        data = Path(cfg['data_root']).expanduser().resolve()
    elif cfg.get('project_root'):
        data = (Path(cfg['project_root']).expanduser() / 'dataset/O-HAZY').resolve()
    else:
        raise ValueError('Cần data_root hoặc project_root trỏ tới checkout VDT_GD2.')
    if not data.is_dir():
        raise FileNotFoundError(f'Không có dataset O-HAZY trong repo: {data}')
    splits = partition(collect_pairs(data), cfg)
    for split, rows in splits.items():
        for row in rows:
            row['hazy_sha256'] = sha256(row['hazy'])
            row['clear_sha256'] = sha256(row['clear'])
            row['split'] = split
    checkpoint = Path(cfg['checkpoint']).expanduser().resolve() if cfg['checkpoint'] else None
    if checkpoint is None:
        root = work / 'weights'
        matches = list(root.rglob(cfg['checkpoint_name'])) if root.is_dir() else []
        if not matches:
            import gdown
            archive = work / 'WDMamba_ckpts.zip'
            if cfg['checkpoint_zip']:
                archive = Path(cfg['checkpoint_zip']).expanduser().resolve()
            elif not archive.is_file():
                partial = archive.with_suffix('.zip.partial')
                result = gdown.download(id=cfg['checkpoint_drive_id'], output=str(partial),
                                        quiet=False, use_cookies=False, resume=True)
                if result is None or not zipfile.is_zipfile(partial):
                    raise RuntimeError('Không tải được checkpoint. Đặt checkpoint hoặc checkpoint_zip ở cell cấu hình.')
                partial.replace(archive)
            safe_extract(archive, root)
            matches = list(root.rglob(cfg['checkpoint_name']))
        if len(matches) != 1:
            raise FileNotFoundError(f'Cần đúng một {cfg["checkpoint_name"]}; tìm thấy {matches}')
        checkpoint = matches[0].resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    inputs = {'data_root': str(data), 'checkpoint': str(checkpoint),
              'checkpoint_sha256': sha256(checkpoint), 'splits': splits}
    dump_json(work / 'inputs.json', inputs)
    write_csv(work / 'split_manifest.csv', [p for rows in splits.values() for p in rows])
    print('Split:', {k: len(v) for k, v in splits.items()}, flush=True)
    print('Checkpoint:', checkpoint, flush=True)
    return inputs


def import_architecture(root):
    # Avoid BasicSR __init__, which imports unrelated training stacks.
    for name, rel in [('basicsr', 'basicsr'), ('basicsr.utils', 'basicsr/utils'),
                      ('basicsr.archs', 'basicsr/archs')]:
        if name in sys.modules:
            raise RuntimeError(f'Runner cần process sạch; đã import {name}.')
        spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        spec.submodule_search_locations = [str(Path(root) / rel)]
        sys.modules[name] = importlib.util.module_from_spec(spec)
    from basicsr.archs.wavemamba_arch import WaveMamba
    from basicsr.archs.detail_enhance_net import DENet
    from basicsr.archs.wavelet import DWT, IWT
    return WaveMamba, DENet, DWT, IWT


def checkpoint_state(path):
    obj = torch.load(path, map_location='cpu', weights_only=False)
    for key in ('params_ema', 'params', 'state_dict', 'model'):
        if isinstance(obj, dict) and isinstance(obj.get(key), dict):
            obj = obj[key]
            break
    return {k.removeprefix('module.'): v for k, v in obj.items()}


def compress_den(den, depth, keep):
    if not 1 <= keep < depth:
        raise ValueError('den_blocks phải nhỏ hơn số block của teacher.')
    selected = np.linspace(0, depth - 1, keep).round().astype(int).tolist()
    if keep == 1:
        selected = [0]
    for name in ('g1', 'g2', 'g3'):
        gp = getattr(den, name).gp
        retained = [gp[i] for i in selected] + [gp[depth]]
        getattr(den, name).gp = nn.Sequential(*retained)
    return selected


def teacher_and_student(cfg, checkpoint):
    WaveMamba, DENet, DWT, IWT = import_architecture(cfg['repo_root'])
    state = checkpoint_state(checkpoint)
    groups = {i: set() for i in (1, 2, 3)}
    for key in state:
        match = re.fullmatch(r'restoration_network\.DE\.g([123])\.gp\.(\d+)\.conv1\.weight', key)
        if match:
            groups[int(match[1])].add(int(match[2]))
    depths = {len(ids) for ids in groups.values()}
    if len(depths) != 1 or next(iter(depths)) not in (4, 6):
        raise ValueError(f'Không nhận diện được DEN checkpoint: {groups}')
    depth = next(iter(depths))
    if any(ids != set(range(depth)) for ids in groups.values()):
        raise ValueError('DEN block index không liên tục.')
    teacher = WaveMamba(in_chn=3, wf=16, n_l_blocks=[1, 2, 2, 4], ffn_scale=2.)
    teacher.restoration_network.DE = DENet(3, depth)
    teacher.load_state_dict(state, strict=True)
    student = copy.deepcopy(teacher)
    keep = int(cfg['den_blocks'])
    # Keep the group-terminal convolution and all FFT/attention/pre/post weights.
    # Group residual topology is preserved; only repeated expensive U-blocks are removed.
    selected = compress_den(student.restoration_network.DE, depth, keep)
    teacher = teacher.cuda().eval().requires_grad_(False)
    student = student.cuda().eval()
    with torch.inference_mode():
        x = torch.rand(1, 3, 64, 96, device='cuda')
        transform, inverse = DWT().cuda(), IWT().cuda()
        err = (inverse(transform(x)) - x).abs().max().item()
        assert err < 1e-6, f'DWT/IWT reconstruction lỗi: {err}'
        assert teacher(x).shape == student(x).shape == x.shape
        assert torch.isfinite(student(x)).all()
    print(f'DEN: {depth} → {keep} block/group; giữ index {selected}; DWT/IWT error={err:.2e}', flush=True)
    return teacher, student, depth


def load_pair(pair, max_side):
    images = []
    for key in ('hazy', 'clear'):
        image = cv2.imread(pair[key], cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f'Không đọc được {pair[key]}')
        images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    x, y = images
    if x.shape != y.shape:
        raise ValueError(f'Cặp O-HAZY khác shape: scene {pair["id"]}; {x.shape}, {y.shape}')
    h, w = x.shape[:2]
    if max_side and max(h, w) > max_side:
        scale = max_side / max(h, w)
        size = (round(w * scale), round(h * scale))
        x, y = [cv2.resize(a, size, interpolation=cv2.INTER_AREA) for a in (x, y)]
    return [torch.from_numpy(a.copy()).permute(2, 0, 1).float().div_(255) for a in (x, y)]


def pad32(x):
    h, w = x.shape[-2:]
    dh, dw = (-h) % 32, (-w) % 32
    return F.pad(x, (0, dw, 0, dh), mode='reflect' if dh < h and dw < w else 'replicate')


def final_output(y):
    return y[-1] if isinstance(y, (tuple, list)) else y


def metrics(pred, gt, border=4):
    a, b = [t.detach().float().clamp(0, 1).cpu().numpy().transpose(1, 2, 0) for t in (pred, gt)]
    if border:
        a, b = [v[border:-border, border:-border] for v in (a, b)]
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    psnr = float(-10 * np.log10(max(mse, 1e-12)))
    # Same 11x11 Gaussian window/sigma and valid-region averaging as BasicSR SSIM.
    ssim = float(structural_similarity(a, b, channel_axis=2, data_range=1.,
                                      gaussian_weights=True, sigma=1.5, use_sample_covariance=False))
    return psnr, ssim


@torch.inference_mode()
def quality(model, pairs, cfg, label, save_dir=None):
    model.eval()
    rows = []
    for pair in pairs:
        x, y = load_pair(pair, cfg['eval_max_side'])
        h, w = x.shape[-2:]
        pred = final_output(model(pad32(x[None].cuda())))[0, :, :h, :w].clamp(0, 1)
        psnr, ssim = metrics(pred, y, cfg['crop_border'])
        rows.append({'model': label, 'id': pair['id'], 'height': h, 'width': w, 'PSNR': psnr, 'SSIM': ssim})
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            a = (pred.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
            cv2.imwrite(str(save_dir / f'{pair["id"]:02d}_pred.png'), cv2.cvtColor(a, cv2.COLOR_RGB2BGR))
    return rows


@torch.inference_mode()
def latency(model, x, cfg):
    model.eval()
    for _ in range(cfg['warmup']):
        model(x)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(cfg['timing_repeats']):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        result = model(x)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
        del result
    return {'latency_ms': float(np.mean(times)), 'latency_std_ms': float(np.std(times)),
            'latency_p50_ms': float(np.percentile(times, 50)), 'latency_p95_ms': float(np.percentile(times, 95)),
            'FPS': 1000. / float(np.mean(times)), 'peak_allocated_GiB': torch.cuda.max_memory_allocated() / 2**30}


def freeze_bn(model):
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def update_ema(ema, model, decay):
    with torch.no_grad():
        for ep, p in zip(ema.parameters(), model.parameters()):
            ep.lerp_(p.detach(), 1 - decay)
        for eb, b in zip(ema.buffers(), model.buffers()):
            eb.copy_(b)


def train(cfg, inputs, teacher, student, output):
    # Full-image teacher context is computed once. No teacher forward per optimizer step.
    # Frozen LFRN phase trains DEN directly on cached coarse images.
    print('Cache coarse/teacher trên train; không cache ảnh test cho training.', flush=True)
    cache = []
    coarse_model = copy.deepcopy(teacher).eval()
    coarse_model.restoration_network.DE = nn.Identity()
    with torch.inference_mode():
        for pair in inputs['splits']['train']:
            x, y = load_pair(pair, cfg['train_max_side'])
            h, w = x.shape[-2:]
            xp = pad32(x[None].cuda())
            coarse = coarse_model(xp)
            target = teacher.restoration_network.DE(coarse)
            cache.append({'id': pair['id'], 'x': x.half(), 'y': y.half(),
                          'coarse': coarse[0, :, :h, :w].half().cpu(),
                          'teacher': target[0, :, :h, :w].half().cpu()})
            print(f'  cached {pair["id"]:02d} ({h}x{w})', flush=True)
    del coarse_model
    torch.cuda.empty_cache()
    if any(min(v['x'].shape[-2:]) < cfg['crop_size'] for v in cache):
        raise ValueError('crop_size lớn hơn ảnh train sau resize. Giảm crop_size hoặc tăng train_max_side.')
    ema = copy.deepcopy(student).eval().requires_grad_(False)
    best = -math.inf
    history = []
    step = 0
    started = time.perf_counter()

    def batch():
        samples = {key: [] for key in ('x', 'y', 'coarse', 'teacher')}
        for _ in range(cfg['batch_size']):
            item = random.choice(cache)
            h, w = item['x'].shape[-2:]
            s = cfg['crop_size']
            top, left = random.randrange(h-s+1), random.randrange(w-s+1)
            flips = (random.random() < .5, random.random() < .5)
            k = random.randrange(4)
            for key in samples:
                v = item[key][:, top:top+s, left:left+s]
                if flips[0]:
                    v = v.flip(-1)
                if flips[1]:
                    v = v.flip(-2)
                samples[key].append(v.rot90(k, (-2, -1)))
        return {key: torch.stack(v).float().cuda(non_blocking=True) for key, v in samples.items()}

    for stage, count in [('den', cfg['den_steps']), ('joint', cfg['joint_steps'])]:
        if not count:
            continue
        for name, p in student.named_parameters():
            p.requires_grad_(stage == 'joint' or name.startswith('restoration_network.DE.'))
        den_params, lfrn_params = [], []
        for name, p in student.named_parameters():
            if p.requires_grad:
                (den_params if name.startswith('restoration_network.DE.') else lfrn_params).append(p)
        lr = cfg['den_lr'] if stage == 'den' else cfg['joint_den_lr']
        groups = [{'params': den_params, 'lr': lr}]
        if lfrn_params:
            groups.append({'params': lfrn_params, 'lr': cfg['joint_lfrn_lr']})
        optimizer = torch.optim.AdamW(groups, weight_decay=cfg['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=count, eta_min=1e-6)
        student.train()
        freeze_bn(student)  # Small real-data batches must not overwrite pretrained BN statistics.
        for local_step in range(1, count+1):
            data = batch()
            optimizer.zero_grad(set_to_none=True)
            if stage == 'den':
                pred = student.restoration_network.DE(data['coarse'])
            else:
                pred = final_output(student(data['x']))
            gt = data['y']
            mse = F.mse_loss(pred, gt)
            l1 = F.l1_loss(pred, gt)
            distill = F.l1_loss(pred, data['teacher'])
            # Decay KD during joint fine-tuning so GT, not teacher bias, controls the result.
            kd = cfg['kd_weight'] * (1 if stage == 'den' else 1-local_step/count)
            loss = mse + cfg['l1_weight']*l1 + kd*distill
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Loss không hữu hạn tại step {step+1}.')
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            update_ema(ema, student, cfg['ema_decay'])
            step += 1
            if step % cfg['log_every'] == 0:
                print(f'{stage} step={step} loss={loss.item():.5f} MSE={mse.item():.5f} '
                      f'train_elapsed={time.perf_counter()-started:.0f}s', flush=True)
            if local_step % cfg['val_every'] == 0 or local_step == count:
                vr = quality(ema, inputs['splits']['val'], cfg, 'val')
                score = float(np.mean([v['PSNR'] for v in vr]))
                history.append({'step': step, 'stage': stage, 'loss': loss.item(), 'val_PSNR': score,
                                'elapsed_s': time.perf_counter()-started})
                write_csv(output / 'history.csv', history)
                print(f'  validation EMA PSNR={score:.4f} dB; best={best:.4f}', flush=True)
                if score > best:
                    best = score
                    torch.save({'params_ema': ema.state_dict(), 'config': cfg, 'step': step,
                                'val_PSNR': best, 'teacher_sha256': inputs['checkpoint_sha256']},
                               output / 'fast_wdmamba_o_hazy_best.pth')
    if not history:
        raise ValueError('Cần ít nhất một stage training có steps > 0.')
    student.load_state_dict(checkpoint_state(output / 'fast_wdmamba_o_hazy_best.pth'), strict=True)
    del ema, cache
    torch.cuda.empty_cache()
    return {'seconds': time.perf_counter()-started, 'steps': step, 'best_val_PSNR': best}


def benchmark(model, pairs, cfg, label, output):
    # Only one model lives on GPU during final benchmarks; parameter memory is comparable.
    model.cuda().eval()
    quality_rows = quality(model, pairs, cfg, label, output / 'predictions' / label)
    timing_rows = []
    for pair in pairs:
        x, _ = load_pair(pair, cfg['eval_max_side'])
        xp = pad32(x[None].cuda())
        timing_rows.append({'model': label, 'id': pair['id'], 'height': x.shape[-2], 'width': x.shape[-1],
                            'padded_height': xp.shape[-2], 'padded_width': xp.shape[-1],
                            **latency(model, xp, cfg)})
        del xp
    fixed = latency(model, torch.zeros(1, 3, 256, 256, device='cuda'), cfg)
    mean_ms = float(np.mean([v['latency_ms'] for v in timing_rows]))
    summary = {'model': label, 'images': len(pairs), 'precision': 'FP32', 'batch_size': 1,
               'eval_max_side': cfg['eval_max_side'], 'PSNR': float(np.mean([v['PSNR'] for v in quality_rows])),
               'SSIM': float(np.mean([v['SSIM'] for v in quality_rows])), 'latency_ms': mean_ms,
               'FPS': 1000/mean_ms, 'FPS_256': fixed['FPS'], 'latency_256_ms': fixed['latency_ms'],
               'params_M': sum(p.numel() for p in model.parameters())/1e6,
               'peak_allocated_GiB': max(v['peak_allocated_GiB'] for v in timing_rows)}
    write_csv(output / f'{label}_image_metrics.csv', quality_rows)
    write_csv(output / f'{label}_latency.csv', timing_rows)
    dump_json(output / f'{label}_256_latency.json', fixed)
    model.cpu()
    torch.cuda.empty_cache()
    return summary, quality_rows


def run(cfg):
    assert torch.cuda.is_available(), 'Cần GPU CUDA để train và benchmark WDMamba.'
    assert cfg['crop_size'] % 32 == 0
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cv2.setNumThreads(0)
    output = Path(cfg['output_root'])
    output.mkdir(parents=True, exist_ok=True)
    inputs = json.loads((Path(cfg['work_root']) / 'inputs.json').read_text(encoding='utf-8'))
    # Validate inputs again; fail if content changed after the manifest was created.
    for rows in inputs['splits'].values():
        for item in rows:
            for key in ('hazy', 'clear'):
                if sha256(item[key]) != item[key+'_sha256']:
                    raise ValueError(f'Dataset đã thay đổi sau prepare: {item[key]}')
    if sha256(inputs['checkpoint']) != inputs['checkpoint_sha256']:
        raise ValueError('Checkpoint đã thay đổi sau prepare.')
    teacher, student, depth = teacher_and_student(cfg, inputs['checkpoint'])
    base_val = quality(teacher, inputs['splits']['val'], cfg, 'teacher_val')
    training = train(cfg, inputs, teacher, student, output)
    teacher.cpu()
    student.cpu()
    torch.cuda.empty_cache()
    summaries, image_rows = [], []
    for model, label in [(teacher, 'WDMamba_O_HAZY'), (student, 'Fast_WDMamba_O_HAZY')]:
        summary, rows = benchmark(model, inputs['splits']['test'], cfg, label, output)
        summaries.append(summary)
        image_rows.extend(rows)
        print(summary, flush=True)
    base, fast = summaries
    delta = fast['PSNR']-base['PSNR']
    speedup = fast['FPS']/base['FPS']
    fixed_speedup = fast['FPS_256']/base['FPS_256']
    paired = [b['PSNR']-a['PSNR'] for a, b in zip(image_rows[:5], image_rows[5:])]
    rng = np.random.default_rng(cfg['seed'])
    ci = np.percentile(np.mean(rng.choice(paired, size=(10000, len(paired)), replace=True), axis=1), [2.5, 97.5]).tolist()
    gates = {'quality': delta >= -cfg['psnr_tolerance_db'], 'speed': speedup >= cfg['target_speedup'],
             'speed_256': fixed_speedup >= cfg['target_speedup']}
    status = 'PASS' if all(gates.values()) else 'REVIEW'
    write_csv(output / 'summary.csv', summaries)
    write_csv(output / 'image_metrics.csv', image_rows)
    dump_json(output / 'config.json', cfg)
    write_csv(output / 'teacher_val_metrics.csv', base_val)
    try:
        commit = subprocess.check_output(['git', '-C', cfg['repo_root'], 'rev-parse', 'HEAD'], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = 'unknown'
    metadata = {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
                'repo_commit': commit, 'teacher_DEN_blocks': depth, 'student_DEN_blocks': cfg['den_blocks'],
                'teacher_sha256': inputs['checkpoint_sha256'], 'training': training, 'gates': gates,
                'status': status, 'delta_PSNR': delta, 'speedup': speedup, 'speedup_256': fixed_speedup,
                'paired_delta_PSNR_bootstrap_CI95': ci, 'protocol': 'RGB float [0,1]; crop_border=4; FP32; batch=1; no TTA; no tiling',
                'note': 'Năm scene test; bootstrap mô tả bất định, không thay thế chạy nhiều seed.'}
    dump_json(output / 'metadata.json', metadata)
    lines = ['# Fast-WDMamba trên O-HAZY', '', f'Trạng thái: **{status}**', '',
             'Mô hình cuối: LFRN nguyên bản + DEN rút gọn còn 1 U-block/group; giữ FFT, channel attention và residual.', '',
             f"Train/val/test: {len(inputs['splits']['train'])}/{len(inputs['splits']['val'])}/5; test scene 41–45.",
             f"eval_max_side={cfg['eval_max_side']}; crop border={cfg['crop_border']}; RGB FP32, batch 1.", '',
             '| Model | PSNR | SSIM | ms/ảnh | FPS | FPS 256 |', '|---|---:|---:|---:|---:|---:|']
    for v in summaries:
        lines.append(f"| {v['model']} | {v['PSNR']:.4f} | {v['SSIM']:.5f} | {v['latency_ms']:.3f} | {v['FPS']:.2f} | {v['FPS_256']:.2f} |")
    lines += ['', f'ΔPSNR={delta:+.4f} dB; speedup={speedup:.2f}x; speedup 256={fixed_speedup:.2f}x.',
              f'Bootstrap CI95 cho mean ΔPSNR trên 5 scene: [{ci[0]:+.4f}, {ci[1]:+.4f}] dB.', '',
              f"Mục tiêu: ΔPSNR ≥ -{cfg['psnr_tolerance_db']:.2f} dB và speedup ≥ {cfg['target_speedup']:.2f}x.",
              'REVIEW nghĩa là chưa đạt mục tiêu, không phải đã đảm bảo PSNR/FPS.', '',
              'Đây là compression/fine-tuning dùng checkpoint O-HAZY và loss GT + distillation, không phải ablation cùng loss/from-scratch.',
              'Teacher đã học 1–40, gồm cả ảnh validation; validation ở đây chỉ chọn checkpoint compression, không phải validation độc lập cho pretraining.',
              'Số điểm dưới max_side=768 không được đồng nhất với protocol paper resize 1600x1200 hoặc full resolution.',
              'Không dùng test để chọn lại kiến trúc sau khi xem báo cáo. Chỉ 5 scene nên kết luận còn hạn chế.']
    (output / 'final_report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    cfg = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    if sys.argv[2] == 'prepare':
        prepare_inputs(cfg)
    elif sys.argv[2] == 'run':
        run(cfg)
    else:
        raise ValueError('Chọn prepare hoặc run.')
