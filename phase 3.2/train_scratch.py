"""Phase 3.2: random initialization, resumable training, eager FP32 evaluation."""
from pathlib import Path
from collections import OrderedDict
import copy
import json
import math
import random
import re
import sys
import time

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import data_utils as data
from swdnet import SWDNet, HaarDown, HaarUp, ARCH_VERSION


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def device_for(cfg):
    device = torch.device(cfg.get('device', 'cuda'))
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Cần GPU CUDA cho notebook. CPU chỉ dành cho smoke test với device=cpu.')
    return device


def configure(cfg):
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cv2.setNumThreads(0)


def signature(cfg):
    keys = ('model', 'seed', 'schedule', 'lr', 'min_lr', 'warmup_steps', 'weight_decay',
            'ema_decay', 'l1_weight', 'edge_weight', 'train_max_side', 'eval_max_side',
            'crop_border', 'val_every', 'early_stop_patience', 'early_stop_min_steps', 'train_precision')
    return {k: cfg[k] for k in keys}


def validate_schedule(cfg):
    if not cfg['schedule'] or cfg['val_every'] < 1 or cfg['cache_images'] < 1:
        raise ValueError('Cần schedule, val_every >= 1 và cache_images >= 1.')
    for stage in cfg['schedule']:
        if stage['steps'] < 1 or stage['batch_size'] < 1 or stage['crop'] < 0:
            raise ValueError('steps/batch_size phải dương; crop >= 0.')
        if stage['crop'] == 0 and stage['batch_size'] != 1:
            raise ValueError('Giai đoạn ảnh đầy đủ cần batch_size=1 vì ảnh khác kích thước.')


def learning_rate(cfg, step, total):
    warm = min(cfg['warmup_steps'], max(0, total-1))
    if step <= warm:
        return cfg['lr']*step/max(warm, 1)
    progress = (step-warm)/max(total-warm, 1)
    return cfg['min_lr']+.5*(cfg['lr']-cfg['min_lr'])*(1+math.cos(math.pi*progress))


def stage_at(schedule, step):
    end = 0
    for stage in schedule:
        end += stage['steps']
        if step <= end:
            return stage
    raise ValueError('Step nằm ngoài schedule.')


class PairCache:
    def __init__(self, pairs, max_side, capacity):
        self.pairs, self.max_side, self.capacity = pairs, max_side, capacity
        self.cache = OrderedDict()

    def sample(self, crop):
        row = random.choice(self.pairs)
        key = row['hazy']
        if key not in self.cache:
            self.cache[key] = data.load_pair(row, self.max_side)
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        x, y = self.cache[key]
        if crop:
            h, w = x.shape[-2:]
            if h < crop or w < crop:
                x, y = [F.pad(v, (0, max(0, crop-w), 0, max(0, crop-h)), mode='replicate') for v in (x, y)]
            h, w = x.shape[-2:]
            top, left = random.randrange(h-crop+1), random.randrange(w-crop+1)
            x, y = [v[:, top:top+crop, left:left+crop] for v in (x, y)]
            if random.random() < .5:
                x, y = x.flip(-1), y.flip(-1)
            if random.random() < .5:
                x, y = x.flip(-2), y.flip(-2)
            turns = random.randrange(4)
            x, y = x.rot90(turns, (-2, -1)), y.rot90(turns, (-2, -1))
        return x, y

    def batch(self, crop, batch_size, device):
        pairs = [self.sample(crop) for _ in range(batch_size)]
        return [torch.stack([p[k] for p in pairs]).to(device) for k in (0, 1)]


def restoration_loss(pred, gt, cfg):
    pred, gt = pred.float(), gt.float()
    mse = F.mse_loss(pred, gt)
    l1 = F.l1_loss(pred, gt)
    # First differences, no external perceptual network or pretrained teacher.
    edge = (F.l1_loss(pred[..., 1:, :]-pred[..., :-1, :], gt[..., 1:, :]-gt[..., :-1, :])
            + F.l1_loss(pred[..., :, 1:]-pred[..., :, :-1], gt[..., :, 1:]-gt[..., :, :-1]))*.5
    return mse+cfg['l1_weight']*l1+cfg['edge_weight']*edge


@torch.no_grad()
def update_ema(ema, model, step, decay):
    decay = min(decay, (1+step)/(10+step))
    for target, source in zip(ema.parameters(), model.parameters()):
        target.lerp_(source, 1-decay)
    for target, source in zip(ema.buffers(), model.buffers()):
        target.copy_(source)


@torch.no_grad()
def quality(model, pairs, cfg, device, predictions=None):
    model.eval()
    rows = []
    for row in pairs:
        x, gt = data.load_pair(row, cfg['eval_max_side'])
        pred = model(x[None].to(device))[0]
        if not torch.isfinite(pred).all():
            raise FloatingPointError(f'Output không hữu hạn: {row["id"]}')
        pred = pred.clamp(0, 1).cpu()
        psnr, ssim = data.metrics(pred, gt, cfg['crop_border'])
        rows.append({'dataset': cfg['dataset_name'], 'id': row['id'], 'PSNR': psnr, 'SSIM': ssim,
                     'height': x.shape[-2], 'width': x.shape[-1]})
        if predictions:
            folder = Path(predictions); folder.mkdir(parents=True, exist_ok=True)
            image = (pred.permute(1, 2, 0).numpy()*255).round().astype(np.uint8)
            name = re.sub(r'[^\w.-]', '_', str(row['id']))
            cv2.imwrite(str(folder/(name+'.png')), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return rows


def train(cfg, inputs, output, resume=False):
    if inputs['evaluation_only']:
        raise ValueError('SOTS chuẩn không có train. Cần TRAIN_ROOT ITS/OTS riêng hoặc benchmark checkpoint đã train.')
    validate_schedule(cfg)
    last, best_path = output/'last.pth', output/'best.pth'
    if not resume and (last.exists() or best_path.exists()):
        raise FileExistsError('Run đã có checkpoint. Đổi RUN_NAME để train từ đầu, hoặc RUN_MODE=resume để tiếp tục.')
    device = device_for(cfg)
    model = SWDNet(**cfg['model']).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    total = sum(s['steps'] for s in cfg['schedule'])
    start, best, stale, history, completed = 0, -math.inf, 0, [], False
    arch_sha = data.sha256(Path(__file__).with_name('swdnet.py'))
    train_dtype = (torch.bfloat16 if device.type == 'cuda' and cfg['train_precision'] == 'bf16'
                   and torch.cuda.is_bf16_supported() else None)
    if resume:
        saved = torch.load(last, map_location=device, weights_only=False)
        if (saved['train_signature'] != signature(cfg) or saved['manifest_sha256'] != inputs['manifest_sha256']
                or saved['architecture_sha256'] != arch_sha or saved['initialization'] != 'random'):
            raise ValueError('Checkpoint resume không khớp kiến trúc, lịch train hoặc manifest dataset.')
        if saved['actual_train_precision'] != ('BF16' if train_dtype else 'FP32'):
            raise ValueError('Precision của runtime resume khác runtime training trước.')
        model.load_state_dict(saved['params']); ema.load_state_dict(saved['params_ema'])
        optimizer.load_state_dict(saved['optimizer'])
        start, best, stale, history = saved['step'], saved['best_val_PSNR'], saved['stale'], saved['history']
        completed = saved['completed']
        random.setstate(saved['random_state']); np.random.set_state(saved['numpy_state'])
        torch.set_rng_state(saved['torch_state'].cpu())
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_states']])
    cache = PairCache(inputs['splits']['train'], cfg['train_max_side'], cfg['cache_images'])
    print(f'SWDNet: random initialization; device={device}; precision={"BF16" if train_dtype else "FP32"}; '
          f'parameters={sum(p.numel() for p in model.parameters()):,}; start={start}/{total}', flush=True)
    started = time.perf_counter()
    step = start

    def base_state():
        return {'arch_version': ARCH_VERSION, 'architecture_sha256': arch_sha, 'model_spec': cfg['model'],
                'dataset_name': cfg['dataset_name'], 'initialization': 'random',
                'manifest_sha256': inputs['manifest_sha256'], 'train_signature': signature(cfg),
                'step': step, 'best_val_PSNR': best, 'eval_max_side': cfg['eval_max_side'],
                'crop_border': cfg['crop_border'], 'actual_train_precision': 'BF16' if train_dtype else 'FP32',
                'params_ema': {k: v.detach().cpu().clone() for k, v in ema.state_dict().items()}}

    if not completed:
        for step in range(start+1, total+1):
            stage = stage_at(cfg['schedule'], step)
            lr = learning_rate(cfg, step, total)
            for group in optimizer.param_groups:
                group['lr'] = lr
            model.train()
            x, gt = cache.batch(stage['crop'], stage['batch_size'], device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=train_dtype or torch.bfloat16, enabled=train_dtype is not None):
                pred = model(x)
                loss = restoration_loss(pred, gt, cfg)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Loss không hữu hạn tại step {step}; giữ checkpoint last gần nhất.')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            update_ema(ema, model, step, cfg['ema_decay'])
            if step % cfg['log_every'] == 0:
                print(f'step={step}/{total} crop={stage["crop"]} loss={loss.item():.6f} lr={lr:.2e}', flush=True)
            if step % cfg['val_every'] == 0 or step == total:
                rows = quality(ema, inputs['splits']['val'], cfg, device)
                score = float(np.mean([r['PSNR'] for r in rows]))
                if not math.isfinite(score):
                    raise FloatingPointError('Validation PSNR không hữu hạn.')
                improved = score > best
                if improved:
                    best, stale = score, 0
                elif step >= cfg['early_stop_min_steps']:
                    stale += 1
                completed = step == total or (cfg['early_stop_patience'] > 0 and stale >= cfg['early_stop_patience'])
                history.append({'step': step, 'crop': stage['crop'], 'loss': loss.item(), 'lr': lr,
                                'val_PSNR': score, 'best_val_PSNR': best, 'elapsed_s_this_session': time.perf_counter()-started})
                data.write_csv(output/'history.csv', history)
                if improved:
                    atomic_save({**base_state(), 'kind': 'best'}, best_path)
                    data.write_csv(output/'best_val_metrics.csv', rows)
                atomic_save({**base_state(), 'kind': 'last', 'params': model.state_dict(),
                             'optimizer': optimizer.state_dict(), 'history': history, 'stale': stale, 'completed': completed,
                             'random_state': random.getstate(), 'numpy_state': np.random.get_state(),
                             'torch_state': torch.get_rng_state(),
                             'cuda_states': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}, last)
                print(f'validation={score:.4f} dB; best={best:.4f}; checkpoint saved', flush=True)
                if completed:
                    break
    return {'initialization': 'random', 'steps': step, 'best_val_PSNR': best,
            'seconds_this_session': time.perf_counter()-started, 'resumed': resume}


@torch.no_grad()
def profile_gflops(model, x):
    ops = {'conv_linear': 0, 'haar': 0}
    handles = []
    states = {m: m.training for m in model.modules()}

    def hook(module, args, y):
        if isinstance(module, nn.Conv2d):
            ops['conv_linear'] += 2*y.numel()*(module.in_channels//module.groups)*math.prod(module.kernel_size)
        elif isinstance(module, nn.Linear):
            ops['conv_linear'] += 2*args[0].numel()*module.out_features
        elif isinstance(module, (HaarDown, HaarUp)):
            ops['haar'] += 4*args[0].numel()  # Three additions and one multiply per output coefficient.
    try:
        model.eval()
        for module in model.modules():
            handles.append(module.register_forward_hook(hook))
        model(x)
    finally:
        for handle in handles:
            handle.remove()
        for module, training in states.items():
            module.training = training
    return {'counted': ops['conv_linear']/1e9, 'supplemental': ops['haar']/1e9,
            'estimated_total': sum(ops.values())/1e9}


@torch.no_grad()
def latency(model, x, cfg):
    model.eval()
    for _ in range(cfg['warmup']):
        model(x)
    values = []
    if x.is_cuda:
        torch.cuda.synchronize(x.device)
        torch.cuda.reset_peak_memory_stats(x.device)
    for _ in range(cfg['timing_repeats']):
        if x.is_cuda:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(); model(x); end.record(); end.synchronize()
            values.append(start.elapsed_time(end))
        else:
            start = time.perf_counter(); model(x); values.append((time.perf_counter()-start)*1000)
    if not values:
        raise ValueError('timing_repeats phải lớn hơn 0.')
    mean = float(np.mean(values))
    return {'latency_ms': mean, 'FPS': 1000/mean, 'p50_ms': float(np.median(values)),
            'p95_ms': float(np.percentile(values, 95)),
            'peak_allocated_GiB': torch.cuda.max_memory_allocated(x.device)/2**30 if x.is_cuda else None}


def evaluate(cfg, inputs, output):
    path = Path(cfg.get('evaluation_checkpoint') or output/'best.pth').expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Cần checkpoint SWDNet đã train: {path}. Không đánh giá trọng số ngẫu nhiên trên SOTS.')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved['arch_version'] != ARCH_VERSION or saved['initialization'] != 'random' or saved['step'] < 1:
        raise ValueError('Checkpoint không thuộc kiến trúc SWDNet train từ đầu.')
    if saved['architecture_sha256'] != data.sha256(Path(__file__).with_name('swdnet.py')):
        raise ValueError('Mã kiến trúc khác checkpoint; dùng đúng phiên bản swdnet.py.')
    cross_dataset = saved['dataset_name'] != cfg['dataset_name']
    if cross_dataset and not inputs['evaluation_only']:
        raise ValueError('Checkpoint thuộc dataset khác; chỉ dùng cross-dataset khi eval_only=True.')
    if not cross_dataset and saved['manifest_sha256'] != inputs['manifest_sha256']:
        raise ValueError('Manifest dataset khác checkpoint training.')
    device = device_for(cfg)
    model = SWDNet(**saved['model_spec']).to(device).eval()
    model.load_state_dict(saved['params_ema'], strict=True)
    # Save quality and partial summary before timing, so a benchmark failure does not lose PSNR.
    rows = quality(model, inputs['splits']['test'], cfg, device, output/'predictions')
    data.write_csv(output/'test_metrics.csv', rows)
    summary = {'dataset': cfg['dataset_name'], 'model': 'SWDNet', 'source_dataset': saved['dataset_name'],
               'device': str(device), 'precision': 'FP32', 'execution': 'eager', 'batch_size': 1,
               'images': len(rows), 'eval_max_side': cfg['eval_max_side'], 'crop_border': cfg['crop_border'],
               'PSNR': float(np.mean([r['PSNR'] for r in rows])), 'SSIM': float(np.mean([r['SSIM'] for r in rows])),
               'params_M': sum(p.numel() for p in model.parameters())/1e6}
    data.dump_json(output/'quality_partial.json', summary)
    timing_rows = []
    for pair in inputs['splits']['test']:
        x, _ = data.load_pair(pair, cfg['eval_max_side'])
        probe = x[None].to(device)
        timing_rows.append({'id': pair['id'], 'height': x.shape[-2], 'width': x.shape[-1],
                            **latency(model, probe, cfg), **profile_gflops(model, probe)})
    probe = torch.zeros(1, 3, 256, 256, device=device)
    f256, t256 = profile_gflops(model, probe), latency(model, probe, cfg)
    avg_ms = float(np.mean([r['latency_ms'] for r in timing_rows]))
    summary.update(latency_ms=avg_ms, FPS=1000/avg_ms, FPS_256=t256['FPS'],
                   GFLOPs_256_counted=f256['counted'], GFLOPs_256_supplemental=f256['supplemental'],
                   GFLOPs_256_estimated_total=f256['estimated_total'],
                   GFLOPs_test_estimated_mean=float(np.mean([r['estimated_total'] for r in timing_rows])))
    data.write_csv(output/'test_timing_gflops.csv', timing_rows)
    data.dump_json(output/'probe_256.json', {'gflops': f256, 'timing': t256})
    data.write_csv(output/'summary.csv', [summary])
    baseline = []
    for pair in inputs['splits']['test']:
        x, gt = data.load_pair(pair, cfg['eval_max_side'])
        psnr, ssim = data.metrics(x, gt, cfg['crop_border'])
        baseline.append({'id': pair['id'], 'PSNR': psnr, 'SSIM': ssim})
    data.write_csv(output/'hazy_input_metrics.csv', baseline)
    metadata = {'status': 'MEASURED', 'initialization': 'random', 'source_dataset': saved['dataset_name'],
                'cross_dataset': cross_dataset, 'checkpoint': str(path), 'checkpoint_sha256': data.sha256(path),
                'selected_step': saved['step'], 'best_val_PSNR': saved['best_val_PSNR'],
                'manifest_sha256': inputs['manifest_sha256'], 'architecture': saved['model_spec'],
                'architecture_sha256': saved['architecture_sha256'], 'torch': torch.__version__,
                'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU smoke test',
                'cuda': torch.version.cuda, 'train_precision': saved['actual_train_precision'],
                'GFLOPs_method': '1 MAC=2 FLOPs; Conv/Linear counted; Haar arithmetic supplemental; normalization, gates, activations, pooling, adds excluded',
                'FPS_scope': 'batch 1, FP32 eager; model forward including internal pad/crop; excludes I/O, transfer, metrics and warmup'}
    data.dump_json(output/'metadata.json', metadata)
    lines = ['# Phase 3.2 — SWDNet train từ đầu', '',
             '| Dataset | Model | PSNR | SSIM | FPS | FPS 256 | GFLOPs counted | Haar bổ sung | GFLOPs tổng ước tính | Params M |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|',
             f"| {cfg['dataset_name']} | SWDNet | {summary['PSNR']:.4f} | {summary['SSIM']:.5f} | {summary['FPS']:.2f} | {summary['FPS_256']:.2f} | {f256['counted']:.3f} | {f256['supplemental']:.3f} | {f256['estimated_total']:.3f} | {summary['params_M']:.3f} |", '',
             f"Input hazy: PSNR={np.mean([r['PSNR'] for r in baseline]):.4f}, SSIM={np.mean([r['SSIM'] for r in baseline]):.5f}.",
             f"Checkpoint: step {saved['step']}; validation PSNR={saved['best_val_PSNR']:.4f}; nguồn train={saved['dataset_name']}.",
             f"Thiết bị: {metadata['gpu']}; train {saved['actual_train_precision']}, đánh giá FP32 eager.",
             'GFLOPs là số phép tính mỗi ảnh; normalization/gating/activation/pooling/elementwise không nằm trong số đếm.',
             'Không suy ra FPS từ GFLOPs; cần đo các baseline trên cùng GPU và evaluator trước khi kết luận tăng tốc.',
             'Khởi tạo ngẫu nhiên, không teacher/KD/pretrained; validation chọn EMA checkpoint, test chỉ lập báo cáo.',
             'O-HAZY chỉ có 36 train/4 val/5 test trong protocol này; test các phase trước đã được xem.',
             'MEASURED không có nghĩa là đã đạt PSNR/FPS của WDMamba pretrained.']
    (output/'final_report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def run(cfg, mode):
    if mode in ('train', 'resume') and cfg.get('evaluation_checkpoint'):
        raise ValueError('EVALUATION_CHECKPOINT chỉ dùng trong benchmark; để trống khi train/resume.')
    configure(cfg)
    inputs = json.loads((Path(cfg['work_root'])/'inputs.json').read_text(encoding='utf-8'))
    data.validate(cfg, inputs)
    output = Path(cfg['output_root']); output.mkdir(parents=True, exist_ok=True)
    if mode in ('train', 'resume'):
        info = train(cfg, inputs, output, resume=mode == 'resume')
        data.dump_json(output/'training.json', info)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    elif mode != 'benchmark':
        raise ValueError('Chọn train, resume hoặc benchmark.')
    data.dump_json(output/'config.json', cfg)
    return evaluate(cfg, inputs, output)


if __name__ == '__main__':
    config = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    if sys.argv[2] == 'prepare':
        data.prepare(config)
    else:
        run(config, sys.argv[2])
