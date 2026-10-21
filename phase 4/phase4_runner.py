"""Phase 4: reproducible adaptation/training and same-session quality/speed evaluation."""
from pathlib import Path
from collections import OrderedDict
import copy
import hashlib
import html
import json
import math
import random
import re
import sys
import time
import zipfile
import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import phase4_data as data
import phase4_arch as arch
from phase4_profile import profile_gflops


def atomic_save(value, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, tmp); tmp.replace(path)


def configure(cfg):
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cv2.setNumThreads(0)


def device_for(cfg):
    device = torch.device(cfg.get('device', 'cuda'))
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Cần GPU CUDA; CPU chỉ dùng cho kiểm thử.')
    return device


def source_hashes(cfg):
    here = Path(__file__).parent
    paths = list(here.glob('phase4_*.py'))
    paths += list((Path(cfg['repo_root'])/'basicsr/archs').glob('*.py'))
    return {p.name: data.sha256(p) for p in sorted(paths)}


def resolve_source(cfg):
    name = data.normalize_dataset_name(cfg['dataset_name'])
    if name == 'i-haze':
        if cfg.get('checkpoint') or cfg.get('student_checkpoint'):
            raise ValueError('I-HAZE phase 4 yêu cầu khởi tạo ngẫu nhiên; để trống cả hai checkpoint.')
        return None
    if cfg.get('checkpoint'):
        path = Path(cfg['checkpoint']).expanduser().resolve()
        if not path.is_file(): raise FileNotFoundError(path)
        return path
    if name in ('sots-its', 'sots-ots', 'reside6k') and cfg.get('reside_checkpoint'):
        path = Path(cfg['reside_checkpoint']).expanduser().resolve()
        if not path.is_file(): raise FileNotFoundError(path)
        return path
    filenames = ['Ohaze_1200_1600_27.22.pth'] if name == 'o-hazy' else ['reside6k_32.15.pth']
    roots = [Path(cfg['project_root']), Path(cfg['work_root'])/'weights']
    roots += [Path(p).expanduser() for p in cfg.get('checkpoint_search_roots', []) if p]
    for root in roots:
        if root.is_dir():
            for filename in filenames:
                found = sorted(root.rglob(filename))
                if found: return found[0].resolve()
    archive = Path(cfg['checkpoint_zip']).expanduser() if cfg.get('checkpoint_zip') else None
    if archive is None and cfg.get('allow_weight_download', False):
        archive = Path(cfg['work_root'])/'WDMamba_ckpts.zip'
        if not archive.is_file():
            import gdown
            tmp = archive.with_suffix('.zip.partial')
            result = gdown.download(id='1Vm-1mtMSc9YIUICq4nWKu-4AHCtYnPpF', output=str(tmp),
                                    quiet=False, use_cookies=False, resume=True)
            if result is None or not zipfile.is_zipfile(tmp):
                raise RuntimeError('Không tải được ZIP trọng số; đặt CHECKPOINT hoặc CHECKPOINT_ZIP.')
            tmp.replace(archive)
    if archive:
        target = (Path(cfg['work_root'])/'weights').resolve(); target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None: raise ValueError('ZIP trọng số lỗi.')
            for item in z.infolist():
                if not (target/item.filename).resolve().is_relative_to(target):
                    raise ValueError('ZIP chứa đường dẫn nằm ngoài thư mục giải nén.')
            z.extractall(target)
        for filename in filenames:
            found = sorted(target.rglob(filename))
            if found: return found[0].resolve()
    # O-HAZY can still improve a local Phase 3 model. Its scope is logged explicitly;
    # gains must then be measured against that model, not against the absent teacher.
    if name == 'o-hazy':
        for phase in ('phase 3.1', 'phase 3'):
            p = Path(cfg['project_root'])/'solution/Haze Clear'/phase/'fast_wdmamba_o_hazy_best.pth'
            if p.is_file(): return p.resolve()
    raise FileNotFoundError(f'Không tìm thấy trọng số cho {name}. Đặt CHECKPOINT riêng cho Indoor/Outdoor '
                            'hoặc CHECKPOINT_ZIP. Không tự dùng trọng số O-HAZY cho SOTS.')


def prepare(cfg):
    inputs = data.prepare(cfg)
    require_adaptation(cfg, inputs)
    source = resolve_source(cfg)
    student = Path(cfg['student_checkpoint']).expanduser().resolve() if cfg.get('student_checkpoint') else None
    if student is None and inputs['dataset_name'] == 'o-hazy' and source is not None:
        # Reuse the recovered Phase 3 DEN when its recorded teacher matches the
        # full reference. Avoid starting again from destructive pruning.
        digest = data.sha256(source)
        for phase in ('phase 3.1', 'phase 3'):
            candidate = Path(cfg['project_root'])/'solution/Haze Clear'/phase/'fast_wdmamba_o_hazy_best.pth'
            if candidate.is_file() and candidate.resolve() != source.resolve():
                obj = torch.load(candidate, map_location='cpu', weights_only=False)
                if obj.get('teacher_sha256') == digest:
                    student = candidate.resolve(); break
    if student is not None and not student.is_file(): raise FileNotFoundError(student)
    if student is not None:
        obj = torch.load(student, map_location='cpu', weights_only=False)
        previous = obj.get('config', {})
        previous_name = data.normalize_dataset_name(previous.get('dataset_name', 'o-hazy'))
        if previous_name != inputs['dataset_name']:
            raise ValueError(f'Student {previous_name} không cùng dataset với {inputs["dataset_name"]}.')
        # No cross-teacher O-HAZY mix when the student provides a provenance hash.
        if source and arch.detect_den_depth(arch.checkpoint_state(source)) > 1:
            if obj.get('teacher_sha256') and obj['teacher_sha256'] != data.sha256(source):
                raise ValueError('Teacher SHA256 của student không khớp model tham chiếu.')
    inputs.update(source=str(source) if source else '', source_sha256=data.sha256(source) if source else '',
                  student_source=str(student) if student else '', student_sha256=data.sha256(student) if student else '',
                  code_sha256=source_hashes(cfg))
    path = Path(cfg['work_root'])/'inputs.json'
    # Preserve identity of a previous run; changing inputs must use a new RUN_NAME.
    output = Path(cfg['output_root'])
    for artifact in (output/'final_best.pth', output/'reference_best.pth'):
        if artifact.is_file():
            saved = torch.load(artifact, map_location='cpu', weights_only=False)
            if saved['identity'] != identity(cfg, inputs):
                raise ValueError('Run đã có trọng số nhưng dữ liệu/cấu hình/mã khác. Đổi RUN_NAME.')
    data.dump_json(path, inputs)
    return inputs


def identity(cfg, inputs):
    keys = ('dataset_name', 'seed', 'den_blocks', 'detail_width', 'reference_depth', 'train_max_side',
            'eval_max_side', 'crop_border', 'adapt_schedule', 'scratch_schedule', 'reference_schedule',
            'l1_weight', 'edge_weight', 'kd_weight', 'ema_decay', 'weight_decay', 'val_every', 'patience')
    return {'config': {**{k: cfg[k] for k in keys},
                       'runtime_only_datasets': cfg.get('runtime_only_datasets', []),
                       'require_compression': cfg.get('require_compression', False)},
            'manifest': inputs['manifest_sha256'],
            'source': inputs['source_sha256'], 'student': inputs['student_sha256'], 'code': inputs['code_sha256']}


def require_adaptation(cfg, inputs):
    if cfg.get('require_compression') and (
            inputs['evaluation_only'] or cfg['dataset_name'] in cfg.get('runtime_only_datasets', [])):
        raise ValueError('REQUIRE_COMPRESSION=True cần train/val riêng để tạo model nhẹ. '
                         'SOTS cần TRAIN_ROOT hoặc AUTO_RESIDE_TRAIN=True; không dùng bản sao Reference.')


def compression_info(reference, final):
    ref_count = sum(p.numel() for p in reference.parameters())
    final_count = sum(p.numel() for p in final.parameters())
    return {'reference_params': ref_count, 'final_params': final_count,
            'parameter_reduction': 1-final_count/max(1, ref_count),
            'architecture_changed': reference.spec != final.spec,
            'compressed': reference.spec != final.spec and final_count < ref_count}


def validate(cfg, inputs):
    data.validate(cfg, inputs)
    for path_key, hash_key in (('source', 'source_sha256'), ('student_source', 'student_sha256')):
        if inputs.get(path_key) and data.sha256(inputs[path_key]) != inputs[hash_key]:
            raise ValueError(f'File đã đổi: {inputs[path_key]}')
    if source_hashes(cfg) != inputs['code_sha256']:
        raise ValueError('Mã đã đổi sau prepare. Chạy prepare với RUN_NAME mới nếu đã train.')


@torch.no_grad()
def quality(model, rows, cfg, device, save_dir=None):
    model.eval(); result = []
    for row in rows:
        x, gt = data.load_pair(row, cfg['eval_max_side'])
        pred = model(x[None].to(device))[0].cpu()
        values = data.metric_values(pred, gt, cfg['crop_border'])
        result.append({'dataset': cfg['dataset_name'], 'id': row['id'], 'height': x.shape[-2],
                       'width': x.shape[-1], **values})
        if save_dir:
            folder = Path(save_dir); folder.mkdir(parents=True, exist_ok=True)
            image = (pred.clamp(0, 1).permute(1, 2, 0).numpy()*255).round().astype(np.uint8)
            name = re.sub(r'[^\w.-]', '_', str(row['id']))
            cv2.imwrite(str(folder/f'{name}.png'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return result


def metric_mean(rows, key):
    return float(np.mean([r[key] for r in rows]))


@torch.no_grad()
def latency(model, x, cfg):
    model.eval()
    for _ in range(cfg['warmup']): model(x)
    values = []
    if x.is_cuda:
        torch.cuda.synchronize(x.device); torch.cuda.reset_peak_memory_stats(x.device)
    for _ in range(cfg['timing_repeats']):
        if x.is_cuda:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(); model(x); end.record(); end.synchronize()
            values.append(start.elapsed_time(end))
        else:
            start = time.perf_counter(); model(x); values.append((time.perf_counter()-start)*1000)
    if not values: raise ValueError('timing_repeats phải dương.')
    mean = float(np.mean(values))
    return {'latency_ms': mean, 'FPS': 1000/mean, 'p50_ms': float(np.median(values)),
            'p95_ms': float(np.percentile(values, 95)),
            'peak_allocated_GiB': torch.cuda.max_memory_allocated(x.device)/2**30 if x.is_cuda else None}


class PairCache:
    def __init__(self, rows, cfg):
        self.rows, self.cfg, self.cache = rows, cfg, OrderedDict()

    def sample(self, crop):
        row = random.choice(self.rows); key = row['hazy']
        if key not in self.cache:
            self.cache[key] = data.load_pair(row, self.cfg['train_max_side'])
            if len(self.cache) > self.cfg['cache_images']: self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        x, y = self.cache[key]
        if crop:
            h, w = x.shape[-2:]; size = min(crop, h, w)
            top, left = random.randrange(h-size+1), random.randrange(w-size+1)
            x, y = [v[:, top:top+size, left:left+size] for v in (x, y)]
            if random.random() < .5: x, y = x.flip(-1), y.flip(-1)
            if random.random() < .5: x, y = x.flip(-2), y.flip(-2)
            turn = random.randrange(4); x, y = x.rot90(turn, (-2, -1)), y.rot90(turn, (-2, -1))
        return x, y


def restoration_loss(pred, gt, cfg, target=None):
    pred, gt = pred.float(), gt.float()
    border = cfg['crop_border']
    if border:
        pred, gt = [v[..., border:-border, border:-border] for v in (pred, gt)]
        if target is not None: target = target[..., border:-border, border:-border]
    edge = (F.l1_loss(pred[..., 1:, :]-pred[..., :-1, :], gt[..., 1:, :]-gt[..., :-1, :])
            + F.l1_loss(pred[..., :, 1:]-pred[..., :, :-1], gt[..., :, 1:]-gt[..., :, :-1]))*.5
    loss = F.mse_loss(pred, gt)+cfg['l1_weight']*F.l1_loss(pred, gt)+cfg['edge_weight']*edge
    if target is not None: loss = loss+cfg['kd_weight']*F.mse_loss(pred, target.float())
    return loss


@torch.no_grad()
def update_ema(ema, model, decay):
    for target, source in zip(ema.parameters(), model.parameters()): target.lerp_(source, 1-decay)
    for target, source in zip(ema.buffers(), model.buffers()): target.copy_(source)


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(value):
    random.setstate(value['python']); np.random.set_state(value['numpy']); torch.set_rng_state(value['torch'])
    if torch.cuda.is_available() and value['cuda']: torch.cuda.set_rng_state_all(value['cuda'])


def train_model(model, cfg, inputs, output, label, schedule, origin, teacher=None, mode='auto'):
    device = device_for(cfg); model.to(device)
    best_path, last_path = output/f'{label}_best.pth', output/f'{label}_last.pth'
    run_identity = identity(cfg, inputs)
    if best_path.is_file():
        saved = torch.load(best_path, map_location='cpu', weights_only=False)
        if saved['identity'] != run_identity: raise ValueError('Trọng số không khớp run hiện tại.')
        if mode in ('auto', 'benchmark'):
            last = torch.load(last_path, map_location='cpu', weights_only=False) if last_path.is_file() else None
            if mode == 'benchmark' or (last and last['completed']):
                model.load_state_dict(saved['params_ema'], strict=True)
                return saved.get('training', {'best_step': saved['step'], 'best_val_PSNR': saved['best_val_PSNR']})
        if mode == 'train': raise FileExistsError(f'Đã có {best_path}; đổi RUN_NAME hoặc dùng resume.')
    elif mode in ('resume', 'benchmark'):
        raise FileNotFoundError(f'Không có {best_path}.')
    if not inputs['splits']['train'] or not inputs['splits']['val']:
        raise ValueError('Cần train/val riêng để cập nhật trọng số; SOTS cần TRAIN_ROOT ITS/OTS.')
    for stage in schedule:
        if stage['steps'] < 1 or stage['crop'] < 0 or stage['accum'] < 1:
            raise ValueError('Schedule không hợp lệ.')
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    if teacher is not None: teacher.to(device).eval().requires_grad_(False)
    initial = quality(ema, inputs['splits']['val'], cfg, device)
    best = metric_mean(initial, 'PSNR'); best_step = step = 0
    history = []; resume = None
    if mode in ('auto', 'resume') and last_path.is_file():
        resume = torch.load(last_path, map_location='cpu', weights_only=False)
        if resume['identity'] != run_identity: raise ValueError('Resume không khớp dữ liệu/cấu hình/mã.')
        model.load_state_dict(resume['params']); ema.load_state_dict(resume['params_ema'])
        best, best_step, step, history = resume['best_val_PSNR'], resume['best_step'], resume['step'], resume['history']
    started = time.perf_counter(); cache = PairCache(inputs['splits']['train'], cfg)
    def artifact():
        return {'arch_version': arch.ARCH_VERSION, 'model_spec': model.spec, 'identity': run_identity,
                'params_ema': cpu_state(ema), 'step': step, 'best_step': best_step, 'best_val_PSNR': best,
                'origin': origin, 'training': {'best_step': best_step, 'best_val_PSNR': best, 'steps': step}}
    if not best_path.is_file():
        atomic_save(artifact(), best_path); data.write_csv(output/f'{label}_initial_val.csv', initial)
    for stage_id, stage in enumerate(schedule):
        if resume and stage_id < resume['stage_id']: continue
        den_only = stage.get('den_only', False)
        for name, param in model.named_parameters():
            param.requires_grad_(not den_only or name.startswith(('trunk.restoration_network.DE.', 'adapter.')))
        den, trunk = [], []
        for name, param in model.named_parameters():
            if param.requires_grad:
                (den if name.startswith(('trunk.restoration_network.DE.', 'adapter.')) else trunk).append(param)
        groups = [{'params': den, 'lr': stage['lr'], 'initial_lr': stage['lr']}]
        if trunk: groups.append({'params': trunk, 'lr': stage.get('trunk_lr', stage['lr']),
                                 'initial_lr': stage.get('trunk_lr', stage['lr'])})
        opt = torch.optim.AdamW(groups, weight_decay=cfg['weight_decay'])
        local_start, stale = 0, 0
        if resume and stage_id == resume['stage_id']:
            opt.load_state_dict(resume['optimizer']); local_start, stale = resume['local_step'], resume['stale']
            restore_rng(resume['rng'])
        for local_step in range(local_start+1, stage['steps']+1):
            model.train()
            if origin != 'random':
                # Preserve inherited running statistics during small-domain adaptation.
                for module in model.modules():
                    if isinstance(module, nn.modules.batchnorm._BatchNorm): module.eval()
            progress = local_step/stage['steps']
            warm = min(200, stage['steps']//10)
            factor = local_step/max(1, warm) if local_step <= warm else .02+.98*.5*(1+math.cos(math.pi*(local_step-warm)/max(1, stage['steps']-warm)))
            for group in opt.param_groups: group['lr'] = group['initial_lr']*factor
            opt.zero_grad(set_to_none=True); loss_sum = 0.
            for _ in range(stage['accum']):
                x, gt = cache.sample(stage['crop']); x, gt = x[None].to(device), gt[None].to(device)
                with torch.no_grad(): target = teacher(x) if teacher is not None else None
                pred = model(x); loss = restoration_loss(pred, gt, cfg, target)
                if not torch.isfinite(loss): raise FloatingPointError(f'Loss NaN/Inf tại step {step+1}.')
                (loss/stage['accum']).backward(); loss_sum += loss.item()/stage['accum']
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
            opt.step(); step += 1
            update_ema(ema, model, min(cfg['ema_decay'], (1+step)/(10+step)))
            if step % cfg['log_every'] == 0:
                print(f'{label}/{stage_id}: {local_step}/{stage["steps"]} loss={loss_sum:.6f}', flush=True)
            if local_step % cfg['val_every'] == 0 or local_step == stage['steps']:
                rows = quality(ema, inputs['splits']['val'], cfg, device); score = metric_mean(rows, 'PSNR')
                if not math.isfinite(score): raise FloatingPointError('Validation PSNR không hữu hạn.')
                if score > best:
                    best, best_step, stale = score, step, 0
                    atomic_save(artifact(), best_path); data.write_csv(output/f'{label}_best_val.csv', rows)
                else: stale += 1
                history.append({'step': step, 'stage': stage_id, 'local_step': local_step, 'loss': loss_sum,
                                'val_PSNR': score, 'best_val_PSNR': best, 'seconds': time.perf_counter()-started})
                data.write_csv(output/f'{label}_history.csv', history)
                end_stage = local_step == stage['steps'] or stale >= cfg['patience']
                atomic_save({**artifact(), 'params': cpu_state(model), 'optimizer': opt.state_dict(),
                             'stage_id': stage_id, 'local_step': stage['steps'] if end_stage else local_step,
                             'stale': stale, 'rng': rng_state(), 'history': history,
                             'completed': end_stage and stage_id == len(schedule)-1}, last_path)
                print(f'{label}: validation={score:.4f}; best={best:.4f} tại step {best_step}', flush=True)
                if end_stage: break
        # Each new stage starts at the validation optimum rather than an overfit iterate.
        saved = torch.load(best_path, map_location='cpu', weights_only=False)
        model.load_state_dict(saved['params_ema']); ema.load_state_dict(saved['params_ema'])
        resume = None
    del ema, cache
    if teacher is not None: teacher.cpu()
    return {'steps': step, 'best_step': best_step, 'best_val_PSNR': best, 'seconds_this_session': time.perf_counter()-started}


@torch.inference_mode(False)
@torch.no_grad()
def choose_runtime(model, cfg, inputs, output):
    device = device_for(cfg); model.to(device).eval()
    if inputs['splits']['val']:
        probes = [data.load_pair(r, cfg['eval_max_side'])[0][None].to(device) for r in inputs['splits']['val'][:4]]
        selection = 'validation'
    else:
        # Synthetic probes only: test images and labels are not consulted for runtime selection.
        gen = torch.Generator(device=device).manual_seed(cfg['seed']+991)
        probes = [torch.rand(1, 3, h, w, device=device, generator=gen) for h,w in ((256,256),(512,768))]
        selection = 'synthetic'
    fused = copy.deepcopy(model).eval(); count = arch.fuse_conv_bn(fused)
    errors = [float((model(x)-fused(x)).abs().max()) for x in probes]
    eager_ms = np.mean([latency(model, x, cfg)['latency_ms'] for x in probes])
    fused_ms = np.mean([latency(fused, x, cfg)['latency_ms'] for x in probes])
    acceptable = max(errors) <= cfg['fusion_max_error']
    use = count > 0 and acceptable and fused_ms < eager_ms*.98
    info = {'execution': 'conv_bn_fused' if use else 'eager', 'selection': selection,
            'fused_pairs': count, 'max_abs_error': max(errors), 'eager_probe_ms': float(eager_ms),
            'fused_probe_ms': float(fused_ms), 'accepted': bool(use)}
    data.dump_json(output/'runtime_selection.json', info)
    result = fused if use else model
    model.cpu()
    if not use: fused.cpu()
    return result.cpu(), info


def evaluate_one(model, label, execution, cfg, inputs, output):
    device = device_for(cfg); model.to(device).eval()
    rows = quality(model, inputs['splits']['test'], cfg, device, output/'predictions'/label)
    # Save quality before profiling/timing so a later failure does not erase it.
    data.write_csv(output/f'{label}_metrics.csv', rows)
    timing, flops = [], []
    shape_cache = {}
    for row in inputs['splits']['test']:
        x, _ = data.load_pair(row, cfg['eval_max_side']); probe = x[None].to(device)
        shape = tuple(probe.shape)
        cached = cfg.get('cache_shape_benchmarks', False) and shape in shape_cache
        if not cached:
            measured = (latency(model, probe, cfg), profile_gflops(model, probe), row['id'])
            shape_cache[shape] = measured
        measured = shape_cache[shape]
        timing.append({'id': row['id'], 'timing_source': 'shape_cache' if cached else 'measured',
                       'representative_id': measured[2], **measured[0]})
        flops.append({'id': row['id'], **measured[1]})
    probe = torch.zeros(1, 3, 256, 256, device=device)
    t256, f256 = latency(model, probe, cfg), profile_gflops(model, probe)
    fps = 1000/metric_mean(timing, 'latency_ms'); mse = metric_mean(rows, 'MSE')
    for quality_row, timing_row in zip(rows, timing):
        quality_row.update(FPS=timing_row['FPS'], FPS_over_MSE=data.efficiency(timing_row['FPS'], quality_row['MSE']))
    data.write_csv(output/f'{label}_metrics.csv', rows)
    data.write_csv(output/f'{label}_timing.csv', timing); data.write_csv(output/f'{label}_gflops.csv', flops)
    data.dump_json(output/f'{label}_256.json', {'timing': t256, 'gflops': f256})
    result = {'dataset': cfg['dataset_name'], 'model': label, 'execution': execution,
              'precision': 'FP32', 'batch_size': 1, 'images': len(rows), 'eval_max_side': cfg['eval_max_side'],
              'crop_border': cfg['crop_border'], 'PSNR': metric_mean(rows, 'PSNR'), 'SSIM': metric_mean(rows, 'SSIM'),
              'MSE': mse, 'latency_ms': metric_mean(timing, 'latency_ms'), 'FPS': fps, 'FPS_256': t256['FPS'],
              'FPS_over_MSE': data.efficiency(fps, mse), 'params_M': sum(p.numel() for p in model.parameters())/1e6,
              'GFLOPs_256_counted': f256['counted_GFLOPs'], 'GFLOPs_256_supplemental': f256['supplemental_GFLOPs'],
              'GFLOPs_256_estimated_total': f256['estimated_total_GFLOPs'],
              'GFLOPs_test_estimated_mean': metric_mean(flops, 'estimated_total_GFLOPs')}
    model.cpu()
    if device.type == 'cuda': torch.cuda.empty_cache()
    print(result, flush=True)
    return result


def assess(reference, final, cfg):
    gates = {'quality': final['PSNR']-reference['PSNR'] >= -cfg['psnr_tolerance_db'],
             'speed': final['FPS']/reference['FPS'] >= cfg['target_speedup'],
             'speed_256': final['FPS_256']/reference['FPS_256'] >= cfg['target_speedup_256'],
             'efficiency': final['FPS_over_MSE'] >= reference['FPS_over_MSE']}
    return gates


def write_report(output, rows, metadata, cfg):
    lines = [f'# Phase 4 — WaveLite-Final / {cfg["dataset_name"]}', '',
             f'Trạng thái: **{metadata["status"]}**.', '',
             '| Model | Thực thi | PSNR | SSIM | MSE | FPS | FPS/MSE | FPS 256 | GFLOPs 256* | Params M |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["model"]} | {r["execution"]} | {r["PSNR"]:.4f} | {r["SSIM"]:.5f} | {r["MSE"]:.8f} | '
                     f'{r["FPS"]:.2f} | {r["FPS_over_MSE"]:.2f} | {r["FPS_256"]:.2f} | '
                     f'{r["GFLOPs_256_estimated_total"]:.3f} | {r["params_M"]:.3f} |')
    ref, final = rows[0], rows[-1]
    compression = metadata.get('compression', {})
    architecture_note = ('Chỉ tối ưu runtime: giữ nguyên kiến trúc và trọng số Reference; chưa có model nén.'
                         if metadata['origin']['method'] == 'runtime_only' else
                         f'Kiến trúc: DEN {metadata["reference_spec"]["depth"]} → '
                         f'{metadata["model_spec"]["depth"]} U-block/group; '
                         f'giảm {100*compression.get("parameter_reduction", 0):.1f}% tham số.')
    # A concise, reviewable highlight uses the actual measured deltas only.
    lines[4:4] = [f'**PSNR {final["PSNR"]:.2f} dB · {final["FPS"]:.1f} FPS · '
                  f'{final["GFLOPs_256_estimated_total"]:.2f} GFLOPs ở 256² · '
                  f'FPS/MSE {final["FPS_over_MSE"]:,.0f}**', '']
    lines += ['', architecture_note, '', f'ΔPSNR: {final["PSNR"]-ref["PSNR"]:+.4f} dB; tăng tốc: {final["FPS"]/ref["FPS"]:.3f}×; '
              f'tỉ lệ FPS/MSE: {final["FPS_over_MSE"]/ref["FPS_over_MSE"]:.3f}×.',
              f'Mục tiêu: giảm PSNR ≤ {cfg["psnr_tolerance_db"]:.2f} dB, FPS ≥ {cfg["target_speedup"]:.2f}× '
              f'và FPS 256 ≥ {cfg["target_speedup_256"]:.2f}×. REVIEW nghĩa là chưa đạt đủ điều kiện.', '',
              'Kiến trúc đích: DWT → nhánh Mamba trên LL, giữ các dải cao → IWT → DEN một U-block '
              'và một residual depthwise block/group '
              'với ba nhánh FFT/attention → residual depthwise detail. Bản runtime giữ cấu trúc đầy đủ '
              'khi chỉ có bộ benchmark; kiến trúc thực đo được lưu trong metadata.',
              f'RGB [0,1], batch 1, FP32, TF32 tắt; max-side={cfg["eval_max_side"]} '
              f'(0 nghĩa là kích thước gốc), crop border={cfg["crop_border"]}. Không TTA/tiling.',
              'PSNR/SSIM/MSE là trung bình đều theo ảnh. FPS = 1000 / latency trung bình theo ảnh. '
              'FPS/MSE = FPS / MSE trung bình; không lấy trung bình tỉ số và không suy MSE từ PSNR trung bình.',
              'FPS đo forward gồm pad/crop, đã warm-up; không gồm đọc ảnh, H2D/D2H và tính metric. '
              'Tất cả model đo lại trong cùng phiên GPU.',
              '*GFLOPs: 1 MAC=2 FLOPs; Conv/Linear và SS2D projections counted; scan/FFT/Haar ước tính bổ sung. '
              'Normalization, activation, pooling và elementwise chưa được đếm đầy đủ. CSV giữ từng thành phần.',
              'Test chỉ dùng lập bảng cuối; validation quyết định trọng số và runtime. '
              'FPS/MSE là metric phụ thuộc thiết bị và protocol, không thay thế điều kiện PSNR riêng.',
              'Thông tin tái lập và nguồn trọng số: metadata.json, config.json, inputs.json và SHA256 mã/dữ liệu.']
    if cfg['dataset_name'] == 'o-hazy':
        lines += ['O-HAZY: test 41–45 đã được xem ở các phase trước; đây là đánh giá trong quá trình phát triển. '
                  'Validation có thể trùng scene từng dùng để huấn luyện mô hình tham chiếu, nên không phải holdout độc lập.']
    if cfg.get('cache_shape_benchmarks'):
        lines += ['Đánh giá chất lượng trên toàn bộ test. Latency/GFLOPs đo một ảnh đại diện cho mỗi '
                  'kích thước đầu vào rồi dùng lại cho các ảnh cùng kích thước; CSV lưu representative_id.']
    (output/'protocol_report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    if cfg.get('preset') == 'quick':
        lines = lines[:6] + lines[6:8+len(rows)] + [
            '', f'**Tăng tốc {final["FPS"]/ref["FPS"]:.2f}× · ΔPSNR {final["PSNR"]-ref["PSNR"]:+.3f} dB · '
            f'FPS/MSE {final["FPS_over_MSE"]/ref["FPS_over_MSE"]:.2f}×.**',
            architecture_note,
            f'Test: {final["images"]} ảnh; FP32, batch 1; max-side {cfg["eval_max_side"]}; '
            f'crop border {cfg["crop_border"]}. FPS/MSE = FPS / MSE RGB [0,1] trung bình theo ảnh.',
            'GFLOPs gồm counted + ước tính scan/FFT/Haar; chưa đếm đầy đủ normalization/elementwise. '
            'FPS không gồm I/O/transfer. REVIEW nghĩa là chưa đạt đầy đủ mục tiêu.',
            ('Latency/GFLOPs được dùng lại theo kích thước ảnh; chất lượng đo toàn bộ test.'
             if cfg.get('cache_shape_benchmarks') else 'Latency đo từng ảnh test.'),
            'Chi tiết: [protocol](protocol_report.md), [metadata](metadata.json), [CSV](summary.csv).']
    (output/'final_report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write_html_report(output, rows, metadata, cfg)
    print('\n'.join(lines), flush=True)


def write_html_report(output, rows, metadata, cfg):
    ref, final = rows[0], rows[-1]
    escape = lambda value: html.escape(str(value))
    cards = [('PSNR', f'{final["PSNR"]:.2f}', 'dB'), ('Tốc độ', f'{final["FPS"]:.1f}', 'FPS'),
             ('Hiệu quả', f'{final["FPS_over_MSE"]:,.0f}', 'FPS/MSE'),
             ('Độ phức tạp', f'{final["GFLOPs_256_estimated_total"]:.2f}', 'GFLOPs / 256², ước tính')]
    body = ''.join(f'<article><small>{escape(title)}</small><strong>{value}</strong><span>{unit}</span></article>'
                   for title,value,unit in cards)
    table = ''.join('<tr'+(' class="selected"' if i == len(rows)-1 else '')+'>'+''.join(
        f'<td>{escape(value)}</td>' for value in (r['model'],r['execution'],f'{r["PSNR"]:.3f}',
        f'{r["SSIM"]:.4f}',f'{r["MSE"]:.6f}',f'{r["FPS"]:.1f}',f'{r["FPS_over_MSE"]:,.0f}',
        f'{r["GFLOPs_256_estimated_total"]:.2f}'))+'</tr>' for i,r in enumerate(rows))
    document = '''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Phase 4 — Kết quả khử sương</title><style>
body{margin:0;background:#f3f6fb;color:#162238;font:16px/1.6 system-ui,sans-serif}main{max-width:1100px;margin:48px auto;padding:28px}
h1{font-size:36px;margin:4px 0}small,.muted{color:#61718c}.badge{padding:6px 14px;border-radius:20px;background:#e5eaf2;font-weight:700}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:16px;margin:28px 0}article{background:white;padding:22px;border-radius:18px;box-shadow:0 5px 25px #182c4810}
article strong{display:block;font-size:38px;letter-spacing:-1px}article span{font-size:13px;color:#61718c}.delta{padding:18px 24px;border-left:4px solid #3978ee;background:#e9f0ff;border-radius:8px}
.table{overflow:auto;margin:24px 0}table{border-collapse:collapse;background:white;width:100%;font-size:14px}td,th{padding:14px;text-align:right;border-bottom:1px solid #e8edf4;white-space:nowrap}td:first-child,th:first-child{text-align:left}.selected{background:#edf6ff;font-weight:700}footer{font-size:13px;color:#61718c}a{color:#2866c7}
</style><main>'''
    document += f'<small>KẾT QUẢ THỰC NGHIỆM · PHASE 4</small><h1>{escape(cfg["dataset_name"].upper())}</h1>'
    document += f'<span class="badge">{escape(metadata["status"])}</span><div class="cards">{body}</div>'
    if metadata['origin']['method'] == 'runtime_only':
        document += '<p>Chỉ tối ưu runtime: giữ nguyên kiến trúc và trọng số Reference; chưa có model nén.</p>'
    document += f'<div class="delta">Tăng tốc <b>{final["FPS"]/ref["FPS"]:.2f}×</b> · '
    document += f'ΔPSNR <b>{final["PSNR"]-ref["PSNR"]:+.3f} dB</b> · '
    document += f'FPS/MSE <b>{final["FPS_over_MSE"]/ref["FPS_over_MSE"]:.2f}×</b></div>'
    document += '<div class="table"><table><tr><th>Model</th><th>Runtime</th><th>PSNR</th><th>SSIM</th><th>MSE</th><th>FPS</th><th>FPS/MSE</th><th>GFLOPs 256*</th></tr>'+table+'</table></div>'
    document += f'<footer>{final["images"]} ảnh test · FP32 · batch 1 · max-side {cfg["eval_max_side"]} · '
    document += f'crop border {cfg["crop_border"]}. FPS/MSE dùng MSE RGB [0,1] trung bình theo ảnh. '
    document += 'FPS loại I/O/transfer. *GFLOPs là ước tính một phần. REVIEW: chưa đạt đủ điều kiện PSNR/FPS.'
    if cfg.get('cache_shape_benchmarks'): document += ' Latency/GFLOPs dùng chung theo shape; metric chất lượng đo mọi ảnh.'
    document += '<p><a href="protocol_report.md">Protocol</a> · <a href="metadata.json">Metadata</a> · <a href="summary.csv">Số liệu CSV</a></p></footer></main></html>'
    (output/'report.html').write_text(document, encoding='utf-8')


def run(cfg, mode='auto'):
    configure(cfg); device_for(cfg)
    output = Path(cfg['output_root']); output.mkdir(parents=True, exist_ok=True)
    inputs = json.loads((Path(cfg['work_root'])/'inputs.json').read_text(encoding='utf-8')); validate(cfg, inputs)
    require_adaptation(cfg, inputs)
    scratch = cfg['dataset_name'] == 'i-haze'
    if scratch:
        if inputs['evaluation_only']: raise ValueError('I-HAZE cần train/val để khởi tạo ngẫu nhiên.')
        reference = arch.reference_model(cfg['repo_root'], depth=cfg['reference_depth'])
        reference_training = train_model(reference, cfg, inputs, output, 'reference', cfg['reference_schedule'], 'random', mode=mode)
        # Independently random initialization: final I-HAZE receives no reference weights or KD.
        # Keep the final initialization identical whether reference training just
        # ran or was restored by auto/resume in a new process.
        configure({**cfg, 'seed': cfg['seed']+1000})
        final = arch.skeleton(cfg['repo_root'], {'kind': 'final', 'depth': 1, 'feature_adapter': True,
                                                'detail_width': cfg['detail_width']})
        # A pipeline interrupted during reference training has no final artifact
        # yet; finishing that reference must still allow the final stage to start.
        final_mode = 'auto' if mode == 'resume' and not (output/'final_best.pth').is_file() else mode
        training = train_model(final, cfg, inputs, output, 'final', cfg['scratch_schedule'], 'random', mode=final_mode)
        origin = {'method': 'random', 'reference_method': 'random_independent', 'source': None,
                  'reference_seed': cfg['seed'], 'final_seed': cfg['seed']+1000}
    else:
        reference = arch.reference_model(cfg['repo_root'], inputs['source'])
        reference_training = None
        if inputs['evaluation_only'] or cfg['dataset_name'] in cfg.get('runtime_only_datasets', []):
            # No adaptation split: avoid irreversible pruning without recovery supervision.
            final = copy.deepcopy(reference)
            training = {'mode': 'runtime_only', 'steps': 0}
            origin = {'method': 'runtime_only', 'source': inputs['source']}
        else:
            start = arch.reference_model(cfg['repo_root'], inputs['student_source']) if inputs['student_source'] else reference
            final, selected = arch.final_from_reference(start, cfg['den_blocks'], cfg['detail_width'])
            if cfg.get('require_compression') and not compression_info(reference, final)['compressed']:
                raise ValueError('Model cuối chưa giảm tham số so với Reference; dừng trước train.')
            print('Before recovery training:', compression_info(reference, final), flush=True)
            origin = {'method': 'inherited', 'source': inputs['source'], 'student_source': inputs['student_source'],
                      'selected_blocks': selected}
            training = train_model(final, cfg, inputs, output, 'final', cfg['adapt_schedule'], 'inherited',
                                   teacher=reference, mode=mode)
    compression = compression_info(reference, final)
    if cfg.get('require_compression') and not compression['compressed']:
        raise ValueError('Model cuối không có kiến trúc nhẹ hơn Reference; không xuất bản sao thành WaveLite.')
    print('Architecture check:', compression, flush=True)
    reference.cpu(); final.cpu()
    deploy, runtime = choose_runtime(copy.deepcopy(final), cfg, inputs, output)
    # Runtime selection is sealed before any test quality is computed.
    rows = []
    jobs = [(reference, 'Reference', 'eager')]
    runtime_only = origin['method'] == 'runtime_only'
    if not (runtime_only and cfg.get('skip_duplicate_eager', False)):
        jobs.append((final, 'Reference_copy' if runtime_only else 'WaveLite_eager', 'eager'))
    jobs.append((deploy, 'Reference_runtime' if runtime_only else 'WaveLite_deploy', runtime['execution']))
    for model, label, execution in jobs:
        rows.append(evaluate_one(model, label, execution, cfg, inputs, output))
        data.write_csv(output/'summary_partial.csv', rows)
    gates = assess(rows[0], rows[-1], cfg)
    if cfg.get('require_compression'): gates['architecture_compression'] = compression['compressed']
    gates['deployment_quality'] = abs(rows[-1]['PSNR']-rows[-2]['PSNR']) <= cfg['deploy_psnr_tolerance_db']
    metadata = {'status': 'PASS' if all(gates.values()) else 'REVIEW', 'gates': gates,
                'origin': origin, 'architecture_version': arch.ARCH_VERSION, 'model_spec': final.spec,
                'reference_spec': reference.spec, 'training': training, 'reference_training': reference_training,
                'runtime': runtime, 'compression': compression, 'identity': identity(cfg, inputs),
                'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU',
                'torch': torch.__version__, 'cuda': torch.version.cuda,
                'FPS_MSE_definition': 'FPS / arithmetic mean per-image RGB MSE on [0,1], same crop as PSNR',
                'metric_scope': 'same-session FP32, batch=1, raw input pad/crop included; I/O and transfer excluded'}
    data.write_csv(output/'summary.csv', rows); data.dump_json(output/'metadata.json', metadata)
    data.dump_json(output/'config.json', cfg)
    if (output/'final_best.pth').is_file(): metadata['final_sha256'] = data.sha256(output/'final_best.pth')

    # Self-contained eager export: reconstruction needs the architecture files,
    # not the original source weights. Fusion remains an inference-only transform.
    atomic_save({'arch_version': arch.ARCH_VERSION, 'model_spec': final.spec,
                 'params_ema': cpu_state(final), 'identity': identity(cfg, inputs),
                 'origin': origin, 'runtime': runtime}, output/'model_final.pth')
    metadata['export_sha256'] = data.sha256(output/'model_final.pth')
    data.dump_json(output/'metadata.json', metadata); write_report(output, rows, metadata, cfg)


if __name__ == '__main__':
    cfg = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')); action = sys.argv[2]
    if action == 'prepare': prepare(cfg)
    elif action in ('auto', 'train', 'resume', 'benchmark'): run(cfg, action)
    else: raise ValueError('Chọn prepare, auto, train, resume hoặc benchmark.')
