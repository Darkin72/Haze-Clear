"""Resume Phase 3, recover PSNR, and benchmark FP32 deployment plus estimated GFLOPs."""
import copy
import gc
import json
import math
import random
import subprocess
import sys
import time
from collections import defaultdict, OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import phase3_core as core


def prepare(cfg):
    inputs = core.prepare_inputs(cfg)
    inputs['phase3_checkpoint'] = None
    inputs['phase3_sha256'] = None
    if not cfg.get('phase3_checkpoint'):
        core.dump_json(Path(cfg['work_root']) / 'inputs.json', inputs)
        print('Initialize student from compressed teacher; no O-HAZY student checkpoint.', flush=True)
        return
    path = Path(cfg['phase3_checkpoint']).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Cần checkpoint Phase 3: {path}')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    previous = saved.get('config', {})
    if int(previous.get('den_blocks', 1)) != 1:
        raise ValueError('V2 này tiếp tục từ Phase 3 có một block DEN/group.')
    name = core.normalize_dataset_name(cfg.get('dataset_name', 'o-hazy'))
    source_name = core.normalize_dataset_name(previous.get('dataset_name', 'o-hazy'))
    expected_source = 'reside6k' if name.startswith('sots-') and inputs['evaluation_only'] else name
    if source_name != expected_source:
        raise ValueError(f'Checkpoint student thuộc {source_name}, cần {expected_source}. Để trống để nén teacher.')
    if not inputs['evaluation_only'] and previous.get('val_ids', cfg['val_ids']) != cfg['val_ids']:
        raise ValueError('Giữ nguyên val_ids của Phase 3.')
    if name == 'o-hazy' and previous.get('eval_max_side', cfg['eval_max_side']) != cfg['eval_max_side']:
        raise ValueError('Giữ nguyên eval_max_side để so sánh với Phase 3.')
    if saved.get('teacher_sha256', inputs['checkpoint_sha256']) != inputs['checkpoint_sha256']:
        raise ValueError('Checkpoint teacher không khớp teacher của Phase 3.')
    inputs['phase3_checkpoint'] = str(path)
    inputs['phase3_sha256'] = core.sha256(path)
    core.dump_json(Path(cfg['work_root']) / 'inputs.json', inputs)
    print('Resume Phase 3:', path, flush=True)


def configure(cfg):
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    core.cv2.setNumThreads(0)


def validate_inputs(cfg, inputs):
    if inputs['dataset_name'] != core.normalize_dataset_name(cfg.get('dataset_name', 'o-hazy')):
        raise ValueError('Dataset selector đã thay đổi; chạy prepare lại.')
    core.assert_disjoint(inputs['splits'])
    for split in inputs['splits'].values():
        for row in split:
            for key in ('hazy', 'clear'):
                if core.sha256(row[key]) != row[key+'_sha256']:
                    raise ValueError(f'Dataset đã thay đổi: {row[key]}')
    for key in ('checkpoint', 'phase3_checkpoint'):
        if not inputs.get(key):
            continue
        hash_key = 'checkpoint_sha256' if key == 'checkpoint' else 'phase3_sha256'
        if core.sha256(inputs[key]) != inputs[hash_key]:
            raise ValueError(f'Checkpoint đã thay đổi: {inputs[key]}')
    data, expected, eval_only = core.dataset_splits(cfg)
    if str(data) != inputs['data_root'] or eval_only != inputs['evaluation_only']:
        raise ValueError('Data root hoặc chế độ đã thay đổi; chạy prepare lại.')
    for name, rows in expected.items():
        keys = lambda values: [(r['id'], r['hazy'], r['clear']) for r in values]
        if keys(rows) != keys(inputs['splits'][name]):
            raise ValueError('Manifest split không khớp config hiện tại; chạy prepare lại.')


def restoration_loss(pred, gt, teacher, cfg):
    border = cfg['crop_border']
    if border:
        pred, gt, teacher = [v[..., border:-border, border:-border] for v in (pred, gt, teacher)]
    return (F.mse_loss(pred, gt) + cfg['l1_weight']*F.l1_loss(pred, gt)
            + cfg['kd_mse_weight']*F.mse_loss(pred, teacher))


def refine(cfg, inputs, teacher, student, output):
    """A short DEN stage followed by full-image joint training at evaluation resolution."""
    if inputs.get('evaluation_only'):
        return {'seconds': 0., 'steps': 0, 'initial_val_PSNR': None, 'best_val_PSNR': None,
                'mode': 'evaluation-only; không fine-tune trên test SOTS'}
    started = time.perf_counter()
    # Bounded LRU: RESIDE-6K must not materialize thousands of full FP32 images in RAM.
    cached = OrderedDict()
    capacity = int(cfg.get('max_cache_images', 40))
    if capacity < 1:
        raise ValueError('max_cache_images phải lớn hơn 0.')
    training_pairs = inputs['splits']['train']
    coarse = copy.deepcopy(student).eval().requires_grad_(False)
    coarse.restoration_network.DE = nn.Identity()
    @torch.no_grad()
    def cache_pair(row):
        key = row['hazy']
        if key not in cached:
            x, y = core.load_pair(row, cfg['train_max_side'])
            h, w = x.shape[-2:]
            xp = core.pad32(x[None].cuda())
            cached[key] = {'id': row['id'], 'x': x, 'y': y,
                           'coarse': coarse(xp)[0, :, :h, :w].cpu().contiguous(),
                           'teacher': teacher(xp)[0, :, :h, :w].cpu().contiguous()}
            if len(cached) > capacity:
                cached.popitem(last=False)
        cached.move_to_end(key)
        return cached[key]
    if len(training_pairs) <= capacity:
        for row in training_pairs:
            cache_pair(row)
            print(f'Cache train {row["id"]}', flush=True)
        coarse.cpu()
        teacher.cpu()
    else:
        print(f'Cache LRU {capacity} ảnh; lấy mẫu từ toàn bộ {len(training_pairs)} ảnh train.', flush=True)
    torch.cuda.empty_cache()
    ema = copy.deepcopy(student).eval().requires_grad_(False)
    initial = core.quality(ema, inputs['splits']['val'], cfg, 'initial_student_val')
    best = float(np.mean([r['PSNR'] for r in initial]))
    history = [{'stage': 'initial', 'step': 0, 'loss': 0., 'val_PSNR': best, 'elapsed_s': 0.}]
    path = output / 'fast_wdmamba_v2_best.pth'

    def save(step, score):
        torch.save({'params_ema': ema.state_dict(), 'config': cfg, 'step': step,
                    'val_PSNR': score, 'teacher_sha256': inputs['checkpoint_sha256'],
                    'phase3_sha256': inputs['phase3_sha256']}, path)

    save(0, best)  # A worse validation checkpoint never replaces the initial Phase 3.
    core.write_csv(output / 'initial_val_metrics.csv', initial)
    step = 0
    for stage, count in [('den_crop', cfg['crop_steps']), ('joint_full', cfg['full_steps'])]:
        if count <= 0:
            continue
        for name, p in student.named_parameters():
            p.requires_grad_(stage == 'joint_full' or name.startswith('restoration_network.DE.'))
        den, lfrn = [], []
        for name, p in student.named_parameters():
            if p.requires_grad:
                (den if name.startswith('restoration_network.DE.') else lfrn).append(p)
        groups = [{'params': den, 'lr': cfg['crop_lr'] if stage == 'den_crop' else cfg['full_den_lr']}]
        if lfrn:
            groups.append({'params': lfrn, 'lr': cfg['full_lfrn_lr']})
        opt = torch.optim.AdamW(groups, weight_decay=cfg['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(count, 1), eta_min=1e-7)
        student.train()
        core.freeze_bn(student)
        stale = 0
        for local_step in range(1, count+1):
            opt.zero_grad(set_to_none=True)
            batch_loss = 0.
            for _ in range(cfg['accum_steps']):
                row = cache_pair(random.choice(training_pairs))
                keys = ('coarse', 'y', 'teacher') if stage == 'den_crop' else ('x', 'y', 'teacher')
                images = [row[k] for k in keys]
                if stage == 'den_crop':
                    h, w = images[0].shape[-2:]
                    s = (min(cfg['crop_size'], h, w)//32)*32
                    if s < 32:
                        raise ValueError('Ảnh train phải có cạnh ngắn ít nhất 32 pixel.')
                    top, left = random.randrange(h-s+1), random.randrange(w-s+1)
                    images = [im[:, top:top+s, left:left+s] for im in images]
                    if random.random() < .5:
                        images = [im.flip(-1) for im in images]
                    if random.random() < .5:
                        images = [im.flip(-2) for im in images]
                x, gt, target = [im[None].cuda() for im in images]
                if stage == 'den_crop':
                    pred = student.restoration_network.DE(x)
                else:
                    # No crop/context mismatch: same full-image pad/crop as evaluator.
                    h, w = x.shape[-2:]
                    pred = core.final_output(student(core.pad32(x)))[..., :h, :w]
                loss = restoration_loss(pred, gt, target, cfg)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Loss không hữu hạn.')
                (loss/cfg['accum_steps']).backward()
                batch_loss += loss.item()/cfg['accum_steps']
            nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
            opt.step()
            scheduler.step()
            core.update_ema(ema, student, cfg['ema_decay'])
            step += 1
            if local_step % cfg['log_every'] == 0:
                print(f'{stage}: step={step}, loss={batch_loss:.6f}', flush=True)
            if local_step % cfg['val_every'] == 0 or local_step == count:
                rows = core.quality(ema, inputs['splits']['val'], cfg, 'V2_val')
                score = float(np.mean([r['PSNR'] for r in rows]))
                history.append({'stage': stage, 'step': step, 'loss': batch_loss, 'val_PSNR': score,
                                'elapsed_s': time.perf_counter()-started})
                core.write_csv(output / 'history_v2.csv', history)
                print(f'Validation={score:.4f}, best={best:.4f}', flush=True)
                if score > best:
                    best, stale = score, 0
                    save(step, score)
                else:
                    stale += 1
                if stale >= cfg['patience']:
                    print(f'Early stop stage {stage} bằng validation.', flush=True)
                    break
        # Start the next stage from the best EMA, not an overfit last iterate.
        student.load_state_dict(core.checkpoint_state(path), strict=True)
        ema.load_state_dict(student.state_dict(), strict=True)
    student.load_state_dict(core.checkpoint_state(path), strict=True)
    del ema, cached, coarse
    teacher.cpu()
    torch.cuda.empty_cache()
    return {'seconds': time.perf_counter()-started, 'steps': step,
            'initial_val_PSNR': history[0]['val_PSNR'], 'best_val_PSNR': best}


@torch.inference_mode(False)
@torch.no_grad()
def fuse_conv_bn(model):
    """Inference-only Conv2d/BatchNorm fusion; no graph or checkpoint key mutation during train."""
    fused = 0
    for module in list(model.modules()):
        if isinstance(module, nn.Sequential):
            names = list(module._modules)
            for left, right in zip(names, names[1:]):
                conv, bn = module._modules[left], module._modules[right]
                if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
                    module._modules[left] = torch.nn.utils.fusion.fuse_conv_bn_eval(conv, bn)
                    module._modules[right] = nn.Identity()
                    fused += 1
    return fused


class GraphReplay(nn.Module):
    """One shape at a time; static output is consumed before the next replay."""
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()
        self.release()

    def release(self):
        self.graph = self.static_x = self.static_y = None
        self.shape = None
        gc.collect()

    @torch.inference_mode()
    def forward(self, x):
        shape = (tuple(x.shape), x.dtype, x.device)
        if self.shape != shape:
            self.release()
            self.static_x = x.clone()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.model(self.static_x)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_y = self.model(self.static_x)
            self.shape = shape
        self.static_x.copy_(x)
        self.graph.replay()
        return self.static_y


@torch.inference_mode(False)
@torch.no_grad()
def choose_deployment(model, cfg, inputs, output):
    """Decide runtime on validation only, before opening any test metric."""
    model.eval().cuda()
    if inputs['splits']['val']:
        selection_source = 'validation'
        x, _ = core.load_pair(inputs['splits']['val'][0], cfg['eval_max_side'])
        xp = core.pad32(x[None].cuda())
    else:
        selection_source = 'synthetic_256'
        # Evaluation-only datasets have no tuning split; select runtime on a synthetic shape probe.
        xp = torch.zeros(1, 3, 256, 256, device='cuda')
    reference = model(xp).clone()
    fused = fuse_conv_bn(model)
    error = (model(xp)-reference).abs().max().item()
    if error > cfg['fusion_max_error']:
        raise ValueError(f'Conv-BN fusion không đạt kiểm tra sai số: {error}')
    eager = core.latency(model, xp, cfg)
    info = {'selection_source': selection_source, 'fused_conv_bn': fused, 'fusion_max_abs_error': error, 'execution': 'fused_eager',
            'validation_eager_ms': eager['latency_ms']}
    graph = GraphReplay(model)
    chosen = model
    if cfg['try_cuda_graph']:
        try:
            y = graph(xp)
            graph_error = (y-model(xp)).abs().max().item()
            if graph_error > cfg['fusion_max_error']:
                raise ValueError(f'Graph output error={graph_error}')
            timing = core.latency(graph, xp, cfg)
            info.update(graph_max_abs_error=graph_error, validation_graph_ms=timing['latency_ms'])
            if timing['latency_ms'] < eager['latency_ms']:
                chosen = graph
                info['execution'] = 'fused_cuda_graph'
        except (RuntimeError, ValueError) as exc:
            info['graph_error'] = str(exc)
    if chosen is model:
        graph.release()
        # Verify fallback in the same ordinary-tensor mode used for later profiling.
        model(xp)
    core.dump_json(output / 'deployment_selection.json', info)
    del xp, reference
    print('Deployment selected:', info, flush=True)
    return chosen, info


def profile_gflops(model, x):
    """Analytical forward hooks; 1 multiply-add = 2 FLOPs, not GFLOP/s.

    Count Conv/Linear arithmetic exactly under dense zero-padding conventions.
    Functional SS2D projections are counted explicitly. Scan, FFT and Haar are
    supplemental estimates; activations, normalization, pooling and elementwise
    bookkeeping are not counted. This is not a complete hardware instruction count.
    """
    rows = defaultdict(float)
    handles = []

    def hook(module, args, result):
        a = args[0]
        name = module.__class__.__name__
        if isinstance(module, (nn.Conv1d, nn.Conv2d)):
            rows['counted_conv_linear'] += 2*result.numel()*(module.in_channels//module.groups)*math.prod(module.kernel_size)
        elif isinstance(module, nn.ConvTranspose2d):
            rows['counted_conv_linear'] += 2*a.numel()*(module.out_channels//module.groups)*math.prod(module.kernel_size)
        elif isinstance(module, nn.Linear):
            rows['counted_conv_linear'] += 2*a.numel()*module.out_features
        elif name == 'SS2D':
            b, h, w, _ = a.shape
            d, n, r = module.d_inner, module.d_state, module.dt_rank
            rows['counted_ss2d_einsum'] += 2*b*4*h*w*d*(2*r+2*n)
            rows['estimated_selective_scan'] += b*4*h*w*d*(8*n+2)
        elif name == 'fft_bench':
            b, c, h, w = a.shape
            rows['estimated_fft'] += 10*b*c*h*w*math.log2(h*w)
        elif name == 'DWTForward':
            # pytorch_wavelets functional Haar convolutions bypass nn.Conv hooks.
            rows['estimated_internal_haar'] += 4*a.numel()

    was_training = {m: m.training for m in model.modules()}
    try:
        model.eval()
        for module in model.modules():
            handles.append(module.register_forward_hook(hook))
        with torch.inference_mode():
            model(x)
    finally:
        for handle in handles:
            handle.remove()
        for module, training in was_training.items():
            module.training = training
    # Outer DWT/IWT are constructed inside forward and cannot be hooked persistently.
    if hasattr(model, 'restoration_network'):
        rows['estimated_outer_haar'] += 8*x.numel()
    counted = sum(v for k, v in rows.items() if k.startswith('counted_'))/1e9
    supplemental = sum(v for k, v in rows.items() if k.startswith('estimated_'))/1e9
    return {'counted_GFLOPs': counted, 'supplemental_GFLOPs': supplemental,
            'estimated_total_GFLOPs': counted+supplemental,
            **{k+'_GFLOPs': v/1e9 for k, v in rows.items()}}


def clear_graph(model):
    if isinstance(model, GraphReplay):
        model.release()
    torch.cuda.empty_cache()


def evaluate(model, cfg, inputs, label, execution, output):
    model.cuda().eval()
    physical = model.model if isinstance(model, GraphReplay) else model
    quality = core.quality(model, inputs['splits']['test'], cfg, label, output/'predictions'/label)
    clear_graph(model)
    timing, flops = [], []
    for row in inputs['splits']['test']:
        x, _ = core.load_pair(row, cfg['eval_max_side'])
        xp = core.pad32(x[None].cuda())
        h, w = xp.shape[-2:]
        # Eager arithmetic counter always profiles the physical model, never a replay.
        f = profile_gflops(physical, xp)
        flops.append({'dataset': cfg.get('dataset_name', 'o-hazy'), 'model': label, 'id': row['id'], 'height': h, 'width': w, **f})
        timing.append({'dataset': cfg.get('dataset_name', 'o-hazy'), 'model': label, 'id': row['id'], 'height': h, 'width': w,
                       **core.latency(model, xp, cfg)})
        clear_graph(model)
        del xp
    probe = torch.zeros(1, 3, 256, 256, device='cuda')
    f256 = profile_gflops(physical, probe)
    t256 = core.latency(model, probe, cfg)
    clear_graph(model)
    summary = {'dataset': cfg.get('dataset_name', 'o-hazy'), 'model': label, 'execution': execution, 'precision': 'FP32',
               'images': len(inputs['splits']['test']),
               'batch_size': 1, 'eval_max_side': cfg['eval_max_side'],
               'PSNR': float(np.mean([v['PSNR'] for v in quality])),
               'SSIM': float(np.mean([v['SSIM'] for v in quality])),
               'latency_ms': float(np.mean([v['latency_ms'] for v in timing])),
               'FPS': 1000/float(np.mean([v['latency_ms'] for v in timing])),
               'FPS_256': t256['FPS'], 'params_M': sum(p.numel() for p in physical.parameters())/1e6,
               'peak_allocated_GiB': max(v['peak_allocated_GiB'] for v in timing),
               'GFLOPs_256_counted': f256['counted_GFLOPs'],
               'GFLOPs_256_supplemental': f256['supplemental_GFLOPs'],
               'GFLOPs_256_estimated_total': f256['estimated_total_GFLOPs'],
               'GFLOPs_test_estimated_mean': float(np.mean([v['estimated_total_GFLOPs'] for v in flops]))}
    core.write_csv(output/f'{label}_metrics.csv', quality)
    core.write_csv(output/f'{label}_timing.csv', timing)
    core.write_csv(output/f'{label}_gflops.csv', flops)
    core.dump_json(output/f'{label}_256.json', {'latency': t256, 'gflops': f256})
    del probe
    model.cpu()
    torch.cuda.empty_cache()
    print(summary, flush=True)
    return summary


def restore_v2(cfg, inputs, student, output):
    """Load the already selected EMA checkpoint; never train or select by test metrics."""
    path = Path(cfg.get('v2_checkpoint') or output/'fast_wdmamba_v2_best.pth').expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Không có checkpoint V2 đã train: {path}. Chế độ benchmark không chạy training.')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    previous = saved.get('config', {})
    if saved.get('teacher_sha256') != inputs['checkpoint_sha256']:
        raise ValueError('Teacher của checkpoint V2 không khớp manifest.')
    if saved.get('phase3_sha256') != inputs.get('phase3_sha256'):
        raise ValueError('Baseline Phase 3 của checkpoint V2 không khớp manifest.')
    for key in ('dataset_name', 'den_blocks', 'eval_max_side', 'crop_border', 'val_ids'):
        if previous.get(key) != cfg.get(key):
            raise ValueError(f'Cấu hình checkpoint V2 không khớp {key}: {previous.get(key)} != {cfg.get(key)}')
    student.load_state_dict(core.checkpoint_state(path), strict=True)
    return {'mode': 'benchmark-only; không train lại', 'checkpoint': str(path),
            'checkpoint_sha256': core.sha256(path), 'selected_step': saved.get('step'),
            'best_val_PSNR': saved.get('val_PSNR')}


def run(cfg, benchmark_only=False):
    if not torch.cuda.is_available():
        raise RuntimeError('Cần GPU CUDA để train/benchmark. CPU chỉ dùng kiểm tra công thức.')
    configure(cfg)
    output = Path(cfg['output_root'])
    output.mkdir(parents=True, exist_ok=True)
    inputs = json.loads((Path(cfg['work_root'])/'inputs.json').read_text(encoding='utf-8'))
    validate_inputs(cfg, inputs)
    teacher, student, depth = core.teacher_and_student(cfg, inputs['checkpoint'])
    resumed = bool(inputs.get('phase3_checkpoint'))
    if resumed:
        student.load_state_dict(core.checkpoint_state(inputs['phase3_checkpoint']), strict=True)
    baseline_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
    training = (restore_v2(cfg, inputs, student, output) if benchmark_only
                else refine(cfg, inputs, teacher, student, output))
    student.eval()
    # Select runtime using validation; test metrics are still unopened.
    deployment, deploy_info = choose_deployment(copy.deepcopy(student), cfg, inputs, output)
    clear_graph(deployment)
    deployment.cpu()
    student.cpu()
    teacher.cpu()
    previous = copy.deepcopy(student)
    previous.load_state_dict(baseline_state, strict=True)
    rows = []
    baseline_label = 'Phase3' if resumed else 'Compressed_init'
    student_label = 'Student_eager' if inputs['evaluation_only'] else 'V2_eager'
    deploy_label = 'Student_deploy' if inputs['evaluation_only'] else 'V2_deploy'
    for model, name, execution in [(teacher, 'WDMamba', 'eager'), (previous, baseline_label, 'eager'),
                                   (student, student_label, 'eager'),
                                   (deployment, deploy_label, deploy_info['execution'])]:
        rows.append(evaluate(model, cfg, inputs, name, execution, output))
        # Keep completed model results if a later GPU operation fails.
        core.write_csv(output/'summary_partial.csv', rows)
    core.write_csv(output/'summary.csv', rows)
    base, old, new, deploy = rows
    if inputs['evaluation_only']:
        gates = {'deployment_quality': abs(deploy['PSNR']-new['PSNR']) <= cfg['deploy_psnr_tolerance_db']}
    else:
        gates = {'PSNR_gain_vs_baseline': new['PSNR']-old['PSNR'] >= cfg['min_psnr_gain_db'],
                 'FPS_gain_vs_baseline': deploy['FPS']/old['FPS'] >= cfg['min_fps_ratio'],
                 'teacher_quality': new['PSNR']-base['PSNR'] >= -cfg['psnr_tolerance_db'],
                 'deployment_quality': abs(deploy['PSNR']-new['PSNR']) <= cfg['deploy_psnr_tolerance_db']}
    status = ('EVALUATION_ONLY' if inputs['evaluation_only'] else 'PASS') if all(gates.values()) else 'REVIEW'
    metadata = {'status': status, 'gates': gates, 'baseline_model': baseline_label,
                'dataset': inputs['dataset_name'], 'evaluation_only': inputs['evaluation_only'],
                'delta_PSNR_vs_baseline': new['PSNR']-old['PSNR'],
                'delta_PSNR_vs_teacher': new['PSNR']-base['PSNR'],
                'deployment_speedup_vs_baseline': deploy['FPS']/old['FPS'],
                'eager_speedup_vs_baseline': new['FPS']/old['FPS'],
                'deployment': deploy_info, 'training': training,
                'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
                'teacher_DEN_blocks': depth, 'phase3_sha256': inputs['phase3_sha256'],
                'teacher_sha256': inputs['checkpoint_sha256'],
                'GFLOPs_method': '1 MAC=2 FLOPs; Conv/Linear + SS2D projections counted; scan/FFT/Haar estimated; elementwise/norm/pooling excluded'}
    core.dump_json(output/'metadata.json', metadata)
    core.dump_json(output/'config.json', cfg)
    lines = [f"# Fast-WDMamba V2 — {inputs['dataset_name']}", '', f"Trạng thái: **{metadata['status']}**", '',
             '| Dataset | Model | Thực thi | PSNR | SSIM | FPS | FPS 256 | GFLOPs counted | GFLOPs bổ sung | GFLOPs tổng ước tính | Params M |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for v in rows:
        lines.append(f"| {v['dataset']} | {v['model']} | {v['execution']} | {v['PSNR']:.4f} | {v['SSIM']:.5f} | {v['FPS']:.2f} | {v['FPS_256']:.2f} | {v['GFLOPs_256_counted']:.3f} | {v['GFLOPs_256_supplemental']:.3f} | {v['GFLOPs_256_estimated_total']:.3f} | {v['params_M']:.3f} |")
    lines += ['', f"ΔPSNR so {baseline_label}: {metadata['delta_PSNR_vs_baseline']:+.4f} dB; "
              f"tốc độ triển khai so baseline: {metadata['deployment_speedup_vs_baseline']:.3f}x.",
              f"ΔPSNR so teacher: {metadata['delta_PSNR_vs_teacher']:+.4f} dB.", '',
              f"Chế độ: {'chỉ đánh giá, không cập nhật trọng số' if inputs['evaluation_only'] else 'fine-tune train, chọn checkpoint bằng validation'}.",
              'GFLOPs gần như giữ nguyên giữa Phase 3 và V2; giảm kernel launch không đồng nghĩa giảm số phép tính.',
              'Đếm trên tensor đã pad, cùng hình dạng và FP32. GFLOPs là số phép tính/ảnh, không phải GFLOP/s.',
              'CSV tách counted và supplemental; không gọi supplemental là số FLOPs chính xác toàn mạng.',
              'Tất cả model được đo lại trong cùng phiên GPU. Không dùng FPS cũ từ file Phase 3 để tính speedup.',
              f"Đánh giá {len(inputs['splits']['test'])} ảnh test của {inputs['dataset_name']}; không dùng test để chọn checkpoint.",
              'Nếu REVIEW, mục tiêu tăng cả PSNR và FPS chưa được xác nhận; giữ nguyên số đo.']
    if inputs['dataset_name'] == 'o-hazy':
        lines += ['O-HAZY giữ test 41–45. Test Phase 3 đã được xem; đây là vòng phát triển tiếp.',
                  'Teacher O-HAZY đã học 1–40, gồm validation compression; điểm validation không độc lập với pretraining.']
    if inputs['dataset_name'] == 'i-haze' and Path(inputs['checkpoint']).name == 'Ohaze_1200_1600_27.22.pth':
        lines += ['I-HAZE dùng teacher O-HAZY làm khởi tạo chuyển miền; không có checkpoint I-HAZE gốc trong cấu hình mặc định.']
    if inputs['dataset_name'].startswith('sots-'):
        lines += ['SOTS là benchmark Indoor/Outdoor; ITS/OTS là các tập train riêng, có thể chỉ định bằng TRAIN_ROOT.',
                  'Nguồn teacher mặc định: RESIDE-6K. Cần kiểm tra giao scene với dữ liệu pretraining trước khi claim tổng quát hóa.']
    (output/'final_report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)


def benchmark(cfg):
    run(cfg, benchmark_only=True)


def auto(cfg):
    inputs = json.loads((Path(cfg['work_root'])/'inputs.json').read_text(encoding='utf-8'))
    checkpoint = Path(cfg.get('v2_checkpoint') or Path(cfg['output_root'])/'fast_wdmamba_v2_best.pth').expanduser()
    if not inputs['evaluation_only'] and (checkpoint.is_file() or cfg.get('v2_checkpoint')):
        print('AUTO: reuse saved V2 checkpoint; benchmark only, no training.', flush=True)
        benchmark(cfg)
    else:
        print('AUTO: run training/evaluation according to the dataset splits.', flush=True)
        run(cfg)


if __name__ == '__main__':
    cfg = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    {'prepare': prepare, 'run': run, 'benchmark': benchmark, 'auto': auto}[sys.argv[2]](cfg)
