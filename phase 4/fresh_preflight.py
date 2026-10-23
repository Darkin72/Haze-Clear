
import json, sys
from pathlib import Path
import torch
cfg_path = Path(sys.argv[1]); sys.path.insert(0, str(cfg_path.parent))
import phase4_arch as arch
import phase4_runner as runner
cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
inputs = json.loads(cfg_path.with_name('inputs.json').read_text(encoding='utf-8'))
runner.configure(cfg)
ref = arch.reference_model(cfg['repo_root'], inputs['source'] or None, depth=cfg['reference_depth'])
models = [ref]
if cfg['dataset_name'] == 'i-haze':
    models.append(arch.skeleton(cfg['repo_root'], {'kind':'final','depth':1,'feature_adapter':True,'detail_width':cfg['detail_width']}))
elif not (inputs['evaluation_only'] or cfg['dataset_name'] in cfg.get('runtime_only_datasets', [])):
    start = runner.adaptation_start(cfg, inputs, ref)
    models.append(arch.final_from_reference(start, cfg['den_blocks'], cfg['detail_width'])[0])
runner.require_adaptation(cfg, inputs)
if cfg.get('require_compression') and (len(models) < 2 or not runner.compression_info(ref, models[-1])['compressed']):
    raise RuntimeError('Preflight: model cuối chưa có kiến trúc nhẹ hơn Reference.')
print('Architecture preflight:', [m.spec for m in models], flush=True)
if len(models) > 1: print('Compression preflight:', runner.compression_info(ref, models[-1]), flush=True)
if cfg['dataset_name'] != 'i-haze' and len(models) > 1:
    print('Checkpoint transfer preflight:', arch.transfer_summary(models[-1]), flush=True)
probe = torch.rand(1, 3, 256, 256, device=cfg['device'])
for model in models:
    model.to(cfg['device']).eval()
    with torch.no_grad(): result = model(probe)
    if result.shape != probe.shape or not torch.isfinite(result).all():
        raise RuntimeError('Forward smoke test không hợp lệ.')
    model.cpu()
print('PREFLIGHT PASS:', cfg['dataset_name'], 'checkpoint strict + forward GPU', flush=True)
