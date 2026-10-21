"""WaveLite-Final: wavelet/Mamba trunk, compact DEN and residual detail."""
from functools import lru_cache
from pathlib import Path
import copy
import importlib.machinery
import importlib.util
import re
import sys
import torch
from torch import nn
import phase4_data as data

ARCH_VERSION = 'p4-wavelite-uden1-dw-detail-v1'


@lru_cache(maxsize=1)
def import_architecture(root):
    for name, rel in [('basicsr', 'basicsr'), ('basicsr.utils', 'basicsr/utils'),
                      ('basicsr.archs', 'basicsr/archs')]:
        if name in sys.modules:
            raise RuntimeError(f'Cần process sạch; đã import {name}.')
        spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        spec.submodule_search_locations = [str(Path(root) / rel)]
        sys.modules[name] = importlib.util.module_from_spec(spec)
    from basicsr.archs.wavemamba_arch import WaveMamba
    from basicsr.archs.detail_enhance_net import DENet
    return WaveMamba, DENet


def checkpoint_state(path):
    obj = torch.load(path, map_location='cpu', weights_only=False)
    for key in ('params_ema', 'params', 'state_dict', 'model'):
        if isinstance(obj, dict) and isinstance(obj.get(key), dict):
            obj = obj[key]
            break
    if not isinstance(obj, dict) or not obj or not all(torch.is_tensor(v) for v in obj.values()):
        raise ValueError(f'Không có state_dict WDMamba hợp lệ: {path}')
    return {k.removeprefix('module.'): v for k, v in obj.items()}


def detect_den_depth(state):
    groups = {i: set() for i in (1, 2, 3)}
    for key in state:
        m = re.fullmatch(r'restoration_network\.DE\.g([123])\.gp\.(\d+)\.conv1\.weight', key)
        if m:
            groups[int(m[1])].add(int(m[2]))
    depths = {len(v) for v in groups.values()}
    if len(depths) != 1 or next(iter(depths)) not in (1, 2, 3, 4, 6):
        raise ValueError(f'Không nhận diện được độ sâu DEN: {groups}')
    depth = next(iter(depths))
    if any(v != set(range(depth)) for v in groups.values()):
        raise ValueError('Chỉ số block DEN không liên tục.')
    return depth


class FeatureAdapter(nn.Module):
    """A second, cheap DEN block; exact identity at initialization."""
    def __init__(self, channels=16):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, 2*channels, 1), nn.SiLU(),
                                  nn.Conv2d(2*channels, 2*channels, 5, padding=2, groups=2*channels),
                                  nn.SiLU(), nn.Conv2d(2*channels, channels, 1))
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, x):
        return x + self.body(x)


class DetailAdapter(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(3, width, 1), nn.SiLU(),
                                  nn.Conv2d(width, width, 3, padding=1, groups=width), nn.SiLU(),
                                  nn.Conv2d(width, 3, 1))
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, x):
        return self.body(x)


class PaddedModel(nn.Module):
    """All measured forwards include pad/crop; no change to image resolution."""
    def __init__(self, trunk, spec, adapter=None):
        super().__init__()
        self.trunk, self.adapter, self.spec = trunk, adapter, spec

    def forward(self, x):
        h, w = x.shape[-2:]
        y = self.trunk(data.pad32(x))
        if isinstance(y, (tuple, list)):
            y = y[-1]
        if self.adapter is not None:
            y = y + self.adapter(y)
        return y[..., :h, :w]


def new_trunk(repo_root, depth):
    WaveMamba, DENet = import_architecture(str(Path(repo_root).resolve()))
    trunk = WaveMamba(in_chn=3, wf=16, n_l_blocks=[1, 2, 2, 4], ffn_scale=2.)
    trunk.restoration_network.DE = DENet(3, depth)
    return trunk


def reference_model(repo_root, path=None, depth=4):
    state = checkpoint_state(path) if path else None
    if state is not None:
        depth = detect_den_depth(state)
    trunk = new_trunk(repo_root, depth)
    if state is not None:
        trunk.load_state_dict(state, strict=True)
    return PaddedModel(trunk, {'kind': 'reference', 'depth': depth, 'detail_width': 0})


def final_from_reference(reference, keep=1, detail_width=16):
    if keep != 1:
        raise ValueError('Phase 4 cố định một U-block và một depthwise block/group.')
    trunk = copy.deepcopy(reference.trunk)
    depth = reference.spec['depth']
    selected = [0]
    for name in ('g1', 'g2', 'g3'):
        group = getattr(trunk.restoration_network.DE, name)
        gp = group.gp
        channels = gp[depth].in_channels
        group.gp = nn.Sequential(copy.deepcopy(gp[0]), FeatureAdapter(channels), copy.deepcopy(gp[depth]))
    return PaddedModel(trunk, {'kind': 'final', 'depth': keep, 'feature_adapter': True, 'detail_width': detail_width},
                       DetailAdapter(detail_width)), selected


def skeleton(repo_root, spec):
    if spec['kind'] == 'reference':
        return reference_model(repo_root, depth=spec['depth'])
    if spec['kind'] != 'final' or spec['depth'] != 1 or not spec.get('feature_adapter'):
        raise ValueError(f'Model spec không hợp lệ: {spec}')
    trunk = new_trunk(repo_root, 1)
    for name in ('g1', 'g2', 'g3'):
        group = getattr(trunk.restoration_network.DE, name)
        group.gp = nn.Sequential(group.gp[0], FeatureAdapter(group.gp[-1].in_channels), group.gp[-1])
    return PaddedModel(trunk, dict(spec), DetailAdapter(spec['detail_width']))


@torch.inference_mode(False)
@torch.no_grad()
def fuse_conv_bn(model):
    count = 0
    for module in list(model.modules()):
        if isinstance(module, nn.Sequential):
            names = list(module._modules)
            for left, right in zip(names, names[1:]):
                conv, bn = module._modules[left], module._modules[right]
                if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
                    module._modules[left] = torch.nn.utils.fusion.fuse_conv_bn_eval(conv, bn)
                    module._modules[right] = nn.Identity()
                    count += 1
    return count
