"""Analytical operation estimates; shared counting rules for every model."""
from collections import defaultdict
import math
import torch
from torch import nn

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
    if any(hasattr(m, 'restoration_network') for m in model.modules()):
        h, w = x.shape[-2:]
        padded_values = math.prod(x.shape[:-2]) * ((h+31)//32*32) * ((w+31)//32*32)
        rows['estimated_outer_haar'] += 8*padded_values
    counted = sum(v for k, v in rows.items() if k.startswith('counted_'))/1e9
    supplemental = sum(v for k, v in rows.items() if k.startswith('estimated_'))/1e9
    return {'counted_GFLOPs': counted, 'supplemental_GFLOPs': supplemental,
            'estimated_total_GFLOPs': counted+supplemental,
            **{k+'_GFLOPs': v/1e9 for k, v in rows.items()}}
