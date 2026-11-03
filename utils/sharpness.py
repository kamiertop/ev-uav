"""Non-adaptive SAM perturbation for the existing Adam/SGD optimizer.

Foret et al., Sharpness-Aware Minimization, ICLR 2021.
The caller computes the first gradient, enters the perturbation context,
clears gradients, and computes a second gradient on exactly the same batch.
Parameters are restored before the base optimizer steps, including on failure.
"""
from contextlib import contextmanager
import math

import torch
from torch.nn.modules.batchnorm import _BatchNorm


@contextmanager
def sam_perturbation(parameters, rho):
    if not math.isfinite(rho) or rho < 0:
        raise ValueError("SAM radius must be finite and nonnegative")
    if rho == 0:
        yield {"sam_perturbation_norm": 0.0}
        return
    parameters = [p for p in parameters if p.requires_grad and p.grad is not None]
    if not parameters:
        raise ValueError("SAM requires first-pass gradients")
    if any(p.grad.is_sparse for p in parameters):
        raise ValueError("SAM supports dense parameter gradients only")
    # Scale before reducing squared gradients to avoid overflow on finite inputs.
    maximum = torch.stack([p.grad.detach().abs().max() for p in parameters]).max()
    if not bool(torch.isfinite(maximum)):
        raise FloatingPointError("Non-finite first-pass SAM gradient")
    if not bool(maximum > 0):
        yield {"sam_perturbation_norm": 0.0}
        return
    dtype = torch.float64 if any(p.dtype == torch.float64 for p in parameters) else torch.float32
    unit_norm = torch.stack([
        (p.grad.detach().to(dtype) / maximum.to(dtype)).square().sum()
        for p in parameters
    ]).sum().sqrt()
    originals = []
    try:
        with torch.no_grad():
            for p in parameters:
                original = p.detach().clone()
                originals.append((p, original))
                delta = (p.grad.detach().to(dtype) / maximum.to(dtype)) * (rho / unit_norm)
                p.add_(delta.to(p.dtype))
        yield {"sam_perturbation_norm": rho}
    finally:
        with torch.no_grad():
            for p, original in originals:
                p.copy_(original)


@contextmanager
def batch_stats_without_running_updates(model):
    """Use training batch statistics on pass two, updating BN buffers only once."""
    states = [(module, module.track_running_stats) for module in model.modules()
              if isinstance(module, _BatchNorm)]
    try:
        for module, _ in states:
            module.track_running_stats = False
        yield
    finally:
        for module, tracking in states:
            module.track_running_stats = tracking
