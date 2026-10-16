"""Stop invalid optimization steps before they contaminate model parameters."""
import math
import torch


def check_loss(loss):
    if not bool(torch.isfinite(loss).all()):
        raise FloatingPointError("Non-finite training loss")


def check_gradients(parameters, max_norm=0.0):
    parameters = [p for p in parameters if p.grad is not None]
    if not parameters:
        return 0.0
    # infinity norm avoids overflowing a sum of squared, otherwise finite grads.
    norm = torch.nn.utils.clip_grad_norm_(
        parameters, max_norm if max_norm > 0 else math.inf,
        norm_type=2.0 if max_norm > 0 else math.inf,
        error_if_nonfinite=True,
    )
    return float(norm)


def check_tensors(named_tensors):
    for name, tensor in named_tensors:
        if tensor is not None and (tensor.is_floating_point() or tensor.is_complex()):
            if not bool(torch.isfinite(tensor).all()):
                raise FloatingPointError(f"Non-finite tensor: {name}")
