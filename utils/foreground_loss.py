"""R1 foreground event BCE, averaged equally over samples.

Only label>0.5 events with positive instance IDs receive this auxiliary loss.
STC continues to supervise background and foreground events without valid IDs.
This loss does not impose temporal connectivity or instance/bin balancing.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class ForegroundEventLoss(nn.Module):
    def forward(self, logits, labels, locations, instance_ids):
        logits = logits.reshape(-1)
        if logits.dtype in (torch.float16, torch.bfloat16):
            logits = logits.float()
        labels, instance_ids = labels.reshape(-1), instance_ids.reshape(-1)
        n = logits.numel()
        if locations.shape != (n, 4) or labels.numel() != n or instance_ids.numel() != n:
            raise ValueError("foreground inputs must align: N logits/labels/IDs and Nx4 locations")
        zero = logits[:0].sum()
        foreground = labels > 0.5
        eligible = foreground & (instance_ids > 0)
        stats = {
            "risk_foreground_events": eligible.sum().detach(),
            "risk_unassigned_foreground": (foreground & ~eligible).sum().detach(),
            "risk_empty_samples": 0,
        }
        if n == 0:
            return zero, stats
        sample_ids, sample_inverse = torch.unique(
            locations[:, 0], sorted=True, return_inverse=True,
        )
        sample_count = sample_ids.numel()
        if not bool(eligible.any()):
            stats["risk_empty_samples"] = sample_count
            return zero, stats
        sample_index = sample_inverse[eligible]
        event_losses = F.softplus(-logits[eligible])
        sums = event_losses.new_zeros(sample_count).scatter_add(0, sample_index, event_losses)
        counts_per_sample = torch.bincount(sample_index, minlength=sample_count)
        risk = (sums / counts_per_sample.clamp_min(1)).mean()
        stats["risk_empty_samples"] = (counts_per_sample == 0).sum().detach()
        return risk, stats


class STCForegroundLoss(nn.Module):
    """Compose an existing STC criterion with an independently logged risk."""

    def __init__(self, stc, cfg):
        super().__init__()
        if not math.isfinite(cfg.risk_weight) or cfg.risk_weight < 0:
            raise ValueError("risk weight must be finite and nonnegative")
        self.stc = stc
        self.weight = cfg.risk_weight
        self.risk = ForegroundEventLoss()

    def forward(self, voxel, p2v_map, predictions, labels, auxiliary,
                locations, instance_ids):
        segmentation = self.stc(voxel, p2v_map, predictions, labels)
        if self.weight == 0:
            return segmentation, {"seg": segmentation.detach(), "risk_disabled": 1}
        event_logits = auxiliary["segmentation_logits"].reshape(-1)[p2v_map.reshape(-1).long()]
        risk, stats = self.risk(event_logits, labels, locations, instance_ids)
        weighted = self.weight * risk
        return segmentation + weighted, {
            "seg": segmentation.detach(), "risk": risk.detach(),
            "risk_weighted": weighted.detach(), **stats,
        }
