"""R1 controls: event IoU alignment and extra supervision of difficult background.

Retains the original R1/STC computation. Hard negatives are training GT background
events with the highest logits, selected independently inside each sample.
This is a diagnostic loss composition, not a claim of a novel mining algorithm.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from utils.iou_loss import EventIoULoss
from utils.trajectory_risk import STCTrajectoryRiskLoss


class HardBackgroundLoss(nn.Module):
    def __init__(self, ratio=3.0, minimum=256):
        super().__init__()
        if not math.isfinite(ratio) or ratio <= 0:
            raise ValueError("hardneg_ratio must be finite and positive")
        if not isinstance(minimum, int) or minimum < 1:
            raise ValueError("hardneg_min must be a positive integer")
        self.ratio, self.minimum = ratio, minimum

    def forward(self, logits, labels, locations):
        logits, labels = logits.reshape(-1), labels.reshape(-1)
        if locations.shape != (logits.numel(), 4) or labels.shape != logits.shape:
            raise ValueError("hard background inputs must align: N logits/labels and Nx4 locations")
        if logits.dtype in (torch.float16, torch.bfloat16):
            logits = logits.float()
        zero = logits[:0].sum()
        stats = {"hardneg_selected": 0, "hardneg_background": 0, "hardneg_empty_samples": 0}
        if logits.numel() == 0:
            return zero, stats
        losses = []
        for sample in torch.unique(locations[:, 0], sorted=True):
            sample_mask = locations[:, 0] == sample
            foreground = labels[sample_mask] > 0.5
            background = logits[sample_mask][~foreground]
            count = background.numel()
            stats["hardneg_background"] += count
            if count == 0:
                losses.append(zero)
                stats["hardneg_empty_samples"] += 1
                continue
            positives = int(foreground.sum().item())
            k = min(count, max(self.minimum, math.ceil(self.ratio * positives)))
            # softplus is monotone: select on logits, backprop through selected BCE.
            selected = torch.topk(background, k, sorted=False).values
            losses.append(F.softplus(selected).mean())
            stats["hardneg_selected"] += k
        return torch.stack(losses).mean(), stats


class STCR1PrecisionLoss(nn.Module):
    def __init__(self, stc, cfg):
        super().__init__()
        if cfg.risk_mode != "event":
            raise ValueError("R1 precision controls require risk_mode=event")
        for name in ("iou_weight", "hardneg_weight"):
            value = getattr(cfg, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        self.base = STCTrajectoryRiskLoss(stc, cfg)
        self.iou_weight, self.negative_weight = cfg.iou_weight, cfg.hardneg_weight
        self.overlap = EventIoULoss("jaccard")
        self.background = HardBackgroundLoss(cfg.hardneg_ratio, cfg.hardneg_min)

    def forward(self, voxel, p2v_map, predictions, labels, auxiliary,
                locations, instance_ids):
        loss, stats = self.base(voxel, p2v_map, predictions, labels, auxiliary,
                                locations, instance_ids)
        if self.iou_weight == 0 and self.negative_weight == 0:
            return loss, stats
        logits = auxiliary["segmentation_logits"].reshape(-1)[p2v_map.reshape(-1).long()]
        if self.iou_weight:
            overlap = self.overlap(logits, labels, locations)
            weighted = self.iou_weight * overlap
            loss = loss + weighted
            stats.update(iou_surrogate=overlap.detach(), iou_weighted=weighted.detach())
        if self.negative_weight:
            background, counts = self.background(logits, labels, locations)
            weighted = self.negative_weight * background
            loss = loss + weighted
            stats.update(hardneg_loss=background.detach(), hardneg_weighted=weighted.detach(), **counts)
        return loss, stats
