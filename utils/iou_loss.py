"""Soft Jaccard on hard event labels, averaged equally over samples.

Includes background events and uses event logits, not voxel counts.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def soft_jaccard(logits, labels):
    if logits.numel() == 0:
        return logits.sum()
    probabilities = logits.sigmoid()
    # Empty foreground: use mean background BCE rather than a constant IoU loss.
    if not bool(labels.any()):
        return F.softplus(logits).mean()
    intersection = (probabilities * labels).sum()
    union = (probabilities + labels - probabilities * labels).sum()
    return 1 - intersection / union.clamp_min(1e-7)


class EventIoULoss(nn.Module):
    def forward(self, logits, labels, locations):
        logits, labels = logits.reshape(-1), labels.reshape(-1)
        if locations.shape != (logits.numel(), 4) or labels.shape != logits.shape:
            raise ValueError("IoU loss requires event-aligned logits, labels and Nx4 locations")
        if logits.dtype in (torch.float16, torch.bfloat16):
            logits = logits.float()
        labels = (labels > 0.5).to(logits.dtype)
        if logits.numel() == 0:
            return logits.sum()
        losses = []
        for sample in torch.unique(locations[:, 0], sorted=True):
            mask = locations[:, 0] == sample
            losses.append(soft_jaccard(logits[mask], labels[mask]))
        return torch.stack(losses).mean()


class STCIoULoss(nn.Module):
    def __init__(self, stc, cfg):
        super().__init__()
        if not math.isfinite(cfg.iou_weight) or cfg.iou_weight < 0:
            raise ValueError("iou_weight must be finite and nonnegative")
        self.stc = stc
        self.weight = cfg.iou_weight
        self.surrogate = EventIoULoss()

    def forward(self, voxel, p2v_map, predictions, labels, auxiliary,
                locations, instance_ids):
        segmentation = self.stc(voxel, p2v_map, predictions, labels)
        if self.weight == 0:
            return segmentation, {"seg": segmentation.detach(), "iou_disabled": 1}
        logits = auxiliary["segmentation_logits"].reshape(-1)[p2v_map.reshape(-1).long()]
        overlap = self.surrogate(logits, labels, locations)
        weighted = self.weight * overlap
        return segmentation + weighted, {
            "seg": segmentation.detach(), "iou_surrogate": overlap.detach(),
            "iou_weighted": weighted.detach(),
        }
