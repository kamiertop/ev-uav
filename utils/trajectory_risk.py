"""Foreground risk controls on events or instance/time bins (no sparse ops).

This is an auxiliary objective, always combined with STC background supervision.
Bin origins are each sampled track's first foreground timestamp, as in the
existing trajectory metrics. Empty bins are unobserved, never negative labels.
The objective is order invariant and does not claim temporal connectivity.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class TrajectoryRisk(nn.Module):
    def __init__(self, mode="term", beta=1.0, bin_ms=50.0):
        super().__init__()
        if mode not in {"event", "balanced", "term"}:
            raise ValueError(f"Unknown risk mode: {mode}")
        if not math.isfinite(beta) or beta < 0:
            raise ValueError("risk beta must be finite and nonnegative")
        if not math.isfinite(bin_ms) or bin_ms <= 0:
            raise ValueError("risk bin duration must be finite and positive")
        self.mode, self.beta, self.bin_ms = mode, beta, bin_ms

    def forward(self, logits, labels, locations, instance_ids):
        """Accept event-aligned tensors; average samples, then tracks, then bins.

        The event control averages all foreground events within each sample.
        Samples without valid foreground IDs contribute zero, including to the
        sample average. All modes use the same eligible foreground events.
        """
        logits = logits.reshape(-1)
        # Keep double precision for gradcheck; use FP32 for half precision inputs.
        if logits.dtype in (torch.float16, torch.bfloat16):
            logits = logits.float()
        labels, instance_ids = labels.reshape(-1), instance_ids.reshape(-1)
        n = logits.numel()
        if locations.shape != (n, 4) or labels.numel() != n or instance_ids.numel() != n:
            raise ValueError("risk inputs must align: N logits/labels/IDs and Nx4 locations")
        # An empty slice produces a differentiable zero without summing extreme logits.
        zero = logits[:0].sum()
        foreground = labels > 0.5
        eligible = foreground & (instance_ids > 0)
        stats = {
            "risk_foreground_events": eligible.sum().detach(),
            "risk_unassigned_foreground": (foreground & ~eligible).sum().detach(),
            "risk_tracks": 0, "risk_bins": 0, "risk_empty_samples": 0,
            "risk_max_bin_weight": zero.detach(),
            "risk_effective_bins": zero.detach(),
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
        tracks, track_index = torch.unique(
            torch.stack((sample_index, instance_ids[eligible].long()), dim=1),
            dim=0, sorted=True, return_inverse=True,
        )
        track_count = tracks.shape[0]
        times = locations[eligible, 3].to(logits.dtype)
        origins = times.new_full((track_count,), float("inf"))
        origins.scatter_reduce_(0, track_index, times, reduce="amin", include_self=True)
        bin_ids = torch.floor((times - origins[track_index]) / self.bin_ms).long()
        bins, bin_index, counts = torch.unique(
            torch.stack((track_index, bin_ids), dim=1), dim=0,
            sorted=True, return_inverse=True, return_counts=True,
        )
        bin_losses = event_losses.new_zeros(bins.shape[0]).scatter_add(
            0, bin_index, event_losses,
        ) / counts
        bin_tracks = bins[:, 0]
        bins_per_track = torch.bincount(bin_tracks, minlength=track_count)
        tracks_per_sample = torch.bincount(tracks[:, 0], minlength=sample_count)

        if self.mode == "term" and self.beta > 0:
            maxima = bin_losses.new_full((track_count,), -float("inf"))
            maxima.scatter_reduce_(0, bin_tracks, bin_losses.detach(), reduce="amax")
            shifted = self.beta * (bin_losses - maxima[bin_tracks])
            exponentials = shifted.exp()
            sums = exponentials.new_zeros(track_count).scatter_add(0, bin_tracks, exponentials)
            # expm1/log1p retain the beta -> 0 limit without subtracting log(n).
            mean_expm1 = shifted.new_zeros(track_count).scatter_add(
                0, bin_tracks, shifted.expm1(),
            ) / bins_per_track
            log_mean = torch.where(
                mean_expm1 > -0.5,
                mean_expm1.clamp_min(-0.5).log1p(),
                (sums / bins_per_track).log(),
            )
            track_risk = maxima + log_mean / self.beta
            weights = exponentials / sums[bin_tracks]
        else:
            track_risk = bin_losses.new_zeros(track_count).scatter_add(
                0, bin_tracks, bin_losses,
            ) / bins_per_track
            weights = 1.0 / bins_per_track[bin_tracks].to(logits.dtype)

        if self.mode == "event":
            sums = event_losses.new_zeros(sample_count).scatter_add(0, sample_index, event_losses)
            counts_per_sample = torch.bincount(sample_index, minlength=sample_count)
            risk = (sums / counts_per_sample.clamp_min(1)).mean()
        else:
            sums = track_risk.new_zeros(sample_count).scatter_add(0, tracks[:, 0], track_risk)
            risk = (sums / tracks_per_sample.clamp_min(1)).mean()

        weight_squares = weights.new_zeros(track_count).scatter_add(0, bin_tracks, weights.square())
        stats.update(
            risk_tracks=track_count, risk_bins=bins.shape[0],
            risk_empty_samples=(tracks_per_sample == 0).sum().detach(),
            risk_max_bin_weight=weights.max().detach(),
            risk_effective_bins=weight_squares.reciprocal().mean().detach(),
        )
        return risk, stats


class STCTrajectoryRiskLoss(nn.Module):
    """Compose an existing STC criterion with an independently logged risk."""

    def __init__(self, stc, cfg):
        super().__init__()
        if not math.isfinite(cfg.risk_weight) or cfg.risk_weight < 0:
            raise ValueError("risk weight must be finite and nonnegative")
        self.stc = stc
        self.weight = cfg.risk_weight
        self.risk = TrajectoryRisk(cfg.risk_mode, cfg.risk_beta, cfg.risk_bin_ms)

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
