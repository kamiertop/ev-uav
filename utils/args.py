"""Shared argument parser — replaces the old configs/ YAML-based system.

Usage::

    from utils import args
    args.parse()
    # access args.cfg.gpu, args.cfg.data_dir, etc.
"""

import argparse
import math
from typing import Any

def _boolean(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError("expected true or false")


EXPERIMENTS = {
    "a0": dict(patch_attention="legacy", loss="stc"),
    "attention_fix": dict(patch_attention="corrected", loss="stc"),
    "a1": dict(patch_attention="sequence", loss="stc"),
    "a1_jaccard": dict(patch_attention="sequence", loss="stc_iou"),
    "r1_event": dict(patch_attention="sequence", loss="stc_risk"),
    "r1_jaccard": dict(patch_attention="sequence", loss="stc_r1_precision",
                       iou_weight=0.1, hardneg_weight=0.0),
    "r1_hardneg": dict(patch_attention="sequence", loss="stc_r1_precision",
                      iou_weight=0.0, hardneg_weight=0.03),
    "r1_precision": dict(patch_attention="sequence", loss="stc_r1_precision",
                        iou_weight=0.1, hardneg_weight=0.03),
}

# Module-level singleton — populated by parse()
cfg: argparse.Namespace | None = None


def parse(extra: list[tuple[str, dict[str, Any]]] | None = None) -> argparse.Namespace:
    """Parse CLI arguments; store result in ``utils.args.cfg`` and return it.

    ``cfg`` is a module-level attribute, so other modules can access it with
    ``from utils import args; args.cfg`` — this is safe because module
    attribute access resolves at runtime, not at import time.
    """
    global cfg

    parser = argparse.ArgumentParser(
        description="EV-UAV Event Point Cloud Segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--experiment", choices=EXPERIMENTS,
                        help="model/loss preset; explicit flags override preset values")

    # ── GPU ─────────────────────────────────────────────────────────
    parser.add_argument("--gpu", default="0", type=str, help="CUDA device index")

    # ── paths ───────────────────────────────────────────────────────
    parser.add_argument("--data_dir", default="/data/ev-uav",
                        help="path to dataset root directory")
    parser.add_argument("--model_path", default="",
                        help="path to pretrained checkpoint (for test)")
    parser.add_argument("--run_name", default="experiment",
                        help="short tag embedded in the run directory")

    # ── data ────────────────────────────────────────────────────────
    parser.add_argument("--input_channel", default=4, type=int)
    parser.add_argument("--whole_t", default=8000, type=int,
                        help="time window in ms")
    parser.add_argument("--res", nargs=2, default=[346, 260], type=int,
                        help="sensor resolution (width height)")
    parser.add_argument("--max_events_num", default=700000, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--train_workers", default=8, type=int)

    # ── model ───────────────────────────────────────────────────────
    parser.add_argument("--width", default=12, type=int,
                        help="network width multiplier")
    parser.add_argument("--block_residual", default=True, type=_boolean, nargs="?", const=True)
    parser.add_argument("--no-block_residual", dest="block_residual", action="store_false")
    parser.add_argument("--block_reps", default=2, type=int)
    parser.add_argument("--use_coords", default=True, type=_boolean, nargs="?", const=True)
    parser.add_argument("--no-use_coords", dest="use_coords", action="store_false")
    parser.add_argument("--attention_backend", default="auto", choices=["auto", "math"],
                        help="SDPA backend; math disables fused kernels for numerical diagnosis")
    parser.add_argument("--patch_attention", default="sequence",
                        choices=["sequence", "legacy", "corrected"],
                        help="sequence: enhanced; corrected: sequence fix only; legacy: released behavior")
    # ── training ────────────────────────────────────────────────────
    parser.add_argument("--diagnose_numerics", action="store_true",
                        help="trace backward NaNs and check inputs, outputs, parameters and buffers; slow")
    parser.add_argument("--grad_clip_norm", default=0.0, type=float,
                        help="gradient L2 norm cap; 0 disables clipping, finite checks always enabled")
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--lr", default=0.01, type=float,
                        help="initial learning rate")
    parser.add_argument("--final_lr", default=0.001, type=float,
                        help="final learning rate for linear decay")
    parser.add_argument("--optim", default="Adam", choices=["Adam", "SGD"])
    parser.add_argument("--scheduler", default="linear",
                        choices=["linear", "step", "none"])
    parser.add_argument(
        "--loss", default="stc", choices=["stc", "stc_risk", "stc_iou", "stc_r1_precision"],
        help="STC baseline, foreground risk, or event IoU surrogate",
    )
    parser.add_argument("--k", default=3, type=int,
                        help="spatial correlation kernel size")
    parser.add_argument("--t", default=5, type=int,
                        help="temporal correlation kernel size in voxels")
    parser.add_argument("--seed", default=37, type=int, help="random seed")
    parser.add_argument("--val_start_epoch", default=0, type=int)
    parser.add_argument("--val_interval", default=1, type=int)
    parser.add_argument(
        "--early_stopping_patience", default=10, type=int,
        help="number of validation rounds without IoU improvement before stopping; 0 disables",
    )
    parser.add_argument(
        "--early_stopping_min_delta", default=1e-4, type=float,
        help="minimum validation IoU improvement counted as progress",
    )
    parser.add_argument(
        "--test_after_train", action=argparse.BooleanOptionalAction,
        default=False,
        help="evaluate the held-out test split after model selection",
    )

    # Jaccard and hard-background supervision; trajectory bins are metrics only.
    parser.add_argument("--iou_weight", default=0.1, type=float)
    parser.add_argument("--hardneg_weight", default=0.03, type=float)
    parser.add_argument("--hardneg_ratio", default=3.0, type=float,
                        help="hard background count per foreground event; training only")
    parser.add_argument("--hardneg_min", default=256, type=int,
                        help="minimum selected background events, capped by availability")
    parser.add_argument("--trajectory_bin_ms", default=50.0, type=float)

    # Foreground event BCE; no instance/time-bin risk variants.
    parser.add_argument("--risk_weight", default=0.1, type=float)

    # ── evaluation ──────────────────────────────────────────────────
    parser.add_argument("--eval", default=True, type=_boolean, nargs="?", const=True)
    parser.add_argument("--no-eval", dest="eval", action="store_false")
    parser.add_argument("--roc", default=True, type=_boolean, nargs="?", const=True)
    parser.add_argument("--no-roc", dest="roc", action="store_false")
    parser.add_argument("--pd_detT", default=50, type=int,
                        help="detection time window for ROC")
    parser.add_argument("--correct_thresh", default=0.0001, type=float)
    parser.add_argument(
        "--trajectory_correct_thresh", default=0.1, type=float,
        help="fraction of GT events recovered for a trajectory bin to count",
    )
    parser.add_argument("--prediction_thresh", default=0.9, type=float)

    if extra:
        for flag, kwargs in extra:
            parser.add_argument(flag, **kwargs)

    selected, _ = parser.parse_known_args()
    if selected.experiment:
        parser.set_defaults(**EXPERIMENTS[selected.experiment])
    cfg = parser.parse_args()
    if cfg.loss in {"stc_iou", "stc_r1_precision"} and (not math.isfinite(cfg.iou_weight) or cfg.iou_weight < 0):
        parser.error("--iou_weight must be finite and nonnegative")
    if cfg.loss == "stc_r1_precision":
        if not math.isfinite(cfg.hardneg_weight) or cfg.hardneg_weight < 0:
            parser.error("--hardneg_weight must be finite and nonnegative")
        if not math.isfinite(cfg.hardneg_ratio) or cfg.hardneg_ratio <= 0 or cfg.hardneg_min < 1:
            parser.error("--hardneg_ratio must be finite and positive; --hardneg_min must be positive")
    if cfg.loss in {"stc_risk", "stc_r1_precision"}:
        if not math.isfinite(cfg.risk_weight) or cfg.risk_weight < 0:
            parser.error("--risk_weight must be finite and nonnegative")
    globals()["cfg"] = cfg
    return cfg
