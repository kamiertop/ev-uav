"""EV-UAV evaluation script (standalone).

Usage::

    uv run python test.py --model_path runs/train_xxx/checkpoints/best_iou_seed37.pt
"""

import torch
import json
from pathlib import Path

from utils import args
from utils.eval import run_test


def main():
    args.parse(extra=[("--split", dict(default="test", choices=["val", "test"],
                                     help="use val for model selection; test only after freezing configuration")),
                      ("--output_json", dict(default="", help="save metrics, per-sequence counts and evaluation flags"))])
    device = f"cuda:{args.cfg.gpu}"
    torch.cuda.set_device(device)
    results = run_test(args.cfg.model_path, args.cfg, device=device, split=args.cfg.split,
                       include_per_sequence=bool(args.cfg.output_json))
    if args.cfg.output_json:
        output = Path(args.cfg.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        serializable = {key: value.item() if hasattr(value, "item") else value
                        for key, value in results.items()}
        output.write_text(json.dumps({"config": vars(args.cfg), "results": serializable},
                                     indent=2, ensure_ascii=False, allow_nan=False))

    if "pd" in results:
        print(f"iou={results['iou']:.4f}  seg_acc={results['seg_acc']:.4f}  "
              f"pd={results['pd']:.4f}  fa={results['fa']:.6g}  "
              f"coverage={results['trajectory_coverage']:.4f}  "
              f"longest={results['trajectory_longest_ratio']:.4f}  "
              f"fragmentation={results['trajectory_fragmentation']:.4f}")
    else:
        print(f"iou={results['iou']:.4f}  seg_acc={results['seg_acc']:.4f}")
    print(f"precision={results['precision']:.4f}  tp={results['tp']}  "
          f"fp={results['fp']}  fn={results['fn']}  fp_per_gt={results['fp_per_gt']}")


if __name__ == "__main__":
    main()
