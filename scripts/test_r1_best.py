"""Frozen test evaluation of the two preserved October 21 R1 candidates."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import uuid


BATCH = '20261021_104437_461ed8'
CANDIDATES = {
    'r1_jaccard': ('20261021_104716', 0.7321335077285767,
                   '8a2498345d65d0271cd6a619d870e82f11d57a6ca8f8d90e5824550bb1baa6d3'),
    'r1_precision': ('20261021_115309', 0.7304908037185669,
                     '1a4ec27ac0135ae6f4e562724bd836e74bee6e69a6597e4bf52c6b2b6d3aea67'),
}


def build_plan(root, runs, data, gpu, dry_run=False):
    plan = []
    for experiment, (suffix, best_iou, expected_sha256) in CANDIDATES.items():
        variant = experiment
        matches = [runs / f'train_r1p_{experiment}_w0.1_s37_{BATCH}_{suffix}']
        matches = [p for p in matches if p.is_dir()]
        if len(matches) != 1:
            raise RuntimeError(f'{variant}: expected one run from batch {BATCH}, found {len(matches)}')
        directory = matches[0]
        cfg = json.loads((directory / 'config.json').read_text())
        expected = dict(experiment=experiment, seed=37, epochs=50, batch_size=1,
                        max_events_num=700000, attention_backend='math',
                        prediction_thresh=0.9, early_stopping_patience=0,
                        test_after_train=False,
                        loss='stc_r1_precision', risk_weight=0.1, iou_weight=0.1,
                        hardneg_weight=0.03 if experiment == 'r1_precision' else 0.0)
        for key, value in expected.items():
            if cfg.get(key) != value:
                raise RuntimeError(f'{directory}: unexpected {key}={cfg.get(key)!r}, expected {value!r}')
        last = json.loads((directory / 'metrics.jsonl').read_text().splitlines()[-1])
        if (last.get('epoch') != 50 or last.get('validation_rounds') != 50
                or last.get('early_stopped') is not False
                or abs(last.get('best_iou', -1) - best_iou) > 1e-7):
            raise RuntimeError(f'Incomplete training or unexpected best validation IoU: {directory}')
        checkpoint = directory / 'checkpoints/best_iou_seed37.pt'
        if not dry_run and not checkpoint.is_file():
            raise RuntimeError(f'Missing checkpoint: {checkpoint}; run on the training server')
        if cfg.get('sam_rho', 0) != 0:
            raise RuntimeError(f'Expected original non-SAM candidate: {directory}')
        if not dry_run and hashlib.sha256(checkpoint.read_bytes()).hexdigest() != expected_sha256:
            raise RuntimeError(f'Checkpoint differs from preserved October 21 weights: {checkpoint}')
        command = [sys.executable, str(root / 'test.py'), '--split', 'test',
                   '--experiment', experiment, '--gpu', gpu, '--data_dir', str(data),
                   '--train_workers', '0', '--batch_size', '1', '--seed', '37',
                   '--prediction_thresh', '0.85', '--model_path', str(checkpoint),
                   '--no-test_after_train']
        # Restore training/model and metric settings explicitly; no threshold sweep.
        for key in ['input_channel', 'whole_t', 'res', 'max_events_num', 'width',
                    'block_residual', 'block_reps', 'use_coords', 'patch_attention',
                    'attention_backend', 'pd_detT', 'correct_thresh', 'trajectory_bin_ms',
                    'trajectory_correct_thresh', 'risk_weight', 'iou_weight', 'hardneg_weight',
                    'hardneg_ratio', 'hardneg_min']:
            value = cfg[key]
            command += ['--' + key] + ([str(v) for v in value] if isinstance(value, list) else [str(value)])
        plan.append(dict(variant=variant, experiment=experiment, seed=37, threshold=0.85,
                         run_dir=str(directory), checkpoint=str(checkpoint),
                         expected_checkpoint_sha256=expected_sha256,
                         command=command))
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--data-dir', default='../ev-uav-data')
    parser.add_argument('--runs-root', default='runs')
    parser.add_argument('--dry-run', action='store_true')
    options = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    runs = Path(options.runs_root).resolve()
    data = Path(options.data_dir).resolve()
    plan = build_plan(root, runs, data, options.gpu, options.dry_run)
    if options.dry_run:
        for entry in plan:
            print(shlex.join(entry['command']))
        print('2 tests at threshold 0.85; dry-run does not check weights or access test data.')
        return
    if not list((data / 'test').glob('*.npz')):
        raise RuntimeError(f'Missing test data: {data / "test"}')
    tag = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
    output = root / 'runs/logs' / f'r1_best_test_{tag}'
    output.mkdir(parents=True, exist_ok=False)
    for entry in plan:
        stem = entry['variant'] + '_s37'
        entry['command'] += ['--output_json', str(output / f'{stem}.json')]
        entry['checkpoint_sha256'] = hashlib.sha256(Path(entry['checkpoint']).read_bytes()).hexdigest()
        directory = Path(entry['run_dir'])
        for name in ['config.json', 'metrics.jsonl']:
            (output / f'{stem}_training_{name}').write_bytes((directory / name).read_bytes())
    (output / 'frozen_plan.json').write_text(json.dumps(plan, indent=2))
    (output / 'test_r1_best.py').write_bytes(Path(__file__).read_bytes())
    rows = []
    try:
        for entry in plan:
            stem = entry['variant'] + '_s37'
            log = output / f'{stem}.log'
            print(f'TEST {stem} threshold=0.85; log: {log}', flush=True)
            with log.open('w') as handle:
                status = subprocess.run(entry['command'], cwd=root, stdout=handle,
                                        stderr=subprocess.STDOUT)
            if status.returncode:
                print('\n'.join(log.read_text(errors='replace').splitlines()[-45:]), flush=True)
                raise RuntimeError(f'Test failed: {log}; previous results preserved')
            result = json.loads((output / f'{stem}.json').read_text())['results']
            row = dict(variant=entry['variant'], split='test', seed=37, threshold=0.85,
                       **{k: result[k] for k in ['iou', 'seg_acc', 'pd', 'fa', 'precision', 'tp', 'fp', 'fn']})
            rows.append(row)
            with (output / 'summary.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(rows)
            print(row, flush=True)
    finally:
        archive = output.with_suffix('.tar.gz')
        with tarfile.open(archive, 'w:gz') as handle:
            handle.add(output, arcname=output.relative_to(root))
        print(f'Result package (completed {len(rows)}/2): {archive}', flush=True)


if __name__ == '__main__':
    main()
