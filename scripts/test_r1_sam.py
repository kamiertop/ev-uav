"""Frozen test evaluation for the four runs in the November 10 SAM batch."""
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


BATCH = '20261110_121715_c91ef8'
VARIANTS = [('r1_jaccard', 0.0), ('r1_jaccard', 0.05),
            ('r1_precision', 0.0), ('r1_precision', 0.05)]


def build_plan(root, runs, data, gpu, dry_run=False):
    plan = []
    for experiment, rho in VARIANTS:
        variant = experiment + ('_sam0.05_e0' if rho else '')
        matches = list(runs.glob(f'train_r1p_{variant}_w0.1_s37_{BATCH}_*'))
        if len(matches) != 1:
            raise RuntimeError(f'{variant}: expected one run from batch {BATCH}, found {len(matches)}')
        directory = matches[0]
        cfg = json.loads((directory / 'config.json').read_text())
        expected = dict(experiment=experiment, seed=37, epochs=50, batch_size=1,
                        max_events_num=700000, attention_backend='math',
                        prediction_thresh=0.9, early_stopping_patience=0,
                        test_after_train=False, sam_rho=rho, sam_start_epoch=0,
                        loss='stc_r1_precision', risk_mode='event', risk_weight=0.1,
                        iou_mode='jaccard', iou_weight=0.1,
                        hardneg_weight=0.03 if experiment == 'r1_precision' else 0.0)
        for key, value in expected.items():
            if cfg.get(key) != value:
                raise RuntimeError(f'{directory}: unexpected {key}={cfg.get(key)!r}, expected {value!r}')
        last = json.loads((directory / 'metrics.jsonl').read_text().splitlines()[-1])
        if (last.get('epoch') != 50 or last.get('validation_rounds') != 50
                or last.get('early_stopped') is not False
                or last.get('best_iou_sam_active') is not bool(rho)):
            raise RuntimeError(f'Incomplete training or wrong SAM checkpoint stage: {directory}')
        checkpoint = directory / 'checkpoints/best_iou_seed37.pt'
        if not dry_run and not checkpoint.is_file():
            raise RuntimeError(f'Missing checkpoint: {checkpoint}; run on the training server')
        command = [sys.executable, str(root / 'test.py'), '--split', 'test',
                   '--experiment', experiment, '--gpu', gpu, '--data_dir', str(data),
                   '--train_workers', '0', '--batch_size', '1', '--seed', '37',
                   '--prediction_thresh', '0.9', '--model_path', str(checkpoint),
                   '--no-test_after_train']
        # Restore training/model and metric settings explicitly; no threshold sweep.
        for key in ['input_channel', 'whole_t', 'res', 'max_events_num', 'width',
                    'block_residual', 'block_reps', 'use_coords', 'patch_attention',
                    'attention_backend', 'pd_detT', 'correct_thresh', 'trajectory_bin_ms',
                    'trajectory_correct_thresh', 'risk_mode', 'risk_weight', 'risk_beta',
                    'risk_bin_ms', 'iou_mode', 'iou_weight', 'hardneg_weight',
                    'hardneg_ratio', 'hardneg_min', 'sam_rho', 'sam_start_epoch']:
            value = cfg[key]
            command += ['--' + key] + ([str(v) for v in value] if isinstance(value, list) else [str(value)])
        plan.append(dict(variant=variant, experiment=experiment, seed=37, threshold=0.9,
                         sam_rho=rho, run_dir=str(directory), checkpoint=str(checkpoint),
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
        print('4 tests at threshold 0.9; dry-run does not check weights or access test data.')
        return
    if not list((data / 'test').glob('*.npz')):
        raise RuntimeError(f'Missing test data: {data / "test"}')
    tag = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
    output = root / 'runs/logs' / f'r1_sam_test_{tag}'
    output.mkdir(parents=True, exist_ok=False)
    for entry in plan:
        stem = entry['variant'] + '_s37'
        entry['command'] += ['--output_json', str(output / f'{stem}.json')]
        entry['checkpoint_sha256'] = hashlib.sha256(Path(entry['checkpoint']).read_bytes()).hexdigest()
        directory = Path(entry['run_dir'])
        for name in ['config.json', 'metrics.jsonl']:
            (output / f'{stem}_training_{name}').write_bytes((directory / name).read_bytes())
    (output / 'frozen_plan.json').write_text(json.dumps(plan, indent=2))
    (output / 'test_r1_sam.py').write_bytes(Path(__file__).read_bytes())
    rows = []
    try:
        for entry in plan:
            stem = entry['variant'] + '_s37'
            log = output / f'{stem}.log'
            print(f'TEST {stem} threshold=0.9; log: {log}', flush=True)
            with log.open('w') as handle:
                status = subprocess.run(entry['command'], cwd=root, stdout=handle,
                                        stderr=subprocess.STDOUT)
            if status.returncode:
                print('\n'.join(log.read_text(errors='replace').splitlines()[-45:]), flush=True)
                raise RuntimeError(f'Test failed: {log}; previous results preserved')
            result = json.loads((output / f'{stem}.json').read_text())['results']
            row = dict(variant=entry['variant'], split='test', seed=37, threshold=0.9,
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
        print(f'Result package (completed {len(rows)}/4): {archive}', flush=True)


if __name__ == '__main__':
    main()
