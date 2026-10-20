"""Evaluate frozen A1/Jaccard candidates from one exact IoU ablation batch."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', required=True, help='e.g. 20261020_163642_2d5b7c')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--data-dir', default='../ev-uav-data')
    parser.add_argument('--runs-root', default='runs')
    parser.add_argument('--dry-run', action='store_true')
    options = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if not options.batch or any(c not in '0123456789abcdef_' for c in options.batch):
        parser.error('invalid batch identifier')
    runs = Path(options.runs_root).resolve()
    plan = []
    # Freeze before accessing test results: each method's uniform val mean optimum.
    for experiment, threshold in [('a1', '0.85'), ('a1_jaccard', '0.9')]:
        for seed in [13, 37, 73]:
            matches = list(runs.glob(f'train_iou_{experiment}_w0.1_s{seed}_{options.batch}_*'))
            if len(matches) != 1:
                raise RuntimeError(f'{experiment} seed {seed}: expected one exact run, found {len(matches)}')
            directory = matches[0]
            cfg = json.loads((directory / 'config.json').read_text())
            expected = dict(experiment=experiment, seed=seed, epochs=50,
                            attention_backend='math', iou_weight=0.1, batch_size=1,
                            test_after_train=False, early_stopping_patience=0,
                            loss='stc' if experiment == 'a1' else 'stc_iou')
            if any(cfg.get(key) != value for key, value in expected.items()):
                raise RuntimeError(f'Unexpected training configuration: {directory}')
            if experiment == 'a1_jaccard' and cfg.get('iou_mode') != 'jaccard':
                raise RuntimeError(f'Wrong surrogate: {directory}')
            last = json.loads((directory / 'metrics.jsonl').read_text().splitlines()[-1])
            if last.get('validation_rounds') != 50 or last.get('early_stopped') is not False:
                raise RuntimeError(f'Incomplete training: {directory}')
            checkpoint = directory / 'checkpoints' / f'best_iou_seed{seed}.pt'
            if not options.dry_run and not checkpoint.is_file():
                raise RuntimeError(f'Missing checkpoint: {checkpoint}; run on the training server')
            command = [sys.executable, str(root / 'test.py'), '--split', 'test',
                       '--experiment', experiment, '--gpu', options.gpu,
                       '--data_dir', str(Path(options.data_dir).resolve()),
                       '--train_workers', '0', '--batch_size', '1', '--seed', str(seed),
                       '--prediction_thresh', threshold, '--model_path', str(checkpoint)]
            # Evaluation does not automatically restore checkpoint configuration.
            for key in ['input_channel', 'whole_t', 'res', 'width', 'block_residual',
                        'block_reps', 'use_coords', 'patch_attention', 'attention_backend',
                        'pd_detT', 'correct_thresh', 'trajectory_bin_ms',
                        'trajectory_correct_thresh', 'iou_mode', 'iou_weight']:
                value = cfg[key]
                command += ['--' + key] + ([str(v) for v in value] if isinstance(value, list) else [str(value)])
            plan.append((experiment, seed, threshold, directory, command))
    if options.dry_run:
        for _, _, _, _, command in plan:
            print(shlex.join(command))
        print('6 evaluations planned; dry-run does not check checkpoint availability or access test data.')
        return
    data = Path(options.data_dir).resolve()
    if not list((data / 'test').glob('*.npz')):
        raise RuntimeError(f'Missing test data: {data / "test"}')
    tag = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
    output = root / 'runs/logs' / f'iou_test_{tag}'
    output.mkdir(parents=True, exist_ok=False)
    manifest = []
    # Save the complete frozen plan before the first test evaluation.
    for experiment, seed, threshold, directory, command in plan:
        stem = f'{experiment}_s{seed}'
        manifest.append(dict(experiment=experiment, seed=seed, threshold=threshold,
                             run_dir=str(directory), command=command + ['--output_json', str(output / f'{stem}.json')]))
        (output / f'{stem}_training_config.json').write_text((directory / 'config.json').read_text())
    (output / 'frozen_plan.json').write_text(json.dumps(manifest, indent=2))
    for entry in manifest:
        stem = f'{entry["experiment"]}_s{entry["seed"]}'
        print(f'TEST {stem} threshold={entry["threshold"]}; log: {output / (stem + ".log")}', flush=True)
        with (output / f'{stem}.log').open('w') as handle:
            subprocess.run(entry['command'], cwd=root, stdout=handle, stderr=subprocess.STDOUT, check=True)
        result = json.loads((output / f'{stem}.json').read_text())['results']
        print({key: result[key] for key in ['iou', 'seg_acc', 'precision', 'pd', 'fa']}, flush=True)
    archive = output.with_suffix('.tar.gz')
    with tarfile.open(archive, 'w:gz') as handle:
        handle.add(output, arcname=output.relative_to(root))
    print(f'ALL DONE. Send back: {archive}', flush=True)


if __name__ == '__main__':
    main()
