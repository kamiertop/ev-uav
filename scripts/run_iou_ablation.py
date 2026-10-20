"""Controlled A1 / Lovasz / soft-Jaccard runs. Never evaluates the test split."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--data-dir', default='../ev-uav-data')
    parser.add_argument('--seeds', nargs='+', type=int, default=[37, 13, 73])
    parser.add_argument('--weight', type=float, choices=[0.03, 0.1], default=0.1)
    parser.add_argument('--dry-run', action='store_true')
    options = parser.parse_args()
    if len(set(options.seeds)) != len(options.seeds):
        parser.error('duplicate seeds')
    os.chdir(Path(__file__).resolve().parents[1])
    tag = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
    bundle = Path('runs/logs') / f'iou_ablation_{tag}'
    common = ['--data_dir', options.data_dir, '--gpu', options.gpu,
              '--train_workers', '0', '--batch_size', '1', '--attention_backend', 'math']
    planned = []
    for seed in options.seeds:
        for experiment in ['a1', 'a1_lovasz', 'a1_jaccard']:
            name = f'iou_{experiment}_w{options.weight}_s{seed}_{tag}'
            command = [sys.executable, 'train.py', '--experiment', experiment, *common,
                       '--run_name', name, '--seed', str(seed), '--epochs', '50',
                       '--max_events_num', '700000', '--optim', 'Adam', '--lr', '0.01',
                       '--final_lr', '0.001', '--scheduler', 'linear',
                       '--prediction_thresh', '0.9', '--iou_weight', str(options.weight),
                       '--val_start_epoch', '0', '--val_interval', '1',
                       '--early_stopping_patience', '0', '--no-test_after_train']
            planned.append((seed, experiment, name, command))
    if options.dry_run:
        for seed, experiment, name, command in planned:
            print(shlex.join(command))
            print(f'  -> val only, best_iou_seed{seed}.pt, thresholds: .7 .8 .85 .9 .95')
        return

    bundle.mkdir(parents=True, exist_ok=False)
    manifest = bundle / 'manifest.jsonl'

    def record(**values):
        with manifest.open('a') as handle:
            handle.write(json.dumps(values) + '\n')

    def run(command, log):
        record(command=command, log=str(log), status='started')
        print(f'Running {shlex.join(command)}\nLog: {log}', flush=True)
        with log.open('w') as handle:
            process = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
        record(command=command, log=str(log), status='ok' if process.returncode == 0 else 'failed',
               returncode=process.returncode)
        if process.returncode:
            raise RuntimeError(f'Command failed; see {log}. Previous results are preserved.')

    import torch
    import HAIS_OP  # noqa: F401 -- check the server extension before training
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    device = torch.device(f'cuda:{options.gpu}')
    free, total = torch.cuda.mem_get_info(device)
    environment = dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(device),
                       free_bytes=free, total_bytes=total, data_dir=str(Path(options.data_dir).resolve()))
    (bundle / 'environment.json').write_text(json.dumps(environment, indent=2))
    print(environment, flush=True)
    # Prior full runs used ~23.9 GiB. Do not start alongside a large competing task.
    if free < 22 * 1024**3:
        raise RuntimeError('Less than 22 GiB free GPU memory; use an idle 24 GiB or larger GPU')
    for split in ['train', 'val']:
        if not list((Path(options.data_dir) / split).glob('*.npz')):
            raise RuntimeError(f'Missing data: {options.data_dir}/{split}')
    run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests'], bundle / 'cpu_tests.log')
    completed = []
    # One epoch checks only execution/memory; formal runs still start from scratch
    # with the original 50-epoch learning-rate schedule.
    for seed, experiment, name, command in planned[:3]:
        if experiment == 'a1':
            continue
        smoke = command.copy()
        smoke[smoke.index('--epochs') + 1] = '1'
        smoke[smoke.index('--run_name') + 1] = 'smoke_' + name
        run(smoke, bundle / f'{experiment}_smoke.log')
    for seed, experiment, name, command in planned:
        log = bundle / f'{experiment}_s{seed}_train.log'
        run(command, log)
        prefix = '[train] Run directory: '
        directories = [Path(line[len(prefix):].strip()) for line in log.read_text().splitlines()
                       if line.startswith(prefix)]
        if len(directories) != 1:
            raise RuntimeError(f'Expected exactly one run directory in {log}')
        directory = directories[0]
        config = json.loads((directory / 'config.json').read_text())
        rows = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines()]
        if config['run_name'] != name or rows[-1].get('validation_rounds') != 50 or rows[-1].get('early_stopped'):
            raise RuntimeError(f'Incomplete or mismatched training run: {directory}')
        checkpoint = directory / 'checkpoints' / f'best_iou_seed{seed}.pt'
        if not checkpoint.is_file():
            raise RuntimeError(f'Missing checkpoint: {checkpoint}')
        completed.append(directory)
        record(run_dir=str(directory), seed=seed, experiment=experiment, checkpoint=str(checkpoint))
        for threshold in ['0.7', '0.8', '0.85', '0.9', '0.95']:
            stem = f'{experiment}_s{seed}_t{threshold}'
            run([sys.executable, 'test.py', '--split', 'val', '--experiment', experiment,
                 *common, '--seed', str(seed), '--prediction_thresh', threshold,
                 '--iou_weight', str(options.weight), '--model_path', str(checkpoint),
                 '--output_json', str(bundle / f'{stem}.json')], bundle / f'{stem}.log')

    # Lightweight transfer: retain checkpoints on the server, include all numeric evidence.
    archive = bundle.with_suffix('.tar.gz')
    with tarfile.open(archive, 'w:gz') as handle:
        handle.add(bundle)
        for directory in completed:
            for filename in ['config.json', 'metrics.jsonl']:
                handle.add(directory / filename)
    print(f'ALL DONE. Send back {archive}; retain server checkpoints.', flush=True)


if __name__ == '__main__':
    main()
