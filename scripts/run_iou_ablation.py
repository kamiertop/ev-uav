"""Controlled A1 / Lovasz / soft-Jaccard runs. Never evaluates the test split."""
import argparse
import hashlib
import math
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import uuid

# Executing ``python scripts/...`` adds scripts/, not the repository root, to
# sys.path. build_ext.py places HAIS_OP.so in the root, as train.py expects.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--suite', choices=['iou', 'r1_precision', 'r1_sam'], default='iou')
    parser.add_argument('--experiments', nargs='+', default=None,
                        help='run only these members of the selected suite; fresh training, not checkpoint resume')
    parser.add_argument('--data-dir', default='../ev-uav-data')
    parser.add_argument('--seeds', nargs='+', type=int, default=None)
    parser.add_argument('--weight', type=float, choices=[0.03, 0.1], default=0.1)
    parser.add_argument('--sam-rhos', nargs='+', type=float, default=None,
                        help='SAM radii; 0 is the unchanged optimizer control')
    parser.add_argument('--sam-start-epoch', type=int, default=0)
    parser.add_argument('--dry-run', action='store_true')
    options = parser.parse_args()
    r1_suite = options.suite != 'iou'
    if options.seeds is None:
        options.seeds = [37] if r1_suite else [37, 13, 73]
    if options.sam_rhos is None:
        options.sam_rhos = [0.0, 0.05] if options.suite == 'r1_sam' else [0.0]
    if (any(not math.isfinite(rho) or rho < 0 for rho in options.sam_rhos)
            or len(set(options.sam_rhos)) != len(options.sam_rhos)):
        parser.error('SAM radii must be unique, finite, nonnegative')
    if not 0 <= options.sam_start_epoch < 50:
        parser.error('--sam-start-epoch must be in [0, 49]')
    if len(set(options.seeds)) != len(options.seeds):
        parser.error('duplicate seeds')
    os.chdir(PROJECT_ROOT)
    tag = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
    suite_name = options.suite if r1_suite else 'iou_ablation'
    bundle = Path('runs/logs') / f'{suite_name}_{tag}'
    common = ['--data_dir', options.data_dir, '--gpu', options.gpu,
              '--train_workers', '0', '--batch_size', '1', '--attention_backend', 'math']
    planned = []
    experiments = (['r1_event', 'r1_jaccard', 'r1_hardneg', 'r1_precision']
                   if r1_suite else ['a1', 'a1_lovasz', 'a1_jaccard'])
    if options.suite == 'r1_sam':
        experiments = ['r1_jaccard', 'r1_precision']
    if options.experiments is not None:
        if len(set(options.experiments)) != len(options.experiments):
            parser.error('duplicate experiments')
        invalid = set(options.experiments) - set(experiments)
        if invalid:
            parser.error(f'experiments outside the selected suite: {sorted(invalid)}')
        experiments = options.experiments
    thresholds = (['0.8', '0.85', '0.9', '0.95', '0.97', '0.98']
                  if r1_suite else ['0.7', '0.8', '0.85', '0.9', '0.95'])
    for seed in options.seeds:
        for experiment in experiments:
            for rho in options.sam_rhos:
                run_prefix = 'r1p' if r1_suite else 'iou'
                variant = experiment if rho == 0 else f'{experiment}_sam{rho:g}_e{options.sam_start_epoch}'
                name = f'{run_prefix}_{variant}_w{options.weight}_s{seed}_{tag}'
                weight = options.weight
                negative = 0.0
                if r1_suite:
                    weight = options.weight if experiment in ['r1_jaccard', 'r1_precision'] else 0.0
                    negative = 0.03 if experiment in ['r1_hardneg', 'r1_precision'] else 0.0
                command = [sys.executable, 'train.py', '--experiment', experiment, *common,
                           '--run_name', name, '--seed', str(seed), '--epochs', '50',
                           '--max_events_num', '700000', '--optim', 'Adam', '--lr', '0.01',
                           '--final_lr', '0.001', '--scheduler', 'linear',
                           '--prediction_thresh', '0.9', '--iou_weight', str(weight),
                           '--val_start_epoch', '0', '--val_interval', '1',
                           '--early_stopping_patience', '0', '--no-test_after_train',
                           '--sam_rho', str(rho), '--sam_start_epoch', str(options.sam_start_epoch if rho else 0)]
                if r1_suite:
                    command += ['--risk_mode', 'event', '--risk_weight', '0.1',
                                '--iou_mode', 'jaccard', '--hardneg_weight', str(negative),
                                '--hardneg_ratio', '3', '--hardneg_min', '256']
                planned.append((seed, variant, name, command))
    if options.dry_run:
        for seed, experiment, name, command in planned:
            print(shlex.join(command))
            print(f'  -> val only, best_iou_seed{seed}.pt, thresholds: {" ".join(thresholds)}')
        return

    bundle.mkdir(parents=True, exist_ok=False)
    (bundle / 'frozen_plan.json').write_text(json.dumps([
        dict(seed=seed, variant=variant, run_name=name, command=command, split='val', thresholds=thresholds)
        for seed, variant, name, command in planned
    ], indent=2))
    # Archive the code actually used, so future baseline drift can be audited.
    sources = [Path('train.py'), Path('test.py'), Path(__file__).relative_to(PROJECT_ROOT)]
    for folder in ['utils', 'model', 'dataset', 'tests']:
        sources.extend(sorted(Path(folder).rglob('*.py')))
    hashes = {}
    for source in sources:
        content = source.read_bytes()
        target = bundle / 'source_snapshot' / source
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        hashes[str(source)] = hashlib.sha256(content).hexdigest()
    (bundle / 'source_sha256.json').write_text(json.dumps(hashes, indent=2))
    manifest = bundle / 'manifest.jsonl'

    def record(**values):
        with manifest.open('a') as handle:
            handle.write(json.dumps(values) + '\n')

    def run(command, log):
        # A GPU can become occupied long after the batch-level preflight.
        # Recheck before every model subprocess; contention during a run still
        # requires exclusive GPU scheduling on the server.
        if 'train.py' in command or 'test.py' in command:
            free_now, _ = torch.cuda.mem_get_info(device)
            if free_now < 22 * 1024**3:
                record(command=command, log=str(log), status='blocked_gpu', free_bytes=free_now)
                raise RuntimeError(
                    f'Only {free_now / 1024**3:.2f} GiB GPU memory free before launching '
                    f'{Path(command[1]).name}; need 22 GiB. Earlier results are preserved. '
                    'Use an idle GPU, then rerun only unfinished experiments with --experiments.'
                )
        record(command=command, log=str(log), status='started')
        print(f'Running {shlex.join(command)}\nLog: {log}', flush=True)
        with log.open('w') as handle:
            process = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
        record(command=command, log=str(log), status='ok' if process.returncode == 0 else 'failed',
               returncode=process.returncode)
        if process.returncode:
            print('\n--- subprocess error (last 45 lines) ---', flush=True)
            for line in log.read_text(errors='replace').splitlines()[-45:]:
                print(line, flush=True)
            raise RuntimeError(f'Command failed (exit {process.returncode}); see {log}. Previous results are preserved.')

    import torch
    try:
        import HAIS_OP  # noqa: F401 -- check the server extension before training
    except ModuleNotFoundError as error:
        if error.name != 'HAIS_OP':
            raise
        raise RuntimeError(
            f'HAIS_OP was not found in {PROJECT_ROOT}. '
            'Build it in this server environment with: uv run python scripts/build_ext.py'
        ) from error
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
    data_manifest = []
    for split in ['train', 'val']:
        for path in sorted((Path(options.data_dir) / split).glob('*.npz')):
            row = dict(split=split, file=path.name, bytes=path.stat().st_size)
            if split == 'val':
                digest = hashlib.sha256()
                with path.open('rb') as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                        digest.update(chunk)
                row['sha256'] = digest.hexdigest()
            data_manifest.append(row)
    (bundle / 'data_manifest.json').write_text(json.dumps(data_manifest, indent=2))
    run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests'], bundle / 'cpu_tests.log')
    completed = []
    # One epoch checks only execution/memory; formal runs still start from scratch
    # with the original 50-epoch learning-rate schedule.
    for seed, experiment, name, command in planned[:len(experiments) * len(options.sam_rhos)]:
        rho = float(command[command.index('--sam_rho') + 1])
        if experiment in ['a1', 'r1_event'] or (options.suite == 'r1_sam' and rho == 0):
            continue
        smoke = command.copy()
        smoke[smoke.index('--epochs') + 1] = '1'
        smoke[smoke.index('--run_name') + 1] = 'smoke_' + name
        smoke[smoke.index('--sam_start_epoch') + 1] = '0'
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
        preset = command[command.index('--experiment') + 1]
        if (config['run_name'] != name or config['experiment'] != preset
                or config['seed'] != seed or rows[-1].get('validation_rounds') != 50
                or rows[-1].get('early_stopped') is not False
                or config['sam_rho'] != float(command[command.index('--sam_rho') + 1])):
            raise RuntimeError(f'Incomplete or mismatched training run: {directory}')
        checkpoint = directory / 'checkpoints' / f'best_iou_seed{seed}.pt'
        if not checkpoint.is_file():
            raise RuntimeError(f'Missing checkpoint: {checkpoint}')
        completed.append(directory)
        record(run_dir=str(directory), seed=seed, experiment=preset, variant=experiment,
               sam_rho=config['sam_rho'], checkpoint=str(checkpoint))
        evaluation_options = []
        for key in ['risk_mode', 'risk_weight', 'iou_mode', 'iou_weight',
                    'hardneg_weight', 'hardneg_ratio', 'hardneg_min', 'sam_rho', 'sam_start_epoch']:
            evaluation_options += ['--' + key, str(config[key])]
        for threshold in thresholds:
            stem = f'{experiment}_s{seed}_t{threshold}'
            run([sys.executable, 'test.py', '--split', 'val', '--experiment', preset,
                 *common, '--seed', str(seed), '--prediction_thresh', threshold,
                 *evaluation_options, '--model_path', str(checkpoint),
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
