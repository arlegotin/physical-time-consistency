"""Prospectively register and execute an unchanged-design seed continuation."""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys

from .io import atomic_json, read_json, sha256_file, utc_now


def next_seed_block(existing):
    from .rejection_subset import validate_confirmation_seeds
    validate_confirmation_seeds(existing)
    total = max(2*len(existing), len(existing)+5)
    return [seed for seed in range(total+3) if seed not in {9000, 9001, 9002}][:total]


def extend_working_scope(root, seeds):
    from .rejection_subset import validate_confirmation_seeds
    validate_confirmation_seeds(seeds)
    path = Path(root) / 'config/rejection_subset_scope.json'
    prior = read_json(path)
    if seeds[:len(prior['seed_ids'])] != prior['seed_ids'] or len(seeds) < len(prior['seed_ids'])+5:
        raise ValueError('Precision must retain all existing clusters and add at least five')
    backup = Path(root) / 'config/prior_rejection_subset_scope.json'
    if backup.exists():
        raise ValueError('Working context was already extended')
    atomic_json(backup, prior)
    atomic_json(path, {**prior, 'seed_ids': seeds})


def prepare(root, request):
    from .gates import current_decision
    from .study_archive import create_snapshot
    root = Path(root).resolve()
    decision = current_decision(root)
    if (request['branch'] != 'precision' or not decision or decision['outcome'] != 'INCONCLUSIVE'
            or decision['validity_status'] != 'valid' or request['parent_decision_id'] != decision['decision_id']):
        raise ValueError('Precision requires the exact valid unresolved predecessor')
    if set(request['contract']) != {str(i) for i in range(1, 13)} or not all(request['contract'].values()):
        raise ValueError('Precision requires all twelve execution contract elements')
    scope = read_json(root / 'config/rejection_subset_scope.json')
    seeds = next_seed_block(scope['seed_ids'])
    if request['action_parameters']['seed_ids'] != seeds or request['statistical_family_size'] != 280:
        raise ValueError('Changed prospective seed block or family')
    registration = read_json(root / 'config/milestones.json')['milestones']['MX']['requests']
    if not any(row['request_id'] == request['request_id'] and row['branch'] == 'precision' for row in registration):
        raise ValueError('Precision request is not registered')
    destination = request['working_context']
    create_snapshot(root, destination)
    working = root / destination
    # This is a working continuation, not an immutable completed-study archive.
    (working / 'STUDY_SNAPSHOT.json').rename(working / 'INITIAL_CONTEXT.json')
    old_manifest = root / request['prior_manifest']
    if sha256_file(old_manifest) != request['prior_manifest_sha256']:
        raise ValueError('Changed prior aggregation evidence')
    old = read_json(old_manifest)
    old_folder = working / 'evidence/MR' / old['run_id']
    if old_folder.exists():
        shutil.rmtree(old_folder)  # Only this newly copied aggregation; original remains intact.
    for path in (working / 'decisions/GR').glob('*.json'):
        path.unlink()  # Prior decisions live in the untouched original context.
    extend_working_scope(working, seeds)
    atomic_json(working / 'config/precision_parent.json', {
        'request': request, 'original_project': str(root),
        'original_decision': decision, 'original_manifest_sha256': sha256_file(old_manifest),
        'existing_raw_jobs_retained': True, 'statistical_look_ledger_retained': True})
    return working


def run_context(working, seeds):
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        env[name] = '1'
    # Fresh imports use the identical pinned engine in the continuation context.
    script = "from pathlib import Path; from poc.rejection_subset import run_rejection; " + \
        "r=run_rejection(Path.cwd(),resume=True,seeds=" + repr(seeds) + "); " + \
        "print('PRECISION_RUN_ID='+r['run_id'],flush=True)"
    subprocess.run([sys.executable, '-c', script], cwd=working, env=env, check=True)
    process = subprocess.run([sys.executable, '-m', 'poc.decision_wire', 'GR'], cwd=working,
                             env=env, capture_output=True, text=True)
    value = json.loads(process.stdout)
    if value.get('validity_status') != 'valid':
        raise RuntimeError('Precision gate is invalid; preserve diagnostics: ' + process.stderr)
    atomic_json(working / 'precision_decision.json', value)
    print(json.dumps(value, indent=2), flush=True)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request', type=Path)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    request = read_json(args.request)
    working = args.root.resolve() / request['working_context']
    if not working.exists():
        working = prepare(args.root, request)
    elif not args.resume or read_json(working / 'config/precision_parent.json')['request'] != request:
        raise ValueError('Existing continuation needs compatible --resume')
    run_context(working, request['action_parameters']['seed_ids'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
