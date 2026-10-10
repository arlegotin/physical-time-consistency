"""Run the six frozen learned-control measurement cells serially."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from .io import atomic_json, read_json, safe_path, sha256_file, utc_now


def validate_jobs(root, registry):
    from .control_jobs import validate_request
    if (registry['schema_version'] != 1 or registry['phase'] != 'exploratory'
            or registry['design_sha256'] != sha256_file(root / 'config/control_development.json')
            or registry['cells_sha256'] != sha256_file(root / 'config/control_development_cells.json')):
        raise ValueError('Changed prospective development design')
    cells = {c['cell_id']: c for c in read_json(root / 'config/control_development_cells.json')['eligible_checkpoint_cells']}
    identities = []
    for job in registry['jobs']:
        path = safe_path(root, job['path'])
        if sha256_file(path) != job['sha256']:
            raise ValueError('Changed representative request hash')
        request = validate_request(read_json(path))
        cell = cells[job['cell_id']]
        if (request['phase'] != 'exploratory' or request['actuator'] != 'direct'
                or request['condition'] != 'clean' or request['seed'] != 9000
                or request['episode_id'] != 0 or request['power_mode'] is not None
                or request['actuator_interval_ns'] != 40_000_000 or request['physics_step_ns'] != 250_000
                or any(request[k] != cell[k] for k in ('model_id', 'model_partition_ns',
                    'manifest_sha256', 'checkpoint_sha256')) or request['model_manifest'] != cell['manifest']):
            raise ValueError('Representative identity differs from its eligible frozen cell')
        identities.append((request['model_id'], tuple(request['model_partition_ns'])))
    expected = {(m, h) for m in ('mlp_dt', 'node_generic', 'power_correct')
                for h in ((40_000_000,), (5_000_000,) * 8)}
    if len(identities) != 6 or set(identities) != expected:
        raise ValueError('Require all six distinct prospective representative cells')
    return registry['jobs']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    root = args.root.resolve()
    registry_path = root / 'config/control_development_representative_jobs.json'
    registry_hash = sha256_file(registry_path)
    jobs = validate_jobs(root, read_json(registry_path))
    if args.dry_run:
        print(dict(execute=False, jobs=jobs, prerequisite='Qualified frozen independent planner',
                   initial_measurements_only=True, expensive_batch_projection='After measured episodes'))
        return 0
    from .contracts import verify_run
    planner = read_json(root / 'config/planner.json')
    if planner['robot_reference_qualification'] != 'qualified' or not planner['final_common_budget_frozen']:
        raise ValueError('Independent robot planner qualification must finish first')
    from .planner_qualification import verify_frozen
    verify_frozen(root)
    receipt = root / 'reports/control_representative_measurements.json'
    episodes = []

    def checkpoint(state, **extra):
        atomic_json(receipt, dict(schema_version=1, state=state, recorded_utc=utc_now(),
            registry_sha256=registry_hash, completed_episodes=len(episodes), expected_episodes=6,
            episodes=episodes, scope='One development task and training seed; descriptive measurement only',
            **extra))

    try:
        for job in jobs:
            if sha256_file(registry_path) != registry_hash:
                raise ValueError('Prospective registry changed during measurements')
            print(job['cell_id'], flush=True)
            run = subprocess.run([sys.executable, '-m', 'poc.control_jobs', job['path']],
                                 cwd=root, capture_output=True, text=True)
            if run.returncode:
                raise RuntimeError(f'Control measurement failed: {run.stdout}\n{run.stderr}')
            manifest = Path(run.stdout.strip().splitlines()[-1])
            verify_run(root, read_json(manifest))
            episodes.append(dict(cell_id=job['cell_id'], manifest=str(manifest.relative_to(root)),
                manifest_sha256=sha256_file(manifest),
                summary=read_json(manifest.parent / 'summary.json'),
                resources=read_json(manifest.parent / 'resources.json')))
            checkpoint('running')
        checkpoint('complete')
        print(receipt, flush=True)
        return 0
    except Exception as error:
        checkpoint('interrupted', reason=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    raise SystemExit(main())
