"""Measure the first fixed same-weight control task before projecting its full batch."""
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
        raise ValueError('Changed prospective same-weight development design')
    cells = {c['cell_id']: c for c in read_json(root / 'config/control_development_cells.json')['eligible_checkpoint_cells']}
    identities = []
    for job in registry['jobs']:
        path = safe_path(root, job['path'])
        if sha256_file(path) != job['sha256']:
            raise ValueError('Changed same-weight request hash')
        r = validate_request(read_json(path)); c = cells[job['cell_id']]
        if (r['phase'] != 'exploratory' or r['condition'] != 'clean' or r['seed'] != 9000
                or r['episode_id'] != 0 or c['condition'] != 'F' or c['loss_mode'] != 'one_step'
                or r['actuator_interval_ns'] != 40_000_000 or r['physics_step_ns'] != 250_000
                or any(r[k] != c[k] for k in ('actuator', 'model_id', 'episode_id', 'power_mode',
                    'model_partition_ns', 'manifest_sha256', 'checkpoint_sha256'))
                or r['model_manifest'] != c['manifest']):
            raise ValueError('Same-weight measurement differs from its eligible frozen cell')
        identities.append((r['actuator'], r['model_id'], r['power_mode'], tuple(r['model_partition_ns'])))
    expected = {(a, m, mode, h) for a in ('direct', 'lagged')
                for m, mode in [('power_correct', 'low'), ('power_correct', 'direct'),
                                ('power_correct', 'lagged'), ('node_structured', None)]
                for h in ((40_000_000,), (5_000_000,) * 8)}
    if len(identities) != 16 or set(identities) != expected:
        raise ValueError('Require all sixteen distinct first-task intervention cells')
    return registry['jobs']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    root = args.root.resolve()
    registry_path = root / 'config/control_attribution_measurement_jobs.json'
    registry_hash = sha256_file(registry_path)
    registry = read_json(registry_path); jobs = validate_jobs(root, registry)
    if args.dry_run:
        print(dict(execute=False, jobs=len(jobs), scope=registry['scope'],
                   resource_projection=registry['resource_projection']))
        return 0
    from .contracts import verify_run
    from .planner_qualification import verify_frozen
    from .control_attribution import attribution_values, verify_nominal_power_outputs
    verify_frozen(root)
    receipt = root / 'reports/control_attribution_measurements.json'
    episodes, rows, folders = [], {}, {}

    def checkpoint(state, **extra):
        atomic_json(receipt, dict(schema_version=1, state=state, updated_utc=utc_now(),
            registry_sha256=registry_hash, completed_episodes=len(episodes), expected_episodes=16,
            episodes=episodes, statistical_status='descriptive_only',
            scope='One fixed development task per actuator and one training seed; not confirmation', **extra))

    try:
        for index, job in enumerate(jobs):
            if sha256_file(registry_path) != registry_hash:
                raise ValueError('Prospective intervention registry changed during execution')
            print(f"Same-weight control measurement {index + 1}/16: {job['cell_id']}", flush=True)
            run = subprocess.run([sys.executable, '-m', 'poc.control_jobs', job['path']],
                                 cwd=root, capture_output=True, text=True)
            if run.returncode:
                raise RuntimeError(f'Intervention measurement failed: {run.stdout}\n{run.stderr}')
            manifest = Path(run.stdout.strip().splitlines()[-1])
            verify_run(root, read_json(manifest))
            summary = read_json(manifest.parent / 'summary.json')
            key = (summary['actuator'], summary['model_id'], summary['power_mode'],
                   summary['model_partition_ns'][0])
            rows[key] = summary; folders[key] = manifest.parent
            episodes.append(dict(cell_id=job['cell_id'], manifest=str(manifest.relative_to(root)),
                manifest_sha256=sha256_file(manifest), summary=summary,
                resources=read_json(manifest.parent / 'resources.json')))
            checkpoint('running')
        contrasts = []
        for actuator in ('direct', 'lagged'):
            identity = verify_nominal_power_outputs(
                folders[(actuator, 'power_correct', 'low', 40_000_000)],
                folders[(actuator, 'power_correct', actuator, 40_000_000)])
            costs = [[rows[(actuator, model, mode, h)]['realized_cost']]
                     for h in (40_000_000, 5_000_000)
                     for model, mode in (('power_correct', 'low'), ('power_correct', actuator),
                                         ('node_structured', None))]
            contrasts.append(dict(actuator=actuator, nominal_identity=identity,
                                  **attribution_values(*costs)))
        checkpoint('complete', contrasts=contrasts)
        print(receipt, flush=True)
        return 0
    except Exception as error:
        checkpoint('interrupted', reason=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    raise SystemExit(main())
