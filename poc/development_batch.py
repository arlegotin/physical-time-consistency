"""Complete the original ten-task direct core control workload, serially."""
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
            or registry['cells_sha256'] != sha256_file(root / 'config/control_development_cells.json')
            or registry['measurement_report_sha256'] != sha256_file(root / 'reports/control_representative_measurements.json')):
        raise ValueError('Changed prospective development design or resource measurement')
    cells = {c['cell_id']: c for c in read_json(root / 'config/control_development_cells.json')['eligible_checkpoint_cells']}
    identities = []
    for job in registry['jobs']:
        path = safe_path(root, job['path'])
        if sha256_file(path) != job['sha256']:
            raise ValueError('Changed development request hash')
        r = validate_request(read_json(path)); c = cells[job['cell_id']]
        if (r['phase'] != 'exploratory' or r['actuator'] != 'direct' or r['condition'] != 'clean'
                or r['seed'] != 9000 or r['power_mode'] is not None
                or c['condition'] != 'A' or c['loss_mode'] != 'shared'
                or r['actuator_interval_ns'] != 40_000_000 or r['physics_step_ns'] != 250_000
                or any(r[k] != c[k] for k in ('model_id', 'episode_id', 'model_partition_ns',
                    'manifest_sha256', 'checkpoint_sha256')) or r['model_manifest'] != c['manifest']):
            raise ValueError('Development identity differs from its frozen eligible cell')
        identities.append((r['model_id'], r['episode_id'], tuple(r['model_partition_ns'])))
    expected = {(m, e, h) for m in ('mlp_dt', 'node_generic', 'power_correct')
                for e in range(10) for h in ((40_000_000,), (5_000_000,) * 8)}
    if len(identities) != 60 or set(identities) != expected:
        raise ValueError('Require all sixty distinct original direct core development cells')
    return registry['jobs']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    root = args.root.resolve()
    registry_path = root / 'config/control_development_direct_jobs.json'
    registry = read_json(registry_path); registry_hash = sha256_file(registry_path)
    jobs = validate_jobs(root, registry)
    if args.dry_run:
        print(dict(execute=False, jobs=len(jobs), existing_jobs_reused_by_exact_key=True,
                   projection=registry['remaining_resource_projection']))
        return 0
    from .contracts import verify_run
    from .planner_qualification import verify_frozen
    from .control_comparison import paired_cost_summary
    verify_frozen(root)
    receipt = root / 'reports/control_development_direct.json'
    episodes, rows = [], []

    def checkpoint(state, **extra):
        atomic_json(receipt, dict(schema_version=1, state=state, updated_utc=utc_now(),
            registry_sha256=registry_hash, completed_episodes=len(episodes), expected_episodes=60,
            episodes=episodes, statistical_status='descriptive_only',
            scope='Ten fixed development tasks at one training seed; not confirmation', **extra))

    try:
        for index, job in enumerate(jobs):
            if sha256_file(registry_path) != registry_hash:
                raise ValueError('Prospective development registry changed during execution')
            print(f"Control development {index + 1}/60: {job['cell_id']}", flush=True)
            run = subprocess.run([sys.executable, '-m', 'poc.control_jobs', job['path']],
                                 cwd=root, capture_output=True, text=True)
            if run.returncode:
                raise RuntimeError(f'Control development failed: {run.stdout}\n{run.stderr}')
            manifest = Path(run.stdout.strip().splitlines()[-1])
            verify_run(root, read_json(manifest))
            summary = read_json(manifest.parent / 'summary.json'); rows.append(summary)
            episodes.append(dict(cell_id=job['cell_id'], manifest=str(manifest.relative_to(root)),
                manifest_sha256=sha256_file(manifest), summary=summary,
                resources=read_json(manifest.parent / 'resources.json')))
            checkpoint('running')
        comparisons = []
        for h in (40_000_000, 5_000_000):
            method = [r for r in rows if r['model_id'] == 'power_correct' and r['model_partition_ns'][0] == h]
            for model in ('mlp_dt', 'node_generic'):
                baseline = [r for r in rows if r['model_id'] == model and r['model_partition_ns'][0] == h]
                comparisons.append(dict(baseline_model=model, model_step_ns=h,
                    **paired_cost_summary(method, baseline, [(9000, e) for e in range(10)])))
        checkpoint('complete', comparisons=comparisons)
        print(receipt, flush=True)
        return 0
    except Exception as error:
        checkpoint('interrupted', reason=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    raise SystemExit(main())
