"""Finish the already specified ten-task, two-plant same-weight intervention."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from .io import atomic_json, read_json, safe_path, sha256_file, utc_now

ACTUATORS = ('direct', 'lagged')
MODES = (('power_correct', 'low'), ('power_correct', 'direct'),
         ('power_correct', 'lagged'), ('node_structured', None))
PARTITIONS = ((40_000_000,), (5_000_000,) * 8)


def expected_keys():
    return {(a, e, m, mode, h[0]) for a in ACTUATORS for e in range(10)
            for m, mode in MODES for h in PARTITIONS}


def verify_completed(root, path, expected_hash):
    """Permit only the previously audited recovery change for historical runs."""
    from .contracts import verify_run
    from .planner_qualification import training_recovery_compatibility
    path = safe_path(root, path)
    if sha256_file(path) != expected_hash:
        raise ValueError('Changed reused intervention manifest')
    record = read_json(path)
    verify_run(root, record, check_dependencies=False)
    compatibility, _ = training_recovery_compatibility(root)
    for name, expected in record['job_inputs'].get('dependencies', {}).items():
        if sha256_file(safe_path(root, name)) != expected and (
                name not in compatibility or compatibility[name]['prior_sha256'] != expected):
            raise ValueError('Changed intervention semantic dependency: ' + name)
    for name, expected in record['job_inputs'].get('wheel_hashes', {}).items():
        if sha256_file(safe_path(root, 'wheelhouse/' + name)) != expected:
            raise ValueError('Changed intervention wheel: ' + name)
    for name, keys in record['job_inputs'].get('configuration_keys', {}).items():
        current = read_json(safe_path(root, name))
        if any(current.get(key) != value for key, value in keys.items()):
            raise ValueError('Changed intervention configuration: ' + name)
    return path


def validate_jobs(root, registry):
    from .control_jobs import validate_request
    for key, name in [('design_sha256', 'config/control_development.json'),
                      ('cells_sha256', 'config/control_development_cells.json'),
                      ('measurement_sha256', 'reports/control_attribution_measurements.json')]:
        if registry[key] != sha256_file(root / name):
            raise ValueError('Changed prospective intervention design or measured projection')
    if registry['schema_version'] != 1 or registry['phase'] != 'exploratory':
        raise ValueError('Only the original exploratory workload is authorized')
    cells = {c['cell_id']: c for c in read_json(root / 'config/control_development_cells.json')['eligible_checkpoint_cells']}
    previous = {e['cell_id']: e for e in read_json(root / 'reports/control_attribution_measurements.json')['episodes']}
    identities = []
    for job in registry['jobs']:
        path = safe_path(root, job['path'])
        if sha256_file(path) != job['sha256']:
            raise ValueError('Changed intervention request hash')
        r = validate_request(read_json(path)); c = cells[job['cell_id']]
        if (r['phase'] != 'exploratory' or r['condition'] != 'clean' or r['seed'] != 9000
                or c['condition'] != 'F' or c['loss_mode'] != 'one_step'
                or r['actuator_interval_ns'] != 40_000_000 or r['physics_step_ns'] != 250_000
                or any(r[k] != c[k] for k in ('actuator', 'model_id', 'episode_id', 'power_mode',
                    'model_partition_ns', 'manifest_sha256', 'checkpoint_sha256'))
                or r['model_manifest'] != c['manifest']):
            raise ValueError('Intervention differs from its original eligible checkpoint cell')
        identities.append((r['actuator'], r['episode_id'], r['model_id'],
                           r['power_mode'], r['model_partition_ns'][0]))
        if 'reuse' in job:
            prior = previous.get(job['cell_id'])
            if prior is None or job['reuse'] != {k: prior[k] for k in ('manifest', 'manifest_sha256')}:
                raise ValueError('Reused intervention differs from its verified first-task receipt')
    if len(identities) != 160 or set(identities) != expected_keys():
        raise ValueError('Complete 160 distinct original intervention cells required')
    return registry['jobs']


def summarize(rows, identities):
    from .control_attribution import attribution_values
    if set(rows) != expected_keys() or len(identities) != 40 or not all(i['matched'] for i in identities):
        raise ValueError('Complete paired tasks and all three nominal power identities required')
    contrasts, task_contrasts = [], []
    for actuator in ACTUATORS:
        costs = [[rows[(actuator, e, model, mode, h)]['realized_cost'] for e in range(10)]
                 for h in (40_000_000, 5_000_000)
                 for model, mode in [('power_correct', 'low'), ('power_correct', actuator),
                                     ('node_structured', None)]]
        # Ten tasks belong to one trained seed; they are not ten training replicas.
        contrasts.append(dict(actuator=actuator, n_training_seeds=1, n_tasks=10,
            **attribution_values(*[[sum(v) / len(v)] for v in costs])))
        for e in range(10):
            task_contrasts.append(dict(actuator=actuator, episode_id=e,
                                      **attribution_values(*[[v[e]] for v in costs])))
    return dict(contrasts=contrasts, task_contrasts=task_contrasts, nominal_identities=identities)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv); root = args.root.resolve()
    registry_path = root / 'config/control_attribution_development_jobs.json'
    registry_hash = sha256_file(registry_path)
    registry = read_json(registry_path); jobs = validate_jobs(root, registry)
    if args.dry_run:
        print(dict(execute=False, jobs=160, reused_first_tasks=16,
                   remaining_resource_projection=registry['remaining_resource_projection']))
        return 0
    from .contracts import verify_run
    from .planner_qualification import verify_frozen
    from .control_attribution import verify_nominal_power_outputs
    verify_frozen(root)
    receipt = root / 'reports/control_attribution_development.json'
    episodes, rows, folders = [], {}, {}

    def checkpoint(state, **extra):
        atomic_json(receipt, dict(schema_version=1, state=state, updated_utc=utc_now(),
            registry_sha256=registry_hash, completed_episodes=len(episodes), expected_episodes=160,
            episodes=episodes, statistical_status='descriptive_only',
            scope='Ten original development tasks on both plants at one training seed; not confirmation', **extra))

    try:
        for index, job in enumerate(jobs):
            if sha256_file(registry_path) != registry_hash:
                raise ValueError('Prospective intervention registry changed during execution')
            print(f"Same-weight development {index + 1}/160: {job['cell_id']}", flush=True)
            if 'reuse' in job:
                manifest = verify_completed(root, job['reuse']['manifest'], job['reuse']['manifest_sha256'])
            else:
                run = subprocess.run([sys.executable, '-m', 'poc.control_jobs', job['path']],
                                     cwd=root, capture_output=True, text=True)
                if run.returncode:
                    raise RuntimeError(f'Intervention job failed: {run.stdout}\n{run.stderr}')
                manifest = Path(run.stdout.strip().splitlines()[-1])
                verify_run(root, read_json(manifest))
            request = read_json(root / job['path'])
            if read_json(manifest.parent / 'request.json') != request:
                raise ValueError('Completed intervention request differs from the registered job')
            summary = read_json(manifest.parent / 'summary.json')
            for key in ('actuator', 'episode_id', 'model_id', 'power_mode', 'model_partition_ns',
                        'seed', 'condition', 'phase', 'actuator_interval_ns', 'physics_step_ns'):
                if summary[key] != request[key]:
                    raise ValueError('Completed intervention identity differs: ' + key)
            if summary['checkpoint_hash'] != request['checkpoint_sha256'] or not summary['complete']:
                raise ValueError('Completed intervention checkpoint or completion mismatch')
            key = (summary['actuator'], summary['episode_id'], summary['model_id'],
                   summary['power_mode'], summary['model_partition_ns'][0])
            rows[key] = summary; folders[key] = manifest.parent
            episodes.append(dict(cell_id=job['cell_id'], manifest=str(manifest.relative_to(root)),
                manifest_sha256=sha256_file(manifest), summary=summary,
                resources=read_json(manifest.parent / 'resources.json'), reused='reuse' in job))
            checkpoint('running')
        identities = []
        for actuator in ACTUATORS:
            for episode in range(10):
                low = folders[(actuator, episode, 'power_correct', 'low', 40_000_000)]
                for mode in ('direct', 'lagged'):
                    identities.append(dict(actuator=actuator, episode_id=episode, mode=mode,
                        **verify_nominal_power_outputs(low,
                            folders[(actuator, episode, 'power_correct', mode, 40_000_000)],
                            comparison_mode=mode)))
        checkpoint('complete', **summarize(rows, identities))
        print(receipt, flush=True)
        return 0
    except Exception as error:
        checkpoint('interrupted', reason=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    raise SystemExit(main())
