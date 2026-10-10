"""Explicit, content-addressed single-episode control execution."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import re
import sys

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'

from .io import atomic_json, read_json, safe_path, sha256_file

MODELS = {'mlp_dt', 'gru_dt', 'node_generic', 'node_structured', 'power_correct',
          'physics_fit', 'accel_ridge', 'sim_oracle', 'zero_command', 'constant_velocity'}
SOURCES = ('poc/__init__.py', 'poc/control_jobs.py', 'poc/io.py', 'poc/contracts.py', 'poc/gates.py',
           'poc/execution.py', 'poc/clocks.py', 'poc/envs.py', 'poc/analytic.py', 'poc/history.py',
           'poc/action_path.py', 'poc/control.py', 'poc/mpc.py', 'poc/control_episode.py',
           'poc/control_artifacts.py', 'poc/control_models.py', 'poc/reference_jobs.py', 'poc/train.py',
           'poc/physics_fit.py', 'poc/models/__init__.py', 'poc/models/common.py', 'poc/models/mlp.py',
           'poc/models/power.py', 'poc/models/sequence.py', 'poc/models/ode.py', 'poc/models/factory.py')


def validate_request(value):
    expected = {'schema_version', 'phase', 'model_id', 'actuator', 'seed', 'episode_id', 'condition',
                'model_partition_ns', 'actuator_interval_ns', 'physics_step_ns', 'power_mode', 'model_manifest',
                'manifest_sha256', 'checkpoint_sha256', 'planner_settings', 'resource_projection'}
    if set(value) != expected or value['schema_version'] != 1 or value['model_id'] not in MODELS:
        raise ValueError('Unknown or incomplete control request')
    if (value['actuator'] not in {'direct', 'lagged'}
            or value['phase'] not in {'calibration', 'exploratory', 'confirmatory', 'replication'}
            or any(type(value[key]) is not int or value[key] < 0 for key in ('seed', 'episode_id'))
            or value['condition'] not in {'clean', 'noise', 'disturbance'}
            or value['power_mode'] not in {None, 'low', 'direct', 'lagged'}):
        raise ValueError('Invalid explicit control workload identity')
    if value['phase'] in {'confirmatory', 'replication'} and value['seed'] in {9000, 9001, 9002}:
        raise ValueError('Development seeds cannot become confirmation')
    partition = value['model_partition_ns']
    if (not isinstance(partition, list) or not partition
            or any(type(t) is not int or t <= 0 for t in partition) or sum(partition) != 40_000_000):
        raise ValueError('Internal partition must cover one complete unchanged observation interval')
    tick, hold = value['physics_step_ns'], value['actuator_interval_ns']
    if (type(tick) is not int or tick <= 0 or hold not in {40_000_000, 20_000_000, 10_000_000}
            or hold % tick or (hold != 40_000_000 and value['condition'] != 'clean')):
        raise ValueError('Unsupported hardware/numerical clock or combined perturbation')
    references = {'sim_oracle', 'zero_command', 'constant_velocity'}
    if value['model_id'] in references:
        if any(value[key] is not None for key in ('model_manifest', 'manifest_sha256', 'checkpoint_sha256')):
            raise ValueError('Untrained policy/oracle has no learned checkpoint')
    elif (not isinstance(value['model_manifest'], str) or not value['model_manifest']
          or any(not isinstance(value[key], str) or not re.fullmatch('[0-9a-f]{64}', value[key])
                 for key in ('manifest_sha256', 'checkpoint_sha256'))):
        raise ValueError('Selected model requires complete prospective manifest/checkpoint hashes')
    if value['power_mode'] is not None and value['model_id'] != 'power_correct':
        raise ValueError('Only shared power checkpoints have an exponent intervention')
    from .mpc import _settings
    _settings(value['planner_settings'])
    if not isinstance(value['resource_projection'], dict) or not value['resource_projection']:
        raise ValueError('Measured projection or initial representative-job estimate required')
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request', type=Path)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    root = args.root.resolve()
    request = validate_request(read_json(args.request))
    numerical = read_json(root / 'config/numerical.json')
    accepted = numerical['physics_step_ns']
    if accepted % request['physics_step_ns'] or request['physics_step_ns'] > accepted:
        raise ValueError('Control physics must use the qualified step or an exact refinement')
    planner = read_json(root / 'config/planner.json')
    if request['planner_settings'] != planner['settings']:
        raise ValueError('All models must use the same model-independent selected planner budget')
    if request['model_id'] not in {'sim_oracle', 'physics_fit', 'accel_ridge', 'zero_command', 'constant_velocity'}:
        if planner['robot_reference_qualification'] != 'qualified' or not planner['final_common_budget_frozen']:
            raise ValueError('Qualify model-independent robot control before learned-model outcomes')
    if args.dry_run:
        print({'execute': False, 'request': request, 'expected_observations': 40,
               'expected_physical_ns': 1_600_000_000, 'expected_candidates':
               40 * request['planner_settings']['candidate_count'] * request['planner_settings']['iterations']})
        return 0
    from .execution import cpu_defaults, run_job
    from .envs import Plant
    from .control_models import load_selected, ConstantVelocity, SimOracle
    from .control_episode import execute_episode
    from .control_artifacts import save_episode
    cpu_defaults(request['seed'])
    key_path = str(args.request.resolve().relative_to(root))
    files = list(SOURCES) + [key_path, 'config/planner.json', 'config/numerical.json',
                            f'assets/two_link_{request["actuator"]}.xml',
                            'config/revisions/owner-objective-2026-10-01.json']
    if request['model_manifest']:
        manifest = safe_path(root, request['model_manifest'])
        files += [request['model_manifest'], str((manifest.parent / 'weights.pt').relative_to(root)),
                  str((manifest.parent / 'normalization.json').relative_to(root))]
    dependencies = {filename: sha256_file(root / filename) for filename in files}
    expected = ('control.episode', 'control.clocks', 'control.provenance')
    def producer(folder):
        kind = request['model_id']
        if kind == 'sim_oracle':
            model = SimOracle(request['actuator'], request['physics_step_ns'], root / 'assets')
        elif kind in {'zero_command', 'constant_velocity'}:
            model = ConstantVelocity(request['actuator'])
        else:
            model = load_selected(safe_path(root, request['model_manifest']), request['manifest_sha256'],
                                  request['checkpoint_sha256'], model_id=kind,
                                  actuator=request['actuator'], seed=request['seed'])
        plant = Plant(request['actuator'], request['physics_step_ns'], root / 'assets')
        atomic_json(folder / 'request.json', request)
        result = execute_episode(plant, model, request['planner_settings'], seed=request['seed'],
            episode_id=request['episode_id'], phase=request['phase'], condition=request['condition'],
            model_partition=tuple(request['model_partition_ns']), actuator_interval_ns=request['actuator_interval_ns'],
            power_mode=request['power_mode'])
        summary = save_episode(folder, result, model_id=kind, checkpoint_hash=request['checkpoint_sha256'], run_id=folder.name)
        if kind == 'zero_command' and any(segment.command.any() for segment in result['action_segments']):
            raise ValueError('Zero-command reference unexpectedly chose a nonzero control')
        imported = {str(Path(module.__file__).resolve().relative_to(root))
                    for module in list(sys.modules.values()) if getattr(module, '__file__', None)
                    and str(module.__file__).endswith('.py') and Path(module.__file__).resolve().is_relative_to(root / 'poc')}
        if imported - set(files) or any(sha256_file(root / f) != h for f, h in dependencies.items()):
            raise ValueError('Control source/import/input provenance changed during the episode')
        summary.update({'resource_projection': request['resource_projection'], 'missing_cells': [],
                        'oracle_physics_steps': getattr(model, 'physics_steps', 0),
                        'model_hashes': {'checkpoint_sha256': request['checkpoint_sha256']} if request['checkpoint_sha256'] else {}})
        checks = [{'check_id': key, 'status': 'pass', 'scope': kind,
                   'reason': 'Complete raw physical episode, fixed clocks and hash closure verified', 'tolerance': None}
                  for key in expected]
        return checks, summary
    milestone = 'M70' if request['phase'] in {'confirmatory', 'replication'} else 'M40'
    record = run_job(root, milestone, request['phase'], files, expected, [], producer,
                     config_keys={key_path: request}, argv=[sys.executable, '-m', 'poc.control_jobs', key_path])
    print(root / 'evidence' / milestone / record['run_id'] / 'manifest.json', flush=True)
    return 0 if record['completion_status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
