"""Fixed first-task replay of seeds 5–9; no new training or statistical look."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from .control_replay import _read_episode
from .io import atomic_json, job_key, job_lock, read_json, safe_path, sha256_file, utc_now

CONTRACT = 'config/revisions/MX-control-precision-replay-0001.json'
SPEC = 'docs/superpowers/specs/2026-10-06-control-precision-replay.md'
REGISTRY = 'config/control_precision_replay_jobs.json'
PARAMETERS = dict(actuators=['direct', 'lagged'], models=['mlp_dt', 'power_correct'],
    seed_ids=list(range(5, 10)), episode_ids=[0], model_step_ns=5_000_000,
    actuator_interval_ns=40_000_000, physics_step_ns=250_000,
    tolerances=dict(q_rad=1e-6, v_rad_s=1e-5, command_nm=1e-6, relative_cost=1e-4))


def contract_sources():
    from .control_jobs import SOURCES
    return set(SOURCES) | {SPEC, 'poc/control_precision_replay.py', 'poc/control_replay.py',
        'poc/control_utility.py', 'poc/control_precision.py', 'poc/control_precision_result.py',
        'config/revisions/MX-control-precision-0001.json', 'config/authorization.json', 'locks/environment.json'}


def replay_requests(root, primary):
    root = Path(root).resolve(); jobs = primary.get('jobs', [])
    expected = {(a, m, s, e) for a in PARAMETERS['actuators'] for m in PARAMETERS['models']
                for s in PARAMETERS['seed_ids'] for e in range(20)}
    if (primary.get('state') != 'complete' or primary.get('kind') != 'primary'
            or primary.get('expected_episodes') != 400 or len(jobs) != 400
            or len({j['cell_id'] for j in jobs}) != 400):
        raise ValueError('Complete unique primary precision registry required')
    from .control_jobs import validate_request
    seen = set(); rows = []
    for job in jobs:
        path = safe_path(root, job['path'])
        if sha256_file(path) != job['sha256']:
            raise ValueError('Changed precision primary request before replay')
        request = validate_request(read_json(path))
        coordinate = (request['actuator'], request['model_id'], request['seed'], request['episode_id'])
        if (coordinate not in expected or coordinate in seen or request['phase'] != 'confirmatory'
                or request['condition'] != 'clean' or request['physics_step_ns'] != 250_000
                or request['actuator_interval_ns'] != 40_000_000 or request['power_mode'] is not None
                or request['model_partition_ns'] != [5_000_000]*8
                or request['planner_settings'] != dict(candidate_count=128, iterations=4, elite_count=16,
                                                       initial_std=1., std_floor=.1)):
            raise ValueError('Replay differs from the exact new precision workload')
        seen.add(coordinate)
        if request['episode_id'] == 0:
            rows.append(dict(cell_id='precision_replay__'+job['cell_id'], parent_cell_id=job['cell_id'],
                parent_request_sha256=job['sha256'], request=dict(request, phase='replication')))
    if seen != expected or len(rows) != 20:
        raise ValueError('Complete precision primary coordinate coverage required')
    return rows


def compare_episode(original, replay):
    first, second = original['summary'], replay['summary']
    keys = ('actuator', 'model_id', 'seed', 'episode_id', 'condition', 'model_partition_ns',
            'actuator_interval_ns', 'physics_step_ns', 'power_mode', 'checkpoint_hash',
            'initial_state', 'target', 'seed_namespaces', 'success', 'fallback_count')
    if (first.get('phase') != 'confirmatory' or second.get('phase') != 'replication'
            or first.get('complete') is not True or second.get('complete') is not True
            or any(first.get(k) != second.get(k) for k in keys)
            or first['actuator'] not in PARAMETERS['actuators'] or first['model_id'] not in PARAMETERS['models']
            or type(first['seed']) is not int or first['seed'] not in PARAMETERS['seed_ids']
            or first['episode_id'] != 0 or first['condition'] != 'clean'
            or first['physics_step_ns'] != 250_000 or first['actuator_interval_ns'] != 40_000_000
            or first['power_mode'] is not None or first['model_partition_ns'] != [5_000_000]*8):
        raise ValueError('Changed precision replay task, phase, checkpoint or clocks')
    dimension = 4 if first['actuator'] == 'direct' else 6
    times = np.arange(0, 1_600_000_001, 250_000, dtype=np.int64)
    states = []; commands = []
    for row in (original, replay):
        state = np.asarray(row['state']); command = np.asarray(row['commands']); summary = row['summary']
        if (not np.array_equal(row['time_ns'], times) or state.shape != (len(times), dimension)
                or command.shape != (40, 2) or not np.isfinite(state).all() or not np.isfinite(command).all()
                or np.any(np.abs(command) > 2) or not np.array_equal(state[0], summary['initial_state'])
                or len(row['decision_statuses']) != 40
                or any(s not in ('ok', 'all_candidates_failed') for s in row['decision_statuses'])
                or type(summary['realized_cost']) not in (int, float) or not math.isfinite(summary['realized_cost'])
                or summary['realized_cost'] < 0 or type(summary['success']) is not bool
                or type(summary['fallback_count']) is not int
                or row['decision_statuses'].count('all_candidates_failed') != summary['fallback_count']):
            raise ValueError('Incomplete/nonfinite precision replay or missing failure accounting')
        states.append(state); commands.append(command)
    if original['decision_statuses'] != replay['decision_statuses']:
        raise ValueError('Changed precision replay decision failure classifications')
    delta = np.abs(states[0]-states[1])
    q = float(delta[:, :2].max()); v = float(delta[:, 2:4].max())
    command = float(np.abs(commands[0]-commands[1]).max())
    absolute = abs(first['realized_cost']-second['realized_cost'])
    relative = absolute/first['realized_cost'] if first['realized_cost'] > 0 else (0. if absolute == 0 else None)
    passed = q <= 1e-6 and v <= 1e-5 and command <= 1e-6 and relative is not None and relative <= 1e-4
    return dict(status='pass' if passed else 'outside_tolerance', diagnosis_required=not passed,
        actuator=first['actuator'], model_id=first['model_id'], seed=first['seed'], episode_id=0,
        model_step_ns=5_000_000, max_q_difference_rad=q, max_v_difference_rad_s=v,
        max_command_difference_nm=command, absolute_cost_difference=absolute, relative_cost_difference=relative,
        max_activation_difference_nm=float(delta[:, 4:].max()) if dimension == 6 else None,
        fresh_training=False, new_independent_training_clusters=0, alpha_spent=0,
        scope='Fresh-process replay of the exact selected precision weights and first tasks')


def _verify_contract(root):
    request = read_json(root/CONTRACT)
    if (request.get('request_id') != 'MX-control-precision-replay-0001'
            or request.get('branch') != 'reproduction' or request.get('status') != 'ready'
            or request.get('action_parameters') != PARAMETERS or request.get('spec') != SPEC
            or request.get('contract') != {str(i): f'Exact element {i} in {SPEC}, Twelve-part execution contract'
                                          for i in range(1, 13)}):
        raise ValueError('Complete exact registered precision replay contract required')
    entry = dict(request_id=request['request_id'], branch='reproduction', path=CONTRACT,
                 command=['python', '-m', 'poc.control_precision_replay', '--resume'])
    if read_json(root/'config/milestones.json')['milestones']['MX']['requests'].count(entry) != 1:
        raise ValueError('Precision replay must have one exact registered sub-contract')
    if set(request.get('input_hashes', {})) != contract_sources():
        raise ValueError('Complete replay source/input closure required')
    for name, digest in request['input_hashes'].items():
        if sha256_file(safe_path(root, name)) != digest:
            raise ValueError('Changed registered replay source/input: '+name)
    authorization = read_json(root/'config/authorization.json')
    if authorization.get('local_compute_authorized') is not True or authorization.get('paid_services_authorized') is not False:
        raise ValueError('Existing local authorization required for precision replay')
    return request


def _parents(root, primary, batch):
    if (batch.get('state') != 'complete' or batch.get('completed_episodes') != 400
            or batch.get('expected_episodes') != 400 or len(batch.get('episodes', [])) != 400
            or batch.get('registry_sha256') != sha256_file(root/'config/control_precision_control_jobs.json')):
        raise ValueError('Complete matching precision primary batch required')
    parents = {r['cell_id']: r for r in batch['episodes']}
    if len(parents) != 400 or set(parents) != {r['cell_id'] for r in primary['jobs']}:
        raise ValueError('Complete unique primary precision episodes required')
    for parent in parents.values():
        if sha256_file(safe_path(root, parent['manifest'])) != parent['manifest_sha256']:
            raise ValueError('Changed primary precision episode manifest')
    return parents


def _registry_names(root, jobs):
    names = {CONTRACT, 'poc/control_precision_replay.py', 'poc/control_replay.py', 'poc/control_utility.py',
        'config/control_precision_control_jobs.json', 'reports/control_precision_primary.json',
        'reports/control_precision_result.json', 'reports/control_precision_numerical.json'}
    names.update(_verify_contract(root)['input_hashes'])
    for job in jobs:
        names.update((job['path'], job['parent_manifest']))
        record = read_json(safe_path(root, job['parent_manifest']))
        names.update(record['job_inputs']['dependencies'])
        names.update(item['path'] for item in record['produced_files'])
    return names


def assert_dispatch_ready(root):
    from .control_precision import assert_reproduction_complete
    assert_reproduction_complete(root)
    path = root/'reports/control_precision_execution_handoff.json'
    if not path.exists() or read_json(path).get('state') not in {
            'result_ready_for_reproduction_and_review', 'named_numerical_followup_required'}:
        raise ValueError('Complete qualified precision supervisor required before replay dispatch')
    numerical = read_json(root/'reports/control_precision_numerical.json')
    result = read_json(root/'reports/control_precision_result.json')
    if numerical.get('state') != 'qualified' or numerical.get('required_actions') != [] or result.get('look_index') != 2:
        raise ValueError('Qualified actual precision result required before selected-weight replay')


def prepare(root):
    root = Path(root).resolve()
    path = root/'reports/control_precision_primary.json'
    batch = read_json(path) if path.exists() else {}
    if (batch.get('state') != 'complete' or batch.get('completed_episodes') != 400
            or batch.get('expected_episodes') != 400 or len(batch.get('episodes', [])) != 400):
        raise ValueError('Complete primary precision batch required before preparing replay')
    assert_dispatch_ready(root); _verify_contract(root)
    target = root/REGISTRY
    if target.exists(): return verify_registry(root, read_json(target))
    from .control_precision import verify_episode_registry
    primary = read_json(root/'config/control_precision_control_jobs.json'); verify_episode_registry(root, primary)
    parents = _parents(root, primary, batch)
    jobs = []; chosen = []
    for row in replay_requests(root, primary):
        parent = parents[row['parent_cell_id']]; manifest = safe_path(root, parent['manifest'])
        if sha256_file(manifest) != parent['manifest_sha256']:
            raise ValueError('Changed new precision episode before replay freeze')
        observed = _read_episode(root, manifest, 'confirmatory', dict(row['request'], phase='confirmatory'))
        filename = root/f'config/control_precision_replay_jobs/{row["cell_id"]}.json'
        if filename.exists() and read_json(filename) != row['request']:
            raise ValueError('Conflicting prospective precision replay request')
        atomic_json(filename, row['request'])
        jobs.append({k: v for k, v in row.items() if k != 'request'} | dict(path=str(filename.relative_to(root)),
            sha256=sha256_file(filename), parent_manifest=parent['manifest'],
            parent_manifest_sha256=parent['manifest_sha256']))
        chosen.append(observed['resources'])
    registry = dict(schema_version=1, state='complete', registry_path=REGISTRY, expected_episodes=20, jobs=jobs,
        input_hashes={n: sha256_file(safe_path(root, n)) for n in sorted(_registry_names(root, jobs))},
        resource_projection=dict(cpu_seconds=sum(r['cpu_seconds'] for r in chosen),
            wall_seconds=sum(r['wall_seconds'] for r in chosen), artifact_bytes=sum(r['disk_bytes'] for r in chosen),
            basis='Actual matched new precision parent episodes; process startup/audits additional'),
        fresh_training=False, new_independent_training_clusters=0, alpha_spent=0)
    atomic_json(target, registry)
    return verify_registry(root, registry)


def verify_registry(root, registry):
    root = Path(root).resolve(); _verify_contract(root)
    if set(registry.get('input_hashes', {})) != _registry_names(root, registry.get('jobs', [])):
        raise ValueError('Incomplete precision replay hash closure')
    for name, digest in registry['input_hashes'].items():
        if sha256_file(safe_path(root, name)) != digest:
            raise ValueError('Changed frozen precision replay dependency: '+name)
    from .control_precision import verify_episode_registry
    primary = read_json(root/'config/control_precision_control_jobs.json'); verify_episode_registry(root, primary)
    expected = replay_requests(root, primary)
    parents = _parents(root, primary, read_json(root/'reports/control_precision_primary.json'))
    if (registry.get('state') != 'complete' or registry.get('registry_path') != REGISTRY
            or registry.get('expected_episodes') != 20 or len(registry.get('jobs', [])) != 20):
        raise ValueError('Complete fixed twenty-replay registry required')
    for row, job in zip(expected, registry['jobs']):
        request = safe_path(root, job['path']); parent = safe_path(root, job['parent_manifest'])
        actual_parent = parents[row['parent_cell_id']]
        if (any(job.get(k) != row[k] for k in ('cell_id', 'parent_cell_id', 'parent_request_sha256'))
                or job['path'] != f'config/control_precision_replay_jobs/{row["cell_id"]}.json'
                or job['parent_manifest'] != actual_parent['manifest']
                or job['parent_manifest_sha256'] != actual_parent['manifest_sha256']
                or sha256_file(request) != job['sha256'] or read_json(request) != row['request']
                or sha256_file(parent) != job['parent_manifest_sha256']):
            raise ValueError('Changed precision replay request or parent manifest')
        _read_episode(root, parent, 'confirmatory', dict(row['request'], phase='confirmatory'))
    return registry


def _observe(root, job, path):
    request = safe_path(root, job['path']); parent = safe_path(root, job['parent_manifest'])
    if sha256_file(request) != job['sha256'] or sha256_file(parent) != job['parent_manifest_sha256']:
        raise ValueError('Changed replay request or primary episode')
    value = read_json(request)
    original = _read_episode(root, parent, 'confirmatory', dict(value, phase='confirmatory'))
    replay = _read_episode(root, path, 'replication', value)
    return dict(cell_id=job['cell_id'], parent_cell_id=job['parent_cell_id'],
        manifest=str(path.relative_to(root)), manifest_sha256=sha256_file(path),
        comparison=compare_episode(original, replay), resources=replay['resources'])


def run_replays(root, registry, *, resume):
    root = Path(root).resolve(); assert_dispatch_ready(root)
    registry_path = safe_path(root, registry['registry_path'])
    if read_json(registry_path) != registry: raise ValueError('Exact immutable replay registry required')
    registry_hash = sha256_file(registry_path); target = root/'reports/control_precision_replay.json'
    jobs = {j['cell_id']: j for j in registry['jobs']}; episodes = []
    if len(jobs) != len(registry['jobs']): raise ValueError('Duplicate precision replay coordinate')
    def save(state, **extra):
        value = dict(state=state, registry_sha256=registry_hash, expected_episodes=len(jobs),
            completed_episodes=len(episodes), episodes=episodes, fresh_training=False,
            new_independent_training_clusters=0, alpha_spent=0, **extra)
        atomic_json(target, value); return value
    with job_lock(job_key({'precision_selected_weight_replay': registry_hash}), root):
        verify_registry(root, registry); reused = {}
        if target.exists():
            previous = read_json(target)
            if not resume or previous['registry_sha256'] != registry_hash:
                raise ValueError('Existing precision replay requires compatible --resume')
            for item in previous['episodes']:
                cell = item['cell_id']; path = safe_path(root, item['manifest'])
                if cell not in jobs or cell in reused or sha256_file(path) != item['manifest_sha256']:
                    raise ValueError('Changed or duplicate completed precision replay')
                if _observe(root, jobs[cell], path) != item:
                    raise ValueError('Precision replay receipt differs from verified raw evidence')
                reused[cell] = item
            if previous['state'] == 'complete':
                if set(reused) != set(jobs) or previous['completed_episodes'] != len(jobs):
                    raise ValueError('Incomplete completed precision replay coverage')
                return previous
        estimate = registry['resource_projection']['artifact_bytes']*(len(jobs)-len(reused))/len(jobs)
        if shutil.disk_usage(root).free < estimate:
            raise RuntimeError('Actual disk prerequisite prevents precision replay; preserve evidence and provision disk')
        try:
            for index, job in enumerate(registry['jobs']):
                verify_registry(root, registry)
                if job['cell_id'] in reused: episodes.append(reused[job['cell_id']]); continue
                print(f'Precision replay {index+1}/{len(jobs)}: {job["cell_id"]}', flush=True)
                attempt = root/'reports/control_precision_replay_attempts'/f'{index:04d}-{utc_now().replace(":", "")}'
                attempt.mkdir(parents=True)
                command = [sys.executable, '-m', 'poc.control_jobs', job['path'], '--root', str(root)]
                atomic_json(attempt/'attempt.json', dict(argv=command, cell_id=job['cell_id'],
                    request_sha256=job['sha256'], started_utc=utc_now(), returncode=None))
                child = subprocess.run(command, cwd=root, capture_output=True, text=True)
                (attempt/'stdout.log').write_text(child.stdout); (attempt/'stderr.log').write_text(child.stderr)
                detail = read_json(attempt/'attempt.json'); detail.update(returncode=child.returncode, ended_utc=utc_now())
                atomic_json(attempt/'attempt.json', detail)
                if child.returncode: raise RuntimeError('Precision replay child incomplete: '+str(attempt.relative_to(root)))
                path = Path(child.stdout.strip().splitlines()[-1]).resolve()
                if not path.is_relative_to(root/'evidence/M70') or path.name != 'manifest.json':
                    raise ValueError('Replay child returned an unexpected manifest path')
                episodes.append(_observe(root, job, path)); save('running')
            return save('complete', diagnosis_required=any(r['comparison']['diagnosis_required'] for r in episodes))
        except BaseException as error:
            save('interrupted', reason=f'{type(error).__name__}: {error}'); raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--dry-run', action='store_true'); parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv); root = args.root.resolve()
    if args.dry_run:
        print(json.dumps(dict(execute=False, expected_episodes=20, seed_ids=list(range(5, 10)),
            episode_ids=[0], fresh_training=False, alpha_spent=0, prerequisite='Complete qualified precision result')))
        return 0
    if sys.version_info[:2] != (3, 11): raise ValueError('Pinned Python 3.11 required for replay')
    from .study_reproduce import validate_runtime_versions
    versions = read_json(root/'locks/environment.json')['versions']
    validate_runtime_versions(versions, {n: importlib.metadata.version(n) for n in versions})
    registry = prepare(root)
    if args.prepare: print(root/REGISTRY); return 0
    result = run_replays(root, registry, resume=args.resume)
    print(json.dumps({k: result[k] for k in ('state', 'completed_episodes', 'diagnosis_required')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
