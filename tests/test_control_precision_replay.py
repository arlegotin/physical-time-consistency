"""New selected-weight replays preserve the original physical tolerances."""
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from poc.io import atomic_json, read_json, sha256_file


def primary_fixture(root):
    jobs = []
    for actuator in ('direct', 'lagged'):
        for model in ('mlp_dt', 'power_correct'):
            for seed in range(5, 10):
                for episode in range(20):
                    cell = f'{actuator}__A__shared__{model}__seed{seed}__h5000000__episode{episode}'
                    path = root/f'config/control_precision_jobs/{cell}.json'
                    request = dict(schema_version=1, phase='confirmatory', model_id=model, actuator=actuator,
                        seed=seed, episode_id=episode, condition='clean', model_partition_ns=[5_000_000]*8,
                        actuator_interval_ns=40_000_000, physics_step_ns=250_000, power_mode=None,
                        model_manifest=f'fit/{actuator}-{model}-{seed}/manifest.json',
                        manifest_sha256='a'*64, checkpoint_sha256='b'*64,
                        planner_settings=dict(candidate_count=128, iterations=4, elite_count=16,
                                              initial_std=1., std_floor=.1),
                        resource_projection=dict(cpu_seconds=80., artifact_bytes=2_500_000))
                    atomic_json(path, request)
                    jobs.append(dict(cell_id=cell, path=str(path.relative_to(root)), sha256=sha256_file(path)))
    return dict(state='complete', kind='primary', expected_episodes=400, jobs=jobs)


def test_exact_first_task_of_every_new_coordinate_preserves_selected_inputs(tmp_path):
    from poc.control_precision_replay import replay_requests
    primary = primary_fixture(tmp_path)
    rows = replay_requests(tmp_path, primary)
    parents = {r['cell_id']: r for r in primary['jobs']}
    assert len(rows) == 20 and len({r['cell_id'] for r in rows}) == 20
    assert {(r['request']['actuator'], r['request']['model_id'], r['request']['seed']) for r in rows} == {
        (a, m, s) for a in ('direct', 'lagged') for m in ('mlp_dt', 'power_correct') for s in range(5, 10)}
    for row in rows:
        parent = parents[row['parent_cell_id']]
        assert row['request'] == dict(read_json(tmp_path/parent['path']), phase='replication')
        assert row['request']['episode_id'] == 0
        assert row['parent_request_sha256'] == parent['sha256']


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'seed', 'clock', 'model', 'phase', 'hash'])
def test_partial_or_changed_primary_cannot_produce_replay_subset(tmp_path, mutation):
    from poc.control_precision_replay import replay_requests
    primary = primary_fixture(tmp_path)
    if mutation == 'missing': primary['jobs'].pop()
    elif mutation == 'duplicate': primary['jobs'][-1] = primary['jobs'][0]
    else:
        job = primary['jobs'][0]; path = tmp_path/job['path']; request = read_json(path)
        if mutation == 'seed': request['seed'] = 9000
        elif mutation == 'clock': request['physics_step_ns'] = 125_000
        elif mutation == 'model': request['model_id'] = 'node_generic'
        elif mutation == 'phase': request['phase'] = 'exploratory'
        else: request['checkpoint_sha256'] = 'c'*64
        atomic_json(path, request)
        if mutation != 'hash': job['sha256'] = sha256_file(path)
    with pytest.raises(ValueError): replay_requests(tmp_path, primary)


def episode(actuator='direct'):
    from poc.control_episode import episode_conditions
    task = episode_conditions(5, 0, actuator, 'confirmatory')
    time = np.arange(0, 1_600_000_001, 250_000, dtype=np.int64)
    state = np.tile(task['initial_state'], (len(time), 1))
    summary = dict(phase='confirmatory', complete=True, actuator=actuator, model_id='power_correct',
        seed=5, episode_id=0, condition='clean', model_partition_ns=[5_000_000]*8,
        actuator_interval_ns=40_000_000, physics_step_ns=250_000, power_mode=None,
        checkpoint_hash='a'*64, initial_state=task['initial_state'].tolist(), target=task['target'].tolist(),
        seed_namespaces={'cem': task['cem_seed_entropy'], 'disturbance': task['disturbance_seed_entropy']},
        realized_cost=2., success=False, fallback_count=0)
    return dict(summary=summary, time_ns=time, state=state, commands=np.zeros((40, 2)),
                decision_statuses=['ok']*40)


@pytest.mark.parametrize('actuator', ['direct', 'lagged'])
def test_new_seed_replay_is_not_a_new_cluster_or_fresh_training(actuator):
    from poc.control_precision_replay import compare_episode
    original = episode(actuator); replay = copy.deepcopy(original); replay['summary']['phase'] = 'replication'
    result = compare_episode(original, replay)
    assert result['status'] == 'pass' and result['diagnosis_required'] is False
    assert result['seed'] == 5 and result['relative_cost_difference'] == 0
    assert result['fresh_training'] is False
    assert result['new_independent_training_clusters'] == 0 and result['alpha_spent'] == 0


@pytest.mark.parametrize('component,amount', [('q', 2e-6), ('v', 2e-5), ('command', 2e-6), ('cost', .001)])
def test_original_physical_tolerances_are_not_relaxed_for_new_seeds(component, amount):
    from poc.control_precision_replay import compare_episode
    original = episode(); replay = copy.deepcopy(original); replay['summary']['phase'] = 'replication'
    if component == 'q': replay['state'][1, 0] += amount
    elif component == 'v': replay['state'][1, 2] += amount
    elif component == 'command': replay['commands'][0, 0] += amount
    else: replay['summary']['realized_cost'] += amount
    result = compare_episode(original, replay)
    assert result['status'] == 'outside_tolerance' and result['diagnosis_required'] is True


@pytest.mark.parametrize('mutation', ['seed', 'target', 'checkpoint', 'phase', 'clock', 'missing', 'nan', 'fallback'])
def test_invalid_replay_identity_or_raw_accounting_is_not_a_measured_mismatch(mutation):
    from poc.control_precision_replay import compare_episode
    original = episode(); replay = copy.deepcopy(original); replay['summary']['phase'] = 'replication'
    if mutation == 'seed': original['summary']['seed'] = replay['summary']['seed'] = 9000
    elif mutation == 'target': replay['summary']['target'][0] += .001
    elif mutation == 'checkpoint': replay['summary']['checkpoint_hash'] = 'b'*64
    elif mutation == 'phase': replay['summary']['phase'] = 'confirmatory'
    elif mutation == 'clock': replay['time_ns'][1] += 1
    elif mutation == 'missing': replay['state'] = replay['state'][:-1]
    elif mutation == 'nan': replay['summary']['realized_cost'] = float('nan')
    else: replay['decision_statuses'][0] = 'all_candidates_failed'
    with pytest.raises(ValueError): compare_episode(original, replay)


def test_prepare_refuses_incomplete_precision_before_writing_requests(tmp_path):
    from poc.control_precision_replay import prepare
    atomic_json(tmp_path/'reports/control_precision_primary.json', dict(state='running'))
    with pytest.raises(ValueError, match='Complete'):
        prepare(tmp_path)
    assert not (tmp_path/'config/control_precision_replay_jobs.json').exists()


def test_complete_runner_reverifies_and_never_relaunches_compatible_episode(tmp_path, monkeypatch):
    import poc.control_precision_replay as replay
    original = episode(); repeated = copy.deepcopy(original); repeated['summary']['phase'] = 'replication'
    original['resources'] = repeated['resources'] = {'cpu_seconds': 1., 'disk_bytes': 10}
    parent = tmp_path/'evidence/M70/parent/manifest.json'; atomic_json(parent, {})
    fresh = tmp_path/'evidence/M70/fresh/manifest.json'; atomic_json(fresh, {})
    request = tmp_path/'config/replay.json'; atomic_json(request, {'fixture': True})
    job = dict(cell_id='replay-cell', parent_cell_id='parent-cell', path=str(request.relative_to(tmp_path)),
        sha256=sha256_file(request), parent_manifest=str(parent.relative_to(tmp_path)),
        parent_manifest_sha256=sha256_file(parent))
    registry = dict(registry_path='config/control_precision_replay_jobs.json', expected_episodes=1,
                    jobs=[job], input_hashes={}, resource_projection={'artifact_bytes': 10})
    atomic_json(tmp_path/registry['registry_path'], registry)
    monkeypatch.setattr(replay, 'verify_registry', lambda root, registry: registry)
    monkeypatch.setattr(replay, 'assert_dispatch_ready', lambda root: None)
    monkeypatch.setattr('poc.io._process_start', lambda pid: 'private-metadata')
    monkeypatch.setattr(replay, '_read_episode', lambda root, path, phase, expected_request:
                        original if phase == 'confirmatory' else repeated)
    calls = []
    def child(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout=str(fresh)+'\n', stderr='')
    monkeypatch.setattr(replay.subprocess, 'run', child)
    first = replay.run_replays(tmp_path, registry, resume=False)
    assert first['state'] == 'complete' and first['completed_episodes'] == 1
    assert first['episodes'][0]['comparison']['status'] == 'pass' and len(calls) == 1
    monkeypatch.setattr(replay.subprocess, 'run', lambda *a, **kw: pytest.fail('Completed replay must be reused'))
    assert replay.run_replays(tmp_path, registry, resume=True) == first
    request.write_text('{}')
    with pytest.raises(ValueError): replay.run_replays(tmp_path, registry, resume=True)


def test_dry_run_does_not_require_or_open_new_outcomes(tmp_path, capsys):
    from poc.control_precision_replay import main
    assert main(['--root', str(tmp_path), '--dry-run']) == 0
    assert '20' in capsys.readouterr().out
    assert not list(tmp_path.iterdir())


def complete_replay_fixture(root, monkeypatch):
    import poc.control_precision_replay as module
    primary = primary_fixture(root)
    atomic_json(root/'config/control_precision_control_jobs.json', primary)
    parents = []
    for job in primary['jobs']:
        manifest = root/f'evidence/M70/{job["cell_id"]}/manifest.json'
        atomic_json(manifest, dict(job_inputs={'dependencies': {}}, produced_files=[]))
        parents.append(dict(cell_id=job['cell_id'], manifest=str(manifest.relative_to(root)),
                            manifest_sha256=sha256_file(manifest)))
    atomic_json(root/'reports/control_precision_primary.json', dict(state='complete', completed_episodes=400,
        expected_episodes=400, registry_sha256=sha256_file(root/'config/control_precision_control_jobs.json'), episodes=parents))
    for name in ('reports/control_precision_result.json', 'reports/control_precision_numerical.json',
                 module.CONTRACT, 'poc/control_precision_replay.py', 'poc/control_replay.py', 'poc/control_utility.py'):
        atomic_json(root/name, {})
    monkeypatch.setattr(module, '_verify_contract', lambda root: {'input_hashes': {}})
    monkeypatch.setattr(module, 'assert_dispatch_ready', lambda root: None)
    monkeypatch.setattr('poc.control_precision.verify_episode_registry', lambda root, registry: registry)
    monkeypatch.setattr(module, '_read_episode', lambda *a, **kw: {'resources': {
        'cpu_seconds': 1., 'wall_seconds': 2., 'disk_bytes': 10}})
    return module.prepare(root)


@pytest.mark.parametrize('mutation', ['missing_hash', 'missing_parent', 'changed_parent', 'duplicate_parent'])
def test_registry_requires_full_hash_and_parent_closure(tmp_path, monkeypatch, mutation):
    import poc.control_precision_replay as module
    registry = complete_replay_fixture(tmp_path, monkeypatch)
    if mutation == 'missing_hash': registry['input_hashes'].pop('poc/control_precision_replay.py')
    elif mutation == 'changed_parent':
        registry['jobs'][0]['parent_manifest'] = registry['jobs'][1]['parent_manifest']
        registry['jobs'][0]['parent_manifest_sha256'] = registry['jobs'][1]['parent_manifest_sha256']
    else:
        path = tmp_path/'reports/control_precision_primary.json'; batch = read_json(path)
        if mutation == 'missing_parent': batch['episodes'].pop()
        else: batch['episodes'][-1] = batch['episodes'][0]
        atomic_json(path, batch)
        registry['input_hashes'][str(path.relative_to(tmp_path))] = sha256_file(path)
    with pytest.raises(ValueError): module.verify_registry(tmp_path, registry)


def test_registry_scope_and_source_subset_are_bound_by_the_registered_contract(tmp_path):
    import poc.control_precision_replay as module
    atomic_json(tmp_path/module.CONTRACT, dict(request_id='MX-control-precision-replay-0001',
        branch='reproduction', status='ready', action_parameters=module.PARAMETERS, spec=module.SPEC,
        contract={str(i): f'Exact element {i} in {module.SPEC}, Twelve-part execution contract' for i in range(1, 13)},
        input_hashes={}))
    entry = dict(request_id='MX-control-precision-replay-0001', branch='reproduction', path=module.CONTRACT,
                 command=['python', '-m', 'poc.control_precision_replay', '--resume'])
    atomic_json(tmp_path/'config/milestones.json', dict(milestones={'MX': {'requests': [entry]}}))
    atomic_json(tmp_path/'config/authorization.json', dict(local_compute_authorized=True, paid_services_authorized=False))
    with pytest.raises(ValueError, match='closure|source|input'): module._verify_contract(tmp_path)


def test_qualified_explicit_numerical_followup_can_return_to_replay(tmp_path, monkeypatch):
    import poc.control_precision_replay as module
    monkeypatch.setattr('poc.control_precision.assert_reproduction_complete', lambda root: None)
    atomic_json(tmp_path/'reports/control_precision_execution_handoff.json',
                dict(state='named_numerical_followup_required'))
    atomic_json(tmp_path/'reports/control_precision_numerical.json', dict(state='qualified', required_actions=[]))
    atomic_json(tmp_path/'reports/control_precision_result.json', dict(look_index=2))
    module.assert_dispatch_ready(tmp_path)
    atomic_json(tmp_path/'reports/control_precision_numerical.json',
                dict(state='unqualified', required_actions=[{'branch': 'physics_full'}]))
    with pytest.raises(ValueError): module.assert_dispatch_ready(tmp_path)
    atomic_json(tmp_path/'reports/control_precision_numerical.json', dict(state='qualified', required_actions=[]))
    atomic_json(tmp_path/'reports/control_precision_execution_handoff.json', dict(state='running_precision'))
    with pytest.raises(ValueError): module.assert_dispatch_ready(tmp_path)


def test_interrupted_child_retains_attempt_and_cannot_claim_complete_replay(tmp_path, monkeypatch):
    import poc.control_precision_replay as module
    path = tmp_path/'config/replay.json'; atomic_json(path, {})
    registry = dict(registry_path=module.REGISTRY, jobs=[dict(cell_id='fixture', path='config/replay.json',
        sha256=sha256_file(path))], resource_projection={'artifact_bytes': 10})
    atomic_json(tmp_path/module.REGISTRY, registry)
    monkeypatch.setattr(module, 'assert_dispatch_ready', lambda root: None)
    monkeypatch.setattr(module, 'verify_registry', lambda root, registry: registry)
    monkeypatch.setattr('poc.io._process_start', lambda pid: 'private-metadata')
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **kw: SimpleNamespace(
        returncode=2, stdout='', stderr='Synthetic interrupted child'))
    with pytest.raises(RuntimeError, match='child incomplete'): module.run_replays(tmp_path, registry, resume=False)
    result = read_json(tmp_path/'reports/control_precision_replay.json')
    assert result['state'] == 'interrupted' and result['completed_episodes'] == 0
    attempts = list((tmp_path/'reports/control_precision_replay_attempts').glob('*/attempt.json'))
    assert len(attempts) == 1 and read_json(attempts[0])['returncode'] == 2
    assert (attempts[0].parent/'stderr.log').read_text() == 'Synthetic interrupted child'


def test_real_mujoco_reader_verifies_fresh_phase_and_complete_raw_trajectory(tmp_path):
    """Small zero-command software fixture, not learned-model confirmation."""
    import subprocess
    from poc.control_jobs import SOURCES
    from poc.control_precision_replay import _read_episode
    repository = Path(__file__).resolve().parents[1]
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True, capture_output=True)
    subprocess.run(['git', '-c', 'user.name=PoC Executor', '-c', 'user.email=poc-executor@example.invalid',
                    'commit', '-q', '--allow-empty', '-m', 'Synthetic replay fixture'],
                   cwd=tmp_path, check=True, capture_output=True)
    for name in (*SOURCES, 'locks/environment.json', 'config/base.json', 'config/numerical.json',
                 'config/planner.json', 'assets/two_link_direct.xml',
                 'config/revisions/owner-objective-2026-10-01.json'):
        path = tmp_path/name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((repository/name).read_bytes())
    for path in (repository/'schemas').glob('*.json'):
        target = tmp_path/'schemas'/path.name; target.parent.mkdir(exist_ok=True)
        target.write_bytes(path.read_bytes())
    atomic_json(tmp_path/'locks/wheels.json', {'wheels': []})
    settings = dict(candidate_count=8, iterations=1, elite_count=1, initial_std=1., std_floor=.1)
    planner = read_json(tmp_path/'config/planner.json'); planner['settings'] = settings
    atomic_json(tmp_path/'config/planner.json', planner)
    base = dict(schema_version=1, phase='confirmatory', model_id='zero_command', actuator='direct', seed=5,
        episode_id=0, condition='clean', model_partition_ns=[40_000_000], actuator_interval_ns=40_000_000,
        physics_step_ns=250_000, power_mode=None, model_manifest=None, manifest_sha256=None,
        checkpoint_sha256=None, planner_settings=settings,
        resource_projection={'basis': 'Small independent MuJoCo replay software fixture'})
    observed = []
    for phase in ('confirmatory', 'replication'):
        request = dict(base, phase=phase); path = tmp_path/f'config/fixture_{phase}.json'
        atomic_json(path, request)
        child = subprocess.run([str(repository/'.venv/bin/python'), '-m', 'poc.control_jobs', str(path)],
                               cwd=tmp_path, capture_output=True, text=True)
        assert child.returncode == 0, child.stderr
        manifest = Path(child.stdout.strip().splitlines()[-1])
        observed.append(_read_episode(tmp_path, manifest, phase, request))
        assert 'poc/control_precision_replay.py' not in read_json(manifest)['job_inputs']['dependencies']
    assert np.array_equal(observed[0]['time_ns'], observed[1]['time_ns'])
    assert np.array_equal(observed[0]['state'], observed[1]['state'])
    assert np.array_equal(observed[0]['commands'], observed[1]['commands'])
    assert observed[0]['summary']['realized_cost'] == observed[1]['summary']['realized_cost']
    with pytest.raises(ValueError, match='request'):
        _read_episode(tmp_path, manifest, 'replication', base)
