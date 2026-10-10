"""Prospective sample identity, isolated inputs and dispatch boundaries."""
import copy
from pathlib import Path

import pytest

from poc.io import atomic_json, read_json, sha256_file


def request_inputs(root):
    """Small independent parent records; no scientific outcomes are generated."""
    repository = Path(__file__).resolve().parents[1]
    for name in ('config/control_confirmation_design.json', 'config/authorization.json',
                 'config/planner.json', 'docs/superpowers/specs/2026-10-06-control-precision.md',
                 'docs/superpowers/plans/2026-10-06-control-precision.md'):
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((repository/name).read_bytes())
    criteria = read_json(root/'config/control_confirmation_design.json')['criteria']
    result = dict(look_index=1, family_size=8, look_alpha=.025, tail_alpha=.0015625,
                  status='limited_scope_results', criteria=[dict(row, status='unresolved') for row in criteria],
                  input_hashes={}, original_ptco_terminal_verdict=None)
    atomic_json(root/'reports/control_confirmation_result.json', result)
    atomic_json(root/'reports/control_confirmation_numerical.json',
                dict(state='qualified', scope='full', required_actions=[], input_hashes={}))
    atomic_json(root/'reports/control_confirmation_batch.json',
                dict(state='complete', completed_episodes=1200, expected_episodes=1200))
    planning = read_json(repository/'reports/control_precision_planning_2026-10-06.json')
    for row in planning['evidence']:
        row['sha256'] = sha256_file(root/row['path'])
    atomic_json(root/'reports/control_precision_planning_2026-10-06.json', planning)
    # Parent producer inventories in these fixtures are empty. The real request
    # additionally binds the actual registered producer and source dependencies.
    atomic_json(root/'config/control_confirmation_cells.json',
                dict(state='complete', evaluator_hashes={}, selected=[]))
    return root


def test_request_keeps_old_clusters_and_adds_only_seeds_5_to_9(tmp_path):
    from poc.control_precision import build_request, verify_request
    root = request_inputs(tmp_path)
    request = build_request(root)
    assert verify_request(root, request) == request
    parameters = request['action_parameters']
    assert parameters['retained_seed_ids'] == [0, 1, 2, 3, 4]
    assert parameters['new_seed_ids'] == [5, 6, 7, 8, 9]
    assert parameters['episode_ids'] == list(range(20))
    assert parameters['criterion_ids'] == ['utility_control/direct/mlp_dt/h5000000',
                                           'utility_control/lagged/mlp_dt/h5000000']
    assert request['resource_projection']['n_new_control_episodes'] == 400
    assert parameters['family_size'] == 8
    assert parameters['look_index'] == 2
    assert parameters['look_alpha'] == .0125
    assert parameters['directional_tail_alpha'] == .00078125
    assert set(request['contract']) == {str(i) for i in range(1, 13)}


@pytest.mark.parametrize('mutation', ['replacement', 'exploratory', 'order', 'tail', 'parent', 'hash', 'allocation'])
def test_prospective_request_rejects_changed_sample_or_evidence(tmp_path, mutation):
    from poc.control_precision import build_request, verify_request
    root = request_inputs(tmp_path)
    request = build_request(root)
    if mutation == 'replacement': request['action_parameters']['retained_seed_ids'][0] = 10
    elif mutation == 'exploratory': request['action_parameters']['new_seed_ids'][-1] = 9000
    elif mutation == 'order': request['action_parameters']['new_seed_ids'].reverse()
    elif mutation == 'tail': request['action_parameters']['directional_tail_alpha'] = .025
    elif mutation == 'parent':
        p = root/'reports/control_confirmation_result.json'
        value = read_json(p); value['look_index'] = 0; atomic_json(p, value)
    elif mutation == 'hash': (root/'config/planner.json').write_text('{}')
    else:
        p = root/'config/authorization.json'
        value = read_json(p); value['local_compute_authorized'] = False; atomic_json(p, value)
    with pytest.raises(ValueError): verify_request(root, request)


def context_inputs(root):
    root.mkdir(parents=True)
    (root/'poc').mkdir(); (root/'poc/engine.py').write_text('original nominal engine\n')
    (root/'schemas').mkdir(); (root/'schemas/manifest.schema.json').write_text('{}')
    for name in ('environment.json', 'wheels.json', 'requirements.lock'):
        path = root/'locks'/name; path.parent.mkdir(exist_ok=True); path.write_text('{}')
    atomic_json(root/'config/base.json', dict(fixture=True))
    definitions = []
    for actuator in ('direct', 'lagged'):
        parameters = dict(actuator=actuator, condition='A', loss_mode='shared', model_id='mlp_dt', seed=5)
        for role in ('train', 'validation'):
            path = root/f'data/{actuator}/A/{role}/manifest.json'
            path.parent.mkdir(parents=True); path.with_name('trajectories.npz').write_bytes((actuator+role).encode())
            atomic_json(path, dict(role=role, actuator=actuator, condition='A',
                                   files={'trajectories.npz': sha256_file(path.with_name('trajectories.npz'))}))
            parameters[role+'_manifest'] = str(path.relative_to(root))
        definitions.append(parameters)
    parents = []
    for gate, milestone in (('G00', 'M00'), ('G10', 'M10'), ('G20', 'M20')):
        folder = root/f'evidence/{milestone}/fixture'; folder.mkdir(parents=True)
        output = folder/'answer.json'; output.write_text('{}')
        manifest = folder/'manifest.json'
        atomic_json(manifest, dict(produced_files=[dict(path=str(output.relative_to(root)), sha256=sha256_file(output))],
                                   job_inputs=dict(dependencies={'poc/engine.py': sha256_file(root/'poc/engine.py')},
                                                   configuration_keys={'config/base.json': {'fixture': True}}, wheel_hashes={}),
                                   parent_decisions=[]))
        atomic_json(folder/'COMPLETED', dict(manifest_sha256=sha256_file(manifest)))
        parent = dict(gate_id=gate, decision_id='fixture',
                      evidence=[dict(path=str(manifest.relative_to(root)), sha256=sha256_file(manifest), role='manifest')])
        atomic_json(root/f'decisions/{gate}/fixture.json', parent); parents.append(parent)
    for name in ('evidence/M30/old/weights.pt', 'data/direct/A/confirmation/trajectories.npz',
                 'data/direct/A/development/trajectories.npz'):
        path = root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'excluded')
    return definitions, parents


def test_nominal_inventory_excludes_labels_and_weights(tmp_path):
    from poc.control_precision import union_nominal_inventory
    source = tmp_path/'source'; definitions, parents = context_inputs(source)
    inventory = union_nominal_inventory(source, [(p, parents) for p in definitions])
    assert set(inventory['gate_inputs']['gates']) == {'G00', 'G10', 'G20'}
    assert 'data/direct/A/train/trajectories.npz' in inventory['files']
    assert 'data/lagged/A/validation/trajectories.npz' in inventory['files']
    assert not any('/confirmation/' in p or '/development/' in p or p.endswith('weights.pt')
                   for p in inventory['files'])
    assert inventory['cached_training_copied'] is False


@pytest.mark.parametrize('mutation', ['labels', 'cache', 'missing_plant', 'missing_gate'])
def test_union_rejects_hidden_label_cache_and_incomplete_validity(tmp_path, mutation):
    from poc.control_precision import union_nominal_inventory
    source = tmp_path/'source'; definitions, parents = context_inputs(source)
    if mutation == 'labels': definitions[0]['train_manifest'] = 'data/direct/A/confirmation/manifest.json'
    elif mutation == 'cache':
        manifest = source/'evidence/M20/fixture/manifest.json'; value = read_json(manifest)
        value['job_inputs']['dependencies']['evidence/M30/old/weights.pt'] = sha256_file(source/'evidence/M30/old/weights.pt')
        atomic_json(manifest, value)
        atomic_json(manifest.parent/'COMPLETED', dict(manifest_sha256=sha256_file(manifest)))
        parents[-1]['evidence'][0]['sha256'] = sha256_file(manifest)
        atomic_json(source/'decisions/G20/fixture.json', parents[-1])
    elif mutation == 'missing_plant': definitions.pop()
    else: parents.pop(0)
    with pytest.raises((ValueError, FileNotFoundError)):
        union_nominal_inventory(source, [(p, parents) for p in definitions])


def test_live_reproduction_blocks_dispatch_but_allows_dry_run(tmp_path):
    from poc.control_precision import assert_reproduction_complete, build_request
    root = request_inputs(tmp_path)
    atomic_json(root/'reports/reproduction_followup.json', dict(state='running'))
    assert build_request(root)['status'] == 'ready'
    with pytest.raises(ValueError, match='reproduction'): assert_reproduction_complete(root)


def reproduction_fixture(root):
    atomic_json(root/'reports/reproduction_followup.json',
                dict(state='reproduction_ready_for_whole_branch_review'))
    for name, count, kind in [('control_replay', 60, 'episodes'), ('control_retraining', 12, 'fits'),
                              ('control_retrained_policy_batch', 480, 'episodes')]:
        atomic_json(root/f'reports/{name}.json', dict(state='complete', diagnosis_required=False,
                    **{'completed_'+kind: count, 'expected_'+kind: count, kind: [{} for _ in range(count)]}))
    atomic_json(root/'reports/control_retrained_policy_result.json',
                dict(status='pass', diagnosis_required=False, criteria_unchanged=True,
                     alpha_spent=0, new_independent_training_clusters=0))


def test_complete_reproduction_requires_actual_batch_and_criterion_stability(tmp_path):
    from poc.control_precision import assert_reproduction_complete
    reproduction_fixture(tmp_path)
    assert_reproduction_complete(tmp_path)
    path = tmp_path/'reports/control_retrained_policy_result.json'
    value = read_json(path); value['criteria_unchanged'] = False; atomic_json(path, value)
    with pytest.raises(ValueError): assert_reproduction_complete(tmp_path)


def test_additive_root_files_do_not_change_archived_source_context(tmp_path):
    from poc.control_precision import union_nominal_inventory
    source = tmp_path/'source'; definitions, parents = context_inputs(source)
    before = union_nominal_inventory(source, [(p, parents) for p in definitions])
    (tmp_path/'poc').mkdir(); (tmp_path/'poc/new_precision.py').write_text('new root coordinator')
    assert union_nominal_inventory(source, [(p, parents) for p in definitions]) == before
    (source/'poc/engine.py').write_text('changed archived engine')
    with pytest.raises(ValueError): union_nominal_inventory(source, [(p, parents) for p in definitions])


def reuse_proofs():
    return [dict(cell_id=f'direct__A__shared__{model}__seed{seed}', status='pass',
                 observed=dict(objective=.01, one_step=.005, four_step=.02, one_count=8192, four_count=7424),
                 selected=dict(objective=.01, one_step=.005, four_step=.02, one_count=8192, four_count=7424),
                 canonical_tensor_sha256='a'*64, manifest=f'evidence/M30/fit{model}{seed}/manifest.json',
                 manifest_sha256='b'*64)
            for seed in range(40) for model in ('power_correct', 'node_generic')]


def test_reuse_is_complete_ascending_and_never_selected_by_prediction_score():
    from poc.control_precision import reuse_coordinates
    proofs = reuse_proofs()
    selected = reuse_coordinates(dict(state='complete', expected_cells=80, completed_cells=80,
                                      passed=True, cells=list(reversed(proofs)), labels='Nominal validation only'))
    assert [r['cell_id'] for r in selected] == [f'direct__A__shared__power_correct__seed{s}' for s in [5, 6, 7, 8, 9]]


def test_actual_archived_nominal_audit_is_the_generic_node_cohort():
    from poc.control_precision import reuse_coordinates
    audit = read_json(Path(__file__).resolve().parents[1]/'reports/selection_integrity_forty.json')
    assert [r['cell_id'] for r in reuse_coordinates(audit)] == [
        'direct__A__shared__power_correct__seed5', 'direct__A__shared__power_correct__seed6',
        'direct__A__shared__power_correct__seed7', 'direct__A__shared__power_correct__seed8',
        'direct__A__shared__power_correct__seed9']


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'bad_score', 'failed', 'prediction_selected'])
def test_reuse_refuses_partial_or_changed_independent_selection_proof(mutation):
    from poc.control_precision import reuse_coordinates
    proofs = reuse_proofs()
    row = next(r for r in proofs if r['cell_id'] == 'direct__A__shared__power_correct__seed5')
    if mutation == 'missing': proofs.remove(row)
    elif mutation == 'duplicate': proofs[-1] = copy.deepcopy(row)
    elif mutation == 'bad_score': row['observed']['objective'] += .001
    elif mutation == 'failed': row['status'] = 'fail'
    audit = dict(state='complete', expected_cells=80, completed_cells=80, passed=True, cells=proofs,
                 labels='Nominal validation only' if mutation != 'prediction_selected' else 'Best fine-time scores')
    with pytest.raises(ValueError): reuse_coordinates(audit)


def test_reused_fit_must_match_same_train_labels_scaler_and_training_engine():
    from poc.control_precision import verify_reuse_semantics
    definition = dict(actuator='direct', condition='A', loss_mode='shared', model_id='power_correct', seed=5)
    data = {'data/direct/A/train/manifest.json': 'a'*64,
            'data/direct/A/train/trajectories.npz': 'b'*64,
            'data/direct/A/validation/manifest.json': 'c'*64,
            'data/direct/A/validation/trajectories.npz': 'd'*64}
    inventory = dict(files={**data, 'poc/train.py': 'e'*64, 'poc/models/power.py': 'f'*64})
    parameters = dict(definition, learning_rate=.001, epochs=200, data_hashes=data)
    record = dict(phase='confirmatory', milestone_id='M30',
                  job_inputs=dict(parameters=parameters, dependencies={'poc/train.py': 'e'*64,
                                                                      'poc/models/power.py': 'f'*64}))
    norm = dict(scale_q=[.1, .2], scale_v=[1., 2.], velocity_input_rms=[1., 1.],
                training_manifest_sha256='a'*64)
    verify_reuse_semantics(record, norm, definition, inventory, norm)
    changed = copy.deepcopy(record); changed['job_inputs']['parameters']['data_hashes'][
        'data/direct/A/validation/trajectories.npz'] = '0'*64
    with pytest.raises(ValueError): verify_reuse_semantics(changed, norm, definition, inventory, norm)
    changed = copy.deepcopy(record); changed['job_inputs']['dependencies']['poc/train.py'] = '0'*64
    with pytest.raises(ValueError): verify_reuse_semantics(changed, norm, definition, inventory, norm)
    with pytest.raises(ValueError):
        verify_reuse_semantics(record, dict(norm, scale_q=[.3, .2]), definition, inventory, norm)


def test_known_recovery_variants_preserve_nominal_calculations_and_are_eligible():
    """Reject a scientific change while allowing the already audited recovery fix."""
    import ast
    from poc.control_precision import verify_reuse_semantics
    root = Path(__file__).resolve().parents[1]
    old = root/'study/precision-look-2/study/precision-look-3/study/precision-look-4'
    new = root/'study/control-development'
    names = {'endpoint_loss', 'optimizer_for', 'epoch_starts', 'histories_for', 'rollout_loss',
             'batch_objective', 'select_checkpoint', 'nominal_convergence', 'TrainingSession',
             'validate_nominal', 'array_scales', 'constant_velocity_error', 'qualify_nominal'}
    def calculations(path):
        return {n.name: ast.dump(n, include_attributes=False)
                for n in ast.parse(path.read_text()).body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names}
    assert set(calculations(old/'poc/train.py')) == names
    assert calculations(old/'poc/train.py') == calculations(new/'poc/train.py')
    audit = read_json(root/'reports/selection_integrity_forty.json')
    proof = next(r for r in audit['cells'] if r['cell_id'] == 'direct__A__shared__power_correct__seed5')
    record = read_json(old/proof['manifest'])
    files = dict(record['job_inputs']['parameters']['data_hashes'])
    for name in record['job_inputs']['dependencies']:
        if (new/name).exists(): files[name] = sha256_file(new/name)
    norm = read_json(new/'data/direct/A/normalization.json')
    definition = dict(actuator='direct', condition='A', loss_mode='shared', model_id='power_correct', seed=5)
    verify_reuse_semantics(record, read_json((old/proof['manifest']).parent/'normalization.json'),
                           definition, dict(files=files), norm)
    changed = copy.deepcopy(record); changed['job_inputs']['dependencies']['poc/train.py'] = '0'*64
    with pytest.raises(ValueError): verify_reuse_semantics(changed, norm, definition, dict(files=files), norm)


def test_nominal_dispatch_cannot_start_while_original_reproduction_runs(tmp_path):
    from poc.control_precision import run_nominal
    atomic_json(tmp_path/'reports/reproduction_followup.json', dict(state='running'))
    with pytest.raises(ValueError, match='reproduction'):
        run_nominal(tmp_path, dict(jobs=[]), resume=True)
    assert not (tmp_path/'reports/control_precision_nominal.json').exists()


def test_prepare_nominal_freezes_all_twenty_coordinates_without_opening_outcomes(tmp_path, monkeypatch):
    import poc.control_precision as module
    root = request_inputs(tmp_path)
    request = module.build_request(root); atomic_json(root/module.REQUEST_PATH, request)
    source = root/'study/control-development'; definitions, parents = context_inputs(source)
    inventory = module.union_nominal_inventory(source, [(p, parents) for p in definitions])
    monkeypatch.setattr(module, '_nominal_inputs', lambda r, q: dict(source='study/control-development',
        inventory=inventory, normalization_by_actuator={'direct': {}, 'lagged': {}}, reuse={}, input_hashes={}))
    registry = module.prepare_nominal(root, request)
    assert len(registry['jobs']) == 20
    assert 'poc/control_precision_result.py' in registry['coordinator_hashes']
    unbound = copy.deepcopy(registry)
    unbound['coordinator_hashes'].pop('poc/control_precision_result.py')
    with pytest.raises(ValueError, match='Complete.*hash'):
        module.verify_nominal_registry(root, unbound)
    assert {(j['definition']['actuator'], j['definition']['model_id'], j['definition']['seed'])
            for j in registry['jobs']} == {(a, m, s) for a in ['direct', 'lagged']
                for m in ['mlp_dt', 'power_correct'] for s in [5, 6, 7, 8, 9]}
    assert registry['new_fits'] == 20 and registry['reused_fits'] == 0
    assert not (root/'reports/control_precision_primary.json').exists()
    assert not (root/module.CONTEXT/'evidence/M30').exists()
    assert read_json(root/module.CONTEXT/'FRESH_TRAINING_CONTEXT.json')['cached_training_copied'] is False
    registry['jobs'].pop()
    with pytest.raises(ValueError): module.verify_nominal_registry(root, registry)


def test_nominal_runner_preserves_failure_and_resumes_verified_selection(tmp_path, monkeypatch):
    import poc.control_precision as module
    reproduction_fixture(tmp_path)
    atomic_json(tmp_path/'config/control_precision_nominal_jobs.json', dict(fixture=True))
    definition = dict(actuator='direct', condition='A', loss_mode='shared', model_id='mlp_dt', seed=5,
                      cell_id='direct__A__shared__mlp_dt__seed5')
    registry = dict(jobs=[dict(cell_id=definition['cell_id'], definition=definition, mode='fit')],
                    context='fresh', request_sha256='a'*64)
    monkeypatch.setattr(module, 'verify_nominal_registry', lambda root, value: value)
    monkeypatch.setattr('poc.io._process_start', lambda pid: 'private-metadata')
    def fail(root, job, registry, attempt):
        attempt.mkdir(parents=True); (attempt/'stderr.log').write_text('training interruption')
        raise RuntimeError('training interruption')
    monkeypatch.setattr(module, '_fit_in_child', fail)
    with pytest.raises(RuntimeError, match='interruption'): module.run_nominal(tmp_path, registry, resume=False)
    receipt = read_json(tmp_path/'reports/control_precision_nominal.json')
    assert receipt['state'] == 'interrupted' and receipt['completed_fits'] == 0
    assert len(list((tmp_path/'reports/control_precision_nominal_attempts').glob('*/stderr.log'))) == 1
    calls = []
    def complete(root, job, registry, attempt):
        calls.append(job['cell_id']); attempt.mkdir(parents=True)
        path = root/'fresh/evidence/M30/fit/manifest.json'; atomic_json(path, dict(fixture=True)); return path
    monkeypatch.setattr(module, '_fit_in_child', complete)
    monkeypatch.setattr(module, '_observe_nominal', lambda root, job, registry, path:
        dict(cell_id=job['cell_id'], actuator='direct', model_id='mlp_dt', seed=5,
             manifest=str(path.relative_to(root)), manifest_sha256=sha256_file(path), qualified=True))
    result = module.run_nominal(tmp_path, registry, resume=True)
    assert result['state'] == 'complete' and result['completed_fits'] == 1
    digest = sha256_file(tmp_path/'reports/control_precision_nominal.json')
    assert module.run_nominal(tmp_path, registry, resume=True) == result
    assert calls == ['direct__A__shared__mlp_dt__seed5']
    assert sha256_file(tmp_path/'reports/control_precision_nominal.json') == digest
    atomic_json(tmp_path/result['selected'][0]['manifest'], dict(modified=True))
    with pytest.raises(ValueError): module.run_nominal(tmp_path, registry, resume=True)


def test_precision_cli_dry_run_does_not_create_training_or_control_outputs(tmp_path, capsys):
    from poc.control_precision import REQUEST_PATH, build_request, main
    root = request_inputs(tmp_path); atomic_json(root/REQUEST_PATH, build_request(root))
    assert main([REQUEST_PATH, '--root', str(root), '--dry-run']) == 0
    import json
    output = json.loads(capsys.readouterr().out)
    assert output['execute'] is False and output['new_control_outcomes_opened'] is False
    assert output['expected_new_primary_episodes'] == 400
    assert output['minimum_new_finer_episodes'] == 100
    assert not (root/'config/control_precision_nominal_jobs.json').exists()
    assert not (root/'reports/control_precision_nominal.json').exists()


def selected_fixture(root, module, monkeypatch):
    """Real hashes and metadata; model restoration is the slow existing boundary."""
    request = module.build_request(root); atomic_json(root/module.REQUEST_PATH, request)
    selected = []
    for actuator in ['direct', 'lagged']:
        for model in ['mlp_dt', 'power_correct']:
            for seed in [5, 6, 7, 8, 9]:
                cell = f'{actuator}__A__shared__{model}__seed{seed}'
                path = root/f'fit/{cell}/manifest.json'
                atomic_json(path, dict(fixture=True)); path.with_name('weights.pt').write_bytes(cell.encode())
                selected.append(dict(cell_id=cell, actuator=actuator, model_id=model, seed=seed, qualified=True,
                    manifest=str(path.relative_to(root)), manifest_sha256=sha256_file(path),
                    checkpoint_sha256=sha256_file(path.with_name('weights.pt')), canonical_tensor_sha256='f'*64))
    registry = dict(fixture=True)
    atomic_json(root/'config/control_precision_nominal_jobs.json', registry)
    result = dict(state='complete', expected_fits=20, completed_fits=20, diagnosis_required=False,
                  registry_sha256=sha256_file(root/'config/control_precision_nominal_jobs.json'), selected=selected)
    atomic_json(root/'reports/control_precision_nominal.json', result)
    from poc.control_jobs import SOURCES
    repository = Path(__file__).resolve().parents[1]
    for name in (*SOURCES, 'poc/control_precision.py', 'poc/control_precision_result.py', 'config/numerical.json', 'assets/two_link_direct.xml',
                 'assets/two_link_lagged.xml', 'config/revisions/owner-objective-2026-10-01.json'):
        path = root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes((repository/name).read_bytes())
    monkeypatch.setattr(module, 'verify_nominal_registry', lambda root, value: value)
    monkeypatch.setattr('poc.control_confirmation._check_selected_fit', lambda root, item: None)
    return request, result


def test_exact_400_and_100_cells_match_checkpoint_freeze(tmp_path, monkeypatch):
    import poc.control_precision as module
    root = request_inputs(tmp_path); request, selected = selected_fixture(root, module, monkeypatch)
    primary = module.prepare_control(root, request, selected)
    finer = module.prepare_finer(root, request, primary)
    assert primary['expected_episodes'] == 400 and len(primary['jobs']) == 400
    assert finer['expected_episodes'] == 100 and len(finer['jobs']) == 100
    seen = set()
    for job in primary['jobs']:
        value = read_json(root/job['path']); seen.add((value['actuator'], value['model_id'], value['seed'], value['episode_id']))
        assert value['model_partition_ns'] == [5_000_000]*8
        assert value['actuator_interval_ns'] == 40_000_000 and value['physics_step_ns'] == 250_000
        assert value['planner_settings']['candidate_count'] == 128
        assert value['planner_settings']['iterations'] == 4 and value['planner_settings']['elite_count'] == 16
    assert seen == {(a, m, s, e) for a in ['direct', 'lagged'] for m in ['mlp_dt', 'power_correct']
                    for s in [5, 6, 7, 8, 9] for e in range(20)}
    by_id = {j['cell_id']: j for j in primary['jobs']}
    for job in finer['jobs']:
        value = read_json(root/job['path']); parent = by_id[job['parent_cell_id']]
        original = read_json(root/parent['path'])
        assert value == dict(original, physics_step_ns=125_000)
        assert value['episode_id'] in [0, 1, 2, 3, 4]
        assert job['parent_request_sha256'] == parent['sha256']
    expanded = module.prepare_finer(root, request, primary, full=True, actuators=('lagged',))
    assert expanded['expected_episodes'] == 200
    assert all(read_json(root/j['path'])['actuator'] == 'lagged' for j in expanded['jobs'])


@pytest.mark.parametrize('mutation', ['checkpoint', 'clock', 'request_hash', 'duplicate', 'missing', 'namespace', 'consumer'])
def test_registry_rejects_changed_request_or_checkpoint_before_dispatch(tmp_path, monkeypatch, mutation):
    import poc.control_precision as module
    root = request_inputs(tmp_path); request, selected = selected_fixture(root, module, monkeypatch)
    registry = module.prepare_control(root, request, selected)
    job = registry['jobs'][0]; path = root/job['path']
    if mutation == 'checkpoint':
        (root/selected['selected'][0]['manifest']).with_name('weights.pt').write_bytes(b'changed')
    elif mutation == 'clock':
        value = read_json(path); value['actuator_interval_ns'] = 20_000_000; atomic_json(path, value)
    elif mutation == 'request_hash': path.write_text('{}')
    elif mutation == 'duplicate': registry['jobs'][-1] = copy.deepcopy(job)
    elif mutation == 'missing': registry['jobs'].pop()
    elif mutation == 'namespace': registry['episode_seed_root'] = 7101
    else: (root/'poc/control_precision_result.py').write_text('changed inference\n')
    with pytest.raises(ValueError): module.verify_episode_registry(root, registry)


def test_resume_reuses_completed_jobs_and_preserves_interruption(tmp_path, monkeypatch):
    import poc.control_precision as module
    reproduction_fixture(tmp_path)
    jobs = []
    for index in [0, 1]:
        path = tmp_path/f'config/episode{index}.json'; atomic_json(path, dict(fixture=index))
        jobs.append(dict(cell_id=f'cell{index}', path=str(path.relative_to(tmp_path)), sha256=sha256_file(path)))
    path = tmp_path/'config/fixture_registry.json'
    registry = dict(kind='primary', registry_path=str(path.relative_to(tmp_path)), jobs=jobs,
                    expected_episodes=2, input_hashes={}, resource_projection={'artifact_bytes': 100})
    atomic_json(path, registry)
    monkeypatch.setattr(module, 'verify_episode_registry', lambda root, value: value)
    monkeypatch.setattr('poc.io._process_start', lambda pid: 'private-metadata')
    calls = []
    def child(root, job, attempt):
        calls.append(job['cell_id']); attempt.mkdir(parents=True)
        if job['cell_id'] == 'cell1' and calls.count('cell1') == 1:
            (attempt/'stderr.log').write_text('interrupted before completing the episode')
            raise RuntimeError('interrupted episode')
        manifest = root/f'evidence/M70/{job["cell_id"]}/manifest.json'
        atomic_json(manifest, dict(completed=True)); return manifest
    def observed(root, job, manifest):
        return dict(cell_id=job['cell_id'], manifest=str(manifest.relative_to(root)), manifest_sha256=sha256_file(manifest),
                    request_sha256=job['sha256'], summary=dict(complete=True, realized_cost=.25, fallback_count=0))
    monkeypatch.setattr(module, '_episode_in_child', child)
    monkeypatch.setattr(module, '_observe_episode', observed)
    with pytest.raises(RuntimeError, match='interrupted'): module.run_episode_registry(tmp_path, registry, resume=False)
    receipt = read_json(tmp_path/'reports/control_precision_primary.json')
    assert receipt['state'] == 'interrupted' and receipt['completed_episodes'] == 1
    assert len(list((tmp_path/'reports/control_precision_episode_attempts').glob('*/stderr.log'))) == 1
    result = module.run_episode_registry(tmp_path, registry, resume=True)
    assert result['state'] == 'complete' and result['completed_episodes'] == 2
    assert calls == ['cell0', 'cell1', 'cell1']
    assert module.run_episode_registry(tmp_path, registry, resume=True) == result
    assert calls == ['cell0', 'cell1', 'cell1']
    value = copy.deepcopy(result); value['episodes'].pop(); atomic_json(tmp_path/'reports/control_precision_primary.json', value)
    with pytest.raises(ValueError): module.run_episode_registry(tmp_path, registry, resume=True)


def test_live_episode_lock_prevents_duplicate_dispatch(tmp_path, monkeypatch):
    import poc.control_precision as module
    from poc.io import job_key, job_lock
    reproduction_fixture(tmp_path)
    registry = dict(registry_path='config/fixture_registry.json', kind='primary', jobs=[])
    atomic_json(tmp_path/registry['registry_path'], registry)
    monkeypatch.setattr('poc.io._process_start', lambda pid: 'private-metadata')
    key = job_key({'control_precision_episode_registry': sha256_file(tmp_path/registry['registry_path'])})
    with job_lock(key, tmp_path):
        with pytest.raises(RuntimeError, match='active|live|running|lock'):
            module.run_episode_registry(tmp_path, registry, resume=True)
    assert not (tmp_path/'reports/control_precision_primary.json').exists()


def test_finer_dispatch_uses_distinct_scope_receipt(tmp_path, monkeypatch):
    import poc.control_precision as module
    reproduction_fixture(tmp_path)
    path = tmp_path/'config/control_precision_finer_jobs.json'
    request_path = tmp_path/'config/finer_fixture.json'; atomic_json(request_path, dict(fixture=True))
    job = dict(cell_id='finer', path=str(request_path.relative_to(tmp_path)), sha256=sha256_file(request_path))
    registry = dict(kind='finer', registry_path=str(path.relative_to(tmp_path)), jobs=[job],
                    input_hashes={}, resource_projection=dict(artifact_bytes=100))
    atomic_json(path, registry)
    monkeypatch.setattr(module, 'verify_episode_registry', lambda root, value: value)
    monkeypatch.setattr('poc.io._process_start', lambda pid: 'private-metadata')
    def child(root, job, attempt):
        manifest = root/'evidence/M70/finer/manifest.json'; atomic_json(manifest, dict(fixture=True)); return manifest
    monkeypatch.setattr(module, '_episode_in_child', child)
    monkeypatch.setattr(module, '_observe_episode', lambda root, job, manifest:
        dict(cell_id=job['cell_id'], manifest=str(manifest.relative_to(root)), manifest_sha256=sha256_file(manifest)))
    result = module.run_episode_registry(tmp_path, registry, resume=False)
    assert read_json(tmp_path/'reports/control_precision_finer.json') == result
    assert not (tmp_path/'reports/control_precision_primary.json').exists()


def test_real_mujoco_child_preserves_manifest_raw_cost_and_producer_import_boundary(tmp_path):
    import subprocess
    import poc.control_precision as module
    from poc.control_jobs import SOURCES
    repository = Path(__file__).resolve().parents[1]
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True, capture_output=True)
    subprocess.run(['git', '-c', 'user.name=PoC Executor', '-c', 'user.email=poc-executor@example.invalid',
                    'commit', '-q', '--allow-empty', '-m', 'Synthetic integration fixture'],
                   cwd=tmp_path, check=True, capture_output=True)
    for name in (*SOURCES, 'locks/environment.json', 'config/base.json', 'config/numerical.json',
                 'config/planner.json', 'assets/two_link_direct.xml',
                 'config/revisions/owner-objective-2026-10-01.json'):
        path = tmp_path/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes((repository/name).read_bytes())
    for path in (repository/'schemas').glob('*.json'):
        target = tmp_path/'schemas'/path.name; target.parent.mkdir(exist_ok=True); target.write_bytes(path.read_bytes())
    atomic_json(tmp_path/'locks/wheels.json', dict(wheels=[]))
    settings = dict(candidate_count=8, iterations=1, elite_count=1, initial_std=1., std_floor=.1)
    planner = read_json(tmp_path/'config/planner.json'); planner['settings'] = settings
    atomic_json(tmp_path/'config/planner.json', planner)
    value = dict(schema_version=1, phase='confirmatory', model_id='zero_command', actuator='direct', seed=5,
                 episode_id=0, condition='clean', model_partition_ns=[40_000_000], actuator_interval_ns=40_000_000,
                 physics_step_ns=250_000, power_mode=None, model_manifest=None, manifest_sha256=None,
                 checkpoint_sha256=None, planner_settings=settings,
                 resource_projection=dict(basis='Small real-plant software fixture, no scientific inference'))
    path = tmp_path/'config/fixture_episode.json'; atomic_json(path, value)
    job = dict(cell_id='fixture', path=str(path.relative_to(tmp_path)), sha256=sha256_file(path))
    manifest = module._episode_in_child(tmp_path, job, tmp_path/'attempts/first')
    observed = module._observe_episode(tmp_path, job, manifest)
    assert observed['raw_audit']['n_observations'] == 40
    assert observed['raw_audit']['n_real_anchors'] == 40
    assert observed['summary']['complete'] is True
    assert observed['raw_audit']['raw_cost'] == pytest.approx(observed['summary']['realized_cost'], abs=1e-12)
    record = read_json(manifest)
    assert 'poc/control_precision.py' not in record['job_inputs']['dependencies']
    assert read_json(tmp_path/'attempts/first/attempt.json')['returncode'] == 0
    assert module._episode_in_child(tmp_path, job, tmp_path/'attempts/second') == manifest
    atomic_json(path, dict(value, episode_id=1))
    with pytest.raises(ValueError): module._observe_episode(tmp_path, job, manifest)
