"""Own-solver study fits, bounded exposures, and committed optimizer recovery."""
import copy
import importlib.util
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from poc.calibration_models import build_model, model_metadata
from poc.calibration_study import prepare, open_store, collect_session, prospective_recipe
from poc.models.common import Scales
from poc.train import array_scales, canonical_tensor_hash

ROOT = Path(__file__).resolve().parents[1]


def training_module():
    assert importlib.util.find_spec('poc.calibration_training'), 'Calibration training module is missing'
    from poc import calibration_training
    return calibration_training


def arrays(count=12):
    rng = np.random.default_rng(6)
    return {'state': rng.normal(0, .1, (count, 26, 6)),
            'command': rng.uniform(-.3, .3, (count, 25, 2)),
            'duration_ns': np.full((count, 25), 40_000_000, dtype=np.int64),
            'time_ns': np.tile(np.arange(26, dtype=np.int64)*40_000_000, (count, 1)),
            'trajectory_id': np.arange(count, dtype=np.int64)}


def session(model_id='affine_flow', count=12, data=None, scales=None, rate=.001):
    module = training_module()
    data = arrays(count) if data is None else data
    scales = array_scales(data) if scales is None else scales
    return module.CalibrationTrainingSession(build_model(model_id, scales, 10), data, scales, 10,
                                             learning_rate=rate)


def same_nested(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            same_nested(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            same_nested(x, y)
    else:
        assert a == b


def test_flat_sampler_counts_every_start_once_and_short_batch():
    fit = session()
    starts = fit.starts.copy()
    assert len(starts) == len({tuple(x) for x in starts}) == 300
    assert set(map(tuple, starts)) == {(i, j) for i in range(12) for j in range(25)}
    fit.advance_updates(2)
    assert fit.cursor == 300 and fit.epoch == 1
    assert [r['one_count'] for r in fit.training_rows] == [256, 44]
    assert sum(r['four_count'] for r in fit.training_rows) == 264
    fit.advance_updates(1)
    assert fit.epoch == 2 and fit.cursor == 256
    assert fit.valid_target_count == 556


@pytest.mark.parametrize('model_id', ['affine_flow', 'affine_rk4', 'node_structured',
                                      'node_quadrature', 'node_actuator_aware', 'physics_residual'])
def test_supervision_calls_own_predict_for_one_and_four_steps(model_id):
    fit = session(model_id, count=1)
    real_predict = fit.model.predict
    calls = []
    def predict(*args, **kwargs):
        calls.append((len(args[0]), torch.is_grad_enabled()))
        return real_predict(*args, **kwargs)
    fit.model.predict = predict
    fit.advance_updates(1)
    assert calls == [(25, True), (22, True), (22, True), (22, True), (22, True)]
    assert fit.training_rows[0]['one_count'] == 25
    assert fit.training_rows[0]['four_count'] == 22


def test_arbitrary_update_resume_reproduces_tensors_optimizer_rng_and_exposure(tmp_path):
    whole, split = session(), session()
    whole.advance_updates(5)
    split.advance_updates(1)
    path = tmp_path / 'resume.pt'
    split.save(path)
    torch.rand(3); random.random()
    restored = session()
    restored.restore(path)
    restored.advance_updates(4)
    assert canonical_tensor_hash(whole.model.state_dict()) == canonical_tensor_hash(restored.model.state_dict())
    same_nested(whole.optimizer.state_dict(), restored.optimizer.state_dict())
    assert whole.rng.bit_generator.state == restored.rng.bit_generator.state
    assert whole.training_rows == restored.training_rows
    assert whole.valid_target_count == restored.valid_target_count
    np.testing.assert_array_equal(whole.starts, restored.starts)
    assert (whole.cursor, whole.epoch, whole.updates) == (restored.cursor, restored.epoch, restored.updates)


@pytest.mark.parametrize('change', ['data', 'scales', 'solver', 'rate', 'seed'])
def test_resume_rejects_changed_training_identity(tmp_path, change):
    fit = session()
    fit.advance_updates(1)
    path = tmp_path / 'resume.pt'
    fit.save(path)
    data = arrays()
    scales = array_scales(data)
    if change == 'data':
        data['state'][0, 0, 0] += .1
    if change == 'scales':
        scales = Scales(scales.q*2, scales.v, scales.v_input)
    model = build_model('affine_rk4' if change == 'solver' else 'affine_flow', scales, 10)
    restored = training_module().CalibrationTrainingSession(model, data, scales, 11 if change == 'seed' else 10,
                                                            learning_rate=.0003 if change == 'rate' else .001)
    with pytest.raises(ValueError, match='identity|recipe|inputs'):
        restored.restore(path)


@pytest.fixture
def unit_store(tmp_path):
    output = tmp_path / 'unit'
    prepare(ROOT, output, unit_fixture={'segment_count': 5, 'max_epochs_per_rate': 5, 'physical_epochs': 5})
    receipt = collect_session(0, root=ROOT, output_root=output)
    return open_store(ROOT, output), receipt


def test_two_rates_selection_finite_poor_fit_and_float64_load(unit_store):
    module = training_module()
    store, collected = unit_store
    receipt = module.fit_model(store, collected, 'affine_rk4', 5, 0)
    assert module.fit_model(store, collected, 'affine_rk4', 5, 0) == receipt
    _, manifest = store.read(receipt)
    result = manifest['summary']
    assert [r['learning_rate'] for r in result['attempts']] == [.001, .0003]
    assert all(r['epochs_completed'] == 5 and r['ceiling_hit'] for r in result['attempts'])
    assert all([row['epoch'] for row in r['validation_history']] == [5] for r in result['attempts'])
    assert all(r['valid_target_exposures'] == 500 for r in result['attempts'])
    assert result['fit_status'] == 'finite' and result['selection_kind'] == 'fitted'
    assert result['empirical_claims_allowed'] is False
    model = module.load_selected(store, receipt)
    assert next(model.parameters()).dtype == torch.float64
    assert model_metadata(model)['model_id'] == 'affine_rk4'


@pytest.mark.parametrize('failing_rates', [{.001}, {.001, .0003}])
def test_allowed_rate_failure_is_retained_and_slower_rate_recovers(unit_store, monkeypatch, failing_rates):
    module = training_module()
    store, collected = unit_store
    advance = module.CalibrationTrainingSession.advance_updates
    def fail(self, count):
        if self.recipe['learning_rate'] in failing_rates:
            raise ValueError('unit injected numerical failure')
        advance(self, count)
    monkeypatch.setattr(module.CalibrationTrainingSession, 'advance_updates', fail)
    receipt = module.fit_model(store, collected, 'node_quadrature', 5, 0)
    _, manifest = store.read(receipt)
    summary = manifest['summary']
    assert len(summary['attempts']) == 2
    assert sum(a['status'] == 'failed' for a in summary['attempts']) == len(failing_rates)
    for a in summary['attempts']:
        if a['status'] == 'failed':
            assert a['error_type'] == 'ValueError' and 'injected' in a['error']
    if len(failing_rates) == 1:
        assert summary['selected']['learning_rate'] == .0003
        module.load_selected(store, receipt)
    else:
        assert summary['fit_status'] == 'failed' and summary['selected'] is None
        with pytest.raises(ValueError, match='failed|selected'):
            module.load_selected(store, receipt)


def test_nominal_reference_has_no_trained_checkpoint_and_is_not_a_fit(unit_store):
    module = training_module()
    store, collected = unit_store
    receipt = module.nominal_reference(store, collected, 5)
    folder, manifest = store.read(receipt)
    assert manifest['stage'] == 'nominal'
    assert manifest['summary']['selection_kind'] == 'unfitted_reference'
    assert not (folder / 'weights.pt').exists()
    model = module.load_selected(store, receipt)
    assert all(not p.requires_grad for p in model.parameters())
    torch.testing.assert_close(model.physical_parameters(), torch.tensor([.10, .02, .02, .05, .05], dtype=torch.float64))


def test_physical_once_per_session_budget_and_solver_audits(unit_store):
    module = training_module()
    store, collected = unit_store
    receipt = module.fit_model(store, collected, 'physics_fit', 5, None)
    assert module.fit_model(store, collected, 'physics_fit', 5, None) == receipt
    with pytest.raises(ValueError, match='initialization|coordinate'):
        module.fit_model(store, collected, 'physics_fit', 5, 0)
    _, manifest = store.read(receipt)
    summary = manifest['summary']
    assert summary['solver_audits_before_fit'] and summary['solver_audits_selected_field']
    assert summary['valid_target_exposures'] == 500
    assert summary['integration_step_s'] > 0
    module.load_selected(store, receipt)


def test_unit_training_controls_cannot_enter_empirical_recipes():
    assert prospective_recipe()['training']['max_epochs_per_rate'] == 1000
    recipe = prospective_recipe(unit_fixture={'max_epochs_per_rate': 5, 'physical_epochs': 5})
    assert recipe['phase'] == 'unit' and not recipe['empirical_claims_allowed']
    with pytest.raises(ValueError):
        prospective_recipe(unit_fixture={'max_epochs_per_rate': 6})


@pytest.mark.parametrize('plateau, maximum, expected_epochs', [(True, 1000, 100), (False, 1000, 1000)])
def test_real_training_policy_validates_every_five_and_stops_at_plateau_or_ceiling(tmp_path, monkeypatch,
                                                                                plateau, maximum, expected_epochs):
    module = training_module()
    data = arrays(1)
    scales = array_scales(data)
    validations = []
    def scheduled_validation(model, valid, scales, loss_mode):
        assert not any(p.grad is None for p in model.parameters())
        epoch = 5*(len(validations)+1)
        validations.append(epoch)
        objective = 1. if plateau else 1./epoch
        return {'objective': objective, 'one_step': objective, 'four_step': 0., 'one_count': 25, 'four_count': 22}
    monkeypatch.setattr(module, 'validate_nominal', scheduled_validation)
    result = module._rate_attempt(tmp_path / 'rate', {'model_id': 'node_quadrature'}, data, data,
                                  scales, 10, .001, maximum)
    assert validations == list(range(5, expected_epochs+1, 5))
    assert result['epochs_completed'] == expected_epochs
    assert result['plateau'] is plateau and result['ceiling_hit'] is (not plateau)
    assert result['valid_target_exposures'] == 25*expected_epochs
    assert result['four_step_start_exposures'] == 22*expected_epochs
    assert result['selected']['epoch'] == (5 if plateau else 1000)


def test_rate_tie_keeps_first_rate_and_earliest_checkpoint(unit_store, monkeypatch):
    module = training_module()
    store, collected = unit_store
    monkeypatch.setattr(module, 'validate_nominal', lambda *a, **kw: {
        'objective': 100., 'one_step': 80., 'four_step': 80., 'one_count': 25, 'four_count': 22})
    receipt = module.fit_model(store, collected, 'affine_flow', 5, 0)
    _, manifest = store.read(receipt)
    summary = manifest['summary']
    assert summary['fit_status'] == 'finite'
    assert summary['selected']['learning_rate'] == .001 and summary['selected']['epoch'] == 5
    assert summary['selected']['objective'] == 100.  # Poor fits survive historical qualification thresholds.
    module.load_selected(store, receipt)


@pytest.mark.parametrize('model_id', ['affine_flow', 'node_quadrature', 'node_actuator_aware'])
def test_full_fit_interruption_restores_selected_tensors_and_exposures(unit_store, tmp_path, monkeypatch, model_id):
    module = training_module()
    store, collected = unit_store
    ordinary = module.fit_model(store, collected, model_id, 5, 0)
    _, original = store.read(ordinary)
    # A separate labeled store uses the same prospective acquisition and seed recipe.
    output = tmp_path / 'interrupted-unit'
    prepare(ROOT, output, unit_fixture={'segment_count': 5, 'max_epochs_per_rate': 5, 'physical_epochs': 5})
    second_session = collect_session(0, root=ROOT, output_root=output)
    second_store = open_store(ROOT, output)
    advance = module.CalibrationTrainingSession.advance_updates
    interrupted = False
    def interrupt(self, count):
        nonlocal interrupted
        if self.epoch == 3 and not interrupted:
            interrupted = True
            raise KeyboardInterrupt('unit interruption after committed epoch')
        advance(self, count)
    with monkeypatch.context() as patch:
        patch.setattr(module.CalibrationTrainingSession, 'advance_updates', interrupt)
        with pytest.raises(KeyboardInterrupt):
            module.fit_model(second_store, second_session, model_id, 5, 0)
    resumed = module.fit_model(second_store, second_session, model_id, 5, 0)
    _, reproduced = second_store.read(resumed)
    assert original['summary']['canonical_tensor_sha256'] == reproduced['summary']['canonical_tensor_sha256']
    assert original['summary']['selected'] == reproduced['summary']['selected']
    for a, b in zip(original['summary']['attempts'], reproduced['summary']['attempts']):
        assert a['validation_history'] == b['validation_history']
        assert a['valid_target_exposures'] == b['valid_target_exposures'] == 500
        assert a['four_step_start_exposures'] == b['four_step_start_exposures'] == 440
    for receipt, owning_store in ((ordinary, store), (resumed, second_store)):
        folder, _ = owning_store.read(receipt)
        packet = module._read_packet(folder / 'rate-0' / 'resume.pt')
        if receipt is ordinary:
            saved = packet
        else:
            same_nested(saved['optimizer'], packet['optimizer'])
            assert saved['numpy_rng'] == packet['numpy_rng']
            assert saved['training_rows'] == packet['training_rows']


def reclose_receipt(store, receipt, manifest):
    """Simulate a self-consistent outer receipt around semantically wrong weights."""
    from poc.io import atomic_json, sha256_file
    folder = store.output_root / receipt['path']
    manifest['files'] = {str(p.relative_to(folder)): sha256_file(p) for p in folder.rglob('*')
                         if p.is_file() and p.name not in {'manifest.json', 'COMPLETED'}}
    atomic_json(folder / 'manifest.json', manifest)
    digest = sha256_file(folder / 'manifest.json')
    atomic_json(folder / 'COMPLETED', {'manifest_sha256': digest})
    return dict(receipt, manifest_sha256=digest)


@pytest.mark.parametrize('substitution', ['solver', 'full_budget', 'canonical'])
def test_selected_rejects_checkpoint_identity_substitution_even_with_closed_outer_hashes(unit_store, substitution):
    module = training_module()
    store, collected = unit_store
    receipt = module.fit_model(store, collected, 'affine_flow', 5, 0)
    folder, manifest = store.read(receipt)
    packet = module._read_packet(folder / 'weights.pt')
    if substitution == 'solver':
        packet['request']['model']['solver']['name'] = 'rk4'
    elif substitution == 'full_budget':
        packet['weights']['scale_q'] *= 2
        # Even canonical hashes agree with the replacement: budget scales still must match.
        packet['canonical_tensor_sha256'] = canonical_tensor_hash(packet['weights'])
        manifest['summary']['canonical_tensor_sha256'] = packet['canonical_tensor_sha256']
    else:
        packet['weights']['network.0.weight'][0, 0] += .5
    module._write_packet(folder / 'weights.pt', packet)
    forged = reclose_receipt(store, receipt, manifest)
    store.read(forged)  # This is an identity failure, not a damaged outer receipt.
    with pytest.raises(ValueError, match='identity|canonical|normalization'):
        module.load_selected(store, forged)


def test_validation_labels_do_not_enter_optimizer_or_scales(unit_store):
    module = training_module()
    store, collected = unit_store
    from poc.io import atomic_bytes
    import io
    folder, manifest = store.read(collected)
    with np.load(folder / 'session.npz') as loaded:
        data = {k: loaded[k].copy() for k in loaded.files}
    data['state'][4, :, :4] += 2.  # The validation segment only.
    def alternate(folder):
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **data)
        atomic_bytes(folder / 'session.npz', buffer.getvalue())
        return manifest['summary']
    alternate_receipt = store.produce('collect', dict(manifest['request'], unit_fixture_case='validation shift'), alternate)
    a = module.fit_model(store, collected, 'node_actuator_aware', 5, 0)
    b = module.fit_model(store, alternate_receipt, 'node_actuator_aware', 5, 0)
    af, am = store.read(a); bf, bm = store.read(b)
    assert am['request']['scale_hash'] == bm['request']['scale_hash']
    assert am['request']['training_data_hash'] == bm['request']['training_data_hash']
    assert am['request']['validation_data_hash'] != bm['request']['validation_data_hash']
    for index in range(2):
        x = module._read_packet(af / f'rate-{index}' / 'resume.pt')
        y = module._read_packet(bf / f'rate-{index}' / 'resume.pt')
        same_nested(x['weights'], y['weights'])
        same_nested(x['optimizer'], y['optimizer'])
        assert x['training_rows'] == y['training_rows']
    assert am['summary']['selected']['objective'] != bm['summary']['selected']['objective']


def test_physical_failed_fit_keeps_method_failure(unit_store, monkeypatch):
    module = training_module()
    store, collected = unit_store
    def fail(*args, **kwargs):
        raise ValueError('synthetic endpoint solver refinement failure')
    monkeypatch.setattr(module, 'optimize_endpoints', fail)
    receipt = module.fit_model(store, collected, 'physics_fit', 5, None)
    _, manifest = store.read(receipt)
    assert manifest['summary']['fit_status'] == 'failed'
    assert 'refinement failure' in manifest['summary']['attempts'][0]['error']
    with pytest.raises(ValueError, match='[Ff]ailed'):
        module.load_selected(store, receipt)


def test_physical_synthetic_endpoints_recover_prior_and_qualified_solver(unit_store):
    module = training_module()
    store, collected = unit_store
    from poc.io import atomic_bytes
    import io
    folder, manifest = store.read(collected)
    with np.load(folder / 'session.npz') as file:
        data = {k: file[k].copy() for k in file.files}
    scales = Scales(*(torch.ones(2) for _ in range(3)))
    prior = build_model('physics_nominal', scales, 0)
    with torch.no_grad():
        state = torch.tensor(data['state'][:, 0], dtype=torch.float32)
        data['state'][:, 0] = state.numpy()
        for index in range(25):
            state = prior.predict(state, torch.tensor(data['command'][:, index], dtype=torch.float32),
                                  .04, integration_step_s=.000625).next_state
            data['state'][:, index+1] = state.numpy()
    def synthetic(folder):
        buffer = io.BytesIO(); np.savez_compressed(buffer, **data)
        atomic_bytes(folder / 'session.npz', buffer.getvalue())
        return manifest['summary']
    synthetic_receipt = store.produce('collect', dict(manifest['request'], unit_fixture_case='known prior'), synthetic)
    fitted = module.fit_model(store, synthetic_receipt, 'physics_fit', 5, None)
    model = module.load_selected(store, fitted)
    _, result = store.read(fitted)
    assert result['summary']['best_validation_loss'] < .01
    assert result['summary']['solver_audits_before_fit'][-1]['max_normalized_endpoint_difference'] < .005
    assert result['summary']['solver_audits_selected_field'][-1]['max_normalized_endpoint_difference'] < .005
    torch.testing.assert_close(model.physical_parameters(), prior.double().physical_parameters(), atol=.003, rtol=.05)


def test_committed_packet_recovers_previous_update_after_torn_write(tmp_path):
    fit = session('node_quadrature')
    fit.advance_updates(1)
    previous_hash = canonical_tensor_hash(fit.model.state_dict())
    path = tmp_path / 'resume.pt'
    fit.save(path)
    fit.advance_updates(1)
    fit.save(path)
    path.write_bytes(b'incomplete write')
    restored = session('node_quadrature')
    restored.restore(path)
    assert restored.updates == 1 and restored.valid_target_count == 256
    assert canonical_tensor_hash(restored.model.state_dict()) == previous_hash


def test_resume_restores_python_and_torch_rng_and_selected_packet(tmp_path):
    fit = session('node_actuator_aware', count=1)
    fit.advance_updates(1)
    fit.validation_rows = [{'epoch': 1, 'objective': 1.}]
    fit.best_weights = copy.deepcopy(fit.model.state_dict())
    path = tmp_path / 'resume.pt'
    fit.save(path)
    expected_torch, expected_python = torch.rand(3), random.random()
    torch.rand(30); random.random()
    resumed = session('node_actuator_aware', count=1)
    resumed.restore(path)
    assert torch.equal(torch.rand(3), expected_torch)
    assert random.random() == expected_python
    assert resumed.validation_rows == [{'epoch': 1, 'objective': 1.}]
    same_nested(resumed.best_weights, fit.best_weights)


def test_physical_whole_fit_resume_matches_selected_parameters_and_solver(unit_store, tmp_path, monkeypatch):
    module = training_module()
    store, collected = unit_store
    now=[0.]
    monkeypatch.setattr(module.time,'perf_counter',lambda:now[0])
    monkeypatch.setattr(module.time,'process_time',lambda:now[0])
    real_optimize=module.optimize_endpoints
    def measured(*args,**kwargs):
        commit=kwargs['on_epoch']
        def epoch(packet):
            now[0]+=10.
            commit(packet)
        kwargs['on_epoch']=epoch
        return real_optimize(*args,**kwargs)
    monkeypatch.setattr(module,'optimize_endpoints',measured)
    original = module.fit_model(store, collected, 'physics_fit', 5, None)
    _, reference = store.read(original)
    output = tmp_path / 'physical-interrupted-unit'
    prepare(ROOT, output, unit_fixture={'segment_count': 5, 'max_epochs_per_rate': 5, 'physical_epochs': 5})
    second_session = collect_session(0, root=ROOT, output_root=output)
    second_store = open_store(ROOT, output)
    optimize = module.optimize_endpoints
    def interrupted(*args, **kwargs):
        commit = kwargs['on_epoch']
        def save_then_interrupt(packet):
            commit(packet)
            if packet['epoch'] == 3:
                raise KeyboardInterrupt('physical optimizer interruption after saved epoch')
        kwargs['on_epoch'] = save_then_interrupt
        return optimize(*args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(module, 'optimize_endpoints', interrupted)
        with pytest.raises(KeyboardInterrupt):
            module.fit_model(second_store, second_session, 'physics_fit', 5, None)
    receipt = module.fit_model(second_store, second_session, 'physics_fit', 5, None)
    _, reproduced = second_store.read(receipt)
    assert reproduced['resources']['wall_s']==reproduced['resources']['cpu_s']==50.
    assert len(reproduced['resources']['attempts'])==2
    assert reproduced['resources']['attempts'][0]['boundary_count']>=3
    assert reference['summary']['canonical_tensor_sha256'] == reproduced['summary']['canonical_tensor_sha256']
    for key in ('best_epoch', 'best_validation_loss', 'valid_target_exposures', 'optimizer_updates',
                'solver_audits_before_fit', 'solver_audits_selected_field', 'integration_step_s', 'evaluation_step_s'):
        assert reference['summary'][key] == reproduced['summary'][key]


def test_failed_rate_preserves_failure_coordinate_and_finite_checkpoint(unit_store, monkeypatch):
    module = training_module()
    store, collected = unit_store
    # First rate reaches a valid selection then its final diagnostic write fails.
    write = module._write_packet
    tripped = False
    def fail_late(path, packet):
        nonlocal tripped
        if Path(path).parent.name == 'rate-0' and packet.get('validated_epoch') == 5 and not tripped:
            tripped = True
            raise RuntimeError('unit failure after nominal checkpoint')
        return write(path, packet)
    monkeypatch.setattr(module, '_write_packet', fail_late)
    receipt = module.fit_model(store, collected, 'node_quadrature', 5, 0)
    _, manifest = store.read(receipt)
    attempt = manifest['summary']['attempts'][0]
    assert attempt['status'] == 'failed' and attempt['selected']['epoch'] == 5
    assert attempt['failure_coordinate'] == {'epoch': 5, 'cursor': 100, 'optimizer_updates': 5}
    assert attempt['training_history'][-1]['one_count'] == 100
    module.load_selected(store, receipt)


def test_load_rejects_mislabeled_budget_accounting_even_when_outer_receipt_is_closed(unit_store):
    module = training_module()
    store, collected = unit_store
    receipt = module.nominal_reference(store, collected, 5)
    _, manifest = store.read(receipt)
    manifest['summary']['account']['budget_s'] = 600
    forged = reclose_receipt(store, receipt, manifest)
    with pytest.raises(ValueError, match='identity|account'):
        module.load_selected(store, forged)


def test_load_rejects_nested_residual_normalization_from_another_budget(unit_store):
    module = training_module()
    store, collected = unit_store
    receipt = module.fit_model(store, collected, 'physics_residual', 5, 0)
    folder, manifest = store.read(receipt)
    packet = module._read_packet(folder / 'weights.pt')
    packet['weights']['nominal.scale_q'] *= 2
    packet['canonical_tensor_sha256'] = canonical_tensor_hash(packet['weights'])
    manifest['summary']['canonical_tensor_sha256'] = packet['canonical_tensor_sha256']
    module._write_packet(folder / 'weights.pt', packet)
    forged = reclose_receipt(store, receipt, manifest)
    with pytest.raises(ValueError, match='normalization'):
        module.load_selected(store, forged)


@pytest.mark.parametrize('model_id', ['node_quadrature', 'physics_fit'])
def test_first_ever_interrupted_packet_marker_recovers_full_fit(unit_store, tmp_path, monkeypatch, model_id):
    module = training_module()
    store, collected = unit_store
    initialization = None if model_id == 'physics_fit' else 0
    original = module.fit_model(store, collected, model_id, 5, initialization)
    _, reference = store.read(original)
    output = tmp_path / 'first-write-interrupted-unit'
    prepare(ROOT, output, unit_fixture={'segment_count': 5, 'max_epochs_per_rate': 5, 'physical_epochs': 5})
    second_session = collect_session(0, root=ROOT, output_root=output)
    second_store = open_store(ROOT, output)
    marker_name = 'physical-resume.complete.json' if model_id == 'physics_fit' else 'resume.complete.json'
    atomic = module.atomic_json
    interrupted_path = None
    def interrupt_marker(path, value):
        nonlocal interrupted_path
        if Path(path).name == marker_name and interrupted_path is None:
            interrupted_path = Path(path).with_suffix('.pt').with_name(
                'physical-resume.pt' if model_id == 'physics_fit' else 'resume.pt')
            assert interrupted_path.exists()
            assert not Path(path).exists()
            raise KeyboardInterrupt('first-ever packet marker write interrupted')
        return atomic(path, value)
    with monkeypatch.context() as patch:
        patch.setattr(module, 'atomic_json', interrupt_marker)
        with pytest.raises(KeyboardInterrupt):
            module.fit_model(second_store, second_session, model_id, 5, initialization)
    assert interrupted_path.exists()
    resumed = module.fit_model(second_store, second_session, model_id, 5, initialization)
    folder, reproduced = second_store.read(resumed)
    assert reference['summary']['canonical_tensor_sha256'] == reproduced['summary']['canonical_tensor_sha256']
    assert reference['summary']['selected'] == reproduced['summary']['selected']
    assert list(folder.rglob('*.interrupted-*.pt'))
    import json
    failures = [json.loads(line) for line in (output / 'events.jsonl').read_text().splitlines()
                if json.loads(line)['event'] == 'failed']
    assert any(e['error_type'] == 'KeyboardInterrupt' for e in failures)
    if model_id == 'physics_fit':
        assert reproduced['summary']['valid_target_exposures'] == 500
        assert reproduced['summary']['training_history'] == reference['summary']['training_history']
    else:
        for a, b in zip(reference['summary']['attempts'], reproduced['summary']['attempts']):
            assert a['valid_target_exposures'] == b['valid_target_exposures'] == 500
            assert a['validation_history'] == b['validation_history']


@pytest.mark.parametrize('model_id', ['node_quadrature', 'physics_fit'])
@pytest.mark.parametrize('fault', ['corrupt', 'missing'])
def test_corrupt_first_committed_packet_is_not_silently_restarted(unit_store, monkeypatch, model_id, fault):
    module = training_module()
    store, collected = unit_store
    initialization = None if model_id == 'physics_fit' else 0
    write = module._write_packet
    committed_path = None
    def commit_then_interrupt(path, packet):
        nonlocal committed_path
        result = write(path, packet)
        if Path(path).name in {'resume.pt', 'physical-resume.pt'} and committed_path is None:
            committed_path = Path(path)
            raise KeyboardInterrupt('first committed boundary interrupted')
        return result
    with monkeypatch.context() as patch:
        patch.setattr(module, '_write_packet', commit_then_interrupt)
        with pytest.raises(KeyboardInterrupt):
            module.fit_model(store, collected, model_id, 5, initialization)
    if fault == 'missing':
        committed_path.unlink()
    else:
        committed_path.write_bytes(b'corrupt committed optimizer state')
    with pytest.raises(ValueError, match='consistent committed'):
        module.fit_model(store, collected, model_id, 5, initialization)


def test_physical_scientific_audits_execute_float64_with_float32_optimization(unit_store, monkeypatch):
    module = training_module()
    store, collected = unit_store
    from poc.physics_fit import PhysicsFit
    real = PhysicsFit.predict
    calls = []
    def observe(self, state, command, h_s, *args, **kwargs):
        calls.append((self.raw.dtype, state.dtype, command.dtype, torch.as_tensor(h_s).dtype,
                      torch.is_grad_enabled()))
        return real(self, state, command, h_s, *args, **kwargs)
    monkeypatch.setattr(PhysicsFit, 'predict', observe)
    receipt = module.fit_model(store, collected, 'physics_fit', 5, None)
    _, manifest = store.read(receipt)
    summary = manifest['summary']
    assert summary['fit_status'] == 'finite'
    scientific = [c for c in calls if c[0] == torch.float64]
    assert len(scientific) >= 4, 'Prior and selected solver refinements must execute in float64'
    assert all(c == (torch.float64, torch.float64, torch.float64, torch.float64, False) for c in scientific)
    updates = [c for c in calls if c[-1]]
    assert updates and all(c[:4] == (torch.float32,)*4 for c in updates)
    assert summary['solver_audit_dtype'] == 'float64'
    assert summary['optimizer_solver_audits_before_fit']
    assert summary['optimizer_solver_audits_selected_field']


def test_float64_selected_audit_cannot_silently_replace_the_trained_physical_solver(unit_store, monkeypatch):
    module = training_module()
    store, collected = unit_store
    audit = module._audit_physical
    count = 0
    def stricter_selected(model, data):
        nonlocal count
        count += 1
        step, rows = audit(model, data)
        if count == 2:
            # An independently qualified field can require more resolution than training used.
            return .000001, rows
        return step, rows
    monkeypatch.setattr(module, '_audit_physical', stricter_selected)
    receipt = module.fit_model(store, collected, 'physics_fit', 5, None)
    folder, manifest = store.read(receipt)
    summary = manifest['summary']
    assert summary['fit_status'] == 'failed' and summary['selected'] is None
    assert 'finer training solver' in summary['attempts'][0]['error']
    assert summary['solver_audit_dtype'] == 'float64'
    assert summary['solver_audits_selected_field']
    assert summary['optimizer_solver_audits_selected_field']
    assert not (folder / 'weights.pt').exists()
    assert module._read_packet(folder / 'physical-resume.pt')['resume']['integration_step_s'] > .000001
    with pytest.raises(ValueError, match='[Ff]ailed'):
        module.load_selected(store, receipt)


@pytest.mark.parametrize('interrupt_rate', [0,1])
def test_actual_rate_resume_resources_include_completed_rate_reuse(tmp_path,monkeypatch,interrupt_rate):
    m=training_module(); data=arrays(1); scales=array_scales(data)
    request={'model_id':'node_actuator_aware'}; now=[0.]
    monkeypatch.setattr(m.time,'perf_counter',lambda:now[0])
    monkeypatch.setattr(m.time,'process_time',lambda:now[0])
    advance=m.CalibrationTrainingSession.advance_updates
    triggered=[False]
    def measured(self,count):
        advance(self,count);now[0]+=10
    monkeypatch.setattr(m.CalibrationTrainingSession,'advance_updates',measured)
    reference=[m._rate_attempt(tmp_path/'reference'/str(i),request,data,data,scales,10,r,5) for i,r in enumerate([.001,.0003])]
    save=m.CalibrationTrainingSession.save
    def interrupt(self,path):
        save(self,path)
        if Path(path).parent.name==str(interrupt_rate) and self.epoch==1 and not triggered[0]:
            triggered[0]=True
            raise KeyboardInterrupt('after committed epoch')
    monkeypatch.setattr(m.CalibrationTrainingSession,'save',interrupt)
    with pytest.raises(KeyboardInterrupt):
        for i,r in enumerate([.001,.0003]):
            m._rate_attempt(tmp_path/'resumed'/str(i),request,data,data,scales,10,r,5)
    now[0]+=1000
    resumed=[m._rate_attempt(tmp_path/'resumed'/str(i),request,data,data,scales,10,r,5) for i,r in enumerate([.001,.0003])]
    assert [r['resources']['wall_s'] for r in resumed]==[50.,50.]
    assert [r['resources']['cpu_s'] for r in resumed]==[50.,50.]
    for i,(a,b) in enumerate(zip(reference,resumed)):
        assert a['selected']==b['selected'] and a['validation_history']==b['validation_history']
        x=m._read_packet(tmp_path/'reference'/str(i)/'selected.pt')
        y=m._read_packet(tmp_path/'resumed'/str(i)/'selected.pt')
        assert x['canonical_tensor_sha256']==y['canonical_tensor_sha256']
