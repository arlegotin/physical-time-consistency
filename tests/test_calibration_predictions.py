"""Model-independent moving-state evaluation and complete physical-unit rows."""
from poc.calibration_records import read_records
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from poc.action_path import ActionSegment
from poc.calibration_study import prepare, open_store, collect_session, prospective_recipe
from poc.calibration_training import nominal_reference, fit_model
from poc.envs import Plant

ROOT = Path(__file__).resolve().parents[1]


def module():
    assert importlib.util.find_spec('poc.calibration_predictions'), 'Prediction stage is missing'
    from poc import calibration_predictions
    return calibration_predictions


@pytest.fixture
def evaluated(tmp_path):
    m = module()
    output = tmp_path / 'explicit-unit'
    prepare(ROOT, output, unit_fixture={'segment_count': 5, 'max_epochs_per_rate': 5,
        'evaluation_trajectory_count': 2, 'evaluation_anchor_times_ns': [320_000_000, 1_200_000_000]})
    store = open_store(ROOT, output)
    evaluation = m.prepare_evaluation(store, root=ROOT)
    truth = m.prediction_truth(store, evaluation, root=ROOT)
    return store, evaluation, truth


def read_arrays(store, receipt, name):
    folder, _ = store.read(receipt)
    with np.load(folder / name, allow_pickle=False) as saved:
        return {k: saved[k].copy() for k in saved.files}


def read_rows(store, receipt):
    folder, manifest = store.read(receipt)
    return read_records(folder / 'rows.jsonl.gz'), manifest


def test_frozen_population_and_explicit_probe_laws():
    m = module()
    recipe = prospective_recipe()
    anchors = m.anchor_times(recipe)
    assert anchors.tolist() == list(range(320_000_000, 1_200_000_001, 40_000_000))
    assert len(anchors) * recipe['evaluation']['trajectory_count'] == 1472
    e = recipe['evaluation']
    assert e['held_probe_channel'] == 2 and e['target_channel'] == 3
    assert e['held_probe_bounds'] == [-1.5, 1.5]
    assert e['seed_session_id'] == 0
    assert e['drift_command'] == [0., 0.]
    assert 'immutable' in e['intervention']
    assert all(t + 400_000_000 <= 1_600_000_000 for t in anchors)
    with pytest.raises(ValueError):
        prospective_recipe(unit_fixture={'evaluation_anchor_times_ns': [0]})


def test_evaluation_is_independent_and_all_horizons_share_warm_anchors(evaluated):
    m = module()
    store, receipt, _ = evaluated
    assert receipt == m.prepare_evaluation(store, root=ROOT)
    arrays = read_arrays(store, receipt, 'evaluation.npz')
    assert arrays['trajectory_state'].shape == (2, 41, 6)
    assert arrays['trajectory_command'].shape == (2, 40, 2)
    assert arrays['state'].shape == (6, 6)  # two common warm anchors plus one secondary initial per trajectory
    assert arrays['time_ns'].tolist() == [320_000_000, 1_200_000_000, 0] * 2
    assert arrays['primary'].tolist() == [True, True, False] * 2
    assert np.any(arrays['state'][:, 4:] != 0)
    assert not np.array_equal(arrays['held_command'], arrays['recorded_command'][:, 0])
    assert arrays['target'].shape == (2, 2)
    # Independent acquisition comes entirely from the role=1 seeded collector.
    from poc.calibration_data import rng_for
    rng = rng_for('development-evaluation', 0, 0, 0)
    expected = np.r_[rng.uniform(-.5, .5, 2), rng.uniform(-.25, .25, 2), rng.uniform(-.3, .3, 2)]
    np.testing.assert_array_equal(arrays['trajectory_state'][0, 0], expected)
    plant = Plant('lagged', 250_000, ROOT / 'assets')
    plant.reset(expected)
    np.testing.assert_array_equal(plant.advance(arrays['trajectory_command'][0, 0], 40_000_000)['state'],
                                  arrays['trajectory_state'][0, 1])


def test_query_partitions_end_exactly_and_split_command_changes():
    m = module()
    path = (ActionSegment(0, 13_000_000, np.array([1., 0.])),
            ActionSegment(13_000_000, 387_000_000, np.array([-1., 0.])))
    for horizon in (80_000_000, 160_000_000, 400_000_000):
        for step in (5_000_000, 10_000_000, 20_000_000, 40_000_000):
            parts = m.query_partition(path, horizon, step)
            assert parts[-1].start_ns + parts[-1].duration_ns == horizon
            assert sum(p.duration_ns for p in parts) == horizon
            assert all(p.duration_ns <= step for p in parts)
            assert any(p.start_ns == 13_000_000 for p in parts)
            assert not any(p.start_ns < 13_000_000 < p.start_ns+p.duration_ns for p in parts)


def test_physical_metrics_use_both_joints_and_endpoint_norms():
    m = module()
    errors = np.array([[3., 4.], [0., 0.]])
    result = m.physical_metrics(errors)
    assert result['rmse'] == 2.5
    assert result['p95'] == pytest.approx(4.75)
    assert result['endpoint_count'] == 2
    assert m.refinement_required(1e-16, 1e-15)
    assert not m.refinement_required(1e-18, 1e-15)
    assert m.refinement_required(1e-30, 0.)


def test_truth_replays_identical_states_and_commands_at_three_resolutions(evaluated):
    m = module()
    store, evaluation, receipt = evaluated
    assert receipt == m.prediction_truth(store, evaluation, root=ROOT)
    data = read_arrays(store, evaluation, 'evaluation.npz')
    truth = read_arrays(store, receipt, 'truth.npz')
    _, manifest = store.read(receipt)
    assert manifest['request']['evaluation_receipt'] == evaluation
    assert manifest['request']['resolution_ns'] == [250_000, 125_000, 62_500]
    assert 'fit_receipt' not in manifest['request']
    assert truth['audit_mask'].tolist() == [True] * 6
    assert truth['endpoint'].shape == (6, 15, 6)
    for k, step in enumerate((250_000, 125_000, 62_500)):
        plant = Plant('lagged', step, ROOT / 'assets')
        plant.reset(data['state'][0])
        expected = plant.advance(data['held_command'][0], 5_000_000)['state']
        actual = truth['endpoint'][0, 0] if k == 0 else truth['refined_endpoint'][0, 0, k-1]
        np.testing.assert_array_equal(actual, expected)
        plant.reset(data['state'][0])
        for command in data['recorded_command'][0]:
            expected = plant.advance(command, 40_000_000)['state']
        actual = truth['endpoint'][0, 14] if k == 0 else truth['refined_endpoint'][0, 14, k-1]
        np.testing.assert_array_equal(actual, expected)
    expected = np.maximum(np.abs(truth['endpoint']-truth['refined_endpoint'][:, :, 0]),
                          np.abs(truth['refined_endpoint'][:, :, 0]-truth['refined_endpoint'][:, :, 1]))
    np.testing.assert_array_equal(truth['envelope'], expected)
    assert np.any(truth['envelope'] > 0)  # retain float64 discrepancies


def test_measurement_has_complete_rows_decomposition_composition_and_reference_label(evaluated):
    m = module()
    store, evaluation, truth = evaluated
    collected = collect_session(0, root=ROOT, output_root=store.output_root)
    fit = nominal_reference(store, collected, 5)
    receipt = m.measure_prediction(store, fit, evaluation, truth)
    assert receipt == m.measure_prediction(store, fit, evaluation, truth)
    rows, manifest = read_rows(store, receipt)
    assert len(rows) == 6 * (6 * 4 + 3 * 4) * 2
    assert manifest['summary']['row_count'] == len(rows)
    assert all(r['model_id'] == 'physics_nominal' and r['comparison'] == 'unfitted_reference' for r in rows)
    assert {r['kind'] for r in rows} == {'endpoint', 'drift', 'command_response', 'composition'}
    assert {r['stratum'] for r in rows} == {'primary', 'initial_secondary'}
    groups = {(r['state_id'], r['scenario'], r['horizon_ns'], r['query_step_ns'], r['kind'], r['component']) for r in rows}
    assert len(groups) == len(rows)
    endpoints = [r for r in rows if r['kind'] == 'endpoint' and r['status'] == 'ok']
    assert endpoints and all(len(r['error']) == 2 for r in endpoints)
    assert all(r['truth_status'] in {'audited', 'requires_refinement', 'not_applicable', 'numerically_equal'} for r in rows)
    for component in ('q', 'v'):
        new = next(r for r in rows if r['state_id'] == 0 and r['horizon_ns'] == 5_000_000 and r['kind'] == 'endpoint' and r['component'] == component)
        drift = next(r for r in rows if r['state_id'] == 0 and r['horizon_ns'] == 5_000_000 and r['kind'] == 'drift' and r['component'] == component)
        response = next(r for r in rows if r['state_id'] == 0 and r['horizon_ns'] == 5_000_000 and r['kind'] == 'command_response' and r['component'] == component)
        np.testing.assert_allclose(response['error'], np.array(new['error'])-drift['error'], atol=1e-15)


def test_verified_failed_fit_produces_every_failure_row(evaluated, monkeypatch):
    m = module()
    from poc import calibration_training
    store, evaluation, truth = evaluated
    collected = collect_session(0, root=ROOT, output_root=store.output_root)
    def fail(*_):
        raise ValueError('unit intentional numerical fit failure')
    monkeypatch.setattr(calibration_training.CalibrationTrainingSession, 'advance_updates', fail)
    fit = fit_model(store, collected, 'affine_flow', 5, 0)
    rows, manifest = read_rows(store, m.measure_prediction(store, fit, evaluation, truth))
    assert len(rows) == 432
    assert all(r['status'] == 'fit_failed' and r['error'] is None for r in rows)
    assert all(g['failure_fraction'] == 1 and g['interpretation_status'] == 'prediction_failed'
               for g in manifest['summary']['metrics'])
    diagnostic, _ = read_rows(store, m.propagation_diagnostic(store, fit, evaluation, truth))
    assert len(diagnostic) == 864 and all(r['status'] == 'fit_failed' for r in diagnostic)


def test_invalid_finite_checkpoint_is_not_a_prediction_failure(evaluated, monkeypatch):
    m = module()
    from poc import calibration_training
    store, evaluation, truth = evaluated
    collected = collect_session(0, root=ROOT, output_root=store.output_root)
    fit = nominal_reference(store, collected, 5)
    def corrupt(*_):
        raise ValueError('Selected canonical tensors mismatch')
    monkeypatch.setattr(calibration_training, 'load_selected', corrupt)
    with pytest.raises(ValueError, match='canonical'):
        m.measure_prediction(store, fit, evaluation, truth)


def test_same_weight_diagnostic_keeps_both_propagators_explicit(evaluated):
    m = module()
    store, evaluation, truth = evaluated
    collected = collect_session(0, root=ROOT, output_root=store.output_root)
    fit = fit_model(store, collected, 'affine_flow', 5, 0)
    rows, manifest = read_rows(store, m.propagation_diagnostic(store, fit, evaluation, truth))
    assert len(rows) == 864
    assert {r['propagator'] for r in rows} == {'affine_flow', 'affine_rk4'}
    assert all(r['comparison'] == 'same_weight_propagation_diagnostic' and not r['own_solver_trained'] for r in rows)
    assert manifest['summary']['canonical_tensor_sha256'] == store.read(fit)[1]['summary']['canonical_tensor_sha256']


def test_truth_identity_and_mutated_inputs_rejected(evaluated):
    m = module()
    store, evaluation, truth = evaluated
    folder, _ = store.read(evaluation)
    with (folder / 'evaluation.npz').open('ab') as saved:
        saved.write(b'corrupted')
    with pytest.raises(ValueError, match='hash'):
        m.prediction_truth(store, evaluation, root=ROOT)


def test_predict_cli_consumes_existing_fits_without_training(evaluated, capsys):
    m = module()
    from poc import calibration_study
    store, _, _ = evaluated
    collected = collect_session(0, root=ROOT, output_root=store.output_root)
    nominal_reference(store, collected, 5)
    assert calibration_study.main(['predict', '--root', str(ROOT), '--output-root', str(store.output_root),
                                  '--model', 'physics_nominal', '--phase', 'unit']) == 0
    result = json.loads(capsys.readouterr().out)
    assert len(result['prediction_receipts']) == 1
    assert result['diagnostic_receipts'] == []
    assert result['missing_coordinates'] == []


def test_adaptive_refinement_is_shared_model_independent_and_bounded(evaluated):
    m = module()
    store, evaluation, receipt = evaluated
    raw = read_arrays(store, receipt, 'truth.npz')
    original = raw['endpoint'].copy()
    statuses = ['ok'] + ['unit_unused'] * (len(original)-1)
    # Exact agreement at the original level cannot interpret a nonzero envelope.
    reference, envelope, supplements = m._adaptive(store, evaluation, receipt, raw,
                                                   {0: [(original[:, 0].copy(), statuses)]})
    assert 1 <= len(supplements) <= 2
    np.testing.assert_array_equal(raw['endpoint'], original)
    arrays = read_arrays(store, evaluation, 'evaluation.npz')
    for supplement in supplements:
        folder, manifest = store.read(supplement)
        request = manifest['request']
        assert set(request) == {'evaluation_receipt', 'truth_receipt', 'coordinates', 'resolution_ns'}
        assert request['coordinates'] == [[0, 0]]
        assert supplement == m.supplemental_truth(store, evaluation, receipt, [[0, 0]], request['resolution_ns'])
        plant = Plant('lagged', request['resolution_ns'], ROOT / 'assets')
        plant.reset(arrays['state'][0])
        expected = plant.advance(arrays['held_command'][0], 5_000_000)['state']
        np.testing.assert_array_equal(read_arrays(store, supplement, 'refinement.npz')['endpoint'][0], expected)
    assert np.isfinite(reference).all() and np.isfinite(envelope).all()


def test_numerical_equality_is_unresolved_without_unbounded_refinement(evaluated):
    m = module()
    store, evaluation, receipt = evaluated
    raw = read_arrays(store, receipt, 'truth.npz')
    for key in ('endpoint', 'refined_endpoint', 'envelope'):
        raw[key] = np.zeros_like(raw[key])
    _, _, supplements = m._adaptive(store, evaluation, receipt, raw,
                                    {0: [(raw['endpoint'][:, 0], ['ok']*6)]})
    assert supplements == []
    row = {'session_id': 0, 'budget_s': 5, 'model_id': 'physics_nominal',
           'initialization_id': None, 'comparison': 'unfitted_reference', 'propagator': 'physics_nominal', 'stratum': 'primary', 'scenario': 'held',
           'horizon_ns': 5_000_000, 'query_step_ns': 5_000_000, 'kind': 'endpoint',
           'component': 'q', 'status': 'ok', 'error': [0., 0.], 'envelope_norm': 0.,
           'truth_status': 'numerically_equal'}
    metric = m.aggregate_prediction_rows([row])[0]
    assert metric['rmse'] == 0 and metric['interpretation_status'] == 'numerically_equal_unresolved'


def test_nonfinite_predictions_keep_full_coverage_and_inputs_immutable(evaluated, monkeypatch):
    m = module()
    from poc import calibration_training
    from poc.models.common import prediction
    store, evaluation, truth = evaluated
    collected = collect_session(0, root=ROOT, output_root=store.output_root)
    fit = nominal_reference(store, collected, 5)
    load = calibration_training.load_selected
    def failing_load(*args):
        model = load(*args)
        def failing(state, command, h_s):
            return prediction(torch.full_like(state, float('nan')), 0)
        model.predict = failing
        return model
    monkeypatch.setattr(calibration_training, 'load_selected', failing_load)
    original = read_arrays(store, evaluation, 'evaluation.npz')['state']
    rows, manifest = read_rows(store, m.measure_prediction(store, fit, evaluation, truth))
    assert len(rows) == 432 and all(r['status'] == 'nonfinite' for r in rows)
    assert all(g['failure_fraction'] == 1 for g in manifest['summary']['metrics'])
    np.testing.assert_array_equal(read_arrays(store, evaluation, 'evaluation.npz')['state'], original)


def test_unaudited_refinement_records_both_predecessor_resolutions(tmp_path):
    m = module()
    output = tmp_path / 'three-trajectory-unit'
    prepare(ROOT, output, unit_fixture={'segment_count': 5, 'evaluation_trajectory_count': 3,
                                       'evaluation_anchor_times_ns': [320_000_000]})
    store = open_store(ROOT, output)
    evaluation = m.prepare_evaluation(store, root=ROOT)
    receipt = m.prediction_truth(store, evaluation, root=ROOT)
    raw = read_arrays(store, receipt, 'truth.npz')
    assert raw['audit_mask'].tolist() == [True, True, False, False, True, True]
    statuses = ['unit_unused'] * 6
    statuses[2] = 'ok'
    _, _, supplements = m._adaptive(store, evaluation, receipt, raw,
                                    {0: [(raw['endpoint'][:, 0].copy(), statuses)]})
    steps = [store.read(r)[1]['request']['resolution_ns'] for r in supplements]
    assert steps[:3] == [125_000, 62_500, 31_250]
    assert len(steps) <= 4
    assert all(store.read(r)[1]['request']['coordinates'] == [[2, 0]] for r in supplements)


def test_fresh_predict_cli_reproduces_first_and_last_coordinates(evaluated):
    import subprocess
    import sys
    from poc.calibration_training import load_selected
    m = module()
    store, evaluation, _ = evaluated
    collected = collect_session(0, root=ROOT, output_root=store.output_root)
    fit = fit_model(store, collected, 'affine_rk4', 5, 0)
    result = subprocess.run([sys.executable, '-m', 'poc.calibration_study', 'predict', '--root', str(ROOT),
                             '--output-root', str(store.output_root), '--phase', 'unit',
                             '--model', 'affine_rk4'], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert len(payload['prediction_receipts']) == len(payload['diagnostic_receipts']) == 1
    assert payload['missing_coordinates'] == []
    rows, _ = read_rows(store, payload['prediction_receipts'][0])
    data = read_arrays(store, evaluation, 'evaluation.npz')
    model = load_selected(store, fit)
    for state_id in (0, 4):
        with torch.no_grad():
            expected = model.predict(torch.from_numpy(data['state'][state_id:state_id+1]),
                                     torch.from_numpy(data['held_command'][state_id:state_id+1]), .005).next_state[0].numpy()
        for component, cut in (('q', slice(0, 2)), ('v', slice(2, 4))):
            row = next(r for r in rows if r['state_id'] == state_id and r['kind'] == 'endpoint'
                       and r['horizon_ns'] == 5_000_000 and r['component'] == component)
            np.testing.assert_array_equal(row['prediction'], expected[cut])


def test_drift_rows_are_displacements_and_truth_provenance_is_explicit(evaluated):
    m = module()
    store, evaluation, truth = evaluated
    collected = collect_session(0, root=ROOT, output_root=store.output_root)
    fit = nominal_reference(store, collected, 5)
    rows, _ = read_rows(store, m.measure_prediction(store, fit, evaluation, truth))
    data = read_arrays(store, evaluation, 'evaluation.npz')
    base = read_arrays(store, truth, 'truth.npz')
    row = next(r for r in rows if r['state_id'] == 0 and r['kind'] == 'drift'
               and r['horizon_ns'] == 5_000_000 and r['component'] == 'q')
    np.testing.assert_allclose(row['truth'], base['endpoint'][0, 1, :2]-data['state'][0, :2], atol=1e-14)
    assert row['truth_coordinates'] == [[0, 1]]
    assert row['truth_resolution_ns'] == [250_000]
    response = next(r for r in rows if r['state_id'] == 0 and r['kind'] == 'command_response'
                    and r['horizon_ns'] == 5_000_000 and r['component'] == 'q')
    assert response['truth_coordinates'] == [[0, 0], [0, 1]]
    assert any(np.linalg.norm(r['error']) > 0 for r in rows if r['kind'] == 'composition' and r['status'] == 'ok')


def test_raw_recomputation_keeps_calibration_coordinates_separate():
    m = module()
    base = {'session_id': 0, 'budget_s': 5, 'model_id': 'affine_flow', 'initialization_id': 0,
            'comparison': 'own_solver_trained', 'propagator': 'affine_flow', 'stratum': 'primary',
            'scenario': 'held', 'horizon_ns': 5_000_000, 'query_step_ns': 5_000_000,
            'kind': 'endpoint', 'component': 'q', 'status': 'ok', 'error': [3., 4.],
            'envelope_norm': 0., 'truth_status': 'audited'}
    metrics = m.aggregate_prediction_rows([base, {**base, 'budget_s': 10, 'error': [0., 0.]}])
    assert len(metrics) == 2
    assert metrics[0]['budget_s'] == 5 and metrics[0]['rmse'] == pytest.approx(np.sqrt(12.5))
    assert metrics[1]['budget_s'] == 10 and metrics[1]['rmse'] == 0
