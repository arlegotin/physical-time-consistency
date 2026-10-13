"""Explicit, nonempirical bounded cohort and raw evidence contracts."""
from pathlib import Path

import numpy as np
import pytest

from poc import calibration_study as study
from poc.calibration_models import HYBRID_MODEL_IDS
from poc.calibration_report import verify

ROOT = Path(__file__).resolve().parents[1]
UNIT = {'segment_count': 5, 'max_epochs_per_rate': 5,
        'evaluation_trajectory_count': 2, 'evaluation_anchor_times_ns': [320_000_000],
        'decision_candidate_count': 2}


def test_hybrid_recipe_and_exact_bounded_membership(tmp_path):
    recipe = study.prospective_recipe(matrix='hybrid')
    assert recipe['models'] == list(HYBRID_MODEL_IDS)
    assert recipe['budgets_s'] == [30, 120]
    assert recipe['collector']['segment_count'] == 600
    assert recipe['evaluation']['trajectory_count'] == 64
    assert recipe['evaluation']['seed_session_id'] == 1
    assert 'session=1/' in recipe['decisions']['bank_law']
    assert recipe['hybrid_component'] == {
        'primary_model_id': 'hybrid_affine_flow', 'anchor_model_id': 'hybrid_accel_rk4',
        'session_ids': [1, 2, 3], 'initialization_ids': [0], 'primary_budget_s': 30,
        'primary_query_step_ns': 40_000_000, 'error_ratio_limit': 1.05,
        'median_ratio_limit': .75, 'p95_ratio_limit': 1.0}
    prepared = study.prepare(ROOT, tmp_path, 'hybrid')
    coordinates = prepared['coordinates']
    assert len(coordinates) == 30
    assert {(c['session_id'], c['budget_s']) for c in coordinates} == {
        (1, 30), (1, 120), (2, 30), (2, 120), (3, 30), (3, 120)}
    assert {c['initialization_id'] for c in coordinates} == {0}
    coverage = verify(study.open_store(ROOT, tmp_path, 'hybrid'), 'hybrid')
    assert coverage['expected_fit_count'] == 30 and coverage['expected_nominal_count'] == 6
    assert {c['coordinate']['model_id'] for c in coverage['missing']
            if c['stage'] == 'propagation'} == set(HYBRID_MODEL_IDS[:3])


def test_legacy_recipes_and_membership_stay_compatible():
    legacy = study.prospective_recipe()
    assert legacy == study.prospective_recipe(matrix='pilot') == study.prospective_recipe(matrix='development')
    assert 'hybrid_component' not in legacy
    assert legacy['budgets_s'] == [30, 120, 600] and legacy['evaluation']['seed_session_id'] == 0
    assert len(study._coordinates(legacy, 'pilot')) == 21
    assert len(study._coordinates(legacy, 'development')) == 195


def test_hybrid_preparation_is_immutable_and_filters_reject_old_draws(tmp_path):
    prepared = study.prepare(ROOT, tmp_path, 'hybrid', unit_fixture=UNIT)
    assert study.prepare(ROOT, tmp_path, 'hybrid', unit_fixture=UNIT) == prepared
    for matrix in ('pilot', 'development'):
        with pytest.raises(ValueError, match='binding|recipe'):
            study.prepare(ROOT, tmp_path, matrix, unit_fixture=UNIT)
    with pytest.raises(ValueError, match='membership'):
        study.collect_session(0, root=ROOT, output_root=tmp_path, matrix='hybrid')
    for filters in ({'session_id': 0}, {'model_id': 'affine_flow'}, {'initialization_id': 1}, {'budget_s': 30}):
        with pytest.raises(ValueError, match='prepared'):
            study.train_coordinates(ROOT, tmp_path, 'hybrid', **filters)
        with pytest.raises(ValueError, match='prepared'):
            study.predict_coordinates(ROOT, tmp_path, 'hybrid', **filters)
    with pytest.raises(ValueError, match='prepared'):
        study.main(['train', '--root', str(ROOT), '--output-root', str(tmp_path), '--matrix', 'hybrid',
                    '--model', 'hybrid_affine_flow', '--initialization', '1'])
    from poc.io import atomic_json
    prepared['coordinates'][0]['session_id'] = 0
    atomic_json(tmp_path/'matrices/hybrid.json', prepared)
    with pytest.raises(ValueError, match='coordinate'):
        study.open_store(ROOT, tmp_path, 'hybrid')


def test_fresh_hybrid_evaluation_probes_targets_and_banks(tmp_path):
    from poc.calibration_predictions import prepare_evaluation, load_evaluation
    from poc.calibration_data import rng_for
    from poc.calibration_decisions import _bank
    study.prepare(ROOT, tmp_path, 'hybrid', unit_fixture=UNIT)
    store = study.open_store(ROOT, tmp_path, 'hybrid')
    arrays = load_evaluation(store, prepare_evaluation(store, root=ROOT))
    rng = rng_for('development-evaluation', 1, 0, 0)
    expected = np.r_[rng.uniform(-.5, .5, 2), rng.uniform(-.25, .25, 2), rng.uniform(-.3, .3, 2)]
    np.testing.assert_array_equal(arrays['trajectory_state'][0, 0], expected)
    probes = rng_for('development-evaluation', 1, 0, 2).uniform(-1.5, 1.5, (41, 2))
    np.testing.assert_array_equal(arrays['held_command'][0], probes[8])
    np.testing.assert_array_equal(arrays['target'][0], rng_for('development-evaluation', 1, 0, 3).uniform(-.5, .5, 2))
    recipe = study.prospective_recipe(matrix='hybrid')
    bank = _bank(arrays, 0, recipe)
    expected_bank = rng_for('development-evaluation', 1, 0, 4).uniform(-1.5, 1.5, (9, 254, 6, 2))[8]
    np.testing.assert_array_equal(bank[2:], expected_bank)


@pytest.fixture(scope='module')
def hybrid_predictions(tmp_path_factory):
    from poc.calibration_training import fit_model
    output = tmp_path_factory.mktemp('nonempirical-hybrid-predictions')
    study.prepare(ROOT, output, 'hybrid', unit_fixture=UNIT)
    store = study.open_store(ROOT, output, 'hybrid')
    collected = study.collect_session(1, root=ROOT, output_root=output, matrix='hybrid')
    fit = fit_model(store, collected, 'hybrid_affine_flow', 5, 0)
    measured = study.predict_coordinates(ROOT, output, 'hybrid', session_id=1,
                                         model_id='hybrid_affine_flow', budget_s=5, initialization_id=0)
    return store, fit, measured


def test_hybrid_fit_uses_prepared_membership_and_new_initialization(tmp_path):
    from poc.calibration_training import fit_model
    study.prepare(ROOT, tmp_path, 'hybrid', unit_fixture=UNIT)
    store = study.open_store(ROOT, tmp_path, 'hybrid')
    collected = study.collect_session(1, root=ROOT, output_root=tmp_path, matrix='hybrid')
    for model_id in HYBRID_MODEL_IDS:
        fit = fit_model(store, collected, model_id, 5, 0)
        summary = store.read(fit)[1]['summary']
        assert summary['fit_status'] == 'finite' and summary['seed'] == 2026101202
        assert summary['model']['nominal_physics_frozen'] and summary['optimization_limited']
    with pytest.raises(ValueError, match='membership'):
        fit_model(store, collected, 'hybrid_affine_flow', 5, 1)


def test_three_map_diagnostic_keeps_same_weights_and_explicit_rows(hybrid_predictions):
    from poc.calibration_records import read_records
    store, fit, measured = hybrid_predictions
    assert len(measured['prediction_receipts']) == len(measured['diagnostic_receipts']) == 1
    folder, manifest = store.read(measured['diagnostic_receipts'][0])
    rows = read_records(folder/'rows.jsonl.gz')
    assert len(rows) == 864  # four states * 36 case/partition/kind rows * two components * three maps
    assert {r['propagator'] for r in rows} == {'hybrid_affine_flow', 'hybrid_affine_rk4', 'hybrid_affine_aware'}
    assert all(r['model_id'] == 'hybrid_affine_flow' and not r['own_solver_trained'] and
               r['comparison'] == 'same_weight_propagation_diagnostic' for r in rows)
    assert manifest['summary']['canonical_tensor_sha256'] == store.read(fit)[1]['summary']['canonical_tensor_sha256']
    coverage = verify(store, 'hybrid')
    assert not coverage['invalid']
    assert coverage['counts']['predictions'] == coverage['counts']['propagation'] == 1


@pytest.mark.parametrize('corruption', ['omission', 'swap', 'own_solver', 'tensor'])
def test_invalid_diagnostic_is_retained_as_invalid_missing_evidence(hybrid_predictions, monkeypatch, corruption):
    from copy import deepcopy
    from poc import calibration_report as report
    store, _, measured = hybrid_predictions
    receipt = measured['diagnostic_receipts'][0]
    folder, manifest = store.read(receipt)
    original_rows, original_read = report._rows, store.read
    def altered_rows(path, name):
        rows = original_rows(path, name)
        if path == folder:
            rows = deepcopy(rows)
            if corruption == 'omission':
                rows = [r for r in rows if r['propagator'] != 'hybrid_affine_aware']
            elif corruption == 'swap':
                rows[0]['propagator'] = 'hybrid_accel_aware'
            elif corruption == 'own_solver':
                rows[0]['own_solver_trained'] = True
        return rows
    def altered_read(r):
        path, saved = original_read(r)
        if r == receipt and corruption == 'tensor':
            saved = deepcopy(saved)
            saved['summary']['canonical_tensor_sha256'] = 'wrong-source-weight'
        return path, saved
    monkeypatch.setattr(report, '_rows', altered_rows)
    monkeypatch.setattr(store, 'read', altered_read)
    coverage = report.verify(store, 'hybrid')
    assert any(r.get('receipt') == receipt for r in coverage['invalid'])
    assert any(r['stage'] == 'propagation' and r['coordinate']['session_id'] == 1 and
               r['coordinate']['model_id'] == 'hybrid_affine_flow' for r in coverage['missing'])


def test_genuine_reduced_hybrid_cohort_workflow(tmp_path, capsys):
    """Acquire/fit/measure all five methods on all three fresh unit sessions."""
    import json
    common = ['--root', str(ROOT), '--output-root', str(tmp_path), '--matrix', 'hybrid', '--phase', 'unit']
    def run(command, *extra):
        assert study.main([command, *common, *extra]) == 0
        return json.loads(capsys.readouterr().out)
    prepared = run('prepare', '--unit-fixture-segments', '5', '--unit-fixture-max-epochs', '5',
        '--unit-fixture-evaluation-trajectories', '2', '--unit-fixture-evaluation-anchors-ns', '320000000',
        '--unit-fixture-decision-candidates', '2')
    assert not prepared['recipe']['empirical_claims_allowed']
    assert len(run('collect')['session_receipts']) == 3
    trained = run('train')
    assert len(trained['fit_receipts']) == 15 and len(trained['nominal_receipts']) == 3
    predicted = run('predict')
    assert len(predicted['prediction_receipts']) == 18 and len(predicted['diagnostic_receipts']) == 9
    assert not predicted['missing_coordinates']
    decisions = run('decisions')
    assert len(decisions['decision_receipts']) == 18 and not decisions['missing_coordinates']
    timed = run('time')
    assert timed['timing_receipt'] and not timed['missing_coordinates']
    store = study.open_store(ROOT, tmp_path, 'hybrid')
    timing_request = store.read(timed['timing_receipt'])[1]['request']
    assert len(timing_request['coordinates']) == len(timing_request['fit_receipts']) == 18
    coverage = run('verify')
    assert coverage['complete'] and not coverage['invalid'] and not coverage['missing']
    assert coverage['expected_fit_count'] == 15 and coverage['expected_nominal_count'] == 3


def test_failed_hybrid_fit_retains_all_three_diagnostic_maps(tmp_path, monkeypatch):
    from poc import calibration_training as training
    from poc.calibration_records import read_records
    study.prepare(ROOT, tmp_path, 'hybrid', unit_fixture=UNIT)
    store = study.open_store(ROOT, tmp_path, 'hybrid')
    collected = study.collect_session(1, root=ROOT, output_root=tmp_path, matrix='hybrid')
    def failed(*_):
        raise ValueError('explicit nonempirical fit failure')
    monkeypatch.setattr(training.CalibrationTrainingSession, 'advance_updates', failed)
    fit = training.fit_model(store, collected, 'hybrid_affine_aware', 5, 0)
    assert store.read(fit)[1]['summary']['fit_status'] == 'failed'
    measured = study.predict_coordinates(ROOT, tmp_path, 'hybrid', session_id=1,
                                         model_id='hybrid_affine_aware', budget_s=5, initialization_id=0)
    folder, _ = store.read(measured['diagnostic_receipts'][0])
    rows = read_records(folder/'rows.jsonl.gz')
    assert len(rows) == 864 and all(r['status'] == 'fit_failed' and r['error'] is None for r in rows)
    assert {r['propagator'] for r in rows} == set(HYBRID_MODEL_IDS[:3])
    coverage = verify(store, 'hybrid')
    assert not coverage['invalid'] and len(coverage['failed']) == 1
