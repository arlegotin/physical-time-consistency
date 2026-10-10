"""Reference aggregation derives errors from matrices, not cached metric columns."""
import importlib
import numpy as np
import pytest


def audit(rows, reference):
    try:
        module = importlib.import_module('poc.study_audit')
    except ImportError as error:
        pytest.fail(f'Missing independent report aggregator: {error}')
    return module.reference_primary_errors(rows, reference)


def fixture():
    matrix = np.eye(2)
    reference = {'jacobians': np.tile(np.concatenate([matrix, 2*matrix]), (2, 1, 1, 1)),
        'horizons_ns': np.array([5_000_000]), 'probe_ids': np.array(['p0', 'p1'])}
    rows = {'q_jac': np.stack([1.5*matrix, -matrix]),
        'v_jac': np.stack([2*matrix, 2*matrix]),
        'reference_row_id': np.array([0, 1]), 'status': np.array(['ok', 'ok']),
        'probe_id': np.array(['p0', 'p1']), 'h_ns': np.array([5_000_000, 5_000_000]),
        'stratum': np.array(['cold', 'cold']), 'cell_id': np.array(['model', 'model']),
        'error_q': np.zeros(2), 'error_v': np.zeros(2)}
    return rows, reference


def test_primary_error_uses_full_signed_matrices_and_ignores_favorable_cached_errors():
    rows, reference = fixture()
    result = audit(rows, reference)
    assert result == [{'cell_id': 'model', 'stratum': 'cold', 'h_ns': 5_000_000,
                       'n_probes': 2, 'primary_error': .625}]


def test_unclassified_model_failure_is_not_dropped_from_a_median():
    rows, reference = fixture()
    rows['status'][1] = 'bad'
    with pytest.raises(ValueError, match='failure'):
        audit(rows, reference)


def test_reference_coordinates_must_agree_instead_of_using_a_convenient_row():
    rows, reference = fixture()
    rows['reference_row_id'][1] = 0
    with pytest.raises(ValueError, match='coordinate'):
        audit(rows, reference)


def paired_fixture():
    scope = {'actuator': 'direct', 'condition': 'A', 'loss_mode': 'shared',
             'competitor': 'node_generic', 'seed_ids': [0, 1]}
    rows = []
    for seed, baseline in ((0, 2.), (1, 8.)):
        for model, error in (('node_generic', baseline), ('power_correct', 1.)):
            rows.append({'cell_id': f'direct__A__shared__{model}__seed{seed}',
                         'stratum': 'cold', 'h_ns': 5_000_000, 'n_probes': 17,
                         'primary_error': error, 'role': 'E01'})
    return rows, scope


def test_paired_report_uses_the_ratio_of_cluster_means_and_keeps_each_seed():
    from poc import study_audit
    rows, scope = paired_fixture()
    result = study_audit.paired_primary_results(rows, scope, strata=('cold',), horizons=(5_000_000,))
    assert len(result) == 1
    assert result[0]['baseline_mean'] == 5.
    assert result[0]['method_mean'] == 1.
    assert result[0]['relative_reduction'] == .8
    assert result[0]['baseline_by_seed'] == [2., 8.]
    assert result[0]['method_by_seed'] == [1., 1.]
    assert result[0]['seed_ids'] == [0, 1]


def test_missing_or_duplicate_paired_cluster_cannot_be_silently_aggregated():
    from poc import study_audit
    rows, scope = paired_fixture()
    for broken in (rows[:-1], rows+[rows[0]]):
        with pytest.raises(ValueError, match='paired'):
            study_audit.paired_primary_results(broken, scope, strata=('cold',), horizons=(5_000_000,))
