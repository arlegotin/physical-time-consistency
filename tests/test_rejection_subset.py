"""A limited negative test retains the full family and cannot certify success."""
import importlib
import numpy as np
import pytest

from poc.criteria import expand_criteria


def api(name):
    try:
        module = importlib.import_module('poc.rejection_subset')
    except ImportError as error:
        pytest.fail(f'Missing rejection-only contract: {error}')
    assert hasattr(module, name), f'Missing rejection-only contract: {name}'
    return getattr(module, name)


def test_scope_keeps_both_strata_and_horizons_without_weak_comparator_selection():
    selected = api('scope_criteria')()
    assert len(selected) == 8
    assert {c['dimensions']['stratum'] for c in selected} == {'cold', 'warm'}
    assert {c['dimensions']['h_ns'] for c in selected} == {5_000_000, 10_000_000}
    assert all(c['dimensions']['competitor'] == 'node_generic' for c in selected)
    assert all(c['dimensions']['actuator'] == 'direct' for c in selected)
    assert sum(c['statistical'] for c in expand_criteria()) == 280


def test_subset_support_returns_to_unfinished_development_and_never_passes_poc():
    decision = api('subset_transition')('GO')
    assert decision['terminal_verdict'] is None
    assert decision['permitted_next_actions'][0]['milestone_id'] == 'M30'
    failed = api('subset_transition')('NO-GO')
    assert failed['terminal_verdict'] == 'POC FAIL'
    assert failed['permitted_next_actions'][0]['milestone_id'] == 'MF'


def test_subset_rejects_missing_seed_and_discovery_seed():
    evaluate = api('evaluate_subset')
    with pytest.raises(ValueError, match='seed'):
        evaluate({}, [0, 1, 2, 3], 1)
    with pytest.raises(ValueError, match='seed'):
        evaluate({}, [0, 1, 2, 3, 9000], 1)


def test_seed_continuation_has_no_cap_and_skips_reserved_discovery_ids():
    seeds = list(range(9000)) + [9003, 9004, 9005]
    assert api('validate_confirmation_seeds')(seeds) == seeds


def test_valid_subset_failure_uses_original_family_corrected_precision_rule():
    criteria = api('scope_criteria')()
    baseline = np.array([.995, .9975, 1., 1.0025, 1.005])
    method = .9 * baseline
    summaries = {}
    for criterion in criteria:
        r = criterion['threshold']
        summaries[criterion['criterion_id']] = {
            'validity': 'valid', 'values': ((1-r)*baseline-method).tolist(),
            'baseline': baseline.tolist(), 'method': method.tolist(), 'seed_ids': list(range(5))}
    rows = api('evaluate_subset')(summaries, list(range(5)), 1)
    reductions = [r for r in rows if r['threshold'] == .5]
    assert len(reductions) == 4
    assert all(r['status'] == 'fail' for r in reductions)
    assert all(r['ratio_interval']['width'] <= .20 for r in reductions)
    assert all('rare_tail' in r['rejection_bounds']['rule'] for r in reductions)
    assert all(r['alpha_spent'] == .025 for r in rows)


def test_no_complete_subset_evidence_cannot_be_green():
    with pytest.raises(ValueError, match='coverage'):
        api('evaluate_subset')({}, list(range(5)), 1)


def test_unstable_negative_point_estimate_remains_unresolved():
    criteria = api('scope_criteria')()
    baseline = np.ones(5)
    method = np.array([.1, .2, .4, 2., 9.])
    summaries = {c['criterion_id']: {'validity': 'valid',
        'values': ((1-c['threshold'])*baseline-method).tolist(),
        'baseline': baseline.tolist(), 'method': method.tolist(), 'seed_ids': list(range(5))}
        for c in criteria}
    rows = api('evaluate_subset')(summaries, list(range(5)), 1)
    assert all(r['status'] == 'unresolved' for r in rows)


def test_empty_registered_extension_request_is_not_executable():
    with pytest.raises(ValueError):
        api('validate_readiness_request')({'branch': 'mechanism-negative'})


def test_mx_accepts_only_a_complete_registered_negative_request():
    request = {'protocol_version': 'PTCO-1.0', 'branch': 'mechanism-negative',
        'request_id': 'unit-only-not-executed', 'return_milestone': 'MR',
        'contract': {str(i): 'Unit fixture reference' for i in range(1, 13)},
        'unresolved_question': 'Does a qualified baseline contradict C02?',
        'live_explanations': ['required advantage exists', 'required advantage absent'],
        'input_evidence': [{'path': 'unit/manifest.json', 'sha256': '0'*64}],
        'new_information': 'Independent confirmation clusters',
        'expected_artifacts': ['qualification and freeze'],
        'resource_projection': {'measured_seconds_per_epoch': 1.},
        'authorization_reference': 'config/authorization.json',
        'action_parameters': {'actuator': 'direct', 'competitor': 'node_generic',
            'method': 'power_correct', 'condition': 'A', 'loss_mode': 'shared',
            'development_seed': 9000, 'initial_confirmation_seeds': [0, 1, 2, 3, 4],
            'statistical_family_size': 280}}
    from poc.milestones.mx import validate_request
    assert validate_request(request) == request
    bad = {**request, 'action_parameters': {**request['action_parameters'], 'statistical_family_size': 8}}
    with pytest.raises(ValueError):
        validate_request(bad)


def test_standard_mr_transition_cannot_escalate_subset_support_to_control():
    from poc.milestones.mr import transition
    rows = [{'criterion_id': c['criterion_id'], 'status': 'pass'} for c in api('scope_criteria')()]
    decision = transition('GO', rows)
    assert decision['terminal_verdict'] is None
    assert decision['permitted_next_actions'][0]['milestone_id'] == 'M30'


def test_confirmation_projection_uses_both_measured_fits_without_runtime_ceiling():
    projection = api('project_confirmation_resources')({'node_generic': 100., 'power_correct': 200.}, 5)
    assert projection['training_wall_seconds'] == 1500.
    assert projection['training_fits'] == 10
    assert projection['E01_jobs'] == 15
    assert projection['cpu_time_ceiling_hours'] is None
    with pytest.raises(ValueError):
        api('project_confirmation_resources')({'node_generic': 100.}, 5)


def test_mr_describes_registered_negative_jobs_without_launching_them(tmp_path):
    from poc.milestones.mr import describe
    from poc.io import atomic_json
    atomic_json(tmp_path / 'config/core_negative_development_freeze.json', {'unit_fixture': True})
    result = describe(tmp_path)
    assert result['negative_only'] is True
    assert result['expected_jobs'][0]['model_fits'] == 10
    assert result['expected_jobs'][0]['E01_jobs'] == 15
    assert result['expected_jobs'][0]['actuator'] == 'direct'


def test_subset_scope_rejects_a_contradictory_actuator_header():
    scope = {'protocol_version': 'PTCO-1.0', 'negative_only': True,
        'actuator': 'direct', 'competitor': 'node_generic', 'condition': 'A', 'loss_mode': 'shared',
        'seed_ids': list(range(5)), 'statistical_family_size': 280,
        'criterion_ids': [c['criterion_id'] for c in api('scope_criteria')()],
        'positive_return_milestone': 'M30', 'readiness_decision_id': 'unit-only-not-a-decision'}
    validate = api('validate_scope')
    assert validate(scope) == scope
    with pytest.raises(ValueError):
        validate({**scope, 'actuator': 'lagged'})
    with pytest.raises(ValueError):
        validate({**scope, 'criterion_ids': scope['criterion_ids'][:-1]})
