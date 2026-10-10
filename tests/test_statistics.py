import hashlib
from collections import Counter

import numpy as np
import pytest
from scipy.stats import t

from poc.criteria import expand_criteria, validate_registry
from poc.criterion_gates import evaluate_criterion
from poc.statistics import (
    bootstrap_indices, contrast_bounds, fieller_improvement,
    look_alpha, plan_precision, reserve_look,
)


def indices(n=5):
    return bootstrap_indices(n, 1)


def test_complete_atomic_registry_is_stable_and_exact():
    records = expand_criteria()
    assert records == expand_criteria()
    statistical = [c for c in records if c['statistical']]
    assert len(statistical) == 280
    assert len({c['criterion_id'] for c in records}) == len(records)
    counts = Counter(c['criterion_id'].split('/')[0] for c in statistical)
    assert counts == {
        'K02.relative': 112, 'K02.absolute': 32, 'K02.order': 16,
        'K02.gain': 16, 'K02.actuator': 4, 'K03.jacobian': 8,
        'K03.attribution': 6, 'K04.shift': 14, 'K04.nominal': 14,
        'K04.task': 2, 'K05.actuator_rate': 28, 'K05.noise': 14,
        'K05.disturbance': 14,
    }
    assert validate_registry(records)['statistical_family_size'] == 280
    with pytest.raises(ValueError, match='registry'):
        validate_registry(records[:-1])


def test_shared_cluster_indices_replay_and_rare_tail_rule():
    a = indices()
    assert a.shape == (20_000, 5)
    np.testing.assert_array_equal(a, indices())
    assert not np.array_equal(a, bootstrap_indices(5, 2))
    assert look_alpha(1) == .025
    x = np.array([1., 2., 3., 4., 5.])
    bounds = contrast_bounds(x, .025/280, a, {})
    expected = x.mean() - t.ppf(1-.025/280, 4)*x.std(ddof=1)/np.sqrt(5)
    assert bounds['lower'] == pytest.approx(expected)
    assert bounds['rule'] == 't_rare_tail_with_central_bootstrap'
    assert bounds['central_bootstrap'][0] > 0
    assert bounds['bootstrap_indices_sha256'] == hashlib.sha256(a.tobytes()).hexdigest()


def test_ordinary_tail_t_bootstrap_and_outward_numerical_expansion():
    x = np.array([.8, .9, 1., 1.1, 1.2])
    idx = indices()
    raw = contrast_bounds(x, .025, idx, {})
    means = x[idx].mean(axis=1)
    tl = x.mean()-t.ppf(.975, 4)*x.std(ddof=1)/np.sqrt(5)
    assert raw['lower'] == pytest.approx(min(tl, np.quantile(means, .025, method='linear')))
    expanded = contrast_bounds(x, .025, idx, {'lower': .1, 'upper': .2})
    assert expanded['lower'] == pytest.approx(raw['lower']-.1)
    assert expanded['upper'] == pytest.approx(raw['upper']+.2)


def test_unbounded_constant_contrast_is_not_population_certainty():
    result = contrast_bounds(np.ones(5), .025, indices(), {})
    assert result['status'] == 'unresolved_variance'
    assert result['lower'] is None


@pytest.mark.parametrize('values,side,expected', [
    (np.ones(5), 'lower', .025**(1/5)),
    (np.zeros(5), 'upper', 1-.025**(1/5)),
    (np.full(5, .7), 'lower', max(0, .7-np.sqrt(np.log(40)/10))),
])
def test_constant_fraction_conservative_bounds(values, side, expected):
    result = contrast_bounds(values, .025, indices(), {}, bounded_fraction=True)
    assert result[side] == pytest.approx(expected)
    assert result['status'] == 'ok'


def test_nonconstant_boundary_fraction_uses_hoeffding():
    result = contrast_bounds(np.array([1., 1., 1., .8, .9]), .025, indices(), {}, bounded_fraction=True)
    assert result['rule'] == 'bounded_hoeffding'
    with pytest.raises(ValueError):
        contrast_bounds(np.array([1., 1., 1., .8, 1.1]), .025, indices(), {}, bounded_fraction=True)


def test_fieller_independent_quadratic_and_unbounded_denominator():
    b = np.array([10., 11., 9., 10.5, 9.5])
    m = np.array([6., 6.7, 5.8, 6.2, 5.7])
    result = fieller_improvement(b, m, .025/280)
    assert result['status'] == 'ok'
    n, c = len(b), t.ppf(1-.025/(2*280), 4)
    vb, vm = b.var(ddof=1), m.var(ddof=1)
    cov = np.cov(b, m, ddof=1)[0, 1]
    roots = sorted(np.roots([
        b.mean()**2-c*c*vb/n,
        -2*b.mean()*m.mean()+2*c*c*cov/n,
        m.mean()**2-c*c*vm/n,
    ]))
    assert result['lower'] == pytest.approx(1-roots[1])
    assert result['upper'] == pytest.approx(1-roots[0])
    assert fieller_improvement(np.array([-1., 1., 0., 1., -1.]), m, .025)['status'] == 'unresolved_denominator'


def test_influential_cluster_blocks_ready_interpretation():
    result = contrast_bounds(np.array([0., .01, .02, .03, 4.]), .025, indices(), {})
    assert result['influence']['unstable']
    assert result['influence']['max_centered_share'] >= .5


def test_gain_does_not_pass_five_perfect_seed_fractions():
    c = next(c for c in expand_criteria() if c['criterion_id'].startswith('K02.gain/'))
    result = evaluate_criterion(c, {'validity': 'valid', 'values': np.ones(5), 'numerical_envelope': {}}, 1)
    assert result['status'] == 'unresolved'
    assert result['pass_bounds']['lower'] < .9


def test_invalid_missing_metrics_fail_before_arithmetic():
    c = next(c for c in expand_criteria() if c['statistical'])
    assert evaluate_criterion(c, {'validity': 'invalid'}, 1)['status'] == 'invalid'
    assert evaluate_criterion(c, {'validity': 'valid'}, 1)['status'] == 'invalid'


def test_threshold_point_and_bound_equality_and_strict_sign():
    c = next(c for c in expand_criteria() if c['criterion_id'].startswith('K02.relative/') and c['dimensions']['contrast'] == 'relative')
    # Constant unbounded values do not manufacture a pass at the inclusive boundary.
    result = evaluate_criterion(c, {'validity': 'valid', 'values': np.zeros(5)}, 1)
    assert result['status'] == 'unresolved'
    good = evaluate_criterion(c, {'validity': 'valid', 'values': [.98, 1., 1.02, .99, 1.01]}, 1)
    assert good['status'] == 'pass'
    negative = evaluate_criterion(c, {'validity': 'valid', 'values': [-1., -.98, -1.02, -.99, -1.01]}, 1)
    assert negative['status'] == 'unresolved'  # percentage rejection needs paired ratio precision


def test_precision_adds_new_clusters_without_seed_cap():
    proposal = plan_precision({'n': 5, 'sd': 1., 'desired_half_width': .1, 'last_seed': 4}, {'sd': 1.2}, 2)
    assert proposal['total_seeds'] >= 10
    assert proposal['new_seed_ids'][0] == 5
    assert proposal['new_seed_ids'][-1] == proposal['total_seeds']-1
    assert proposal['planning_sd'] == pytest.approx(1.25*1.2)
    assert proposal['alpha'] == .0125
    assert proposal['predicted_half_width'] <= .1


def test_precise_paired_rejection_and_imprecise_negative_are_different():
    c = next(c for c in expand_criteria() if c['criterion_id'].startswith('K02.relative/') and c['dimensions']['contrast'] == 'relative')
    b = np.array([10., 10.01, 9.99, 10.02, 9.98])
    m = np.array([8., 8.015, 7.99, 8.025, 7.98])
    result = evaluate_criterion(c, {'validity': 'valid', 'values': .5*b-m, 'baseline': b, 'method': m}, 1)
    assert result['status'] == 'fail'
    assert result['ratio_interval']['width'] < .2
    uncertain = evaluate_criterion(c, {'validity': 'valid', 'values': .5*b-m,
                                      'baseline': b, 'method': m,
                                      'ratio_numerical_envelope': {'lower': .2, 'upper': .2}}, 1)
    assert uncertain['status'] == 'unresolved'


def test_fraction_leave_one_out_checks_the_actual_gain_boundary():
    n = 36
    result = contrast_bounds(np.ones(n), .025, bootstrap_indices(n, 1), {},
                             bounded_fraction=True, influence_boundary=.9)
    assert result['lower'] >= .9
    assert result['influence']['unstable']


def test_look_ledger_idempotence_and_new_evidence_spends_again(tmp_path):
    record = reserve_look(tmp_path, {'run_hashes': ['a'], 'criteria': 'frozen'})
    assert record['look_index'] == 1 and record['alpha'] == .025
    assert reserve_look(tmp_path, {'run_hashes': ['a'], 'criteria': 'frozen'}) == record
    next_record = reserve_look(tmp_path, {'run_hashes': ['a', 'b'], 'criteria': 'frozen'})
    assert next_record['look_index'] == 2
    assert next_record['cumulative_alpha'] == .0375
    assert next_record['rejection_alpha'] == .0125/280
