"""Planner qualification aggregates fixed tasks and independent search streams."""
import numpy as np
import pytest

from poc.mpc_calibration import calibration_conditions, qualification_summary


def test_fixed_tasks_keep_targets_identical_across_search_streams():
    first = calibration_conditions(3, 0, 64, 2)
    second = calibration_conditions(3, 4, 64, 2)
    np.testing.assert_array_equal(first['state'], second['state'])
    np.testing.assert_array_equal(first['target'], second['target'])
    assert not np.array_equal(first['standard_normals'], second['standard_normals'])
    np.testing.assert_array_equal(first['standard_normals'], calibration_conditions(3, 0, 64, 2)['standard_normals'])
    assert first['standard_normals'].shape == (2, 64, 6, 2)


def rows(value=.8):
    return [{'episode_id': episode, 'stream_id': stream, 'status': 'ok',
             'achieved_headroom_fraction': value, 'zero_cost': 10., 'optimal_cost': 2.,
             'cem_cost': 10. - 8. * value}
            for stream in range(5) for episode in range(16)]


def test_qualification_needs_coverage_and_model_independent_search_headroom():
    summary = qualification_summary(rows())
    assert summary['qualified']
    assert summary['minimum_achieved_fraction'] == .8
    assert summary['spread_of_stream_means'] == 0
    weak = rows()
    weak[0]['achieved_headroom_fraction'] = .49
    weak[0]['cem_cost'] = 10. - 8. * .49
    assert not qualification_summary(weak)['qualified']
    for bad in (rows()[:-1], rows() + rows()[:1]):
        with pytest.raises(ValueError):
            qualification_summary(bad)


def test_uninformative_tasks_or_variable_search_are_not_silently_dropped():
    cases = rows()
    cases[0].update(status='no_headroom', achieved_headroom_fraction=None, zero_cost=2., optimal_cost=2., cem_cost=2.)
    assert not qualification_summary(cases)['qualified']
    cases = rows()
    for row in cases:
        if row['stream_id'] == 0:
            row['achieved_headroom_fraction'] = .99
            row['cem_cost'] = 10. - 8. * .99
    summary = qualification_summary(cases)
    assert summary['spread_of_stream_means'] == pytest.approx(.19)
    assert not summary['qualified']
