"""Independent checks of the executed physical command path."""
import numpy as np
import pytest

from poc.action_path import (ActionSegment, compile_knots, effort_integral,
                             mean_command, path_checksum, restrict_path,
                             subdivide_path)


def knots():
    return np.array([[0., 2.], [2., 0.], [-2., 1.], [1., -1.], [.5, .2], [-.5, -.2]])


@pytest.mark.parametrize('interval,count', [(40_000_000, 6), (20_000_000, 12), (10_000_000, 24)])
def test_sample_and_hold_has_fixed_knots_and_exact_physical_clock(interval, count):
    values = knots()
    path = compile_knots(values, interval)
    assert len(path) == count
    assert [s.start_ns for s in path] == list(range(0, 240_000_000, interval))
    assert all(s.duration_ns == interval for s in path)
    assert sum(s.duration_ns for s in path) == 240_000_000
    for segment in path:
        index, remainder = divmod(segment.start_ns, 40_000_000)
        expected = values[-1] if index == 5 else (
            (1 - remainder / 40_000_000) * values[index]
            + remainder / 40_000_000 * values[index + 1])
        np.testing.assert_allclose(segment.command, expected, atol=1e-15)
    if interval == 40_000_000:
        np.testing.assert_array_equal(np.stack([s.command for s in path]), values)


def test_commands_do_not_alias_input_or_other_candidates():
    values = knots()
    first, second = compile_knots(values, 20_000_000), compile_knots(values, 20_000_000)
    digest = path_checksum(first)
    values[:] = 0
    assert path_checksum(first) == digest == path_checksum(second)
    with pytest.raises(ValueError):
        first[0].command[0] = 7
    assert not np.shares_memory(first[0].command, second[0].command)


def test_effort_and_completed_mean_integrate_holds_without_interpolating_them():
    path = (ActionSegment(0, 20_000_000, np.array([2., 0.])),
            ActionSegment(20_000_000, 20_000_000, np.array([0., 1.])))
    assert effort_integral(path, 0, 40_000_000) == pytest.approx(.10)
    assert effort_integral(path, 10_000_000, 30_000_000) == pytest.approx(.05)
    np.testing.assert_allclose(mean_command(path, 0, 40_000_000), [1., .5])
    np.testing.assert_allclose(mean_command(path, 10_000_000, 30_000_000), [1., .5])
    np.testing.assert_array_equal(mean_command(path, 20_000_000, 40_000_000), [0., 1.])
    # Averaging squared commands differs from squaring their average.
    assert effort_integral(path, 0, 40_000_000) != pytest.approx(.04 * 1.25)


def test_prediction_steps_split_before_command_and_disturbance_boundaries():
    path = compile_knots(knots(), 20_000_000)
    split = subdivide_path(path, 7_500_000, boundaries=(13_000_000, 21_000_000))
    assert sum(s.duration_ns for s in split) == 240_000_000
    boundaries = {13_000_000, 21_000_000, *range(0, 240_000_001, 20_000_000)}
    assert all(not any(s.start_ns < b < s.start_ns + s.duration_ns for b in boundaries)
               for s in split)
    assert max(s.duration_ns for s in split) <= 7_500_000
    assert effort_integral(split, 0, 240_000_000) == pytest.approx(
        effort_integral(path, 0, 240_000_000))
    chunk = restrict_path(path, 10_000_000, 40_000_000)
    assert chunk[0].start_ns == 10_000_000
    assert chunk[-1].start_ns + chunk[-1].duration_ns == 40_000_000
    assert sum(s.duration_ns for s in chunk) == 30_000_000


@pytest.mark.parametrize('bad', [np.zeros((5, 2)), np.zeros((6, 3)),
                               np.full((6, 2), np.nan), np.full((6, 2), 2.01)])
def test_invalid_knots_are_rejected(bad):
    with pytest.raises(ValueError):
        compile_knots(bad, 40_000_000)


@pytest.mark.parametrize('interval', [0, -1, True, 20_000_000., 30_000_000])
def test_actuator_rates_are_exact_declared_integer_intervals(interval):
    with pytest.raises(ValueError):
        compile_knots(knots(), interval)


def test_gaps_overlaps_uncovered_windows_and_noninteger_clocks_fail():
    first = ActionSegment(0, 20_000_000, np.zeros(2))
    for start in (19_000_000, 21_000_000):
        with pytest.raises(ValueError):
            effort_integral((first, ActionSegment(start, 20_000_000, np.zeros(2))), 0, 40_000_000)
    for start, end in ((0, 21_000_000), (1, 0), (0, 0), (0, 20_000_000.)):
        with pytest.raises(ValueError):
            effort_integral((first,), start, end)
    for start, duration in ((True, 1), (-1, 1), (0, 0), (0, 1.)):
        with pytest.raises(ValueError):
            ActionSegment(start, duration, np.zeros(2))
