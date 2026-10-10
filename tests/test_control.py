"""Physical cost is charged at common anchors with one terminal term."""
import numpy as np
import pytest

from poc.action_path import ActionSegment, compile_knots, subdivide_path
from poc.control import anchor_components, anchor_cost


def test_constant_planning_cost_matches_independent_formula_and_is_partition_invariant():
    states = np.tile([.3, -.1, .2, -.4], (6, 1))
    target, command = np.array([.1, .2]), np.array([.7, -.8])
    path = compile_knots(np.tile(command, (6, 1)), 40_000_000)
    q2 = (.3 - .1)**2 + (-.1 - .2)**2
    v2 = .2**2 + (-.4)**2
    expected = .24 * (q2 + .05 * v2 + .001 * (.7**2 + .8**2)) + 5 * q2
    assert anchor_cost(states, target, path) == pytest.approx(expected)
    for step in (40_000_000, 20_000_000, 5_000_000, 2_500_000):
        assert anchor_cost(states, target, subdivide_path(path, step)) == pytest.approx(expected)


def test_episode_cost_charges_terminal_once_and_ignores_observed_activation():
    states = np.zeros((40, 6))
    states[:, :2] = [1., -2.]
    states[:, 4:] = 100  # Known activation is not an extra learned/control-cost target.
    path = (ActionSegment(0, 1_600_000_000, np.zeros(2)),)
    parts = anchor_components(states, np.zeros(2), path)
    assert parts['tracking'] == pytest.approx(1.6 * 5)
    assert parts['velocity'] == parts['effort'] == 0
    assert parts['terminal'] == 25
    assert parts['total'] == pytest.approx(33)
    assert anchor_cost(states, np.zeros(2), path) == pytest.approx(33)


def test_changing_state_anchor_uses_endpoint_and_effort_uses_executed_squared_commands():
    states = np.array([[1., 0., 2., 0.], [0., 2., 0., 1.]])
    path = (ActionSegment(0, 40_000_000, np.array([2., 0.])),
            ActionSegment(40_000_000, 40_000_000, np.array([0., 1.])))
    parts = anchor_components(states, np.zeros(2), path)
    assert parts['tracking'] == pytest.approx(.04 * (1 + 4))
    assert parts['velocity'] == pytest.approx(.04 * .05 * (4 + 1))
    assert parts['effort'] == pytest.approx(.001 * (.04 * 4 + .04 * 1))
    assert parts['terminal'] == 20


def test_extra_substep_states_cannot_be_counted_as_extra_forty_ms_cost_samples():
    path = compile_knots(np.zeros((6, 2)), 40_000_000)
    with pytest.raises(ValueError):
        anchor_cost(np.zeros((48, 4)), np.zeros(2), path)


@pytest.mark.parametrize('states,target,weight', [
    (np.zeros((0, 4)), np.zeros(2), 5),
    (np.zeros((6, 3)), np.zeros(2), 5),
    (np.full((6, 4), np.nan), np.zeros(2), 5),
    (np.zeros((6, 4)), np.zeros(3), 5),
    (np.zeros((6, 4)), np.full(2, np.inf), 5),
    (np.zeros((6, 4)), np.zeros(2), -1),
])
def test_invalid_cost_inputs_fail_instead_of_becoming_an_optimistic_score(states, target, weight):
    with pytest.raises(ValueError):
        anchor_cost(states, target, compile_knots(np.zeros((6, 2)), 40_000_000), weight)
