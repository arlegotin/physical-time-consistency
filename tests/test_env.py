"""Compiled smooth mechanics, held actuation and isolated branch replay."""
import importlib
from pathlib import Path

import mujoco
import numpy as np
import pytest


def plant(actuator, step=500_000):
    assert importlib.util.find_spec('poc.envs'), 'Coupled robot plant is not implemented'
    from poc.envs import Plant
    return Plant(actuator, step, Path(__file__).resolve().parents[1] / 'assets')


@pytest.mark.parametrize('actuator', ['direct', 'lagged'])
def test_compiled_parameters_units_and_no_hidden_state(actuator):
    robot = plant(actuator)
    manifest = robot.compiled_manifest()
    assert [manifest[key] for key in ('nq','nv','nu','na')] == [2,2,2,0 if actuator == 'direct' else 2]
    np.testing.assert_array_equal(manifest['gravity'], [0,0,0])
    np.testing.assert_array_equal(manifest['joint_limited'], [False,False])
    np.testing.assert_array_equal(manifest['dof_armature'], [.02,.02])
    np.testing.assert_array_equal(manifest['dof_damping'], [.05,.05])
    np.testing.assert_array_equal(manifest['actuator_gear'], [[1,0,0,0,0,0]]*2)
    assert manifest['integrator'] == 'RK4'
    assert manifest['contact_disabled'] and manifest['autoreset_disabled']
    assert manifest['actuator_actearly'] == [False,False]
    assert manifest['actuator_dyntype'] == (['none']*2 if actuator == 'direct' else ['filter']*2)
    if actuator == 'lagged':
        assert manifest['actuator_dynprm'][0][0] == .08
    assert manifest['body_mass'][1:] == [1.,.7]
    assert len(manifest['body_iquat']) == 3


@pytest.mark.parametrize('actuator', ['direct', 'lagged'])
def test_snapshot_replay_branch_order_and_copied_state(actuator):
    robot = plant(actuator)
    state = np.array([.4,-.8,.7,-.2]+([.9,-.6] if actuator == 'lagged' else []))
    robot.reset(state, time_s=.32)
    copy = robot.state()
    copy[:] = 999
    np.testing.assert_array_equal(robot.state(), state)
    snapshot = robot.snapshot()
    command = np.array([.2,-.3])
    first = robot.advance(command, 10_000_000, external_torque=np.array([0,.1]), log_every_step=True)
    robot.restore(snapshot)
    second = robot.advance(command, 10_000_000, external_torque=np.array([0,.1]))
    np.testing.assert_array_equal(first['state'], second['state'])
    assert first['time_ns'] == 330_000_000
    assert len(first['trajectory']['time_ns']) == 20
    assert abs(robot.data.time-.33) <= 1e-10
    outcomes = {}
    for sign in (1,-1,-1,1):
        robot.restore(snapshot)
        perturbed = command+np.array([.02*sign,0])
        end = robot.advance(perturbed, 5_000_000)['state']
        if sign in outcomes:
            np.testing.assert_array_equal(end, outcomes[sign])
        outcomes[sign] = end
        np.testing.assert_array_equal(snapshot['state'], state)


def test_filtered_activation_and_external_torque_are_separate():
    robot = plant('lagged', 125_000)
    initial = np.array([.2,-.3,.1,-.2,.8,-.7])
    command = np.array([-1.1,1.2])
    robot.reset(initial)
    clean = robot.advance(command, 40_000_000)['state']
    expected = initial[4:]+(command-initial[4:])*(-np.expm1(-.04/.08))
    np.testing.assert_allclose(clean[4:], expected, atol=1e-12, rtol=1e-12)
    robot.reset(initial)
    disturbed = robot.advance(command, 40_000_000, external_torque=np.array([0,.1]))['state']
    np.testing.assert_array_equal(clean[4:], disturbed[4:])
    assert not np.array_equal(clean[:4], disturbed[:4])
    # Omitted external torque clears a previous hold's disturbance.
    robot.advance(command, 40_000_000)
    np.testing.assert_array_equal(robot.data.qfrc_applied, [0,0])


def test_bounds_ticks_nonfinite_and_warning_detection():
    robot = plant('direct')
    for command, duration in (([2.001,0],5_000_000), ([float('nan'),0],5_000_000), ([0,0],1), ([0,0],0), ([0],5_000_000)):
        with pytest.raises(ValueError):
            robot.advance(np.array(command), duration)
    with pytest.raises(ValueError):
        robot.reset(np.array([0,0,float('inf'),0]))
    snapshot = robot.snapshot()
    robot.data.warning[mujoco.mjtWarning.mjWARN_BADQPOS].number = 1
    with pytest.raises(RuntimeError, match='warning'):
        robot.advance(np.zeros(2), 500_000)
    robot.restore(snapshot)
    assert robot.advance(np.zeros(2), 500_000)['status'] == 'ok'
