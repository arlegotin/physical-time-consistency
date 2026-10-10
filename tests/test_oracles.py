"""Independent rigid-body equations, inertial frames and controlled exact limits."""
import ast
import importlib
from pathlib import Path

import mujoco
import numpy as np
import pytest

from poc.analytic import coefficients
from poc.envs import Plant


def oracle():
    assert importlib.util.find_spec('poc.oracle_rigidbody'), 'Independent rigid-body oracle is not implemented'
    return importlib.import_module('poc.oracle_rigidbody')


def test_inertia_principal_frame_rotation_and_positive_definite_mass():
    module = oracle()
    compiled = Plant('direct').compiled_manifest()
    parameters = module.parameters_from_compiled(compiled)
    # Capsule principal z is the small axial inertia; body z is transverse.
    np.testing.assert_allclose(parameters['inertia_body_zz'], [.009446875,.004813786764705882], atol=1e-15)
    assert parameters['theta1'] == pytest.approx(.009446875+.15**2+.7*.3**2)
    assert parameters['theta2'] == pytest.approx(.004813786764705882+.7*.125**2)
    for angle in np.linspace(-np.pi,np.pi,1001):
        assert np.linalg.eigvalsh(module.mass_matrix(np.array([.3,angle]),parameters))[0] > 0
    tree = ast.parse(Path(module.__file__).read_text())
    assert not any(isinstance(node,(ast.Import,ast.ImportFrom)) and 'mujoco' in ast.unparse(node) for node in ast.walk(tree))


@pytest.mark.parametrize('actuator', ['direct','lagged'])
@pytest.mark.parametrize('zero_velocity', [False, True])
def test_acceleration_matches_compiled_plant_at_randomized_states(actuator, zero_velocity):
    module, robot = oracle(), Plant(actuator)
    parameters = module.parameters_from_compiled(robot.compiled_manifest())
    rng = np.random.Generator(np.random.PCG64(8202))
    for _ in range(128):
        state = rng.uniform(-2,2,4 if actuator == 'direct' else 6)
        if zero_velocity:
            state[2:4] = 0
        command = rng.uniform(-2,2,2)
        robot.reset(state)
        robot.data.ctrl[:] = command
        mujoco.mj_forward(robot.model,robot.data)
        result = module.rhs(0.,state[:4],command,state[4:] if actuator == 'lagged' else None,parameters)
        np.testing.assert_allclose(result[:2],state[2:4],atol=0,rtol=0)
        np.testing.assert_allclose(result[2:],robot.data.qacc,atol=1e-10,rtol=1e-11)


def test_coriolis_sign_damping_armature_and_energy_identity():
    module = oracle()
    parameters = {'theta1':.15,'theta2':.03,'theta3':.02,'damping':[.05,.05],'armature':[.02,.02],'tau_s':.08}
    q, v = np.array([.1,np.pi/2]), np.array([1.,2.])
    c = np.array([-.02*(2*1*2+2**2), .02*1**2])
    torque = c+.05*v
    acceleration = module.rhs(0.,np.r_[q,v],torque,None,parameters)[2:]
    np.testing.assert_allclose(acceleration,0,atol=1e-12)
    expected = np.array([[.15+.03+.02,.03],[.03,.03+.02]])
    np.testing.assert_allclose(module.mass_matrix(q,parameters),expected,atol=1e-15)
    torque = np.array([.7,-.6])
    acceleration = module.rhs(0.,np.r_[q,v],torque,None,parameters)[2:]
    derivative = -.02*np.sin(q[1])*v[1]*np.array([[2,1],[1,0]])
    energy_rate = v@module.mass_matrix(q,parameters)@acceleration+.5*v@derivative@v
    assert energy_rate == pytest.approx(v@torque-.05*(v@v),abs=1e-12)


@pytest.mark.parametrize('actuator', ['direct','lagged'])
def test_dop853_matches_independent_constant_mass_exact_map(actuator):
    module = oracle()
    parameters = {'theta1':2.,'theta2':1.,'theta3':0.,'damping':[0,0],'armature':[0,0],'tau_s':.08}
    matrix = np.array([[3.,1.],[1.,1.]])
    state = np.array([.3,-.4,.7,-.8]+([.2,-.1] if actuator == 'lagged' else []))
    command = np.array([1.,-1.2])
    for h in (0.,.0025,.005,.04,.08):
        result = module.integrate(state,command,h,parameters,actuator)
        assert result['success'] and result['status'] == 0
        assert result['endpoint_time_s'] == h
        q,v = state[:2],state[2:4]
        if actuator == 'direct':
            acceleration = np.linalg.solve(matrix,command)
            expected = np.r_[q+h*v+.5*h*h*acceleration,v+h*acceleration]
        else:
            kw,kv,kq = coefficients(h)
            a0,delta = np.linalg.solve(matrix,state[4:]),np.linalg.solve(matrix,command-state[4:])
            expected = np.r_[q+h*v+.5*h*h*a0+kq*delta,v+h*a0+kv*delta,state[4:]+kw*(command-state[4:])]
        np.testing.assert_allclose(result['state'],expected,atol=1e-12,rtol=1e-12)


def test_unresolved_stable_zero_jacobian_is_not_a_valid_reference():
    assert importlib.util.find_spec('poc.sensitivity'), 'Reference numerical qualification is not implemented'
    from poc.sensitivity import resolved_reference
    assert not resolved_reference(np.zeros((2,2)),1e-20)
    assert not resolved_reference(np.eye(2)*1e-8,1e-9)
    assert resolved_reference(np.eye(2)*1e-5,1e-9)


def test_reference_gate_recomputes_raw_metrics_instead_of_trusting_green_summary(tmp_path):
    from poc.sensitivity import _reference_measurements
    from poc.milestones.m20 import CHECKS,evaluate
    from poc.io import atomic_json
    folder = tmp_path / 'evidence/M20/synthetic'
    folder.mkdir(parents=True)
    horizons,eps = np.array([.0025,.005,.01]),np.array([.01,.02,.04])
    subset = np.r_[np.arange(8),np.arange(64,72)]
    strata = np.r_[np.zeros(64,dtype=int),np.ones(64,dtype=int)]
    parameters = {'theta1':2.,'theta2':1.,'theta3':0.,'damping':[0,0],'armature':[0,0],'tau_s':.08}
    inverse = np.linalg.solve(np.array([[3.,1.],[1.,1.]]),np.eye(2))
    summary = {'measurements':{},'numerical':{'qualified_level':0,'slope_window_s':horizons.tolist()}}
    for actuator in ('direct','lagged'):
        dimension = 4 if actuator == 'direct' else 6
        states,commands = np.zeros((128,dimension)),np.zeros((128,2))
        gains = []
        for h in horizons:
            kw,kv,kq = coefficients(h)
            gains.append(np.r_[.5*h*h*inverse,h*inverse] if actuator == 'direct' else np.r_[kq*inverse,kv*inverse,kw*np.eye(2)])
        primary = np.broadcast_to(np.asarray(gains),(128,3,dimension,2)).copy()
        jacobian = np.repeat(primary[:,:,None],3,axis=2)
        endpoints = np.empty((128,3,3,2,2,dimension))
        for epsilon_id,epsilon in enumerate(eps):
            for axis in (0,1):
                for sign_id,sign in enumerate((-1,1)):
                    endpoints[:,:,epsilon_id,axis,sign_id] = sign*epsilon*primary[:,:,:,axis]
        independent = {'endpoints':endpoints[:,:,1],'jacobians':primary}
        tightened = {key:value[subset] for key,value in independent.items()}
        measurements,envelope,_ = _reference_measurements(actuator,states,commands,strata,horizons,horizons.tolist(),eps,
                              (endpoints,jacobian),(endpoints,jacobian),independent,tightened,subset,parameters)
        summary['measurements'][actuator] = measurements
        atomic_json(folder / (actuator+'_oracle_parameters.json'),parameters)
        np.savez_compressed(folder / f'reference_{actuator}_level0.npz',states=states,commands=commands,strata=strata,
                            horizons_s=horizons,perturbations_nm=eps,coarse_endpoints=endpoints,fine_endpoints=endpoints,
                            coarse_jacobians=jacobian,fine_jacobians=jacobian,numerical_envelope=envelope,tight_subset=subset,
                            oracle_status=np.zeros((128,3,2,2),dtype=int),tight_status=np.zeros((16,3,2,2),dtype=int),
                            **{'oracle_'+key:value for key,value in independent.items()},
                            **{'tight_'+key:value for key,value in tightened.items()})
    summary['measurements']['direct']['fd_discrepancy_p99'] = .004  # Within the budget but contrary to raw arrays.
    atomic_json(folder / 'summary.json',summary)
    with pytest.raises(ValueError,match='raw arrays'):
        evaluate(tmp_path,[{'run_id':'synthetic','milestone_id':'M20','phase':'calibration'}],
                 {key:{'check_id':key,'status':'pass','reason':'Synthetic fixture'} for key in CHECKS})
