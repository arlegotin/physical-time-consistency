"""Independent planar rigid-body equations and DOP853 validation integration."""
from __future__ import annotations

import math
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation


def parameters_from_compiled(manifest: dict) -> dict:
    masses = np.asarray(manifest['body_mass'],dtype=np.float64)[1:3]
    centers = np.asarray(manifest['body_ipos'],dtype=np.float64)[1:3,0]
    inertias = np.asarray(manifest['body_inertia'],dtype=np.float64)[1:3]
    quaternions = np.asarray(manifest['body_iquat'],dtype=np.float64)[1:3]
    tensors = []
    for inertia, quaternion in zip(inertias,quaternions):
        rotation = Rotation.from_quat(quaternion[[1,2,3,0]]).as_matrix()
        tensors.append(rotation@np.diag(inertia)@rotation.T)
    i1,i2 = (float(tensor[2,2]) for tensor in tensors)
    m1,m2 = masses
    c1,c2 = centers
    length = float(manifest['body_pos'][2][0])
    return {'theta1':float(i1+m1*c1*c1+m2*length*length), 'theta2':float(i2+m2*c2*c2),
            'theta3':float(m2*length*c2), 'armature':manifest['dof_armature'], 'damping':manifest['dof_damping'],
            'tau_s':.08, 'inertia_body_zz':[i1,i2], 'inertia_body_tensors':[tensor.tolist() for tensor in tensors],
            'source_xml_sha256':manifest['xml_sha256'], 'privileged_validation_only':True}


def mass_matrix(q: np.ndarray, parameters: dict) -> np.ndarray:
    first, second, cross = (parameters[name] for name in ('theta1','theta2','theta3'))
    coupling = cross*np.cos(q[1])
    armature = parameters['armature']
    return np.array([[first+second+2*coupling+armature[0],second+coupling],
                     [second+coupling,second+armature[1]]],dtype=np.float64)


def rhs(t_s: float, qv: np.ndarray, command: np.ndarray, initial_w: np.ndarray | None, parameters: dict) -> np.ndarray:
    q,v = qv[:2],qv[2:4]
    cross = parameters['theta3']*np.sin(q[1])
    coriolis = np.array([-cross*(2*v[0]*v[1]+v[1]**2),cross*v[0]**2])
    effort = command if initial_w is None else initial_w+(command-initial_w)*(-np.expm1(-t_s/parameters['tau_s']))
    acceleration = np.linalg.solve(mass_matrix(q,parameters),effort-coriolis-np.asarray(parameters['damping'])*v)
    return np.concatenate((v,acceleration))


def integrate(state: np.ndarray, command: np.ndarray, h_s: float, parameters: dict, actuator: str,
              rtol: float = 1e-11, atol: float = 1e-13, max_step_s: float = .00025) -> dict:
    state,command = np.asarray(state,dtype=np.float64),np.asarray(command,dtype=np.float64)
    if actuator not in {'direct','lagged'} or state.shape != (4 if actuator == 'direct' else 6,) or command.shape != (2,):
        raise ValueError('Oracle expects fully observed planar q/v[/w] and two physical commands')
    if not np.isfinite(state).all() or not np.isfinite(command).all() or not math.isfinite(h_s) or h_s < 0:
        raise ValueError('Oracle inputs must be finite, with nonnegative seconds')
    if min(rtol,atol,max_step_s) <= 0 or not all(math.isfinite(x) for x in (rtol,atol,max_step_s)):
        raise ValueError('Positive finite ODE tolerances and maximum step required')
    if h_s == 0:
        return {'state':state.copy(),'status':0,'message':'Exact zero-time no-op','success':True,'nfev':0,'endpoint_time_s':0.,'wall_seconds':0.}
    w = state[4:].copy() if actuator == 'lagged' else None
    started = time.perf_counter_ns()
    solution = solve_ivp(lambda t,x:rhs(t,x,command,w,parameters),(0.,float(h_s)),state[:4],method='DOP853',
                         rtol=rtol,atol=atol,max_step=max_step_s,t_eval=[float(h_s)])
    if not solution.success or solution.t[-1] != h_s or not np.isfinite(solution.y).all():
        raise RuntimeError(f'Independent integration failed: {solution.status} {solution.message}')
    endpoint = solution.y[:,-1]
    if actuator == 'lagged':
        endpoint = np.concatenate((endpoint,w+(command-w)*(-np.expm1(-h_s/parameters['tau_s']))))
    return {'state':endpoint,'success':bool(solution.success),'status':int(solution.status),'message':solution.message,
            'nfev':int(solution.nfev),'endpoint_time_s':float(solution.t[-1]),'wall_seconds':(time.perf_counter_ns()-started)/1e9,
            'method':'DOP853','rtol':rtol,'atol':atol,'max_step_s':max_step_s}

