"""Reference-only action-Jacobian calibration, before any learned-model audit."""
from __future__ import annotations

import numpy as np

from .clocks import to_ns
from .envs import Plant
from .execution import cpu_defaults, parent, run_job
from .io import append_event, atomic_json, read_json, utc_now
from .oracle_rigidbody import integrate, mass_matrix, parameters_from_compiled


def resolved_reference(jacobian: np.ndarray, uncertainty: float) -> bool:
    norm = np.linalg.norm(jacobian)
    return bool(np.isfinite(norm) and np.isfinite(uncertainty) and uncertainty >= 0 and norm > max(100*uncertainty,1e-14))


def _calibration_probes(root, actuator):
    code = 0 if actuator == 'direct' else 1
    robot = Plant(actuator,asset_root=root / 'assets')
    states,commands,strata,entropy = [],[],[],[]
    for stratum in (0,1):
        for probe_id in range(64):
            seed = [8201,code,stratum,probe_id]
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
            state = np.r_[rng.uniform(-np.pi,np.pi,2),rng.uniform(-2,2,2)]
            if actuator == 'lagged':
                state = np.r_[state,rng.uniform(-2,2,2)]
            if probe_id < 4:
                state[1] = 0 if probe_id%2 == 0 else np.pi
                state[2:4] = [2 if probe_id < 2 else -2, -2 if probe_id%2 else 2]
            robot.reset(state)
            if stratum:
                for _ in range(8):
                    robot.advance(rng.uniform(-2,2,2),40_000_000)
                state = robot.state()
            states.append(state)
            commands.append(rng.uniform(-1.5,1.5,2))
            strata.append(stratum)
            entropy.append(seed)
    return np.asarray(states),np.asarray(commands),np.asarray(strata,dtype=np.int64),np.asarray(entropy,dtype=np.int64)


def _plant_grid(root, actuator, states, commands, strata, horizons, step_ns, perturbations):
    robot = Plant(actuator,step_ns,root / 'assets')
    endpoints = np.empty((len(states),len(horizons),len(perturbations),2,2,states.shape[1]))
    for probe_id,(state,command) in enumerate(zip(states,commands)):
        for horizon_id,h in enumerate(horizons):
            for epsilon_id,epsilon in enumerate(perturbations):
                for axis in range(2):
                    for sign_id,sign in enumerate((-1,1)):
                        robot.reset(state,time_s=.32 if strata[probe_id] else 0.)
                        perturbed = command+sign*epsilon*np.eye(2)[axis]
                        endpoints[probe_id,horizon_id,epsilon_id,axis,sign_id] = robot.advance(perturbed,to_ns(h))['state']
    difference = endpoints[:,:,:,:,1,:]-endpoints[:,:,:,:,0,:]
    jacobians = np.swapaxes(difference/(2*np.asarray(perturbations)[None,None,:,None,None]),-1,-2)
    return endpoints,jacobians


def _oracle_grid(actuator,states,commands,horizons,parameters,epsilon,*,tight=False):
    endpoints = np.empty((len(states),len(horizons),2,2,states.shape[1]))
    nfev = np.empty((len(states),len(horizons),2,2),dtype=np.int64)
    status = np.empty_like(nfev)
    messages = np.empty(nfev.shape,dtype='U128')
    for probe_id,(state,command) in enumerate(zip(states,commands)):
        for horizon_id,h in enumerate(horizons):
            for axis in range(2):
                for sign_id,sign in enumerate((-1,1)):
                    result = integrate(state,command+sign*epsilon*np.eye(2)[axis],h,parameters,actuator,
                                       rtol=1e-12 if tight else 1e-11,atol=1e-14 if tight else 1e-13,
                                       max_step_s=.000125 if tight else .00025)
                    endpoints[probe_id,horizon_id,axis,sign_id] = result['state']
                    nfev[probe_id,horizon_id,axis,sign_id] = result['nfev']
                    status[probe_id,horizon_id,axis,sign_id] = result['status']
                    messages[probe_id,horizon_id,axis,sign_id] = result['message']
    jacobians = np.swapaxes((endpoints[:,:,:,1,:]-endpoints[:,:,:,0,:])/(2*epsilon),-1,-2)
    return {'endpoints':endpoints,'jacobians':jacobians,'nfev':nfev,'status':status,'message':messages}


def _slopes(jacobians,horizons,window,component):
    indices = [list(horizons).index(h) for h in window]
    norms = np.linalg.norm(jacobians[:,indices,component,:],axis=(-2,-1))
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError('Slope has unresolved nonpositive/nonfinite reference gain')
    log_h = np.log(window)
    regression = (log_h-log_h.mean())/np.sum((log_h-log_h.mean())**2)
    return np.log(norms)@regression


def _reference_measurements(actuator,states,commands,strata,horizons,window,eps,coarse,fine,oracle,tight,subset,parameters):
    coarse_end,coarse_j = coarse
    fine_end,fine_j = fine
    primary = fine_j[:,:,1]
    measures = {'fd_discrepancy_p99':0.,'fd_discrepancy_max':0.,'physics_jacobian_relative_max':0.,
                'oracle_jacobian_relative_max':0.,'slope_change_max':0.,'window_coverage_min':1.,'unresolved_rows':0,
                'solver_refinement_jacobian_relative_max':0.,'minimum_signal_to_uncertainty':float('inf')}
    envelopes = np.empty((len(states),len(horizons),2))
    slopes = []
    for component_id,component in enumerate((slice(0,2),slice(2,4))):
        truth = primary[:,:,component,:]
        norm = np.linalg.norm(truth,axis=(-2,-1))
        perturb = np.max(np.linalg.norm(fine_j[:,:,[0,2],component,:]-truth[:,:,None],axis=(-2,-1)),axis=2)
        numerical = np.linalg.norm(coarse_j[:,:,1,component,:]-truth,axis=(-2,-1))
        independent = np.linalg.norm(oracle['jacobians'][:,:,component,:]-truth,axis=(-2,-1))
        refinement = np.linalg.norm(tight['jacobians'][:,:,component,:]-oracle['jacobians'][subset,:,component,:],axis=(-2,-1))
        subtraction = np.max(np.abs(fine_end[:,:,:, :, :,component]),axis=(2,3,4,5))*8*np.finfo(float).eps/eps[0]
        eta = np.maximum.reduce((perturb,numerical,independent,subtraction))
        envelopes[:,:,component_id] = eta
        measures['unresolved_rows'] += sum(not resolved_reference(row,uncertainty)
                                          for row,uncertainty in zip(truth.reshape(-1,2,2),eta.ravel()))
        measures['minimum_signal_to_uncertainty'] = min(measures['minimum_signal_to_uncertainty'],float(np.min(norm/np.maximum(eta,1e-300))))
        measures['fd_discrepancy_p99'] = max(measures['fd_discrepancy_p99'],float(np.quantile(perturb/norm,.99,method='linear')))
        measures['fd_discrepancy_max'] = max(measures['fd_discrepancy_max'],float(np.max(perturb/norm)))
        measures['physics_jacobian_relative_max'] = max(measures['physics_jacobian_relative_max'],float(np.max(numerical/norm)))
        measures['oracle_jacobian_relative_max'] = max(measures['oracle_jacobian_relative_max'],float(np.max(independent/norm)))
        measures['solver_refinement_jacobian_relative_max'] = max(measures['solver_refinement_jacobian_relative_max'],float(np.max(refinement/norm[subset])))
        endpoint_error = np.max(np.abs(coarse_end[:,:,:,:,:,component]-fine_end[:,:,:,:,:,component]),axis=(2,3,4,5))
        independent_error = np.max(np.abs(oracle['endpoints'][:,:,:,:,component]-fine_end[:,:,1,:,:,component]),axis=(2,3,4))
        label = 'q' if component_id == 0 else 'v'
        measures['physics_endpoint_'+label+'_p99'] = float(np.quantile(endpoint_error,.99,method='linear'))
        measures['oracle_endpoint_'+label+'_p99'] = float(np.quantile(independent_error,.99,method='linear'))
        measures['solver_endpoint_'+label+'_max'] = float(np.max(np.abs(tight['endpoints'][:,:,:,:,component]-oracle['endpoints'][subset,:,:, :,component])))
        slope = _slopes(primary,horizons,window,component)
        old_slope = _slopes(coarse_j[:,:,1],horizons,window,component)
        measures['slope_change_max'] = max(measures['slope_change_max'],float(np.max(np.abs(slope-old_slope))))
        expected = (2,1)[component_id] if actuator == 'direct' else (3,2)[component_id]
        for stratum in (0,1):
            coverage = float(np.mean(np.abs(slope[strata == stratum]-expected) <= .10))
            measures['window_coverage_min'] = min(measures['window_coverage_min'],coverage)
            measures[f'{label}_slope_coverage_{"cold" if stratum == 0 else "warm"}'] = coverage
        slopes.append(slope)
    measures['activation_error_max'] = 0.
    if actuator == 'lagged':
        for epsilon_id,epsilon in enumerate(eps):
            for axis in range(2):
                for sign_id,sign in enumerate((-1,1)):
                    u = commands+sign*epsilon*np.eye(2)[axis]
                    expected = states[:,None,4:]+(u[:,None,:]-states[:,None,4:])*(-np.expm1(-np.asarray(horizons)[None,:,None]/.08))
                    measures['activation_error_max'] = max(measures['activation_error_max'],float(np.max(np.abs(fine_end[:,:,epsilon_id,axis,sign_id,4:]-expected))))
    measures['minimum_inertia_eigenvalue'] = float(min(np.linalg.eigvalsh(mass_matrix(state[:2],parameters))[0] for state in states))
    return measures,envelopes,np.stack(slopes,axis=1)


REFERENCE_LIMITS = {'fd_discrepancy_p99':.005,'fd_discrepancy_max':.01,'physics_jacobian_relative_max':.005,
                    'oracle_jacobian_relative_max':.01,'slope_change_max':.02,'activation_error_max':1e-9,
                    'physics_endpoint_q_p99':1e-7,'physics_endpoint_v_p99':1e-6,
                    'oracle_endpoint_q_p99':2e-7,'oracle_endpoint_v_p99':2e-6,
                    'solver_endpoint_q_max':1e-7,'solver_endpoint_v_max':1e-6,
                    'solver_refinement_jacobian_relative_max':.005,'unresolved_rows':0}


def qualify_reference(root):
    from .milestones.m20 import CHECKS
    cpu_defaults(8201)
    predecessor = parent(root,'G10')
    files = ['poc/__init__.py','poc/__main__.py','poc/contracts.py','poc/gates.py','poc/io.py','poc/execution.py',
             'poc/envs.py','poc/clocks.py','poc/oracle_rigidbody.py','poc/sensitivity.py',
             'poc/milestones/__init__.py','poc/milestones/m10.py','poc/milestones/m20.py',
             'assets/two_link_direct.xml','assets/two_link_lagged.xml','tests/test_env.py','tests/test_oracles.py']
    def produce(folder):
        libraries = {actuator:_calibration_probes(root,actuator) for actuator in ('direct','lagged')}
        parameters = {}
        for actuator in libraries:
            compiled = Plant(actuator,asset_root=root / 'assets').compiled_manifest()
            parameters[actuator] = parameters_from_compiled(compiled)
            atomic_json(folder / (actuator+'_compiled.json'),compiled)
            atomic_json(folder / (actuator+'_oracle_parameters.json'),parameters[actuator])
        window = [.0025,.005,.01]
        step,level,eps = 500_000,0,[.01,.02,.04]
        history = []
        while True:
            # Include every nominal label interval as well as all sensitivity horizons.
            horizons = sorted(set([.0025,.005,.01,.02,.03,.04,.05,.06,.08]+window))
            while to_ns(window[0]) < 10*(step//2):
                if step%2:
                    raise RuntimeError('Further refinement needs the registered rational-clock numerical branch')
                step //= 2
            fine_step = step//2
            if step%2 or fine_step <= 0 or any(to_ns(h)%step for h in horizons):
                raise RuntimeError('Reference durations require a rational numerical tick before further refinement')
            results = {}
            for actuator,(states,commands,strata,entropy) in libraries.items():
                append_event(folder / 'events.jsonl',{'event':'reference_grid_start','actuator':actuator,'level':level,'step_ns':step,'utc':utc_now()})
                coarse = _plant_grid(root,actuator,states,commands,strata,horizons,step,eps)
                fine = _plant_grid(root,actuator,states,commands,strata,horizons,fine_step,eps)
                oracle = _oracle_grid(actuator,states,commands,horizons,parameters[actuator],eps[1])
                subset = np.r_[np.arange(8),np.arange(64,72)]
                tight = _oracle_grid(actuator,states[subset],commands[subset],horizons,parameters[actuator],eps[1],tight=True)
                measurements,envelope,slopes = _reference_measurements(actuator,states,commands,strata,horizons,window,eps,coarse,fine,oracle,tight,subset,parameters[actuator])
                output = {'states':states,'commands':commands,'strata':strata,'seed_entropy':entropy,'horizons_s':np.asarray(horizons),
                          'perturbations_nm':np.asarray(eps),'coarse_step_ns':np.array(step),'fine_step_ns':np.array(fine_step),
                          'coarse_endpoints':coarse[0],'fine_endpoints':fine[0],'coarse_jacobians':coarse[1],'fine_jacobians':fine[1],
                          'numerical_envelope':envelope,'reference_slopes':slopes,'tight_subset':subset,
                          **{'oracle_'+key:value for key,value in oracle.items()},**{'tight_'+key:value for key,value in tight.items()}}
                np.savez_compressed(folder / f'reference_{actuator}_level{level}.npz',**output)
                results[actuator] = measurements
                append_event(folder / 'events.jsonl',{'event':'reference_grid_end','actuator':actuator,'level':level,'measurements':measurements,'utc':utc_now()})
            history.append({'level':level,'coarse_step_ns':step,'fine_step_ns':fine_step,'window_s':window.copy(),'measurements':results})
            numeric = all(all(np.isfinite(values[key]) and values[key] <= limit for key,limit in REFERENCE_LIMITS.items()) for values in results.values())
            order = all(values['window_coverage_min'] >= .95 for values in results.values())
            atomic_json(folder / 'convergence_history.json',{'levels':history})
            if numeric and order:
                break
            if not numeric:
                step = fine_step
            elif not order:
                window = [h/2 for h in window]
            level += 1
        numerical = {'schema_version':1,'protocol_version':'PTCO-1.0','physics_step_ns':fine_step,'physics_step_s':fine_step/1e9,
                     'fd_epsilon_nm':eps[1],'fd_sweep_nm':eps,'slope_window_ns':[to_ns(h) for h in window],
                     'slope_window_s':window,'qualified_level':level,'qualification_run_id':folder.name,'parent_gate':'G20',
                     'independent_oracle':{'method':'DOP853','rtol':1e-11,'atol':1e-13,'max_step_s':.00025},
                     'label_budget':{'q_rad':1e-7,'v_rad_s':1e-6}}
        atomic_json(folder / 'numerical.json',numerical)
        atomic_json(root / 'config/numerical.json',numerical)
        import subprocess,sys
        with (folder / 'stdout.log').open('ab') as stdout,(folder / 'stderr.log').open('ab') as stderr:
            tests = subprocess.run([sys.executable,'-m','pytest','-q','tests/test_env.py','tests/test_oracles.py'],cwd=root,stdout=stdout,stderr=stderr)
        if tests.returncode:
            raise ValueError('Independent-oracle/plant focused tests failed')
        checks = [{'check_id':key,'status':'pass','reason':'Compiled physics, independent equations, complete finite-difference/refinement arrays and reference-only window passed',
                   'scope':'M20','tolerance':None} for key in CHECKS]
        return checks,{'measurements':results,'missing_cells':[],'numerical':numerical,'phase':'calibration','learned_models_audited':False}
    return run_job(root,'M20','calibration',files,CHECKS,[predecessor],produce)
