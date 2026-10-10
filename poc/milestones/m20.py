from pathlib import Path

CHECKS = ('compiled_physics','activation','independent_rigidbody','finite_difference','physics_refinement',
          'signal_resolution','slope_window','snapshot_replay','warnings','units','tests')


def describe(root: Path):
    return {'expected_cells':list(CHECKS),'input_paths':['assets/two_link_direct.xml','assets/two_link_lagged.xml',
            'config/base.json','locks/environment.json','locks/requirements.lock','decisions/gate_inputs.json'],
            'expected_jobs':[{'job_id':'M20-reference-calibration','status':'verify_or_produce','predecessor':'G10'}],
            'resource_projection':{'initial_reference_level':True,'mujoco_endpoint_branches':55_296,
                                   'independent_endpoint_branches':9_216,'tightened_endpoint_branches':1_152,
                                   'mujoco_physics_steps':5_483_520,'wall_seconds':None,
                                   'note':'Further reference-only refinement is conditional; no owner-imposed caps'}}


def run(root: Path,resume: bool = False):
    from poc.sensitivity import qualify_reference
    return qualify_reference(root)


def evaluate(root: Path,evidence: list[dict],seen: dict):
    import numpy as np
    from poc.io import read_json,run_directory
    from poc.sensitivity import REFERENCE_LIMITS,resolved_reference,_reference_measurements
    if len(evidence) != 1 or evidence[0]['phase'] != 'calibration' or set(seen) != set(CHECKS) or any(check['status'] != 'pass' for check in seen.values()):
        raise ValueError('G20 requires complete, valid reference calibration')
    folder = run_directory(root,evidence[0])
    summary = read_json(folder / 'summary.json')
    numerical = summary['numerical']
    for actuator in ('direct','lagged'):
        values = summary['measurements'][actuator]
        for key,limit in REFERENCE_LIMITS.items():
            if key not in values or not np.isfinite(values[key]) or values[key] > limit:
                raise ValueError('Unqualified reference budget: '+key)
        if values['window_coverage_min'] < .95:
            raise ValueError('Reference slope window is not asymptotic in each stratum')
        with np.load(folder / f'reference_{actuator}_level{numerical["qualified_level"]}.npz',allow_pickle=False) as arrays:
            if arrays['states'].shape != (128,4 if actuator == 'direct' else 6) or np.any(arrays['oracle_status']) or np.any(arrays['tight_status']):
                raise ValueError('Reference coverage or independent integration status is invalid')
            primary = arrays['fine_jacobians'][:,:,1]
            independent = {key:arrays['oracle_'+key] for key in ('endpoints','jacobians')}
            tightened = {key:arrays['tight_'+key] for key in ('endpoints','jacobians')}
            derived,_,_ = _reference_measurements(actuator,arrays['states'],arrays['commands'],arrays['strata'],
                  arrays['horizons_s'],numerical['slope_window_s'],arrays['perturbations_nm'],
                  (arrays['coarse_endpoints'],arrays['coarse_jacobians']),(arrays['fine_endpoints'],arrays['fine_jacobians']),
                  independent,tightened,arrays['tight_subset'],read_json(folder / (actuator+'_oracle_parameters.json')))
            if any(not np.isclose(values[key],value,atol=1e-15,rtol=1e-12) for key,value in derived.items()):
                raise ValueError('Reported reference metrics disagree with complete raw arrays')
            for i,component in enumerate((slice(0,2),slice(2,4))):
                if not all(resolved_reference(j,eta) for j,eta in zip(primary[:,:,component,:].reshape(-1,2,2),arrays['numerical_envelope'][:,:,i].ravel())):
                    raise ValueError('A stable zero/unresolved reference cannot qualify')
    evaluations = [{'criterion_id':'K01.oracle.'+key,'status':'pass','estimate':1.,'bounds':None,'threshold':None,
                    'operator':'deterministic','reason':seen[key]['reason']} for key in CHECKS]
    return 'GO','valid',evaluations
