"""Fixed-initial-state new-command derivatives, separate from reference calibration."""
from __future__ import annotations

import copy
import numpy as np
import torch

from .envs import Plant
from .models.common import require_same_state


def action_jacobians(model,state,command,h_s,history=None,power_mode=None,solver_substeps=None):
    audit = copy.deepcopy(model).double().eval()
    x = torch.tensor(np.asarray(state,dtype=np.float64)[None,:],dtype=torch.float64)
    original = x.clone()
    u = torch.tensor(np.asarray(command,dtype=np.float64),dtype=torch.float64,requires_grad=True)
    packet = None
    work = {'network_calls':0,'vector_field_calls':0,'history_tokens_processed':0}
    failed = False
    def evaluate(value):
        nonlocal packet,failed
        kwargs = {'history':history}
        if power_mode is not None:
            kwargs['power_mode'] = power_mode
        if solver_substeps is not None:
            kwargs['solver_substeps'] = solver_substeps
        packet = audit.predict(x,value[None,:],h_s,**kwargs)
        require_same_state(original,x)
        failed = failed or packet.status != ('ok',)
        for name in work:
            work[name] += getattr(packet,name)
        return packet.next_state[0]
    jacobian = torch.autograd.functional.jacobian(evaluate,u,create_graph=False,strict=False).detach().numpy()
    epsilons = np.array([1e-4,2e-4])
    if np.any(np.abs(np.asarray(command))+epsilons.max()>2):
        raise ValueError('Model central-difference probes must lie strictly inside torque bounds')
    endpoints = np.empty((2,2,2,len(jacobian)),dtype=np.float64)
    with torch.no_grad():
        for epsilon_id,epsilon in enumerate(epsilons):
            for axis in range(2):
                for sign_id,sign in enumerate((-1,1)):
                    endpoints[epsilon_id,axis,sign_id] = evaluate(u+sign*epsilon*torch.eye(2,dtype=torch.float64)[axis]).numpy()
    central = np.swapaxes((endpoints[:,:,1]-endpoints[:,:,0])/(2*epsilons[:,None,None]),-1,-2)
    envelopes = []
    for component in (slice(0,2),slice(2,4)):
        discrepancy = np.max(np.linalg.norm(central[:,component,:]-jacobian[component],axis=(-2,-1)))
        subtraction = 8*np.finfo(float).eps*np.max(np.abs(endpoints[:,:,:,component]))/epsilons.min()
        envelopes.append(float(max(discrepancy,subtraction)))
    status = 'ok' if not failed and np.isfinite(jacobian).all() and np.isfinite(central).all() else 'model_failure'
    return {'status':status,'q_jac':jacobian[:2],'v_jac':jacobian[2:4],
            'w_jac':jacobian[4:] if len(jacobian) == 6 else None,
            'state_jac':jacobian,'initial_state':np.asarray(state).copy(),'command':np.asarray(command).copy(),
            'central_endpoints':endpoints,'central_jacobians':central,'central_epsilons_nm':epsilons,
            'model_numerical_envelope':envelopes,**work}


def reference_jacobians(plant,state,command,h_ns,epsilon_nm):
    if type(h_ns) is not int or h_ns < 0 or h_ns%plant.physics_step_ns:
        raise ValueError('Reference horizon must be nonnegative integer numerical ticks')
    if not np.isfinite(epsilon_nm) or epsilon_nm <= 0:
        raise ValueError('Reference central differences require positive physical torque perturbations')
    command = np.asarray(command,dtype=np.float64)
    if command.shape != (2,) or not np.isfinite(command).all() or np.any(np.abs(command)+epsilon_nm>2):
        raise ValueError('Both command perturbations must remain inside actuator bounds')
    state = np.asarray(state,dtype=np.float64).copy()
    branch = Plant(plant.actuator,plant.physics_step_ns,plant.xml_path.parent)
    endpoints = np.empty((2,2,len(state)),dtype=np.float64)
    for axis in range(2):
        for sign_id,sign in enumerate((-1,1)):
            branch.reset(state,time_s=plant.time_ns/1e9)
            require_same_state(state,branch.state())
            u = command+sign*epsilon_nm*np.eye(2)[axis]
            endpoints[axis,sign_id] = branch.advance(u,h_ns)['state'] if h_ns else branch.state()
    jacobian = ((endpoints[:,1]-endpoints[:,0])/(2*epsilon_nm)).T
    return {'status':'ok','endpoints':endpoints,'state_jac':jacobian,'q_jac':jacobian[:2],'v_jac':jacobian[2:4],
            'w_jac':jacobian[4:] if len(state) == 6 else None,'initial_state':state,'command':command,
            'epsilon_nm':epsilon_nm,'h_ns':h_ns,'physics_step_ns':plant.physics_step_ns}


def run_e01(job,root,resume=False):
    """Audit every probe/horizon; immutable cell caches preserve complete ± branches."""
    from pathlib import Path
    import io
    import hashlib
    from .audit_jobs import audit_job
    from .io import atomic_bytes,atomic_json,canonical_json,read_json,sha256_file
    from .metrics import jacobian_metrics,fit_slope
    from .probes import load_audit_inputs,make_probe_library,summarize_probes,reference_library,write_metric_csv
    root = Path(root).resolve()
    parameters = dict(job['semantic_inputs'])
    model,corpus,normalization,metadata = load_audit_inputs(parameters,job['phase'],root)
    library = make_probe_library(corpus,parameters['actuator'],metadata['corpus']['condition'],metadata['corpus']['role'])
    numerical = read_json(root/'config/numerical.json')
    horizons = sorted(set([2_500_000,5_000_000,10_000_000,20_000_000,40_000_000,80_000_000]+
                          [int(round(h*1e9)) for h in numerical['slope_window_s']]))
    cell = f'{parameters["actuator"]}__{parameters["condition"]}__{parameters["loss_mode"]}__{parameters["model_id"]}__seed{parameters["seed"]}'
    if parameters.get('power_mode'):
        cell += '__power_'+parameters['power_mode']
    if parameters.get('solver_substeps'):
        if parameters['model_id'] not in ('node_generic','node_structured') or parameters['solver_substeps'] != 2:
            raise ValueError('The prespecified solver audit uses exactly two half steps of the same NODE field')
        cell += '__solver_2'
    def produce(folder):
        reference = reference_library(library,parameters['actuator'],horizons,numerical,root)
        buffer = io.BytesIO()
        np.savez_compressed(buffer,**{key:value for key,value in library.items() if key != 'histories'})
        atomic_bytes(folder/'probes.npz',buffer.getvalue())
        buffer = io.BytesIO(); np.savez_compressed(buffer,**reference)
        atomic_bytes(folder/'reference.npz',buffer.getvalue())
        cache_identity = {'checkpoint':metadata['checkpoint_hash'],'power_mode':parameters.get('power_mode'),
                          'solver_substeps':parameters.get('solver_substeps'),
                          'input_hashes':metadata['input_hashes'],
                          'environment_sha256':sha256_file(root/'locks/environment.json'),
                          'reference_sha256':sha256_file(folder/'reference.npz'),
                          'model_sensitivity_sha256':sha256_file(Path(__file__)),
                          'model_sources':{path:sha256_file(root/path) for path in
                            ('poc/history.py','poc/probes.py','poc/models/common.py','poc/models/power.py','poc/models/mlp.py',
                             'poc/models/sequence.py','poc/models/ode.py','poc/models/factory.py')}}
        key = hashlib.sha256(canonical_json(cache_identity)).hexdigest()
        cache = root/'data'/'audit_cache'/'model_jacobians'/key
        cache.mkdir(parents=True,exist_ok=True)
        rows = {name:[] for name in ('cell_id','model_id','checkpoint_hash','seed','actuator','training_condition',
                'loss_mode','stratum','probe_id','h_ns','reference_row_id','status','q_jac','v_jac',
                'error_q','error_v','error_q_lower','error_q_upper','error_v_lower','error_v_upper',
                'gain_q','gain_v','gain_q_lower','gain_q_upper','gain_v_lower','gain_v_upper',
                'model_envelope_q','model_envelope_v','alignment_q','alignment_v','singular_values_q','singular_values_v',
                'central_endpoints','central_jacobians','network_calls','vector_field_calls','history_tokens_processed')}
        for probe in range(len(library['states'])):
            for horizon_id,horizon in enumerate(horizons):
                path = cache/f'{probe}-{horizon}.npz'
                marker = path.with_suffix('.json')
                if path.exists() and marker.exists():
                    if read_json(marker)['sha256'] != sha256_file(path):
                        raise ValueError('Completed model-Jacobian cell hash mismatch')
                    with np.load(path,allow_pickle=False) as archive:
                        measured = {name:archive[name] for name in archive.files}
                else:
                    measured = action_jacobians(model,library['states'][probe],library['commands'][probe],horizon/1e9,
                                    library['histories'][probe],parameters.get('power_mode'),parameters.get('solver_substeps'))
                    buffer = io.BytesIO()
                    np.savez_compressed(buffer,**{name:value for name,value in measured.items() if value is not None})
                    atomic_bytes(path,buffer.getvalue())
                    atomic_json(marker,{'sha256':sha256_file(path),'probe_id':str(library['probe_ids'][probe]),
                                        'h_ns':horizon,'cache_identity':key})
                entry = {'cell_id':cell,'model_id':parameters['model_id'],'checkpoint_hash':metadata['checkpoint_hash'],
                    'seed':parameters['seed'],'actuator':parameters['actuator'],'training_condition':parameters['condition'],
                    'loss_mode':parameters['loss_mode'],'stratum':library['strata'][probe],
                    'probe_id':library['probe_ids'][probe],'h_ns':horizon,
                    'reference_row_id':probe*len(horizons)+horizon_id,'status':str(measured['status'])}
                for component,index in (('q',0),('v',1)):
                    metric = jacobian_metrics(measured[component+'_jac'],reference['jacobians'][probe,horizon_id,index*2:index*2+2],
                              {'reference':reference['numerical_envelope'][probe,horizon_id,index],
                               'model':measured['model_numerical_envelope'][index]})
                    if metric['status'] != 'ok':
                        entry['status'] = metric['status']
                    entry[component+'_jac'] = measured[component+'_jac']
                    entry['error_'+component] = metric['error'] if metric['error'] is not None else np.nan
                    entry['gain_'+component] = metric['gain'] if metric['gain'] is not None else np.nan
                    for kind in ('error','gain'):
                        bounds = metric[kind+'_bounds']
                        for side,i in (('lower',0),('upper',1)):
                            entry[kind+'_'+component+'_'+side] = bounds[i] if bounds is not None else np.nan
                    entry['model_envelope_'+component] = measured['model_numerical_envelope'][index]
                    entry['alignment_'+component] = metric['alignment'] if metric['alignment'] is not None else np.nan
                    entry['singular_values_'+component] = metric['singular_values'] if metric['singular_values'] is not None else [np.nan]*2
                for name in ('central_endpoints','central_jacobians','network_calls','vector_field_calls','history_tokens_processed'):
                    entry[name] = measured[name]
                for name in rows:
                    rows[name].append(entry[name])
        typed = {name:np.asarray(values) for name,values in rows.items()}
        buffer = io.BytesIO(); np.savez_compressed(buffer,**typed)
        atomic_bytes(folder/'jacobians.npz',buffer.getvalue())
        failed = int(np.count_nonzero(typed['status'] != 'ok'))
        summaries = None if failed else summarize_probes(typed,[cell])
        slopes = []
        window = [int(round(h*1e9)) for h in numerical['slope_window_s']]
        for probe in range(len(library['states'])):
            for component,index in (('q',0),('v',1)):
                selected = [probe*len(horizons)+horizons.index(h) for h in window]
                norms = np.linalg.norm(typed[component+'_jac'][selected],axis=(1,2))
                zeta = typed['model_envelope_'+component][selected]
                fit = fit_slope(np.array(window)/1e9,norms,np.c_[np.maximum(0,norms-zeta),norms+zeta])
                oracle_norms = np.array([np.linalg.norm(reference['jacobians'][probe,horizons.index(h),index*2:index*2+2]) for h in window])
                oracle_eta = np.array([reference['numerical_envelope'][probe,horizons.index(h),index] for h in window])
                oracle = fit_slope(np.array(window)/1e9,oracle_norms,np.c_[oracle_norms-oracle_eta,oracle_norms+oracle_eta])
                slopes.append({'probe_id':str(library['probe_ids'][probe]),'stratum':str(library['strata'][probe]),
                    'component':component,'theoretical_order':(2 if component == 'q' else 1)+int(parameters['actuator']=='lagged'),
                    'oracle_slope':oracle['slope'],**fit})
        atomic_json(folder/'slopes.json',{'rows':slopes,'window_ns':window})
        common = {'run_id':folder.name,'experiment_id':'E01','phase':job['phase'],
                  'model_id':parameters['model_id'],'actuator':parameters['actuator'],
                  'condition':parameters['condition'],'loss_mode':parameters['loss_mode'],
                  'seed':parameters['seed'],'workload_id':parameters['corpus_manifest']}
        write_metric_csv(folder/'slopes.csv',[{**common,'metric':'action_jacobian_slope','statistic':'three_point_OLS',
             'value':row['slope'],'unit':'dimensionless','sample_count':3,'status':row['status'],
             'evidence':'jacobians.npz','probe_id':row['probe_id'],'stratum':row['stratum'],
             'component':row['component'],'window_ns':window,'slope_bounds':row['slope_bounds'],
             'oracle_slope':row['oracle_slope'],'theoretical_order':row['theoretical_order']} for row in slopes])
        metric_rows = []
        if summaries:
            for stratum,by_h in summaries[cell].items():
                for h,metrics in by_h.items():
                    for name,value in metrics.items():
                        if name == 'n_probes':
                            continue
                        metric_rows.append({**common,'stratum':stratum,'h_ns':int(h),'metric':name,
                            'statistic':'per_seed_probe_statistic','value':value,'unit':'dimensionless',
                            'sample_count':metrics['n_probes'],'status':'ok','evidence':'jacobians.npz'})
        else:
            metric_rows = [{**common,'metric':'normalized_jacobian_error','statistic':'median','value':None,
                           'unit':'dimensionless','sample_count':len(library['states']),
                           'status':'model_failure','evidence':'jacobians.npz'}]
        write_metric_csv(folder/'metrics_by_seed.csv',metric_rows)
        atomic_json(folder/'coordinates.json',{'coordinate_order':'q1,q2,v1,v2[,w1,w2]',
                 'command_unit':'N m','jacobian_shape':[2,2], 'cell_id':cell,'horizons_ns':horizons,
                 'model_metadata':metadata['model'],'normalization':normalization})
        return [{'check_id':'E01.coverage','status':'pass','scope':'E01','tolerance':None,
                  'reason':'Every cold/warm state, horizon and raw derivative branch retained; model failures explicit'}],{
                 'cell_id':cell,'metrics':summaries,'slopes':slopes,'row_count':len(typed['h_ns']),
                 'failed_rows':failed,'missing_cells':[],'checkpoint_hash':metadata['checkpoint_hash'],
                 'model_metadata':metadata['model'],'network_calls':int(typed['network_calls'].sum()),
                 'vector_field_calls':int(typed['vector_field_calls'].sum()),
                 'history_tokens_processed':int(typed['history_tokens_processed'].sum())}
    return audit_job(root,'E01',job,metadata,('E01.coverage',),produce)
