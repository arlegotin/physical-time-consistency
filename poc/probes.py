"""Cold/warm libraries and complete, stratum-specific sensitivity aggregation."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch


def write_metric_csv(path,rows):
    import csv
    import json
    import io
    from .io import atomic_bytes
    if not rows:
        raise ValueError('Empty metric output is not completed evidence')
    fields = list(dict.fromkeys(name for row in rows for name in row))
    text = io.StringIO(newline='')
    writer = csv.DictWriter(text,fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({key:json.dumps(value,allow_nan=False) if isinstance(value,(list,dict)) else value
                         for key,value in row.items()})
    atomic_bytes(path,text.getvalue().encode('utf-8'))

from .generate import rng_for
from .history import HistoryBatch,HistoryToken


def make_probe_library(arrays,actuator,condition,role):
    values = {name:[] for name in ('states','commands','time_ns','strata','trajectory_ids','anchor_ids','probe_ids','histories')}
    for trajectory,identifier in enumerate(arrays['trajectory_id']):
        warm = int(np.searchsorted(arrays['time_ns'][trajectory],320_000_000))
        if warm >= arrays['state'].shape[1]:
            raise ValueError('Trajectory has no completed warm-state anchor at 320 ms')
        commands = rng_for(actuator,condition,role,int(identifier),3).uniform(-1.5,1.5,(2,2))
        for k,(stratum,anchor) in enumerate((('cold',0),('warm',warm))):
            time = int(arrays['time_ns'][trajectory,anchor])
            tokens = tuple(HistoryToken(int(arrays['time_ns'][trajectory,j]),arrays['state'][trajectory,j],
                          arrays['command'][trajectory,j],int(arrays['duration_ns'][trajectory,j]),
                          'observed',f'{role}-trajectory-{identifier}-anchor-{j}') for j in range(anchor))
            values['histories'].append(HistoryBatch((tokens,),np.array([time],dtype=np.int64)).eligible())
            for name,value in {'states':arrays['state'][trajectory,anchor], 'commands':commands[k],
                'time_ns':time,'strata':stratum,'trajectory_ids':int(identifier),'anchor_ids':anchor,
                'probe_ids':f'{role}-{stratum}-{identifier}'}.items():
                values[name].append(value)
    return {key:list(value) if key == 'histories' else np.asarray(value) for key,value in values.items()}


def summarize_probes(rows,expected_cells):
    if set(rows['cell_id']) != set(expected_cells):
        raise ValueError('Sensitivity model-cell coverage mismatch')
    if any(len(value) != len(rows['cell_id']) for value in rows.values()):
        raise ValueError('Missing or truncated typed sensitivity rows')
    if np.any(rows['status'] != 'ok') or not np.isfinite(rows['error_q']).all() or not np.isfinite(rows['error_v']).all():
        raise ValueError('Failed/missing sensitivity rows cannot be dropped from medians')
    result = {}
    for cell in expected_cells:
        result[cell] = {}
        for stratum in ('cold','warm'):
            result[cell][stratum] = {}
            selected = (rows['cell_id'] == cell)&(rows['stratum'] == stratum)
            for h in sorted(set(rows['h_ns'][selected])):
                mask = selected&(rows['h_ns'] == h)
                q,v = rows['error_q'][mask],rows['error_v'][mask]
                entry = {'n_probes':int(mask.sum()),'primary_error':float(np.median((q+v)/2)),
                         'q_median':float(np.median(q)),'v_median':float(np.median(v)),
                         'q_p90':float(np.quantile(q,.9,method='linear')),
                         'v_p90':float(np.quantile(v,.9,method='linear'))}
                for metric in ('error','gain'):
                    for component in ('q','v'):
                        for side in ('lower','upper'):
                            key = f'{metric}_{component}_{side}'
                            if key in rows:
                                numbers = rows[key][mask]
                                entry[key+'_median'] = float(np.median(numbers))
                                entry[key+'_p90'] = float(np.quantile(numbers,.9,method='linear'))
                if 'error_q_lower' in rows:
                    for side in ('lower','upper'):
                        entry['primary_error_'+side] = float(np.median((rows['error_q_'+side][mask]+rows['error_v_'+side][mask])/2))
                for component in ('q','v'):
                    if 'gain_'+component in rows:
                        gains = rows['gain_'+component][mask]
                        entry['gain_'+component+'_median'] = float(np.median(gains))
                        entry['gain_'+component+'_fraction'] = float(np.mean((gains >= .5)&(gains <= 1.5)))
                        if 'gain_'+component+'_lower' in rows:
                            low,high = rows['gain_'+component+'_lower'][mask],rows['gain_'+component+'_upper'][mask]
                            entry['gain_'+component+'_certain_fraction'] = float(np.mean((low >= .5)&(high <= 1.5)))
                            entry['gain_'+component+'_possible_fraction'] = float(np.mean((high >= .5)&(low <= 1.5)))
                result[cell][stratum][str(int(h))] = entry
            if not result[cell][stratum]:
                raise ValueError('Cold/warm probe-stratum coverage is incomplete')
    return result


def load_audit_inputs(parameters,phase,root):
    from .contracts import verify_run
    from .dataset import load_corpus
    from .io import read_json,sha256_file,run_directory
    from .models.common import Scales
    from .models.factory import build_model
    root = Path(root)
    training_path = root/parameters['training_manifest']
    training = read_json(training_path)
    verify_run(root,training)
    folder = run_directory(root,training)
    model_summary = read_json(folder/'summary.json')
    identity = model_summary['parameters']
    for name in ('model_id','actuator','condition','seed'):
        if identity[name] != parameters[name]:
            raise ValueError('Audit cell differs from its frozen checkpoint: '+name)
    corpus_path = root/parameters['corpus_manifest']
    corpus = read_json(corpus_path)
    if corpus['role'] not in ('development','confirmation'):
        raise ValueError('Model audit requires a declared development or sealed confirmation workload')
    arrays = load_corpus(corpus_path,phase,corpus['role'],'evaluation')
    if corpus['actuator'] != parameters['actuator']:
        raise ValueError('Audit actuator and physical workload disagree')
    freeze_path = root/parameters['freeze_path']
    freeze = read_json(freeze_path)
    if freeze['protocol_version'] != 'PTCO-1.0' or freeze['statistical_family_size'] != 280:
        raise ValueError('Model audit requires a full preregistered recipe/evaluator freeze')
    if sha256_file(folder/'weights.pt') not in freeze['checkpoint_hashes']:
        raise ValueError('Checkpoint was not frozen before opening the model audit')
    normalization = read_json(folder/'normalization.json')
    scales = Scales(*(torch.tensor(normalization[key]) for key in ('scale_q','scale_v','velocity_input_rms')))
    model = build_model(identity['model_id'],identity['actuator'],scales,identity['seed'])
    model.load_state_dict(torch.load(folder/'weights.pt',weights_only=True,map_location='cpu'))
    hashes = {str(path.relative_to(root)):sha256_file(path) for path in
              (training_path,folder/'weights.pt',folder/'normalization.json',corpus_path,
               corpus_path.parent/'trajectories.npz',freeze_path)}
    return model,arrays,normalization,{'model':identity,'corpus':corpus,'input_hashes':hashes,
               'checkpoint_hash':sha256_file(folder/'weights.pt'),'training_record':training}


def reference_library(library,actuator,horizons,numerical,root):
    """Reusable reference-only measurements, with raw FD, refined-physics and ODE branches."""
    import hashlib
    import io
    from .envs import Plant
    from .io import atomic_json,atomic_bytes,canonical_json,read_json,sha256_file
    from .model_sensitivity import reference_jacobians
    from .oracle_rigidbody import parameters_from_compiled,integrate
    root = Path(root)
    step = numerical['physics_step_ns']
    identity = {'actuator':actuator,'horizons_ns':horizons,'step_ns':step,
                'states':library['states'].tolist(),'commands':library['commands'].tolist(),
                'time_ns':library['time_ns'].tolist(),
                'sources':{path:sha256_file(root/path) for path in
                    ('poc/probes.py','poc/model_sensitivity.py','poc/envs.py','poc/oracle_rigidbody.py',
                     'assets/two_link_'+actuator+'.xml')}}
    key = hashlib.sha256(canonical_json(identity)).hexdigest()
    folder = root/'data'/'audit_cache'/'reference'/key
    folder.mkdir(parents=True,exist_ok=True)
    base = Plant(actuator,step,root/'assets')
    refined = Plant(actuator,step//2,root/'assets')
    physical = parameters_from_compiled(base.compiled_manifest())
    rows = {name:[] for name in ('endpoints','jacobians','refined_endpoints','refined_jacobians',
          'oracle_endpoints','oracle_jacobians','oracle_nfev','numerical_envelope')}
    for probe,(state,command,time) in enumerate(zip(library['states'],library['commands'],library['time_ns'])):
        hold_rows = {name:[] for name in rows}
        for horizon in horizons:
            path = folder/f'{probe}-{horizon}.npz'
            marker = path.with_suffix('.json')
            if path.exists() and marker.exists():
                if sha256_file(path) != read_json(marker)['sha256']:
                    raise ValueError('Completed reference-Jacobian cell hash mismatch')
                with np.load(path,allow_pickle=False) as archive:
                    row = {name:archive[name] for name in archive.files}
            else:
                base.reset(state,float(time)/1e9); refined.reset(state,float(time)/1e9)
                fds = [reference_jacobians(base,state,command,int(horizon),epsilon) for epsilon in (.01,.02,.04)]
                fine = reference_jacobians(refined,state,command,int(horizon),.02)
                oracle = np.empty((2,2,len(state)),dtype=np.float64)
                counts = np.empty((2,2),dtype=np.int64)
                for axis in range(2):
                    for sign_id,sign in enumerate((-1,1)):
                        result = integrate(state,command+sign*.02*np.eye(2)[axis],horizon/1e9,physical,actuator)
                        if not result['success'] or result['status'] != 0:
                            raise ValueError('Independent reference integration failed')
                        oracle[axis,sign_id],counts[axis,sign_id] = result['state'],result['nfev']
                oracle_j = ((oracle[:,1]-oracle[:,0])/(2*.02)).T
                jac = fds[1]['state_jac']
                eta = []
                for component in (slice(0,2),slice(2,4)):
                    norm = np.linalg.norm(jac[component])
                    fd_errors = [np.linalg.norm(item['state_jac'][component]-jac[component]) for item in (fds[0],fds[2])]
                    step_error = np.linalg.norm(fine['state_jac'][component]-jac[component])
                    oracle_error = np.linalg.norm(oracle_j[component]-jac[component])
                    subtraction = 8*np.finfo(float).eps*np.max(np.abs(fds[1]['endpoints'][:,:,component]))/.02
                    uncertainty = float(max(*fd_errors,step_error,oracle_error,subtraction))
                    if norm <= max(100*uncertainty,1e-14) or max(fd_errors) > .01*norm or step_error > .005*norm or oracle_error > .01*norm:
                        raise ValueError('Reference-only sensitivity cell requires numerical discrimination')
                    eta.append(uncertainty)
                row = {'endpoints':np.stack([item['endpoints'] for item in fds]),'jacobians':jac,
                       'refined_endpoints':fine['endpoints'],'refined_jacobians':fine['state_jac'],
                       'oracle_endpoints':oracle,'oracle_jacobians':oracle_j,'oracle_nfev':counts,
                       'numerical_envelope':np.array(eta)}
                buffer = io.BytesIO(); np.savez_compressed(buffer,**row)
                atomic_bytes(path,buffer.getvalue())
                atomic_json(marker,{'sha256':sha256_file(path),'reference_identity':key,
                                    'probe_id':str(library['probe_ids'][probe]),'h_ns':horizon})
            for name in rows:
                hold_rows[name].append(row[name])
        for name in rows:
            rows[name].append(hold_rows[name])
    result = {name:np.asarray(values) for name,values in rows.items()}
    result.update(states=library['states'],commands=library['commands'],time_ns=library['time_ns'],
                  horizons_ns=np.asarray(horizons,dtype=np.int64),perturbations_nm=np.array([.01,.02,.04]),
                  physics_step_ns=np.array(step),probe_ids=library['probe_ids'])
    return result
