"""Measured fresh-data prediction/control evidence for PTCO-AFF-0.2."""
from __future__ import annotations

import argparse
import io
import math
import os
from pathlib import Path
import time

for _variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_variable]='1'

import numpy as np
import torch

from .affine_fresh import (FREEZE as FIT_FREEZE, MODELS, PROTOCOL, SEEDS, build_fresh_model,
                          fresh_rng, load_fresh_arrays, load_freeze, selected_freeze, source_files)
from .affine_fresh import job_configuration
from .affine_experiment import verify_bindings
from .io import atomic_bytes,atomic_json,job_key,job_lock,read_json,sha256_file,utc_now
from .models.common import Scales

FREEZE='config/affine_confirmation_freeze.json'
CONFIGURATIONS=(('affine_flow',5000000),('node_structured',5000000),('node_structured',40000000))
COARSE_CONFIGURATIONS=(('affine_flow',40000000),)
COARSE_SPEC='docs/superpowers/specs/2026-10-08-affine-coarse-comparison.md'


def fresh_probes(arrays):
    states,commands,strata,identities=[],[],[],[]
    for i,identity in enumerate(arrays['trajectory_id']):
        warm=int(np.searchsorted(arrays['time_ns'][i],320000000))
        if warm>=arrays['state'].shape[1]:
            raise ValueError('Fresh warm probe lacks completed physical history')
        states.extend([arrays['state'][i,0].copy(),arrays['state'][i,warm].copy()])
        commands.extend(fresh_rng('confirmation',int(identity),3).uniform(-1.5,1.5,(2,2)))
        strata.extend(['cold','warm']);identities.extend([2*int(identity),2*int(identity)+1])
    return dict(states=np.asarray(states,dtype=np.float64),commands=np.asarray(commands,dtype=np.float64),
                strata=np.asarray(strata),probe_ids=np.asarray(identities,dtype=np.int64),
                horizons_ns=np.array([5000000,10000000],dtype=np.int64))


def prediction_rows(raw,coordinates):
    from .shared_state_diagnostic import decompose
    for key in ('probe_ids','strata','horizons_ns'):
        if not np.array_equal(raw[key],coordinates[key]):
            raise ValueError('Fresh prediction coordinate mismatch')
    shape=(len(raw['probe_ids']),len(raw['horizons_ns']),6)
    fields=('truth_zero','truth_command','model_zero','model_command','model_composed')
    if any(raw[k].shape!=shape or not np.isfinite(raw[k]).all() for k in fields):
        raise ValueError('Fresh prediction requires matching finite endpoint vectors')
    errors=decompose(raw['model_zero'],raw['model_command'],raw['truth_zero'],
                     raw['truth_command'],raw['model_composed'])
    rows=[]
    for stratum in ('cold','warm'):
        mask=raw['strata']==stratum
        if not mask.any():raise ValueError('Fresh prediction missing stratum')
        for hi,horizon in enumerate(raw['horizons_ns']):
            for component,index in (('q',slice(0,2)),('v',slice(2,4))):
                row=dict(stratum=stratum,h_ns=int(horizon),component=component,n_probes=int(mask.sum()))
                for metric,value in errors.items():
                    norm=np.linalg.norm(value[mask,hi,index],axis=-1)
                    row.update({metric+'_mean':float(norm.mean()),metric+'_median':float(np.median(norm)),
                                metric+'_p90':float(np.quantile(norm,.9,method='linear'))})
                rows.append(row)
    return rows


def paired_comparison(baseline,method,look,numerical_error):
    from .statistics import bootstrap_indices,contrast_bounds,fieller_improvement,look_alpha
    b,m=np.asarray(baseline,dtype=float),np.asarray(method,dtype=float)
    if (b.ndim!=1 or b.shape!=m.shape or len(b)<2 or not np.isfinite(b).all() or not np.isfinite(m).all()
            or np.any(b<0) or np.any(m<0) or not math.isfinite(numerical_error) or numerical_error<0):
        raise ValueError('Complete paired nonnegative training-seed statistics required')
    indices=bootstrap_indices(len(b),look);alpha=look_alpha(look)
    contrast=contrast_bounds(b-m,alpha,indices,{'lower':2*numerical_error,'upper':2*numerical_error})
    ratio_envelope=0. if numerical_error==0 else 2*numerical_error/max(float(b.min())-numerical_error,1e-300)
    ratio=fieller_improvement(b,m,.05,{'lower':ratio_envelope,'upper':ratio_envelope},indices)
    return dict(baseline_by_seed=b.tolist(),method_by_seed=m.tolist(),contrast=contrast,ratio=ratio,
                baseline_mean=float(b.mean()),method_mean=float(m.mean()),
                point_improvement=None if b.mean()<=0 else float(1-m.mean()/b.mean()))


def holm_control_comparisons(comparisons,look,numerical_error):
    from scipy.stats import t
    from .statistics import bootstrap_indices,contrast_bounds,look_alpha
    if len(comparisons) not in (2,3):
        raise ValueError('Complete prespecified two/three contrast control family required for Holm')
    ordered=[]
    for i,row in enumerate(comparisons):
        x=np.asarray(row['baseline_by_seed'])-np.asarray(row['method_by_seed'])
        sd=float(x.std(ddof=1));indices=bootstrap_indices(len(x),look)
        p=1. if sd==0 else float(t.sf((x.mean()-2*numerical_error)/(sd/np.sqrt(len(x))),len(x)-1))
        # Invert the same conservative percentile-bound procedure used below.
        p=max(p,float(np.mean(x[indices].mean(axis=1)<=2*numerical_error)))
        ordered.append((p,i,x,indices))
    outputs=[dict(r) for r in comparisons]; preceding=True
    for rank,(p,i,x,indices) in enumerate(sorted(ordered,key=lambda item:(item[0],item[1]))):
        alpha=look_alpha(look)/(len(comparisons)-rank)
        bounds=contrast_bounds(x,alpha,indices,{'lower':2*numerical_error,'upper':2*numerical_error})
        passed=(preceding and p<alpha and bounds['status']=='ok' and bounds['lower']>0
                and not bounds['influence']['unstable'])
        preceding=bool(passed)
        outputs[i]['holm']=dict(rank=rank+1,alpha=alpha,p_value=p,bounds=bounds,supported=bool(passed))
    return outputs


def refinement_envelope(original,refined,seeds,configurations=CONFIGURATIONS):
    def key(r):return (r['model_id'],r['model_partition_ns'][0],r['seed'],r['episode_id'])
    expected={(name,step,seed,0) for seed in seeds for name,step in configurations}
    old={key(r):r for r in original if r['episode_id']==0};new={key(r):r for r in refined}
    if (len(new)!=len(refined) or set(new)!=expected or set(old)!=expected):
        raise ValueError('Control refinement subset coverage mismatch')
    rows=[]
    for identity in sorted(expected):
        a,b=old[identity],new[identity]
        if (a['initial_state']!=b['initial_state'] or a['target']!=b['target']
                or not a['complete'] or not b['complete']
                or not np.isfinite([a['realized_cost'],b['realized_cost']]).all()):
            raise ValueError('Control refinement paired states/costs invalid')
        delta=abs(a['realized_cost']-b['realized_cost'])
        relative=delta/max(a['realized_cost'],1e-300)
        rows.append(dict(cell=list(identity),original_cost=a['realized_cost'],refined_cost=b['realized_cost'],
                    absolute_cost_difference=delta,relative_cost_difference=relative,
                    success_unchanged=a['success']==b['success']))
    return dict(absolute_cost_envelope=max(r['absolute_cost_difference'] for r in rows),rows=rows,
                converged=all(r['relative_cost_difference']<=.02 and r['success_unchanged'] for r in rows),
                limitation='Empirical first-task-per-seed physics envelope, not a certified population bound')


def control_summary(rows,seeds,episodes_per_seed,configurations=CONFIGURATIONS):
    expected={(name,step,seed,task) for seed in seeds for task in range(episodes_per_seed)
              for name,step in configurations}
    coordinates=[(r['model_id'],r['model_partition_ns'][0],r['seed'],r['episode_id']) for r in rows]
    if len(coordinates)!=len(set(coordinates)) or set(coordinates)!=expected:
        raise ValueError('Fresh control coordinate coverage mismatch')
    tasks={};groups={}
    for row,coordinate in zip(rows,coordinates):
        if (row['complete'] is not True or row['phase']!='confirmatory' or row['actuator']!='lagged'
                or row['actuator_interval_ns']!=40000000
                or not all(math.isfinite(row[k]) and row[k]>=0 for k in
                  ('realized_cost','planner_latency_median_seconds','q_rmse_rad','fallback_count'))):
            raise ValueError('Invalid fresh control episode')
        pair=(row['seed'],row['episode_id']);inputs=(row['initial_state'],row['target'])
        if pair in tasks and tasks[pair]!=inputs:raise ValueError('Fresh paired task inputs differ')
        tasks[pair]=inputs;groups.setdefault(coordinate[:3],[]).append(row)
    metrics=[]
    for (name,step,seed),episodes in sorted(groups.items()):
        metrics.append(dict(model_id=name,model_step_ns=step,seed=seed,n_episodes=len(episodes),
                     cost=float(np.mean([r['realized_cost'] for r in episodes])),
                     planning_seconds=float(np.mean([r['planner_latency_median_seconds'] for r in episodes])),
                     q_rmse_rad=float(np.mean([r['q_rmse_rad'] for r in episodes])),
                     successes=sum(r['success'] for r in episodes),fallback_count=sum(r['fallback_count'] for r in episodes)))
    return dict(seed_metrics=metrics,episode_count=len(rows),seeds=list(seeds),episodes_per_seed=episodes_per_seed)


def reserve_analysis_look(root,look,identity):
    directory=Path(root)/'evidence/affine-fresh-looks'
    if type(look) is not int or look<1:raise ValueError('Positive statistical look required')
    path=directory/f'look-{look:03d}.json'
    value=dict(protocol_id=PROTOCOL,look_index=look,alpha=.05/(2**look),evidence_identity=identity)
    with job_lock(job_key(dict(protocol_id=PROTOCOL,stage='statistical_look_ledger')),Path(root)):
        if path.exists():
            existing=read_json(path)
            if existing!=value:raise ValueError('Changed evidence requires the next statistical look')
            return existing
        previous=sorted(directory.glob('look-*.json'))
        if look!=len(previous)+1:raise ValueError('Statistical look cannot skip or reset spending')
        if any(read_json(p)['evidence_identity']==identity for p in previous):
            raise ValueError('Unchanged evidence already has a statistical look; reaggregate that look')
        atomic_json(path,value)
    return value


def validate_profiles(rows,seeds):
    expected={(name,seed,batch,h) for name in MODELS for seed in seeds for batch in (1,128) for h in (.005,.01,.04)}
    coordinates=[(r['model_id'],r['seed'],r['batch_size'],r['h_s']) for r in rows]
    if len(coordinates)!=len(set(coordinates)) or set(coordinates)!=expected:
        raise ValueError('Fresh timing coordinate coverage mismatch')
    for row in rows:
        values=np.asarray(row['samples_ns'],dtype=float)
        if (values.shape!=(500 if row['batch_size']==1 else 100,) or not np.isfinite(values).all()
                or np.any(values<=0) or row['median_ns']!=float(np.median(values))
                or row['p95_ns']!=float(np.quantile(values,.95,method='linear'))):
            raise ValueError('Fresh timing raw-to-summary mismatch')


def retrain_metrics(first,repeated,original_score,repeated_score):
    if set(first)!=set(repeated) or any(np.asarray(first[k]).shape!=np.asarray(repeated[k]).shape for k in first):
        raise ValueError('Training reproduction tensor coordinates differ')
    differences=[];limits=[]
    for key in first:
        a,b=np.asarray(first[key]),np.asarray(repeated[key])
        if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Nonfinite reproduction tensor')
        differences.append(float(np.max(np.abs(a-b))))
        limits.append(1e-6*(1+float(np.max(np.abs(a)))))
    score_difference=abs(original_score-repeated_score)/max(abs(original_score),1e-300)
    return dict(max_parameter_difference=max(differences),validation_relative_difference=score_difference,
                **{'pass':all(d<=limit for d,limit in zip(differences,limits)) and score_difference<=1e-4})


def _sources():
    return sorted(set(source_files())|{'poc/affine_confirmation.py','poc/statistics.py'})


def _npz(path,values):
    buffer=io.BytesIO();np.savez_compressed(buffer,**values);atomic_bytes(path,buffer.getvalue())


def _receipt(root,record):
    if record['completion_status']!='complete':
        raise ValueError('Fresh measurement failed: '+record['run_id'])
    path=f'evidence/{record["milestone_id"]}/{record["run_id"]}/manifest.json'
    return dict(path=path,sha256=sha256_file(root/path))


def _verify_entry(root,entry):
    from .contracts import verify_run
    path=root/entry['path'];verify_bindings(root,{entry['path']:entry['sha256']})
    verify_run(root,read_json(path));return path.parent


def prepare(root):
    root=Path(root).resolve()
    if (root/FREEZE).exists():return evaluation_freeze(root)
    fitting=load_freeze(root);selected=selected_freeze(root)
    corpus_entry=read_json(root/'reports/affine_fresh_data_confirmation.json')
    folder=_verify_entry(root,corpus_entry)
    arrays=load_fresh_arrays(folder/'corpus.json','evaluation');probes=fresh_probes(arrays)
    inputs={name:sha256_file(root/name) for name in (FIT_FREEZE,COARSE_SPEC,'config/affine_fresh_selected.json',
              'reports/affine_fresh_data_confirmation.json',corpus_entry['path'],
              str((folder/'corpus.json').relative_to(root)),str((folder/'trajectories.npz').relative_to(root)))}
    for entry in selected['selections'].values():
        model_folder=_verify_entry(root,entry)
        for name in ('manifest.json','weights.pt','normalization.json'):
            path=str((model_folder/name).relative_to(root));inputs[path]=sha256_file(root/path)
    value=dict(protocol_id=PROTOCOL,created_utc=utc_now(),fit_freeze_sha256=sha256_file(root/FIT_FREEZE),
               selections=selected['selections'],corpus=corpus_entry,seeds=list(SEEDS),
               episodes_per_seed=fitting['initial_episodes_per_seed'],configurations=[list(c) for c in CONFIGURATIONS],
               coarse_configurations=[list(c) for c in COARSE_CONFIGURATIONS],control_family_size=3,
               coordinates={k:probes[k].tolist() for k in ('probe_ids','strata','horizons_ns')},
               planner_settings=fitting['planner_settings'],physics_step_ns=fitting['control_physics_step_ns'],
               source_hashes={name:sha256_file(root/name) for name in _sources()},input_hashes=inputs)
    atomic_json(root/FREEZE,value);return value


def evaluation_freeze(root):
    value=read_json(root/FREEZE)
    if (value['protocol_id']!=PROTOCOL or value['configurations']!=[list(c) for c in CONFIGURATIONS]
            or value.get('coarse_configurations')!=[list(c) for c in COARSE_CONFIGURATIONS]
            or value.get('control_family_size')!=3
            or value['fit_freeze_sha256']!=sha256_file(root/FIT_FREEZE)):
        raise ValueError('Fresh evaluation design differs from frozen fit')
    verify_bindings(root,{**value['source_hashes'],**value['input_hashes']});return value


def load_model(root,name,seed,freeze,replica=False):
    from .train import canonical_tensor_hash
    entry=(read_json(root/f'reports/affine_fresh_replicas/{name}__seed{seed}.json') if replica
           else freeze['selections'][f'{name}:{seed}'])
    folder=_verify_entry(root,entry);summary=read_json(folder/'summary.json')
    if (summary['protocol_id']!=PROTOCOL or summary['model_id']!=name or summary['seed']!=seed
            or summary['actuator']!='lagged' or summary['qualification']['qualified'] is not True):
        raise ValueError('Fresh model identity/qualification mismatch')
    norm=read_json(folder/'normalization.json')
    scales=Scales(*(torch.tensor(norm[k],dtype=torch.float32) for k in
                    ('scale_q','scale_v','velocity_input_rms')))
    model=build_fresh_model(name,scales,seed)
    if sha256_file(folder/'weights.pt')!=entry['checkpoint_sha256']:
        raise ValueError('Fresh selected weights changed')
    weights=torch.load(folder/'weights.pt',weights_only=True,map_location='cpu')
    if canonical_tensor_hash(weights)!=entry['canonical_tensor_sha256']:
        raise ValueError('Fresh selected canonical tensors changed')
    model.load_state_dict(weights,strict=True);return model.double().eval()


def _files(root,freeze,extra=()):
    return sorted(set(freeze['source_hashes'])|set(freeze['input_hashes'])|{FREEZE}|set(extra))


def references(root):
    from .envs import Plant
    from .execution import cpu_defaults,run_job
    from .oracle_rigidbody import integrate,parameters_from_compiled
    root=Path(root).resolve();freeze=evaluation_freeze(root);cpu_defaults(SEEDS[0])
    corpus_folder=_verify_entry(root,freeze['corpus'])
    probes=fresh_probes(load_fresh_arrays(corpus_folder/'corpus.json','evaluation'))
    def produce(folder):
        shape=(len(probes['states']),2,6);grids={}
        for tick in (125000,62500):
            plant=Plant('lagged',tick,root/'assets');truth={k:np.empty(shape) for k in ('zero','command')}
            for i,state in enumerate(probes['states']):
                for hi,horizon in enumerate(probes['horizons_ns']):
                    for key,command in (('zero',np.zeros(2)),('command',probes['commands'][i])):
                        plant.reset(state);truth[key][i,hi]=plant.advance(command,int(horizon))['state']
            grids[tick]=truth
        plant=Plant('lagged',62500,root/'assets');parameters=parameters_from_compiled(plant.compiled_manifest())
        oracle_rows=[];envelopes={'q':0.,'v':0.};activation_error=0.
        for key in ('zero','command'):
            difference=grids[125000][key]-grids[62500][key]
            for c,index in (('q',slice(0,2)),('v',slice(2,4))):
                envelopes[c]=max(envelopes[c],float(np.linalg.norm(difference[:,:,index],axis=-1).max()))
            for i,state in enumerate(probes['states']):
                for hi,horizon in enumerate(probes['horizons_ns']):
                    command=np.zeros(2) if key=='zero' else probes['commands'][i]
                    exact_w=state[4:]+(command-state[4:])*(-np.expm1(-float(horizon)/1e9/.08))
                    activation_error=max(activation_error,float(np.max(np.abs(grids[62500][key][i,hi,4:]-exact_w))))
        for stratum in ('cold','warm'):
            for i in np.flatnonzero(probes['strata']==stratum)[:32]:
                for hi,horizon in enumerate(probes['horizons_ns']):
                    for key,command in (('zero',np.zeros(2)),('command',probes['commands'][i])):
                        result=integrate(probes['states'][i],command,int(horizon)/1e9,parameters,'lagged',max_step_s=.000125)
                        tighter=integrate(probes['states'][i],command,int(horizon)/1e9,parameters,'lagged',
                                          rtol=1e-12,atol=1e-14,max_step_s=.0000625)
                        row=dict(probe_id=int(probes['probe_ids'][i]),h_ns=int(horizon),branch=key,
                                 endpoint=result['state'].tolist(),tighter_endpoint=tighter['state'].tolist(),nfev=result['nfev'])
                        for c,index in (('q',slice(0,2)),('v',slice(2,4))):
                            discrepancy=max(float(np.linalg.norm(result['state'][index]-grids[62500][key][i,hi,index])),
                                            float(np.linalg.norm(result['state'][index]-tighter['state'][index])))
                            row[c+'_discrepancy']=discrepancy;envelopes[c]=max(envelopes[c],discrepancy)
                        oracle_rows.append(row)
        if envelopes['q']>1e-7 or envelopes['v']>1e-6 or activation_error>1e-9:
            raise ValueError('Fresh reference failed plant/oracle/activation numerical budget')
        raw={**probes,**{'truth_'+k:v for k,v in grids[62500].items()},
             **{'coarser_'+k:v for k,v in grids[125000].items()}}
        _npz(folder/'references.npz',raw);atomic_json(folder/'independent_oracle.json',oracle_rows)
        summary=dict(protocol_id=PROTOCOL,missing_cells=[],n_probes=len(probes['states']),
                     numerical_envelope=envelopes,activation_error=activation_error,
                     qualification='Empirical independent/refinement envelope; not mathematically certified')
        return [dict(check_id='fresh.references',status='pass',scope='lagged',tolerance=None,
                     reason='Independent resets, step refinement, rigid-body DOP853 and exact activation')],summary
    keys=job_configuration(root,'references',freeze)
    record=run_job(root,'M40','confirmatory',_files(root,freeze,keys),('fresh.references',),[],produce,
                   config_keys=keys)
    receipt=_receipt(root,record);atomic_json(root/'reports/affine_fresh_references.json',receipt);return receipt


def _profile(model,state,command,name,seed):
    rows=[]
    with torch.no_grad():
        for batch in (1,128):
            x,u=state.repeat(batch,1),command.repeat(batch,1)
            for h in (.005,.01,.04):
                for _ in range(50):model.predict(x,u,h)
                samples=[]
                for _ in range(500 if batch==1 else 100):
                    started=time.perf_counter_ns();output=model.predict(x,u,h);samples.append(time.perf_counter_ns()-started)
                    if any(s!='ok' for s in output.status):raise ValueError('Failed fresh query cannot count as fast inference')
                rows.append(dict(model_id=name,seed=seed,batch_size=batch,h_s=h,samples_ns=samples,
                     median_ns=float(np.median(samples)),p95_ns=float(np.quantile(samples,.95,method='linear')),
                     network_dispatches=output.network_calls,vector_field_calls=output.vector_field_calls,
                     neural_rows=batch*(2 if name=='affine_flow' else 4)))
    return rows


def measure(root):
    from .execution import cpu_defaults,run_job
    from .shared_state_diagnostic import model_endpoints
    root=Path(root).resolve();freeze=evaluation_freeze(root);cpu_defaults(SEEDS[0])
    reference_entry=read_json(root/'reports/affine_fresh_references.json');reference_folder=_verify_entry(root,reference_entry)
    with np.load(reference_folder/'references.npz',allow_pickle=False) as archive:
        reference={k:archive[k].copy() for k in archive.files}
    extra=(reference_entry['path'],str((reference_folder/'references.npz').relative_to(root)))
    def produce(folder):
        rows,profiles=[] ,[]
        state=torch.tensor(reference['states'][0:1],dtype=torch.float64)
        command=torch.tensor(reference['commands'][0:1],dtype=torch.float64)
        for si,seed in enumerate(freeze['seeds']):
            order=MODELS if si%2==0 else tuple(reversed(MODELS))
            for name in order:
                model=load_model(root,name,seed,freeze)
                raw={k:reference[k] for k in ('probe_ids','strata','horizons_ns','truth_zero','truth_command')}
                raw.update({k:np.empty_like(raw['truth_command']) for k in ('model_zero','model_command','model_composed')})
                for hi,horizon in enumerate(reference['horizons_ns']):
                    values=model_endpoints(model,reference['states'],reference['commands'],int(horizon))
                    for key,value in values.items():raw['model_'+key][:,hi]=value
                statistics=prediction_rows(raw,freeze['coordinates'])
                rows.extend([{**r,'model_id':name,'seed':seed} for r in statistics])
                _npz(folder/f'{name}__seed{seed}.npz',raw)
                profiles.extend(_profile(model,state,command,name,seed))
                print(f'Fresh predictions/profile {name} seed{seed}',flush=True)
        atomic_json(folder/'profiles.json',profiles)
        summary=dict(protocol_id=PROTOCOL,missing_cells=[],rows=rows,profiles=profiles,
                     new_model_rows=len(freeze['seeds'])*2*len(reference['probe_ids'])*2,
                     reference=reference_entry,numerical_envelope=read_json(reference_folder/'summary.json')['numerical_envelope'])
        return [dict(check_id='fresh.predictions',status='pass',scope=PROTOCOL,tolerance=None,
                     reason='Complete paired probe vectors, frozen selections and isolated timed queries')],summary
    keys=job_configuration(root,'predictions',dict(freeze=freeze,reference=reference_entry))
    record=run_job(root,'M40','confirmatory',_files(root,freeze,set(extra)|set(keys)),('fresh.predictions',),[],produce,
                   config_keys=keys)
    receipt=_receipt(root,record);atomic_json(root/'reports/affine_fresh_measurement.json',receipt);return receipt


def control_job(root,name,step,seed,task,freeze,kind='primary'):
    from .execution import cpu_defaults,run_job
    from .envs import Plant
    from .control_episode import execute_episode
    from .control_artifacts import save_episode
    if kind not in ('primary','replay','finer_physics','coarse_primary','coarse_replay','coarse_finer'):
        raise ValueError('Unknown fresh control producer')
    cpu_defaults(seed)
    request=dict(protocol_id=PROTOCOL,model_id=name,model_step_ns=step,seed=seed,episode_id=task,kind=kind,
                 freeze_sha256=sha256_file(root/FREEZE),physics_step_ns=125000 if kind in ('finer_physics','coarse_finer') else freeze['physics_step_ns'])
    def produce(folder):
        model=load_model(root,name,seed,freeze);plant=Plant('lagged',request['physics_step_ns'],root/'assets')
        result=execute_episode(plant,model,freeze['planner_settings'],seed=seed,episode_id=task,
                phase='confirmatory' if kind in ('primary','coarse_primary') else 'replication',condition='clean',
                model_partition=(step,)*(40000000//step),actuator_interval_ns=40000000)
        summary=save_episode(folder,result,model_id=name,
                   checkpoint_hash=freeze['selections'][f'{name}:{seed}']['checkpoint_sha256'],run_id=folder.name)
        summary.update(protocol_id=PROTOCOL,kind=kind,missing_cells=[],
                       neural_rows=summary['work']['scalar_candidate_evaluations']*(2 if name=='affine_flow' else 4))
        atomic_json(folder/'request.json',request)
        return [dict(check_id='fresh.control',status='pass',scope=f'{name}:{step}:{seed}:{task}',tolerance=None,
                     reason='Complete physical episode, fixed hardware clocks and paired task/CEM stream')],summary
    keys=job_configuration(root,'control',request)
    record=run_job(root,'M40','confirmatory' if kind in ('primary','coarse_primary') else 'replication',_files(root,freeze,keys),
                   ('fresh.control',),[],produce,config_keys=keys)
    return _receipt(root,record)


def controls(root,kind='primary'):
    root=Path(root).resolve();freeze=evaluation_freeze(root)
    path=root/f'reports/affine_fresh_{kind}_batch.json'
    batch=read_json(path) if path.exists() else dict(state='partial',completed={},freeze_sha256=sha256_file(root/FREEZE))
    if batch['freeze_sha256']!=sha256_file(root/FREEZE):raise ValueError('Fresh batch freeze changed')
    count=freeze['episodes_per_seed'] if kind in ('primary','coarse_primary') else 1
    configurations=COARSE_CONFIGURATIONS if kind.startswith('coarse_') else CONFIGURATIONS
    jobs=[(name,step,seed,task) for seed in freeze['seeds'] for task in range(count) for name,step in configurations]
    with job_lock(job_key(dict(protocol_id=PROTOCOL,freeze=sha256_file(root/FREEZE),kind=kind)),root):
        for name,step,seed,task in jobs:
            key=f'{name}__step{step}__seed{seed}__task{task}'
            if key in batch['completed']:
                _verify_entry(root,batch['completed'][key]);continue
            print(f'{len(batch["completed"])}/{len(jobs)} {kind} {key}',flush=True)
            batch['completed'][key]=control_job(root,name,step,seed,task,freeze,kind)
            atomic_json(path,batch)
        batch['state']='complete';atomic_json(path,batch)
    return batch


def aggregate(root,look=1):
    from .affine_measurement import recompute_episode_cost
    root=Path(root).resolve();freeze=evaluation_freeze(root)
    measurement_entry=read_json(root/'reports/affine_fresh_measurement.json');folder=_verify_entry(root,measurement_entry)
    batch_path=root/'reports/affine_fresh_primary_batch.json'
    completed_batch=(read_json(batch_path) if batch_path.exists() and read_json(batch_path)['state']=='complete' else None)
    measurement=read_json(folder/'summary.json');rows=[]
    validate_profiles(measurement['profiles'],freeze['seeds'])
    for seed in freeze['seeds']:
        for name in MODELS:
            with np.load(folder/f'{name}__seed{seed}.npz',allow_pickle=False) as raw:
                values={k:raw[k].copy() for k in raw.files}
            rows.extend([{**r,'model_id':name,'seed':seed} for r in prediction_rows(values,freeze['coordinates'])])
    if sorted(rows,key=lambda r:(r['seed'],r['model_id'],r['stratum'],r['h_ns'],r['component'])) != sorted(
        measurement['rows'],key=lambda r:(r['seed'],r['model_id'],r['stratum'],r['h_ns'],r['component'])):
        raise ValueError('Fresh raw prediction summaries disagree')
    predictions=[]
    for horizon in (5000000,10000000):
        for component in ('q','v'):
            paired={}
            for name in MODELS:
                paired[name]=[float(np.mean([r['total_error_mean'] for r in rows if r['seed']==seed
                     and r['model_id']==name and r['h_ns']==horizon and r['component']==component])) for seed in freeze['seeds']]
            comparison=paired_comparison(paired['node_structured'],paired['affine_flow'],look,
                                            measurement['numerical_envelope'][component])
            predictions.append(dict(h_ns=horizon,component=component,**comparison))
    timing=[]
    for batch in (1,128):
        for h in (.005,.01,.04):
            paired={}
            for name in MODELS:
                paired[name]=[next(p['median_ns'] for p in measurement['profiles'] if p['seed']==seed
                              and p['model_id']==name and p['batch_size']==batch and p['h_s']==h) for seed in freeze['seeds']]
            timing.append(dict(batch_size=batch,h_s=h,**paired_comparison(paired['node_structured'],paired['affine_flow'],look,0.)))
    control_rows=[];control_comparisons=[];control=None;control_convergence=None;finer_sha256=None;coarse_hashes={}
    if batch_path.exists():
        batch=read_json(batch_path)
        if batch['state']=='complete':
            for entry in batch['completed'].values():
                episode_folder=_verify_entry(root,entry);recompute_episode_cost(episode_folder)
                control_rows.append(read_json(episode_folder/'summary.json'))
            coarse_path=root/'reports/affine_fresh_coarse_primary_batch.json'
            coarse=read_json(coarse_path)
            if coarse['state']!='complete':raise ValueError('Complete the prospective coarse affine comparison before inference')
            for entry in coarse['completed'].values():
                episode_folder=_verify_entry(root,entry);recompute_episode_cost(episode_folder)
                control_rows.append(read_json(episode_folder/'summary.json'))
            all_configurations=CONFIGURATIONS+COARSE_CONFIGURATIONS
            control=control_summary(control_rows,freeze['seeds'],freeze['episodes_per_seed'],all_configurations)
            coarse_hashes['primary']=sha256_file(coarse_path)
            finer_path=root/'reports/affine_fresh_finer_physics_batch.json'
            finer=read_json(finer_path)
            if finer['state']!='complete':raise ValueError('Complete control refinement before inference')
            refined=[]
            for entry in finer['completed'].values():
                refined_folder=_verify_entry(root,entry);recompute_episode_cost(refined_folder)
                refined.append(read_json(refined_folder/'summary.json'))
            coarse_finer_path=root/'reports/affine_fresh_coarse_finer_batch.json'
            coarse_finer=read_json(coarse_finer_path)
            if coarse_finer['state']!='complete':raise ValueError('Complete coarse affine physics refinement before inference')
            for entry in coarse_finer['completed'].values():
                refined_folder=_verify_entry(root,entry);recompute_episode_cost(refined_folder)
                refined.append(read_json(refined_folder/'summary.json'))
            coarse_hashes['finer']=sha256_file(coarse_finer_path)
            control_convergence=refinement_envelope(control_rows,refined,freeze['seeds'],all_configurations)
            finer_sha256=sha256_file(finer_path)
            if not control_convergence['converged']:
                raise ValueError('Control physics refinement needs full affected-cell extension')
            for method_step,step in ((5000000,5000000),(5000000,40000000),(40000000,40000000)):
                paired={}
                for name,partition in (('affine_flow',method_step),('node_structured',step)):
                    paired[name]=[next(r['cost'] for r in control['seed_metrics'] if r['seed']==seed
                           and r['model_id']==name and r['model_step_ns']==partition) for seed in freeze['seeds']]
                control_comparisons.append(dict(method_model_step_ns=method_step,baseline_model_step_ns=step,
                   **paired_comparison(paired['node_structured'],paired['affine_flow'],look,
                                      control_convergence['absolute_cost_envelope'])))
            control_comparisons=holm_control_comparisons(control_comparisons,look,
                                       control_convergence['absolute_cost_envelope'])
    primary=[p for p in predictions if p['h_ns']==5000000]+[t for t in timing if t['batch_size']==128 and t['h_s']==.005]
    supported=all(p['contrast']['status']=='ok' and p['contrast']['lower']>0
                  and not p['contrast']['influence']['unstable'] for p in primary)
    solver=[]
    for p in predictions:
        for seed in freeze['seeds']:
            baseline=[r for r in rows if r['model_id']=='node_structured' and r['seed']==seed
                      and r['h_ns']==p['h_ns'] and r['component']==p['component']]
            delta=abs(p['baseline_by_seed'][freeze['seeds'].index(seed)]-p['method_by_seed'][freeze['seeds'].index(seed)])
            discrepancy=float(np.mean([r['composition_discrepancy_mean'] for r in baseline]))
            solver.append(dict(seed=seed,h_ns=p['h_ns'],component=p['component'],node_solver_discrepancy=discrepancy,
                               between_model_difference=delta,requires_solver_review=discrepancy>.1*delta))
    supported=supported and not any(r['requires_solver_review'] for r in solver if r['h_ns']==5000000)
    look_receipt=reserve_analysis_look(root,look,dict(measurement=measurement_entry,
                    control_batch_sha256=sha256_file(batch_path) if completed_batch else None,
                    finer_batch_sha256=finer_sha256,coarse_batch_sha256=coarse_hashes,
                    evaluation_freeze_sha256=sha256_file(root/FREEZE)))
    result=dict(protocol_id=PROTOCOL,look_index=look,alpha_spent=look_receipt['alpha'],primary_combined_gain_supported=supported,
                predictions=predictions,timing=timing,control=control,control_comparisons=control_comparisons,
                control_convergence=control_convergence,
                look_receipt=look_receipt,node_solver_audit=solver,
                measurement=measurement_entry,control_episodes=control_rows,
                limitation='Inference conditional on fresh fixed train/probe corpora and this known-lag two-link robot; seed-cluster assumptions approximate.')
    atomic_json(root/'reports/affine_fresh_result.json',result);return result


def reproduce(root):
    from .affine_measurement import recompute_episode_cost
    from .shared_state_diagnostic import model_endpoints
    from .execution import cpu_defaults
    root=Path(root).resolve();freeze=evaluation_freeze(root);cpu_defaults(SEEDS[0])
    measurement_folder=_verify_entry(root,read_json(root/'reports/affine_fresh_measurement.json'))
    reference_folder=_verify_entry(root,read_json(root/'reports/affine_fresh_references.json'))
    with np.load(reference_folder/'references.npz',allow_pickle=False) as archive:
        reference={k:archive[k].copy() for k in archive.files}
    subset=np.r_[np.flatnonzero(reference['strata']=='cold')[:64],np.flatnonzero(reference['strata']=='warm')[:64]]
    predictions=[];training=[];control_rows=[]
    for seed in freeze['seeds']:
        for name in MODELS:
            model=load_model(root,name,seed,freeze)
            with np.load(measurement_folder/f'{name}__seed{seed}.npz',allow_pickle=False) as archive:
                saved={k:archive[k].copy() for k in archive.files}
            for hi,horizon in enumerate(reference['horizons_ns']):
                values=model_endpoints(model,reference['states'][subset],reference['commands'][subset],int(horizon))
                for branch,value in values.items():
                    delta=value-saved['model_'+branch][subset,hi]
                    q,v=float(np.max(np.abs(delta[:,:2]))),float(np.max(np.abs(delta[:,2:4])))
                    predictions.append(dict(model_id=name,seed=seed,h_ns=int(horizon),branch=branch,
                              max_q_difference=q,max_v_difference=v,**{'pass':q<=1e-6 and v<=1e-5}))
            if seed in (freeze['seeds'][0],freeze['seeds'][-1]):
                repeated=load_model(root,name,seed,freeze,replica=True)
                original_folder=_verify_entry(root,freeze['selections'][f'{name}:{seed}'])
                repeated_entry=read_json(root/f'reports/affine_fresh_replicas/{name}__seed{seed}.json')
                repeated_folder=_verify_entry(root,repeated_entry)
                old_score=read_json(original_folder/'summary.json')['validation']['objective']
                new_score=read_json(repeated_folder/'summary.json')['validation']['objective']
                metrics=retrain_metrics({k:t.numpy() for k,t in model.state_dict().items()},
                                        {k:t.numpy() for k,t in repeated.state_dict().items()},old_score,new_score)
                training.append(dict(model_id=name,seed=seed,**metrics))
    primary=read_json(root/'reports/affine_fresh_primary_batch.json')
    for kind in ('replay','finer_physics','coarse_replay','coarse_finer'):
        batch=read_json(root/f'reports/affine_fresh_{kind}_batch.json')
        if kind.startswith('coarse_'):primary=read_json(root/'reports/affine_fresh_coarse_primary_batch.json')
        expected=len(freeze['seeds'])*(1 if kind.startswith('coarse_') else 3)
        if batch['state']!='complete' or len(batch['completed'])!=expected:
            raise ValueError('Fresh reproduction control coverage incomplete')
        for key,entry in batch['completed'].items():
            folder=_verify_entry(root,entry);old_folder=_verify_entry(root,primary['completed'][key])
            recompute_episode_cost(folder);recompute_episode_cost(old_folder)
            old,new=read_json(old_folder/'summary.json'),read_json(folder/'summary.json')
            with np.load(old_folder/'true_trajectories.npz',allow_pickle=False) as a, np.load(folder/'true_trajectories.npz',allow_pickle=False) as b:
                old_time,new_time=a['time_ns'],b['time_ns'];indices=np.searchsorted(new_time,old_time)
                if not np.array_equal(new_time[indices],old_time):raise ValueError('Replayed physical clocks differ')
                delta=b['state'][indices]-a['state'];q,v=np.max(np.abs(delta[:,:2])),np.max(np.abs(delta[:,2:4]))
            def commands(p):
                from json import loads
                return np.array([loads(line)['command'] for line in (p/'action_segments.jsonl').read_text().splitlines()])
            command_difference=float(np.max(np.abs(commands(folder)-commands(old_folder))))
            cost_difference=abs(new['realized_cost']-old['realized_cost'])/max(old['realized_cost'],1e-300)
            passed=(q<=1e-6 and v<=1e-5 and command_difference<=1e-6 and cost_difference<=1e-4) if kind in ('replay','coarse_replay') else (
                cost_difference<=.02 and old['success']==new['success'])
            control_rows.append(dict(kind=kind,cell=key,max_q_difference=float(q),max_v_difference=float(v),
                       max_command_difference=command_difference,cost_relative_difference=cost_difference,
                       original_cost=old['realized_cost'],replayed_cost=new['realized_cost'],
                       cost_absolute_difference=abs(new['realized_cost']-old['realized_cost']),
                       success_unchanged=old['success']==new['success'],**{'pass':bool(passed)}))
    result=dict(protocol_id=PROTOCOL,predictions=predictions,training=training,controls=control_rows,
                **{'pass':all(r['pass'] for r in predictions+training+control_rows)})
    atomic_json(root/'reports/affine_fresh_reproduction.json',result);return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','references','measure','controls','replay','finer',
                                         'coarse-controls','coarse-replay','coarse-finer','aggregate','reproduce'))
    parser.add_argument('--root',type=Path,default=Path.cwd());parser.add_argument('--look',type=int,default=1)
    args=parser.parse_args(argv);root=args.root.resolve()
    if args.command=='prepare':
        value=prepare(root);print(dict(protocol_id=PROTOCOL,seeds=value['seeds'],state='evaluation_frozen'))
    elif args.command=='references':print(references(root))
    elif args.command=='measure':print(measure(root))
    elif args.command in {'controls','replay','finer','coarse-controls','coarse-replay','coarse-finer'}:
        value=controls(root,{'controls':'primary','replay':'replay','finer':'finer_physics',
                            'coarse-controls':'coarse_primary','coarse-replay':'coarse_replay','coarse-finer':'coarse_finer'}[args.command])
        print(dict(state=value['state'],completed=len(value['completed'])))
    elif args.command=='reproduce':
        value=reproduce(root);print(dict(reproduction_pass=value['pass'],training_pairs=len(value['training']),
                                   replay_cells=len(value['controls'])))
    else:
        value=aggregate(root,args.look);print(dict(primary_combined_gain_supported=value['primary_combined_gain_supported'],
                    control_completed=value['control'] is not None,look_index=value['look_index']))


if __name__=='__main__':main()
