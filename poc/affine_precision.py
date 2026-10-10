"""Add independent seed clusters without altering frozen affine producers."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

for _variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_variable]='1'

import numpy as np
import torch

from .affine_confirmation import (CONFIGURATIONS, COARSE_CONFIGURATIONS, MODELS, PROTOCOL,
    FREEZE as ORIGINAL_FREEZE, _files, _npz, _profile, _receipt, _verify_entry, control_job,
    control_summary, evaluation_freeze, holm_control_comparisons, load_model,
    paired_comparison, prediction_rows, refinement_envelope, reserve_analysis_look,
    retrain_metrics, validate_profiles)
from .affine_experiment import verify_bindings
from .affine_fresh import job_configuration, SEEDS
from .io import atomic_bytes, atomic_json, job_key, job_lock, read_json, sha256_file, utc_now

REQUEST='config/affine_precision_request_001.json'
FREEZE='config/affine_precision_freeze_001.json'
SPEC='docs/superpowers/specs/2026-10-08-affine-precision-continuation.md'
ALL_CONFIGURATIONS=CONFIGURATIONS+COARSE_CONFIGURATIONS
PREFIX='reports/affine_precision_001'


def precision_seeds(request):
    old,new=request['old_seeds'],request['new_seeds']
    if (old!=list(SEEDS) or len(new)<5 or new!=list(range(old[-1]+1,old[-1]+1+len(new)))
            or request['total_seeds']!=len(old)+len(new)):
        raise ValueError('Invalid ascending independent precision seed registry')
    return old+new


def merge_measurements(blocks,seeds,coordinates):
    raws={};profiles=[]
    for block in blocks:
        for key,raw in block['raws'].items():
            if key in raws:raise ValueError('Duplicate precision measurement coordinate')
            raws[key]=raw
        profiles.extend(block['profiles'])
    expected={(name,seed) for seed in seeds for name in MODELS}
    if set(raws)!=expected:raise ValueError('Precision measurement coverage mismatch')
    validate_profiles(profiles,seeds)
    rows=[]
    for seed in seeds:
        for name in MODELS:
            rows.extend(dict(model_id=name,seed=seed,**r) for r in prediction_rows(raws[name,seed],coordinates))
    return rows,profiles


def prediction_inference(rows,profiles,seeds,look,envelope):
    predictions=[];timing=[];solver=[]
    for horizon in (5000000,10000000):
        for component in ('q','v'):
            paired={name:[float(np.mean([r['total_error_mean'] for r in rows if r['seed']==seed
                and r['model_id']==name and r['h_ns']==horizon and r['component']==component]))
                for seed in seeds] for name in MODELS}
            comparison=paired_comparison(paired['node_structured'],paired['affine_flow'],look,envelope[component])
            predictions.append(dict(h_ns=horizon,component=component,**comparison))
            for i,seed in enumerate(seeds):
                baseline=[r for r in rows if r['model_id']=='node_structured' and r['seed']==seed
                          and r['h_ns']==horizon and r['component']==component]
                discrepancy=float(np.mean([r['composition_discrepancy_mean'] for r in baseline]))
                delta=abs(paired['node_structured'][i]-paired['affine_flow'][i])
                solver.append(dict(seed=seed,h_ns=horizon,component=component,node_solver_discrepancy=discrepancy,
                                   between_model_difference=delta,requires_solver_review=discrepancy>.1*delta))
    for batch in (1,128):
        for h in (.005,.01,.04):
            paired={name:[next(p['median_ns'] for p in profiles if p['seed']==seed and p['model_id']==name
                  and p['batch_size']==batch and p['h_s']==h) for seed in seeds] for name in MODELS}
            timing.append(dict(batch_size=batch,h_s=h,**paired_comparison(paired['node_structured'],paired['affine_flow'],look,0.)))
    primary=[p for p in predictions if p['h_ns']==5000000]+[t for t in timing if t['batch_size']==128 and t['h_s']==.005]
    supported=all(p['contrast']['status']=='ok' and p['contrast']['lower']>0
                  and not p['contrast']['influence']['unstable'] for p in primary)
    supported=supported and not any(s['requires_solver_review'] for s in solver if s['h_ns']==5000000)
    return dict(predictions=predictions,timing=timing,node_solver_audit=solver,primary_combined_gain_supported=supported)


def prepare(root):
    root=Path(root).resolve()
    if (root/FREEZE).exists():return frozen(root)
    request=read_json(root/REQUEST);seeds=precision_seeds(request);base=evaluation_freeze(root)
    if request['original_evaluation_freeze_sha256']!=sha256_file(root/ORIGINAL_FREEZE):
        raise ValueError('Precision original evaluation changed')
    for entry in (request['parent_result'],request['parent_look']):
        verify_bindings(root,{entry['path']:entry['sha256']})
    selections=dict(base['selections']);inputs=dict(base['input_hashes'])
    for seed in request['new_seeds']:
        for name in MODELS:
            entry=read_json(root/f'reports/affine_fresh_fits/{name}__seed{seed}.json')
            if entry['model_id']!=name or entry['seed']!=seed:raise ValueError('Precision selected model/seed mismatch')
            folder=_verify_entry(root,entry);summary=read_json(folder/'summary.json')
            if not summary['qualification']['qualified']:raise ValueError('Unqualified nominal precision fit')
            selections[f'{name}:{seed}']=entry
            for filename in ('manifest.json','weights.pt','normalization.json'):
                path=str((folder/filename).relative_to(root));inputs[path]=sha256_file(root/path)
    reference=read_json(root/'reports/affine_fresh_references.json');_verify_entry(root,reference)
    original=read_json(root/'reports/affine_fresh_measurement.json');_verify_entry(root,original)
    for name in (REQUEST,SPEC,ORIGINAL_FREEZE,request['parent_result']['path'],request['parent_look']['path'],
                 original['path'],reference['path']):inputs[name]=sha256_file(root/name)
    inputs[str((root/reference['path']).parent.relative_to(root)/'references.npz')]=sha256_file((root/reference['path']).parent/'references.npz')
    source_hashes={**base['source_hashes'],'poc/affine_precision.py':sha256_file(root/'poc/affine_precision.py')}
    value={**base,'created_utc':utc_now(),'seeds':seeds,'new_seeds':request['new_seeds'],
           'selections':selections,'input_hashes':inputs,'source_hashes':source_hashes,
           'original_measurement':original,'reference':reference,'request_sha256':sha256_file(root/REQUEST)}
    for seed in seeds:
        for name in MODELS:load_model(root,name,seed,value)
    atomic_json(root/FREEZE,value);return value


def frozen(root):
    value=read_json(root/FREEZE);request=read_json(root/REQUEST)
    if value['seeds']!=precision_seeds(request) or value['request_sha256']!=sha256_file(root/REQUEST):
        raise ValueError('Precision frozen seed design changed')
    evaluation_freeze(root)
    verify_bindings(root,{**value['source_hashes'],**value['input_hashes']})
    # Original control producer binds this additive registry through its files too.
    return {**value,'input_hashes':{**value['input_hashes'],FREEZE:sha256_file(root/FREEZE)}}


def measure(root):
    from .execution import cpu_defaults,run_job
    from .shared_state_diagnostic import model_endpoints
    root=Path(root).resolve();freeze=frozen(root);cpu_defaults(freeze['seeds'][0])
    folder=_verify_entry(root,freeze['reference'])
    with np.load(folder/'references.npz',allow_pickle=False) as a:reference={k:a[k].copy() for k in a.files}
    def produce(output):
        rows=[];profiles=[]
        state=torch.tensor(reference['states'][:1],dtype=torch.float64)
        command=torch.tensor(reference['commands'][:1],dtype=torch.float64)
        for seed in freeze['new_seeds']:
            order=MODELS if freeze['seeds'].index(seed)%2==0 else tuple(reversed(MODELS))
            for name in order:
                model=load_model(root,name,seed,freeze)
                raw={k:reference[k] for k in ('probe_ids','strata','horizons_ns','truth_zero','truth_command')}
                raw.update({k:np.empty_like(raw['truth_command']) for k in ('model_zero','model_command','model_composed')})
                for hi,horizon in enumerate(reference['horizons_ns']):
                    for branch,value in model_endpoints(model,reference['states'],reference['commands'],int(horizon)).items():
                        raw['model_'+branch][:,hi]=value
                rows.extend(dict(model_id=name,seed=seed,**r) for r in prediction_rows(raw,freeze['coordinates']))
                _npz(output/f'{name}__seed{seed}.npz',raw)
                profiles.extend(_profile(model,state,command,name,seed))
                print('Precision predictions/profile',name,seed,flush=True)
        validate_profiles(profiles,freeze['new_seeds']);atomic_json(output/'profiles.json',profiles)
        return [dict(check_id='fresh.precision_predictions',status='pass',scope=PROTOCOL,tolerance=None,
                     reason='Complete new independent seed coordinates, original references and isolated paired timing')],dict(
                protocol_id=PROTOCOL,rows=rows,profiles=profiles,missing_cells=[],seeds=freeze['new_seeds'],
                numerical_envelope=read_json(folder/'summary.json')['numerical_envelope'])
    keys=job_configuration(root,'precision-predictions',dict(freeze_sha256=sha256_file(root/FREEZE),reference=freeze['reference']))
    record=run_job(root,'M40','confirmatory',_files(root,freeze,keys),('fresh.precision_predictions',),[],produce,config_keys=keys)
    entry=_receipt(root,record);atomic_json(root/(PREFIX+'_measurement.json'),entry);return entry


def measurements(root,freeze):
    entries=[freeze['original_measurement'],read_json(root/(PREFIX+'_measurement.json'))];blocks=[];folders={}
    for entry in entries:
        folder=_verify_entry(root,entry);summary=read_json(folder/'summary.json');raws={}
        seeds=sorted({r['seed'] for r in summary['rows']})
        for seed in seeds:
            for name in MODELS:
                with np.load(folder/f'{name}__seed{seed}.npz',allow_pickle=False) as a:raws[name,seed]={k:a[k].copy() for k in a.files}
                folders[name,seed]=folder
        recomputed,_=merge_measurements([dict(raws=raws,profiles=summary['profiles'])],seeds,freeze['coordinates'])
        sort=lambda rows:sorted(rows,key=lambda r:(r['seed'],r['model_id'],r['stratum'],r['h_ns'],r['component']))
        if sort(recomputed)!=sort(summary['rows']):raise ValueError('Precision raw-to-summary mismatch')
        blocks.append(dict(raws=raws,profiles=summary['profiles']))
    rows,profiles=merge_measurements(blocks,freeze['seeds'],freeze['coordinates'])
    return rows,profiles,entries,folders


def controls(root,kind='primary'):
    root=Path(root).resolve();freeze=frozen(root)
    if kind not in ('primary','replay','finer_physics'):raise ValueError('Unknown precision control batch')
    path=root/(PREFIX+'_'+kind+'_batch.json');digest=sha256_file(root/FREEZE)
    batch=read_json(path) if path.exists() else dict(state='partial',completed={},freeze_sha256=digest)
    if batch['freeze_sha256']!=digest:raise ValueError('Precision control freeze changed')
    count=freeze['episodes_per_seed'] if kind=='primary' else 1
    jobs=[(name,step,seed,task) for seed in freeze['seeds'] for task in range(count) for name,step in ALL_CONFIGURATIONS]
    with job_lock(job_key(dict(protocol_id=PROTOCOL,precision=digest,kind=kind)),root):
        for name,step,seed,task in jobs:
            key=f'{name}__step{step}__seed{seed}__task{task}'
            if key in batch['completed']:_verify_entry(root,batch['completed'][key]);continue
            print(f'{len(batch["completed"])}/{len(jobs)} {kind} {key}',flush=True)
            batch['completed'][key]=control_job(root,name,step,seed,task,freeze,kind)
            atomic_json(path,batch)
        batch['state']='complete';atomic_json(path,batch)
    return batch


def episode_rows(root,kind,freeze):
    from .affine_measurement import recompute_episode_cost
    path=root/(PREFIX+'_'+kind+'_batch.json');batch=read_json(path)
    if batch['state']!='complete' or batch['freeze_sha256']!=sha256_file(root/FREEZE):
        raise ValueError('Complete compatible precision control batch required')
    rows=[]
    for entry in batch['completed'].values():
        folder=_verify_entry(root,entry);recompute_episode_cost(folder);rows.append(read_json(folder/'summary.json'))
    return rows


def aggregate(root,look):
    root=Path(root).resolve();freeze=frozen(root)
    rows,profiles,entries,_=measurements(root,freeze)
    reference_folder=_verify_entry(root,freeze['reference'])
    envelope=read_json(reference_folder/'summary.json')['numerical_envelope']
    result=prediction_inference(rows,profiles,freeze['seeds'],look,envelope)
    primary_path=root/(PREFIX+'_primary_batch.json');complete=primary_path.exists() and read_json(primary_path)['state']=='complete'
    control=None;comparisons=[];convergence=None;episodes=[];identity={}
    if complete:
        episodes=episode_rows(root,'primary',freeze)
        control=control_summary(episodes,freeze['seeds'],freeze['episodes_per_seed'],ALL_CONFIGURATIONS)
        refined=episode_rows(root,'finer_physics',freeze)
        convergence=refinement_envelope(episodes,refined,freeze['seeds'],ALL_CONFIGURATIONS)
        if not convergence['converged']:raise ValueError('Control physics refinement requires extension')
        for a,n in ((5000000,5000000),(5000000,40000000),(40000000,40000000)):
            paired={name:[next(r['cost'] for r in control['seed_metrics'] if r['seed']==seed and r['model_id']==name
                   and r['model_step_ns']==step) for seed in freeze['seeds']] for name,step in (('affine_flow',a),('node_structured',n))}
            comparisons.append(dict(method_model_step_ns=a,baseline_model_step_ns=n,**paired_comparison(
                paired['node_structured'],paired['affine_flow'],look,convergence['absolute_cost_envelope'])))
        comparisons=holm_control_comparisons(comparisons,look,convergence['absolute_cost_envelope'])
        identity={kind:sha256_file(root/(PREFIX+'_'+kind+'_batch.json')) for kind in ('primary','finer_physics')}
    receipt=reserve_analysis_look(root,look,dict(measurements=entries,precision_freeze_sha256=sha256_file(root/FREEZE),controls=identity))
    result.update(protocol_id=PROTOCOL,seeds=freeze['seeds'],look_index=look,alpha_spent=receipt['alpha'],
                  look_receipt=receipt,measurement_entries=entries,control=control,control_comparisons=comparisons,
                  control_convergence=convergence,control_episodes=episodes,
                  limitation='Known80ms-lag smooth two-link robot; inference conditional on fixed fresh training/probe corpus; approximate seed-cluster assumptions.')
    atomic_json(root/(PREFIX+'_result.json'),result);return result


def reproduce(root):
    from .execution import cpu_defaults
    from .shared_state_diagnostic import model_endpoints
    root=Path(root).resolve();freeze=frozen(root);cpu_defaults(freeze['seeds'][0])
    _,_,_,folders=measurements(root,freeze);reference_folder=_verify_entry(root,freeze['reference'])
    with np.load(reference_folder/'references.npz',allow_pickle=False) as a:reference={k:a[k].copy() for k in a.files}
    subset=np.r_[np.flatnonzero(reference['strata']=='cold')[:64],np.flatnonzero(reference['strata']=='warm')[:64]]
    predictions=[];training=[];control_rows=[]
    for seed in freeze['seeds']:
        for name in MODELS:
            model=load_model(root,name,seed,freeze)
            with np.load(folders[name,seed]/f'{name}__seed{seed}.npz',allow_pickle=False) as a:saved={k:a[k].copy() for k in a.files}
            for hi,horizon in enumerate(reference['horizons_ns']):
                for branch,value in model_endpoints(model,reference['states'][subset],reference['commands'][subset],int(horizon)).items():
                    delta=value-saved['model_'+branch][subset,hi];q,v=float(np.abs(delta[:,:2]).max()),float(np.abs(delta[:,2:4]).max())
                    predictions.append(dict(model_id=name,seed=seed,h_ns=int(horizon),branch=branch,
                                       max_q_difference=q,max_v_difference=v,**{'pass':q<=1e-6 and v<=1e-5}))
            if seed in (freeze['seeds'][0],freeze['seeds'][-1]):
                repeated=load_model(root,name,seed,freeze,replica=True)
                old_folder=_verify_entry(root,freeze['selections'][f'{name}:{seed}'])
                new_folder=_verify_entry(root,read_json(root/f'reports/affine_fresh_replicas/{name}__seed{seed}.json'))
                metrics=retrain_metrics({k:v.numpy() for k,v in model.state_dict().items()},
                     {k:v.numpy() for k,v in repeated.state_dict().items()},
                     read_json(old_folder/'summary.json')['validation']['objective'],read_json(new_folder/'summary.json')['validation']['objective'])
                training.append(dict(model_id=name,seed=seed,**metrics))
    primary=read_json(root/(PREFIX+'_primary_batch.json'))
    for kind in ('replay','finer_physics'):
        rows=episode_rows(root,kind,freeze)
        expected={(name,step,seed,0) for seed in freeze['seeds'] for name,step in ALL_CONFIGURATIONS}
        actual={(r['model_id'],r['model_partition_ns'][0],r['seed'],r['episode_id']) for r in rows}
        if actual!=expected or len(rows)!=len(expected):raise ValueError('Precision replay coverage mismatch')
        batch=read_json(root/(PREFIX+'_'+kind+'_batch.json'))
        for key,entry in batch['completed'].items():
            folder=_verify_entry(root,entry);old_folder=_verify_entry(root,primary['completed'][key])
            old,new=read_json(old_folder/'summary.json'),read_json(folder/'summary.json')
            with np.load(old_folder/'true_trajectories.npz',allow_pickle=False) as a,np.load(folder/'true_trajectories.npz',allow_pickle=False) as b:
                idx=np.searchsorted(b['time_ns'],a['time_ns'])
                if not np.array_equal(b['time_ns'][idx],a['time_ns']):raise ValueError('Replayed physical clocks differ')
                delta=b['state'][idx]-a['state'];q,v=float(np.abs(delta[:,:2]).max()),float(np.abs(delta[:,2:4]).max())
            from json import loads
            commands=lambda p:np.array([loads(line)['command'] for line in (p/'action_segments.jsonl').read_text().splitlines()])
            command_difference=float(np.abs(commands(folder)-commands(old_folder)).max())
            cost=abs(new['realized_cost']-old['realized_cost'])/max(old['realized_cost'],1e-300)
            passed=(q<=1e-6 and v<=1e-5 and command_difference<=1e-6 and cost<=1e-4) if kind=='replay' else (cost<=.02 and old['success']==new['success'])
            control_rows.append(dict(kind=kind,cell=key,max_q_difference=q,max_v_difference=v,
                max_command_difference=command_difference,cost_relative_difference=cost,**{'pass':passed}))
    result=dict(protocol_id=PROTOCOL,predictions=predictions,training=training,controls=control_rows,
                **{'pass':all(r['pass'] for r in predictions+training+control_rows)})
    atomic_json(root/(PREFIX+'_reproduction.json'),result);return result


def demo(root):
    from .affine_release import render_fresh_replay
    from .affine_measurement import recompute_episode_cost
    from .efficiency_result import clip_data
    root=Path(root).resolve();freeze=frozen(root);rows=episode_rows(root,'primary',freeze);clips=[];bindings={}
    batch=read_json(root/(PREFIX+'_primary_batch.json'))
    control_summary(rows,freeze['seeds'],freeze['episodes_per_seed'],ALL_CONFIGURATIONS)
    for entry in batch['completed'].values():
        folder=_verify_entry(root,entry);row=read_json(folder/'summary.json');cost=recompute_episode_cost(folder)
        with np.load(folder/'true_trajectories.npz',allow_pickle=False) as a:clip=clip_data(a['time_ns'],a['state'],a['target'])
        clips.append(dict(model=row['model_id']+'_'+str(row['model_partition_ns'][0]//1000000),seed=row['seed'],
                    task=row['episode_id'],cost=cost,planner_seconds=row['planner_latency_median_seconds'],**clip))
        bindings[entry['path']]=entry['sha256']
    output=root/'reports/affine_fresh_demo.html';atomic_bytes(output,render_fresh_replay(clips,freeze['seeds'],freeze['episodes_per_seed']).encode())
    receipt=dict(protocol_id=PROTOCOL,clip_count=len(clips),output_path=str(output.relative_to(root)),output_sha256=sha256_file(output),
                 source_sha256=sha256_file(Path(__file__)),evidence=bindings)
    atomic_json(root/'reports/affine_fresh_demo_receipt.json',receipt);return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','measure','controls','replay','finer','aggregate','reproduce','demo'))
    parser.add_argument('--root',type=Path,default=Path.cwd());parser.add_argument('--look',type=int,default=2)
    args=parser.parse_args(argv);root=args.root.resolve()
    if args.command=='aggregate':value=aggregate(root,args.look);print(dict(primary_gain=value['primary_combined_gain_supported'],control=value['control'] is not None,look=args.look))
    elif args.command in ('controls','replay','finer'):
        value=controls(root,{'controls':'primary','replay':'replay','finer':'finer_physics'}[args.command]);print(dict(state=value['state'],completed=len(value['completed'])))
    else:
        value={'prepare':prepare,'measure':measure,'reproduce':reproduce,'demo':demo}[args.command](root)
        print({k:v for k,v in value.items() if k in ('path','sha256','seeds','pass','clip_count')})


if __name__=='__main__':main()
