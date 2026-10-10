"""Raw-linked decision and shareable replay for the targeted compute study."""
from __future__ import annotations
import argparse
import csv
import io
import json
import math
from pathlib import Path
import numpy as np
from scipy.stats import t
from .compute_study import CONFIGS,MODELS,PREFIX,ALLOCATION,FREEZE,context
from .io import atomic_bytes,atomic_json,read_json,sha256_file,job_key,utc_now
from .affine_confirmation import paired_comparison
from .statistics import bootstrap_indices,contrast_bounds,look_alpha


def reduction(baseline,method):
    return None if baseline<=0 else float(1-method/baseline)


def compute_match(seconds):
    values=np.asarray(seconds,dtype=float)
    if values.ndim!=1 or len(values)<2 or not np.isfinite(values).all() or (values<=0).any():
        raise ValueError('Complete positive planner timings required')
    reference=float(values[-2])  # CONFIGS ends with RK4 fine/coarse.
    ratios=values/reference
    return dict(reference_seconds=reference,ratios=ratios.tolist(),matched=bool(np.all(np.abs(ratios-1)<=.1)),
                limitation='Development-selected budgets; actual task timing can differ')


def control_table(rows,seeds,episodes):
    expected={(n,h,s,e) for n,h in CONFIGS for s in seeds for e in range(episodes)}
    coordinates=[(r['model_id'],r['step_ns'],r['seed'],r['episode_id']) for r in rows]
    if len(coordinates)!=len(set(coordinates)) or set(coordinates)!=expected:
        raise ValueError('Missing/duplicate control cells')
    paired={};groups={}
    for r,key in zip(rows,coordinates):
        if (r['complete'] is not True or any(not math.isfinite(r[k]) or r[k]<0 for k in
                ('realized_cost','planning_median_seconds','planning_p95_seconds','q_rmse_rad','terminal_q_error_rad',
                 'velocity_rms_rad_s','effort_integral','fallback_count'))):
            raise ValueError('Invalid completed episode')
        work=r.get('work');count=r.get('settings',{}).get('candidate_count')
        if (not isinstance(work,dict) or type(count) is not int or count<2
                or any(type(work.get(k)) is not int or work[k]<0 for k in
                    ('network_calls','vector_field_calls','scalar_candidate_evaluations','history_tokens_processed'))):
            raise ValueError('Complete nonnegative measured work required')
        task=(r['initial_state'],r['target']);pair=key[2:]
        if pair in paired and paired[pair]!=task:raise ValueError('Paired task differs across models')
        paired[pair]=task;groups.setdefault(key[:3],[]).append(r)
    configurations=[];seed_rows=[]
    for name,h in CONFIGS:
        selected=[r for r in rows if (r['model_id'],r['step_ns'])==(name,h)]
        row=dict(model_id=name,step_ns=h,n_episodes=len(selected),successes=sum(r['success'] for r in selected),
                 fallback_count=sum(r['fallback_count'] for r in selected))
        for k in ('realized_cost','planning_median_seconds','planning_p95_seconds','q_rmse_rad','terminal_q_error_rad',
                  'velocity_rms_rad_s','effort_integral'):
            row[k]=float(np.mean([r[k] for r in selected]))
        row['network_calls_per_decision']=float(np.mean([r['work']['network_calls']/40 for r in selected]))
        row['neural_rows_per_decision']=float(np.mean([r['work']['network_calls']*r['settings']['candidate_count']/40 for r in selected]))
        row['candidate_transitions_per_decision']=float(np.mean([r['work']['scalar_candidate_evaluations']/40 for r in selected]))
        configurations.append(row)
        for s in seeds:
            subset=groups[name,h,s]
            seed_rows.append(dict(model_id=name,step_ns=h,seed=s,cost=float(np.mean([r['realized_cost'] for r in subset])),
                                 planning_seconds=float(np.mean([r['planning_median_seconds'] for r in subset]))))
    return dict(configurations=configurations,seed_rows=seed_rows,episode_count=len(rows),
                actual_compute=compute_match([r['planning_median_seconds'] for r in configurations]))


def holm_eight(comparisons,look,envelope):
    if len(comparisons)!=8:raise ValueError('All eight prespecified contrasts required')
    ordered=[];out=[dict(r) for r in comparisons]
    for i,r in enumerate(comparisons):
        x=np.array(r['baseline_by_seed'])-np.array(r['method_by_seed']);sd=float(x.std(ddof=1))
        indices=bootstrap_indices(len(x),look)
        p=1. if sd==0 else float(t.sf((x.mean()-2*envelope)/(sd/np.sqrt(len(x))),len(x)-1))
        p=max(p,float(np.mean(x[indices].mean(1)<=2*envelope)))
        ordered.append((p,i,x,indices))
    preceding=True
    for rank,(p,i,x,indices) in enumerate(sorted(ordered,key=lambda r:(r[0],r[1]))):
        alpha=look_alpha(look)/(8-rank)
        bound=contrast_bounds(x,alpha,indices,{'lower':2*envelope,'upper':2*envelope})
        robust=(bound['central_bootstrap'][0]>2*envelope) if 20000*alpha<100 else True
        supported=bool(preceding and bound['status']=='ok' and bound['lower']>0 and p<alpha
                       and robust and not bound['influence']['unstable'])
        preceding=supported
        out[i]['holm']=dict(rank=rank+1,alpha=alpha,p_value=p,bounds=bound,supported=supported)
    return out


def independent_cost(raw):
    times=np.asarray(raw['time_ns']);state=np.asarray(raw['state']);commands=np.asarray(raw['commands'])
    target=np.asarray(raw['target'])
    if (times.ndim!=1 or len(times)<41 or times[0]!=0 or times[-1]!=1_600_000_000
            or state.shape!=(len(times),6) or not np.isfinite(state).all() or not np.isfinite(commands).all()
            or target.shape!=(2,) or not np.isfinite(target).all()
            or commands.shape!=(40,2) or (np.abs(commands)>2).any() or not np.all(np.diff(times)==times[1])):
        raise ValueError('Incomplete or invalid physical raw episode')
    indices=np.searchsorted(times,np.arange(1,41)*40_000_000)
    if not np.array_equal(times[indices],np.arange(1,41)*40_000_000):raise ValueError('Missing common anchor')
    anchors=state[indices]
    if not np.array_equal(anchors,raw['anchors']):raise ValueError('Saved anchors differ from physical truth')
    q,v=anchors[:,:2]-raw['target'],anchors[:,2:4]
    tracking=math.fsum(float(x*x)*.04 for x in q.ravel())
    velocity=math.fsum(float(x*x)*.002 for x in v.ravel())
    effort=math.fsum(float(x*x)*.00004 for x in commands.ravel())
    terminal=math.fsum(float(x*x)*5 for x in q[-1])
    return math.fsum((tracking,velocity,effort,terminal))


def audit_candidates(raw,fallback_count):
    costs=np.asarray(raw['candidate_costs']);knots=np.asarray(raw['candidate_knots'])
    statuses=np.asarray(raw['candidate_statuses']);reasons=np.asarray(raw['candidate_reasons'])
    ids=np.asarray(raw['selected_ids']);chosen=np.asarray(raw['selected_knots'])
    if (costs.ndim!=2 or costs.shape[0]!=40 or knots.shape!=(*costs.shape,6,2)
            or statuses.shape!=costs.shape or reasons.shape!=costs.shape or ids.shape!=(40,)
            or chosen.shape!=(40,6,2) or not np.isfinite(knots).all() or (np.abs(knots)>2).any()
            or np.isnan(costs).any() or np.isneginf(costs).any()):
        raise ValueError('Incomplete candidate bank')
    failed=~np.isfinite(costs)
    if not np.array_equal(failed,statuses!='ok') or np.any(failed & (reasons=='')):
        raise ValueError('Unclassified or censored candidate failure')
    count=0
    for decision,selected in enumerate(ids):
        if selected==-1:
            count+=1
            if np.isfinite(costs[decision]).any() or chosen[decision].any():raise ValueError('Invalid zero fallback')
        elif (selected<0 or selected>=costs.shape[1] or failed[decision,selected]
              or not np.array_equal(chosen[decision],knots[decision,selected])):
            raise ValueError('Executed candidate is unscored or failed')
    if count!=fallback_count or not np.array_equal(chosen[:,0],raw['commands']):
        raise ValueError('Undisclosed fallback or altered executed command')
    return count


def validate_clips(clips,seeds,episodes):
    expected={(n,h,s,e) for n,h in CONFIGS for s in seeds for e in range(episodes)}
    ids=[(c['model_id'],c['step_ns'],c['seed'],c['episode_id']) for c in clips]
    if set(ids)!=expected or len(ids)!=len(set(ids)):raise ValueError('Missing/duplicate demo clips')
    pairs={}
    for c,k in zip(clips,ids):
        if (np.asarray(c['q']).shape!=(161,2) or not np.isfinite(c['q']).all()
                or not np.allclose(c['time_s'],np.arange(161)/100,atol=1e-12,rtol=0)
                or not np.isfinite(c['target']).all() or np.asarray(c['target']).shape!=(2,)
                or any(not math.isfinite(c[x]) or c[x]<0 for x in ('cost','planner_seconds'))):
            raise ValueError('Invalid saved demonstration trajectory')
        task=(c['q'][0],c['target'])
        if k[2:] in pairs and pairs[k[2:]]!=task:raise ValueError('Unpaired demo target/start')
        pairs[k[2:]]=task
    return clips


def extend_design(initial,request):
    from .compute_study_precision import validate_request
    seeds=validate_request(request)
    if initial['seeds']!=request['prior_seeds'] or initial['episodes_per_seed']!=request['episodes_per_seed']:
        raise ValueError('Precision changes initial paired design')
    return {**initial,'seeds':seeds}


def extend_episode_design(initial,request):
    from .compute_study_episodes import validate_request
    validate_request(request)
    if initial['seeds']!=request['seeds'] or initial['episodes_per_seed']!=request['prior_episodes_per_seed']:
        raise ValueError('Episode precision changes existing training clusters')
    return {**initial,'episodes_per_seed':request['episodes_per_seed']}


def reproduction_ids(look):
    if look in (1,2):return (0,)
    if look==3:return (0,10)
    raise ValueError('Unregistered reproduction design')


def look_design(root,look):
    f,store=context(root);extra=None
    if look in (2,3):
        from .compute_study_precision import context as precision_context
        continuation,extra=precision_context(root)
        if sha256_file(Path(root)/'reports/provenance/compute_study_look_001_result.json')!=continuation['request']['prior_result_sha256']:
            raise ValueError('Precision parent result changed')
        f=extend_design(f,continuation['request'])
        if look==3:
            from .compute_study_episodes import context as episode_context,PARENT
            episode,_=episode_context(root)
            if sha256_file(Path(root)/PARENT)!=episode['request']['prior_result_sha256']:
                raise ValueError('Episode precision parent result changed')
            f=extend_episode_design(f,episode['request'])
    elif look!=1:raise ValueError('Unregistered analysis look')
    return f,store,extra


def merge_profiles(first,added):
    profiles=list(first)+list(added)
    keys=[(r['model_id'],r['seed'],r['batch_size'],r['h_s']) for r in profiles]
    if len(keys)!=len(set(keys)):raise ValueError('Duplicate timing profiles')
    return profiles


def records(root,kind,look=1):
    f,store=context(root);ledger=read_json(Path(root)/f'{PREFIX}_{kind}.json')
    if ledger['state']!='complete':raise ValueError('Incomplete '+kind+' cohort')
    rows=[];paths=[]
    for receipt in ledger['completed']:
        folder,summary=store.read(receipt);rows.append(summary);paths.append(folder)
    if look>=2:
        from .compute_study_precision import records as added_records,joined_cells
        additional,additional_paths,extra=added_records(root,kind)
        rows=joined_cells(rows,additional);paths+=additional_paths;ledger=dict(initial=ledger,precision=extra)
    if look==3:
        from .compute_study_episodes import records as episode_records
        additional,additional_paths,episode_ledger=episode_records(root,kind)
        rows=joined_cells(rows,additional);paths+=additional_paths;ledger=dict(previous=ledger,episode_precision=episode_ledger)
    return rows,paths,ledger


def reproduction(root,primary,paths,look=1):
    f,store,extra=look_design(root,look);replay,replay_paths,_=records(root,'replay',look);finer,finer_paths,_=records(root,'finer',look)
    key=lambda r:(r['model_id'],r['step_ns'],r['seed'],r['episode_id'])
    expected={(n,h,s,e) for n,h in CONFIGS for s in f['seeds'] for e in reproduction_ids(look)}
    old={key(r):(r,p) for r,p in zip(primary,paths) if r['episode_id'] in reproduction_ids(look)}
    rows=[];envelope=0.
    for label,data,folders in (('replay',replay,replay_paths),('finer',finer,finer_paths)):
        if set(key(r) for r in data)!=expected or len(data)!=len(expected):raise ValueError('Reproduction coverage mismatch')
        for r,p in zip(data,folders):
            original,op=old[key(r)]
            with np.load(op/'episode.npz',allow_pickle=False) as a,np.load(p/'episode.npz',allow_pickle=False) as b:
                if not np.array_equal(a['initial_state'],b['initial_state']) or not np.array_equal(a['target'],b['target']):
                    raise ValueError('Reproduction task changed')
                if label=='replay':
                    qerr=float(np.max(np.abs(a['state'][:,:2]-b['state'][:,:2])))
                    verr=float(np.max(np.abs(a['state'][:,2:4]-b['state'][:,2:4])))
                    uerr=float(np.max(np.abs(a['commands']-b['commands'])))
                    if qerr>1e-6 or verr>1e-5 or uerr>1e-6:raise ValueError('Control replay tolerance failed')
                else:qerr=verr=uerr=None
            delta=abs(r['realized_cost']-original['realized_cost']);relative=delta/max(original['realized_cost'],1e-300)
            if relative>(1e-4 if label=='replay' else .02) or r['success']!=original['success']:
                raise ValueError('Control reproduction/convergence failed; full affected refinement required')
            if label=='finer':envelope=max(envelope,delta)
            rows.append(dict(kind=label,cell=list(key(r)),cost_difference=delta,q_difference=qerr,v_difference=verr,command_difference=uerr))
    training=[]
    for seed in (f['seeds'][0],f['seeds'][-1]):
        if look>=2 and seed not in context(root)[0]['seeds']:
            from .compute_study_precision import fit_path
            a,sa=extra.read(read_json(fit_path(root,'node_quadrature',seed)))
            b,sb=extra.read(read_json(fit_path(root,'node_quadrature',seed,True)))
        else:
            a,sa=store.read(read_json(Path(root)/f'{PREFIX}_fits/seed{seed}.json'))
            b,sb=store.read(read_json(Path(root)/f'{PREFIX}_replicas/seed{seed}.json'))
        if (sa['canonical_tensor_sha256']!=sb['canonical_tensor_sha256']
                or sa['qualification']['selected']!=sb['qualification']['selected']):
            raise ValueError('New comparator training replication differs')
        training.append(dict(seed=seed,canonical_tensor_sha256=sa['canonical_tensor_sha256'],exact=True))
    return dict(status='pass',controls=rows,training=training,numerical_cost_envelope=envelope)


def training_table(root,freeze,store,look=1):
    from .affine_confirmation import _verify_entry
    fits=[];initial_seeds=context(root)[0]['seeds'];extra=look_design(root,look)[2]
    for name in MODELS:
        for seed in freeze['seeds']:
            if seed not in initial_seeds:
                from .compute_study_precision import fit_path
                folder,summary=extra.read(read_json(fit_path(root,name,seed)));resource=read_json(folder/'manifest.json')['resources']
            elif name=='node_quadrature':
                receipt=read_json(Path(root)/f'{PREFIX}_fits/seed{seed}.json')
                folder,summary=store.read(receipt);manifest=read_json(folder/'manifest.json')
                resource=manifest['resources']
            else:
                receipt=freeze['original_selections'][f'{name}:{seed}'];folder=_verify_entry(Path(root),receipt)
                summary=read_json(folder/'summary.json');resource=read_json(folder/'resources.json')
            fits.append(dict(model_id=name,seed=seed,epochs=summary['qualification']['epochs'],
                validation_objective=summary['qualification']['selected']['objective'],
                cpu_seconds=resource['cpu_seconds'],wall_seconds=resource['wall_seconds']))
    averages=[]
    for name in MODELS:
        subset=[r for r in fits if r['model_id']==name]
        averages.append(dict(model_id=name,mean_cpu_seconds=float(np.mean([r['cpu_seconds'] for r in subset])),
            mean_validation_objective=float(np.mean([r['validation_objective'] for r in subset])),
            min_epochs=min(r['epochs'] for r in subset),max_epochs=max(r['epochs'] for r in subset)))
    return dict(fits=fits,averages=averages,limitation='Nominal qualification policy, different attained fit quality; not equal-accuracy training speed')


def aggregate(root,look=1):
    root=Path(root);f,store,extra=look_design(root,look);rows,paths,ledger=records(root,'primary',look)
    table=control_table(rows,f['seeds'],f['episodes_per_seed']);rep=reproduction(root,rows,paths,look)
    audit=[]
    for row,path in zip(rows,paths):
        with np.load(path/'episode.npz',allow_pickle=False) as data:
            cost=independent_cost(data);audit_candidates(data,row['fallback_count'])
        discrepancy=abs(cost-row['realized_cost'])
        if discrepancy>1e-12*(1+cost):raise ValueError('Independent raw cost mismatch')
        audit.append(discrepancy)
    comparisons=[]
    for method_step in (5_000_000,40_000_000):
        for baseline in ('node_quadrature','node_structured'):
            for baseline_step in (5_000_000,40_000_000):
                select=lambda name,h:[next(r['cost'] for r in table['seed_rows'] if r['model_id']==name and r['step_ns']==h and r['seed']==s) for s in f['seeds']]
                comparison=paired_comparison(select(baseline,baseline_step),select('affine_flow',method_step),look,rep['numerical_cost_envelope'])
                comparisons.append(dict(method_step_ns=method_step,baseline=baseline,baseline_step_ns=baseline_step,**comparison))
    comparisons=holm_eight(comparisons,look,rep['numerical_cost_envelope'])
    precision=all(r['ratio']['status']=='ok' and r['ratio']['width']<=.04 and not r['contrast']['influence']['unstable'] for r in comparisons)
    useful=[h for h in (5_000_000,40_000_000) if all(r['holm']['supported'] for r in comparisons if r['method_step_ns']==h)]
    failfree=all(r['fallback_count']==0 for r in rows)
    allocation=read_json(root/ALLOCATION);store.read(allocation['receipt'])
    allocation_check=read_json(root/f'{PREFIX}_allocation_check.json')
    if (not allocation_check['matched'] or allocation_check['allocation_sha256']!=sha256_file(root/ALLOCATION)
            or allocation_check['source_sha256']!=sha256_file(root/'reports/provenance/check_compute_allocation.py')):
        raise ValueError('Missing/stale prospective direct-timer allocation check')
    measurement=read_json(root/f'{PREFIX}_measurement.json');folder,measured=store.read(measurement)
    profiles=measured['profiles'];extra_folder=None;extra_measurement=None;initial_seeds=context(root)[0]['seeds']
    if extra is not None:
        from .compute_study_precision import PREFIX as precision_prefix
        extra_measurement=read_json(root/f'{precision_prefix}_measurement.json');extra_folder,added=extra.read(extra_measurement)
        profiles=merge_profiles(profiles,added['profiles'])
    reference_folder,reference_summary=store.read(measured['reference'])
    with np.load(reference_folder/'references.npz',allow_pickle=False) as data:reference={k:data[k].copy() for k in data.files}
    prediction=[];by_seed={}
    for seed in f['seeds']:
        for name in MODELS:
            source=folder if seed in initial_seeds else extra_folder
            with np.load(source/f'{name}__seed{seed}.npz',allow_pickle=False) as data:raw={k:data[k].copy() for k in data.files}
            errors=dict(total=raw['model_command']-reference['truth_command'],drift=raw['model_zero']-reference['truth_zero'],
                response=(raw['model_command']-raw['model_zero'])-(reference['truth_command']-reference['truth_zero']),
                solver=raw['model_composed']-raw['model_command'])
            for stratum in ('cold','warm','all'):
                mask=np.ones(512,dtype=bool) if stratum=='all' else reference['strata']==stratum
                for hi,h in enumerate(reference['horizons_ns']):
                    for component,index in (('q',slice(0,2)),('v',slice(2,4))):
                        for metric,error in errors.items():
                            value=float(np.linalg.norm(error[mask,hi,index],axis=-1).mean())
                            by_seed[name,seed,stratum,int(h),component,metric]=value
    for stratum in ('cold','warm','all'):
        for h in reference['horizons_ns']:
            for component in ('q','v'):
                for metric in ('total','drift','response','solver'):
                    for baseline in ('node_quadrature','node_structured'):
                        values=lambda name:[by_seed[name,s,stratum,int(h),component,metric] for s in f['seeds']]
                        result=paired_comparison(values(baseline),values('affine_flow'),look,reference_summary['numerical_envelope'][component])
                        prediction.append(dict(stratum=stratum,h_ns=int(h),component=component,metric=metric,baseline=baseline,**result))
    timing=[]
    for batch in (1,128):
        for h in (.005,.01,.04):
            for baseline in ('node_quadrature','node_structured'):
                values=lambda name:[next(p['median_ns'] for p in profiles if p['model_id']==name and p['seed']==s and p['batch_size']==batch and p['h_s']==h) for s in f['seeds']]
                timing.append(dict(batch_size=batch,h_s=h,baseline=baseline,**paired_comparison(values(baseline),values('affine_flow'),look,0.)))
    identity=dict(primary=ledger,measurement=measurement,extra_measurement=extra_measurement,allocation_sha256=sha256_file(root/ALLOCATION),reproduction=rep,
                  report_source_sha256=sha256_file(Path(__file__)),look=look)
    look_path=root/f'{PREFIX}_look{look}.json';spending=dict(protocol_id=f['protocol_id'],look=look,alpha=look_alpha(look),identity=job_key(identity))
    if look_path.exists() and read_json(look_path)!=spending:raise ValueError('Changed evidence requires next analysis look')
    if not look_path.exists():
        prior=list(root.glob(PREFIX+'_look*.json'))
        if look!=len(prior)+1:raise ValueError('Cannot reset/skip analysis spending')
        atomic_json(look_path,spending)
    resources=[]
    resource_paths=sorted((store.base/'jobs').glob('*/manifest.json'))
    if extra is not None:resource_paths+=sorted((extra.base/'jobs').glob('*/manifest.json'))
    if look==3:
        from .compute_study_episodes import context as episode_context
        _,episode_store=episode_context(root)
        resource_paths+=sorted((episode_store.base/'jobs').glob('*/manifest.json'))
    for p in resource_paths:
        m=read_json(p);resources.append(dict(job_key=m['job_key'],stage=m['identity']['stage'],**m['resources'],artifact_bytes=sum((p.parent/k).stat().st_size for k in m['files'])))
    result=dict(protocol_id=f['protocol_id'],state='complete' if (precision or useful) else 'precision_required',look=spending,
        useful_control_intervals_ns=useful if failfree and table['actual_compute']['matched'] else [],
        control=table,control_comparisons=comparisons,precise_reportable=precision,fallback_free=failfree,
        predictions=prediction,timing=timing,reproduction=rep,independent_cost_max_difference=max(audit),
        allocation=allocation,allocation_check=allocation_check,resources=resources,training=training_table(root,f,store,look),seeds=f['seeds'],episodes_per_seed=f['episodes_per_seed'],
        limitations=['One known-lag smooth robot; fixed original training corpus and selected models',
                     'New tasks/probes; earlier findings motivated this comparator',
                     'Central ratio intervals are descriptive, not simultaneous eight-contrast bounds',
                     'Prediction/timing rows are secondary; no selected universal superiority claim',
                     'Offline planning delays are not physical latency'],created_utc=utc_now())
    output=root/f'{PREFIX}_result.json'
    if output.exists():
        previous=read_json(output)
        if previous['look']==spending:result['created_utc']=previous['created_utc']
    immutable=root/f'reports/provenance/compute_study_look_{look:03d}_result.json'
    if immutable.exists() and read_json(immutable)!=result:raise ValueError('Changed result requires a new look or explicit reporting correction')
    atomic_json(immutable,result);atomic_json(output,result);atomic_json(root/f'{PREFIX}_reproduction.json',rep)
    csv_buffer=io.StringIO();writer=csv.DictWriter(csv_buffer,fieldnames=list(table['configurations'][0]));writer.writeheader();writer.writerows(table['configurations'])
    atomic_bytes(root/f'{PREFIX}_metrics.csv',csv_buffer.getvalue().encode())
    write_findings(root,result);return result


def write_findings(root,result):
    useful=result['useful_control_intervals_ns']
    headline=('AffineFlow has a supported controller-cost advantage against every tested NODE option at a matched allocation.' if useful else
              'The matched-compute comparison does not establish an AffineFlow controller advantage over every tested NODE option.')
    lines=['# Compute-matched NODE follow-up', '',headline,'',
        'The controller tries motor commands in a learned model, chooses a plan, and applies its first 40 milliseconds to the simulated arm. AffineFlow is our model; NODE learns continuous acceleration and advances it numerically. The cost score combines target error, speed and commanded torque. Lower is better. This comparison gives the models roughly the same CPU planning time.','',
        f"Complete paired matrix: {result['control']['episode_count']} episodes, {len(result['seeds'])} training clusters, three neural models at 5/40 ms. Hardware sensing, planning and command holds remain 40 ms. Budgets were selected from timing before independent task outcomes.",'',
        '| Model | Query | Candidates × iterations | Mean cost | Tracking RMSE | Successes | Mean episode planner median |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in result['control']['configurations']:
        budget=result['allocation']['settings'][f"{row['model_id']}:{row['step_ns']}"]
        lines.append(f"| {row['model_id']} | {row['step_ns']/1e6:g} ms | {budget['candidate_count']} × 4 | {row['realized_cost']:.6f} | {row['q_rmse_rad']:.6f} rad | {row['successes']}/{row['n_episodes']} | {row['planning_median_seconds']:.4f} s |")
    lines.extend(['',f"Actual episode planning means remain within the 10% matching tolerance: {result['control']['actual_compute']['matched']}. All timing ratios and p95 measurements are retained in the machine result. No physical wall-clock delay was injected.",'',
        '| Affine query | NODE comparator | Cost reduction | Central 95% interval | Holm-supported gain? |',
        '|---|---|---:|---:|---|'])
    for r in result['control_comparisons']:
        ci=r['ratio'];interval='unresolved' if ci['status']!='ok' else f"{100*ci['lower']:.2f}% to {100*ci['upper']:.2f}%"
        lines.append(f"| {r['method_step_ns']/1e6:g} ms | {r['baseline']}, {r['baseline_step_ns']/1e6:g} ms | {100*r['point_improvement']:.2f}% | {interval} | {r['holm']['supported']} |")
    lines.extend(['','No percentage improvement floor is required. The eight prospectively declared cost comparisons share Holm correction at the recorded look alpha. A positive point estimate with an interval crossing zero remains unresolved. Success is the strict final-200-ms criterion and is retained separately from the weighted cost. A narrow non-detection is not equivalence.','',
                  '## Prediction and isolated inference','',
                  '| Comparator | Probe group | Component | Affine error | NODE error | Reduction at 5 ms |',
                  '|---|---|---|---:|---:|---:|'])
    for r in result['predictions']:
        if r['h_ns']==5_000_000 and r['metric']=='total':
            lines.append(f"| {r['baseline']} | {r['stratum']} | {r['component']} | {r['method_mean']:.6g} | {r['baseline_mean']:.6g} | {100*r['point_improvement']:.2f}% |")
    lines.extend(['','These secondary rows retain all cold and warm states; negative reductions mean AffineFlow is worse. Position uses radians and velocity rad/s. Absolute errors, finite command response and solver discrepancy are available for 5,10,40 ms. They are not proof of universal model superiority.','',
                  '| Comparator | Batch | Affine µs | NODE µs | Inference-time reduction at 5 ms |','|---|---:|---:|---:|---:|'])
    for r in result['timing']:
        if r['h_s']==.005:lines.append(f"| {r['baseline']} | {r['batch_size']} | {r['method_mean']/1000:.2f} | {r['baseline_mean']/1000:.2f} | {100*r['point_improvement']:.2f}% |")
    lines.extend(['','Quadrature NODE and AffineFlow both use two network evaluations; RK4 NODE uses four. Quadrature uses the same 17,922-parameter acceleration network and seed initialization as RK4; AffineFlow has 18,182 parameters and a torque-affine prior. The comparator was trained through its own solver using the same fixed data, objective, nominal-only optimizer opportunity and selection rule. Training time must be read beside nominal fitting error, rather than called equal-accuracy speed.','',
        '| Model | Mean CPU minutes per fit | Mean nominal objective | Epoch range |',
        '|---|---:|---:|---:|'])
    for r in result['training']['averages']:
        lines.append(f"| {r['model_id']} | {r['mean_cpu_seconds']/60:.2f} | {r['mean_validation_objective']:.6g} | {r['min_epochs']}–{r['max_epochs']} |")
    lines.extend(['','Measured workload per planning decision (40 decisions per episode):','',
        '| Model | Query | Batched network calls | Neural rows (calls × batch) | Candidate transitions |',
        '|---|---:|---:|---:|---:|'])
    for r in result['control']['configurations']:
        lines.append(f"| {r['model_id']} | {r['step_ns']/1e6:g} ms | {r['network_calls_per_decision']:g} | {r['neural_rows_per_decision']:g} | {r['candidate_transitions_per_decision']:g} |")
    resources=result['resources']
    lines.extend(['',f"New sealed jobs consumed {sum(r['cpu_seconds'] for r in resources)/3600:.3f} CPU hours and {sum(r['wall_seconds'] for r in resources)/3600:.3f} recorded job-wall hours. Saved artifacts occupy {sum(r['artifact_bytes'] for r in resources)/2**30:.3f} GiB. Maximum recorded process peak RSS is {max(r['peak_rss_bytes'] for r in resources)/2**20:.1f} MiB; it is a process high-water mark, not a reset per-job allocation. Previously trained AffineFlow/RK4 fits are listed separately above. Raw effort (squared torque integral, not energy), velocity RMS, terminal error and p95 planning times remain in the machine table."])
    lines.extend(['',
        '## Reproduction and evidence','',
        f"Fresh-process control replay and finer-physics subset: {len(result['reproduction']['controls'])} checks pass. Two new-comparator training replications reproduce exact selected tensors. Independent arithmetic recomputes every episode cost; maximum difference {result['independent_cost_max_difference']:.3g}. Empirical cost envelope: {result['reproduction']['numerical_cost_envelope']:.3g}. Zero intrinsic fallbacks: {result['fallback_free']}.",'',
        'New raw artifacts are on the owner-authorized Data volume, with explicit root, content-addressed jobs, source/model/data bindings and complete-file hashes. Earlier results and frozen producers remain intact. A source-only Git checkout needs the indexed artifact store and original checkpoints to reproduce the measurements.','',
        'Machine results: [compute_study_result.json](compute_study_result.json); practical table: [compute_study_metrics.csv](compute_study_metrics.csv); saved simulation replay: [compute_study_demo.html](compute_study_demo.html). Exact commands and scope are in [compute_study_reproduce.md](compute_study_reproduce.md).','',
        'Scope: one fully observed smooth known-lag arm, offline CPU MPC and this training corpus. No contact, unknown motor dynamics, physical robot, or stronger-than-all-calibrated-physics claim. A larger robot is not justified merely to rescue a null result.'])
    atomic_bytes(Path(root)/f'{PREFIX}_findings.md',('\n'.join(lines)+'\n').encode())


def demo(root,look=1):
    root=Path(root);f,store,_=look_design(root,look);rows,paths,_=records(root,'primary',look);control_table(rows,f['seeds'],f['episodes_per_seed']);clips=[]
    for row,path in zip(rows,paths):
        with np.load(path/'episode.npz',allow_pickle=False) as a:
            stride=10_000_000//row['physics_step_ns']
            if abs(independent_cost(a)-row['realized_cost'])>1e-12:raise ValueError('Demo cost differs from truth')
            clips.append(dict(model_id=row['model_id'],step_ns=row['step_ns'],seed=row['seed'],episode_id=row['episode_id'],
                time_s=(a['time_ns'][::stride]/1e9).tolist(),q=a['state'][::stride,:2].tolist(),target=a['target'].tolist(),
                cost=row['realized_cost'],planner_seconds=row['planning_median_seconds'],candidates=row['settings']['candidate_count']))
    validate_clips(clips,f['seeds'],f['episodes_per_seed'])
    payload=json.dumps(clips,separators=(',',':'),allow_nan=False).replace('<','\\u003c')
    output=root/f'{PREFIX}_demo.html';atomic_bytes(output,TEMPLATE.replace('MEASURED_PAYLOAD',payload).encode())
    atomic_json(root/f'{PREFIX}_demo_receipt.json',dict(clip_count=len(clips),output_sha256=sha256_file(output),
        result_sha256=sha256_file(root/f'{PREFIX}_result.json'),primary_sha256=sha256_file(root/f'{PREFIX}_primary.json'),look=look,
        precision_primary_sha256=sha256_file(root/'reports/compute_study_precision_002_primary.json') if look>=2 else None,
        episode_primary_sha256=sha256_file(root/'reports/compute_study_episode_003_primary.json') if look==3 else None))


TEMPLATE='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Compute-matched robot simulation</title>
<style>body{font:16px system-ui;margin:24px;color:#202832;background:#f7f8fa}p{max-width:1000px;line-height:1.5}select,button,input{font:inherit;margin:8px}#cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px}.card{padding:16px;border:1px solid #ddd;background:white}canvas{width:100%;height:260px}.stats{font-size:14px;line-height:1.6}</style>
<h1>Three learned robot models, matched planning allocation</h1><p>These are actual saved simulations. Each controller sees the same arm and target every 40 ms; candidate budgets were chosen from timing before these tasks. The two NODE versions use two or four neural evaluations. Both NODE and AffineFlow can use fine 5 ms or coarse 40 ms queries. See the accompanying report for complete results and uncertainty.</p>
<label>Training pair <select id="seed"></select></label><label>Task <select id="task"></select></label><button id="play">Play / pause</button><br>
<input id="scrub" aria-label="Physical time" type="range" min="0" max="160" value="0" style="width:min(700px,90%)"><span id="time"></span>
<p>Dashed red arms mark the target. Motion follows physical simulation time. CPU planning time is displayed separately; this is offline simulation.</p><div id="cards"></div>
<script type="application/json" id="clips">MEASURED_PAYLOAD</script><script>
const clips=JSON.parse(document.getElementById('clips').textContent),byId=x=>document.getElementById(x),names={affine_flow:'AffineFlow',node_quadrature:'NODE, two evaluations',node_structured:'NODE, four evaluations'};
const configs=Object.keys(names).flatMap(n=>[5000000,40000000].map(h=>[n,h]));let selected=[],position=0,playing=false,last=0;
for(const seed of [...new Set(clips.map(c=>c.seed))])byId('seed').add(new Option(String(seed),String(seed)));
for(const task of [...new Set(clips.map(c=>c.episode_id))])byId('task').add(new Option(String(task+1),String(task)));
function choose(){selected=configs.map(([n,h])=>clips.find(c=>c.model_id===n&&c.step_ns===h&&c.seed===+byId('seed').value&&c.episode_id===+byId('task').value));byId('cards').innerHTML=selected.map((c,i)=>'<div class="card"><b>'+names[c.model_id]+', '+c.step_ns/1000000+' ms</b><canvas id="c'+i+'" width="300" height="260"></canvas><div class="stats" id="s'+i+'"></div></div>').join('');position=0;draw();}
function arm(ctx,q,color,dash){const x=150,y=140,scale=225,x1=x+scale*.3*Math.cos(q[0]),y1=y-scale*.3*Math.sin(q[0]),x2=x1+scale*.25*Math.cos(q[0]+q[1]),y2=y1-scale*.25*Math.sin(q[0]+q[1]);ctx.beginPath();ctx.setLineDash(dash?[6,5]:[]);ctx.lineWidth=dash?3:8;ctx.strokeStyle=color;ctx.moveTo(x,y);ctx.lineTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();ctx.setLineDash([]);}
function draw(){const frame=Math.min(160,Math.round(position));byId('time').textContent=(frame/100).toFixed(2)+' s';byId('scrub').value=frame;selected.forEach((c,i)=>{const ctx=byId('c'+i).getContext('2d');ctx.clearRect(0,0,300,260);arm(ctx,c.target,'#cf4b4b',true);arm(ctx,c.q[frame],'#087e8b',false);byId('s'+i).textContent='Episode cost '+c.cost.toFixed(5)+' (lower is better). Planner median '+c.planner_seconds.toFixed(4)+' s; '+c.candidates+' candidates ×4. Joint error '+Math.hypot(c.q[frame][0]-c.target[0],c.q[frame][1]-c.target[1]).toFixed(4)+' rad.';});}
byId('seed').onchange=choose;byId('task').onchange=choose;byId('scrub').oninput=()=>{position=+byId('scrub').value;draw();};byId('play').onclick=()=>{if(position>=160)position=0;playing=!playing;last=0;};
function animate(now){if(playing){if(last)position=Math.min(160,position+(now-last)/10);draw();if(position>=160)playing=false;}last=now;requestAnimationFrame(animate);}choose();requestAnimationFrame(animate);
</script></html>'''


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('aggregate','demo','verify'))
    parser.add_argument('--root',type=Path,default=Path.cwd());parser.add_argument('--look',type=int,default=1)
    args=parser.parse_args(argv)
    if args.command=='demo':demo(args.root,args.look)
    else:
        result=aggregate(args.root,args.look)
        print(dict(state=result['state'],episodes=result['control']['episode_count'],useful_control_intervals_ns=result['useful_control_intervals_ns'],compute_matched=result['control']['actual_compute']['matched']))


if __name__=='__main__':main()
