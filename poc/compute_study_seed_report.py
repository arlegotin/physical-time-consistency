"""Additive fourth look; immutable prior reports remain available."""
from __future__ import annotations
import argparse
import csv
import io
import json
from pathlib import Path
import numpy as np
from . import compute_study_report as prior
from .compute_study import CONFIGS, MODELS, PREFIX, ALLOCATION, context as initial_context
from .compute_study_seed_precision import validate_request, context, receipt_path, fit_path
from .compute_study_precision import joined_cells
from .io import atomic_bytes, atomic_json, read_json, sha256_file, job_key, utc_now
from .affine_confirmation import paired_comparison
from .statistics import look_alpha


def extend_design(parent,request):
    seeds=validate_request(request)
    if (parent['seeds']!=request['prior_seeds'] or parent['episodes_per_seed']!=20
            or parent['comparison_family_size']!=8):
        raise ValueError('Changed previous paired design')
    return {**parent,'seeds':seeds}


def join_rows(old,new):
    return joined_cells(old,new)


def reproduction_coordinates(seeds):
    return {(name,h,seed,episode) for name,h in CONFIGS for seed in seeds for episode in (0,10)}


def validate_reproduction_cells(rows,seeds):
    coordinates=[(r['model_id'],r['step_ns'],r['seed'],r['episode_id']) for r in rows]
    expected=reproduction_coordinates(seeds)
    if len(coordinates)!=len(expected) or set(coordinates)!=expected:
        raise ValueError('Fourth reproduction coverage mismatch')


def cohort_identity(ledger):
    if 'completed' in ledger:
        return dict(kind=ledger['kind'],state=ledger['state'],
                    completed=sorted(ledger['completed'],key=lambda r:json.dumps(r,sort_keys=True)))
    return {key:cohort_identity(value) for key,value in ledger.items()}


def creation_time(root,spending):
    immutable=Path(root)/'reports/provenance/compute_study_look_004_result.json'
    if immutable.exists():
        result=read_json(immutable)
        if result['look']!=spending:raise ValueError('Changed immutable fourth look')
        return result['created_utc']
    return utc_now()


def look_design(root):
    parent,store,previous=prior.look_design(root,3)
    added,extra=context(root)
    return extend_design(parent,added['request']),store,previous,extra


def records(root,kind):
    from .compute_study_seed_precision import records as added_records
    old,old_paths,old_ledger=prior.records(root,kind,3)
    new,new_paths,new_ledger=added_records(root,kind)
    return join_rows(old,new),old_paths+new_paths,dict(previous=old_ledger,precision_004=new_ledger)


def reproduction(root,primary,paths):
    parent=prior.reproduction(root,primary,paths,3)
    from .compute_study_seed_precision import records as added_records
    added,store=context(root)
    key=lambda r:(r['model_id'],r['step_ns'],r['seed'],r['episode_id'])
    originals={key(r):(r,p) for r,p in zip(primary,paths)}
    rows=list(parent['controls']);envelope=parent['numerical_cost_envelope']
    for label in ('replay','finer'):
        data,folders,_=added_records(root,label)
        validate_reproduction_cells(data,added['request']['new_seeds'])
        for row,path in zip(data,folders):
            original,original_path=originals[key(row)]
            with np.load(original_path/'episode.npz',allow_pickle=False) as a,np.load(path/'episode.npz',allow_pickle=False) as b:
                if not np.array_equal(a['initial_state'],b['initial_state']) or not np.array_equal(a['target'],b['target']):
                    raise ValueError('Fourth reproduction task changed')
                if label=='replay':
                    qerr=float(np.max(np.abs(a['state'][:,:2]-b['state'][:,:2])))
                    verr=float(np.max(np.abs(a['state'][:,2:4]-b['state'][:,2:4])))
                    uerr=float(np.max(np.abs(a['commands']-b['commands'])))
                    if qerr>1e-6 or verr>1e-5 or uerr>1e-6:
                        raise ValueError('Fourth control replay tolerance failed')
                else:qerr=verr=uerr=None
            delta=abs(row['realized_cost']-original['realized_cost'])
            relative=delta/max(original['realized_cost'],1e-300)
            if relative>(1e-4 if label=='replay' else .02) or row['success']!=original['success']:
                raise ValueError('Control reproduction/convergence failed; full affected refinement required')
            if label=='finer':envelope=max(envelope,delta)
            rows.append(dict(kind=label,cell=list(key(row)),cost_difference=delta,
                             q_difference=qerr,v_difference=verr,command_difference=uerr))
    seed=added['request']['new_seeds'][-1]
    _,a=store.read(read_json(fit_path(root,'node_quadrature',seed)))
    _,b=store.read(read_json(fit_path(root,'node_quadrature',seed,True)))
    if (a['canonical_tensor_sha256']!=b['canonical_tensor_sha256']
            or a['qualification']['selected']!=b['qualification']['selected']):
        raise ValueError('Final comparator training replication differs')
    training=[parent['training'][0],dict(seed=seed,canonical_tensor_sha256=a['canonical_tensor_sha256'],exact=True)]
    return dict(status='pass',controls=rows,training=training,prior_training_checks=parent['training'],
                numerical_cost_envelope=envelope)


def training_table(root):
    parent,old_store,_=prior.look_design(root,3)
    fits=prior.training_table(root,parent,old_store,3)['fits']
    added,store=context(root)
    for name in MODELS:
        for seed in added['request']['new_seeds']:
            folder,summary=store.read(read_json(fit_path(root,name,seed)))
            resource=read_json(folder/'manifest.json')['resources']
            fits.append(dict(model_id=name,seed=seed,epochs=summary['qualification']['epochs'],
                validation_objective=summary['qualification']['selected']['objective'],
                cpu_seconds=resource['cpu_seconds'],wall_seconds=resource['wall_seconds']))
    averages=[]
    for name in MODELS:
        subset=[r for r in fits if r['model_id']==name]
        averages.append(dict(model_id=name,mean_cpu_seconds=float(np.mean([r['cpu_seconds'] for r in subset])),
            mean_validation_objective=float(np.mean([r['validation_objective'] for r in subset])),
            min_epochs=min(r['epochs'] for r in subset),max_epochs=max(r['epochs'] for r in subset)))
    return dict(fits=fits,averages=averages,
                limitation='Nominal qualification policy, different attained fit quality; not equal-accuracy training speed')


def aggregate(root):
    root=Path(root);look=4;freeze,store,previous,extra=look_design(root)
    rows,paths,ledger=records(root,'primary')
    table=prior.control_table(rows,freeze['seeds'],freeze['episodes_per_seed'])
    rep=reproduction(root,rows,paths);audit=[]
    for row,path in zip(rows,paths):
        with np.load(path/'episode.npz',allow_pickle=False) as data:
            cost=prior.independent_cost(data);prior.audit_candidates(data,row['fallback_count'])
        discrepancy=abs(cost-row['realized_cost'])
        if discrepancy>1e-12*(1+cost):raise ValueError('Independent raw cost mismatch')
        audit.append(discrepancy)
    comparisons=[]
    for method_step in (5_000_000,40_000_000):
        for baseline in ('node_quadrature','node_structured'):
            for baseline_step in (5_000_000,40_000_000):
                select=lambda name,h:[next(r['cost'] for r in table['seed_rows'] if r['model_id']==name and r['step_ns']==h and r['seed']==s) for s in freeze['seeds']]
                comparison=paired_comparison(select(baseline,baseline_step),select('affine_flow',method_step),look,rep['numerical_cost_envelope'])
                comparisons.append(dict(method_step_ns=method_step,baseline=baseline,baseline_step_ns=baseline_step,**comparison))
    comparisons=prior.holm_eight(comparisons,look,rep['numerical_cost_envelope'])
    precise=all(r['ratio']['status']=='ok' and r['ratio']['width']<=.04 and not r['contrast']['influence']['unstable'] for r in comparisons)
    useful=[h for h in (5_000_000,40_000_000) if all(r['holm']['supported'] for r in comparisons if r['method_step_ns']==h)]
    failfree=all(r['fallback_count']==0 for r in rows)
    allocation=read_json(root/ALLOCATION);store.read(allocation['receipt'])
    allocation_check=read_json(root/f'{PREFIX}_allocation_check.json')
    if (not allocation_check['matched'] or allocation_check['allocation_sha256']!=sha256_file(root/ALLOCATION)
            or allocation_check['source_sha256']!=sha256_file(root/'reports/provenance/check_compute_allocation.py')):
        raise ValueError('Missing/stale prospective direct-timer allocation check')
    measurement=read_json(root/f'{PREFIX}_measurement.json');folder,measured=store.read(measurement)
    from .compute_study_precision import PREFIX as previous_prefix
    previous_measurement=read_json(root/f'{previous_prefix}_measurement.json')
    previous_folder,previous_measured=previous.read(previous_measurement)
    new_measurement=read_json(receipt_path(extra.base,'measurement'))
    extra_folder,extra_measured=extra.read(new_measurement)
    profiles=prior.merge_profiles(prior.merge_profiles(measured['profiles'],previous_measured['profiles']),extra_measured['profiles'])
    reference_folder,reference_summary=store.read(measured['reference'])
    with np.load(reference_folder/'references.npz',allow_pickle=False) as data:
        reference={k:data[k].copy() for k in data.files}
    initial_seeds=initial_context(root)[0]['seeds'];added_seeds=context(root)[0]['request']['new_seeds']
    prediction=[];by_seed={}
    for seed in freeze['seeds']:
        for name in MODELS:
            source=folder if seed in initial_seeds else extra_folder if seed in added_seeds else previous_folder
            with np.load(source/f'{name}__seed{seed}.npz',allow_pickle=False) as data:
                raw={k:data[k].copy() for k in data.files}
            errors=dict(total=raw['model_command']-reference['truth_command'],drift=raw['model_zero']-reference['truth_zero'],
                response=(raw['model_command']-raw['model_zero'])-(reference['truth_command']-reference['truth_zero']),
                solver=raw['model_composed']-raw['model_command'])
            for stratum in ('cold','warm','all'):
                mask=np.ones(512,dtype=bool) if stratum=='all' else reference['strata']==stratum
                for hi,h in enumerate(reference['horizons_ns']):
                    for component,index in (('q',slice(0,2)),('v',slice(2,4))):
                        for metric,error in errors.items():
                            by_seed[name,seed,stratum,int(h),component,metric]=float(np.linalg.norm(error[mask,hi,index],axis=-1).mean())
    for stratum in ('cold','warm','all'):
        for h in reference['horizons_ns']:
            for component in ('q','v'):
                for metric in ('total','drift','response','solver'):
                    for baseline in ('node_quadrature','node_structured'):
                        values=lambda name:[by_seed[name,s,stratum,int(h),component,metric] for s in freeze['seeds']]
                        result=paired_comparison(values(baseline),values('affine_flow'),look,reference_summary['numerical_envelope'][component])
                        prediction.append(dict(stratum=stratum,h_ns=int(h),component=component,metric=metric,baseline=baseline,**result))
    timing=[]
    for batch in (1,128):
        for h in (.005,.01,.04):
            for baseline in ('node_quadrature','node_structured'):
                values=lambda name:[next(p['median_ns'] for p in profiles if p['model_id']==name and p['seed']==s and p['batch_size']==batch and p['h_s']==h) for s in freeze['seeds']]
                timing.append(dict(batch_size=batch,h_s=h,baseline=baseline,**paired_comparison(values(baseline),values('affine_flow'),look,0.)))
    identity=dict(primary=cohort_identity(ledger),measurement=measurement,previous_measurement=previous_measurement,new_measurement=new_measurement,
                  allocation_sha256=sha256_file(root/ALLOCATION),reproduction=rep,
                  prior_report_source_sha256=sha256_file(root/'poc/compute_study_report.py'),
                  report_source_sha256=sha256_file(Path(__file__)),look=look)
    spending=dict(protocol_id=freeze['protocol_id'],look=look,alpha=look_alpha(look),identity=job_key(identity))
    look_path=root/f'{PREFIX}_look{look}.json'
    if look_path.exists() and read_json(look_path)!=spending:raise ValueError('Changed evidence requires next analysis look')
    if not look_path.exists():
        if len(list(root.glob(PREFIX+'_look*.json')))!=3:raise ValueError('Cannot reset/skip analysis spending')
        atomic_json(look_path,spending)
    from .compute_study_episodes import context as episode_context
    resources=[]
    for source_store in (store,previous,episode_context(root)[1],extra):
        for path in sorted((source_store.base/'jobs').glob('*/manifest.json')):
            manifest=read_json(path)
            resources.append(dict(job_key=manifest['job_key'],stage=manifest['identity']['stage'],**manifest['resources'],
                                  artifact_bytes=sum((path.parent/k).stat().st_size for k in manifest['files'])))
    result=dict(protocol_id=freeze['protocol_id'],state='complete' if precise or useful else 'precision_required',look=spending,
        useful_control_intervals_ns=useful if failfree and table['actual_compute']['matched'] else [],
        control=table,control_comparisons=comparisons,precise_reportable=precise,fallback_free=failfree,
        predictions=prediction,timing=timing,reproduction=rep,independent_cost_max_difference=max(audit),
        allocation=allocation,allocation_check=allocation_check,resources=resources,training=training_table(root),
        seeds=freeze['seeds'],episodes_per_seed=freeze['episodes_per_seed'],
        limitations=['One known-lag smooth robot; fixed original training corpus and selected models',
                     'New tasks/probes; earlier findings motivated this comparator',
                     'Central ratio intervals are descriptive, not simultaneous eight-contrast bounds',
                     'Prediction/timing rows are secondary; no selected universal superiority claim',
                     'Offline planning delays are not physical latency'],created_utc=creation_time(root,spending))
    immutable=root/'reports/provenance/compute_study_look_004_result.json'
    if immutable.exists() and read_json(immutable)!=result:
        raise ValueError('Changed result requires a new look or explicit reporting correction')
    atomic_json(immutable,result);atomic_json(root/f'{PREFIX}_result.json',result)
    atomic_json(root/f'{PREFIX}_reproduction.json',rep)
    buffer=io.StringIO();writer=csv.DictWriter(buffer,fieldnames=list(table['configurations'][0]))
    writer.writeheader();writer.writerows(table['configurations'])
    atomic_bytes(root/f'{PREFIX}_metrics.csv',buffer.getvalue().encode())
    prior.write_findings(root,result)
    return result


def demo(root):
    root=Path(root);freeze,_,_,extra=look_design(root);rows,paths,_=records(root,'primary')
    prior.control_table(rows,freeze['seeds'],freeze['episodes_per_seed']);clips=[]
    for row,path in zip(rows,paths):
        with np.load(path/'episode.npz',allow_pickle=False) as a:
            stride=10_000_000//row['physics_step_ns']
            if abs(prior.independent_cost(a)-row['realized_cost'])>1e-12:raise ValueError('Demo cost differs from truth')
            clips.append(dict(model_id=row['model_id'],step_ns=row['step_ns'],seed=row['seed'],episode_id=row['episode_id'],
                time_s=(a['time_ns'][::stride]/1e9).tolist(),q=a['state'][::stride,:2].tolist(),target=a['target'].tolist(),
                cost=row['realized_cost'],planner_seconds=row['planning_median_seconds'],candidates=row['settings']['candidate_count']))
    prior.validate_clips(clips,freeze['seeds'],freeze['episodes_per_seed'])
    payload=json.dumps(clips,separators=(',',':'),allow_nan=False).replace('<','\\u003c')
    output=root/f'{PREFIX}_demo.html';atomic_bytes(output,prior.TEMPLATE.replace('MEASURED_PAYLOAD',payload).encode())
    atomic_json(root/f'{PREFIX}_demo_receipt.json',dict(clip_count=len(clips),output_sha256=sha256_file(output),
        result_sha256=sha256_file(root/f'{PREFIX}_result.json'),look=4,
        primary_sha256=sha256_file(root/f'{PREFIX}_primary.json'),
        precision_primary_sha256=sha256_file(root/'reports/compute_study_precision_002_primary.json'),
        episode_primary_sha256=sha256_file(root/'reports/compute_study_episode_003_primary.json'),
        seed_primary_sha256=sha256_file(receipt_path(extra.base,'primary'))))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('aggregate','demo','verify'))
    parser.add_argument('--root',type=Path,default=Path.cwd());args=parser.parse_args(argv)
    if args.command=='demo':demo(args.root)
    else:
        result=aggregate(args.root)
        print(dict(state=result['state'],episodes=result['control']['episode_count'],
                   useful_control_intervals_ns=result['useful_control_intervals_ns'],
                   compute_matched=result['control']['actual_compute']['matched']))


if __name__=='__main__':main()
