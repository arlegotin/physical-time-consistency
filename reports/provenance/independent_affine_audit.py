"""Read-only scalar recomputation of measured motion, timing and physical costs."""
from pathlib import Path
import json
import math
import statistics
import sys

import numpy as np
from scipy.stats import t

from poc.io import atomic_json,read_json,sha256_file,utc_now
from poc.affine_confirmation import _verify_entry

root=Path.cwd()
precision=(root/'config/affine_precision_freeze_001.json').exists() and (root/'reports/affine_precision_001_measurement.json').exists()
freeze=read_json(root/('config/affine_precision_freeze_001.json' if precision else 'config/affine_confirmation_freeze.json'))
result=read_json(root/('reports/affine_precision_001_result.json' if precision else 'reports/affine_fresh_result.json'))
entries=[freeze['original_measurement'],read_json(root/'reports/affine_precision_001_measurement.json')] if precision else [read_json(root/'reports/affine_fresh_measurement.json')]
reference_folder=_verify_entry(root,read_json(root/'reports/affine_fresh_references.json'))
with np.load(reference_folder/'references.npz',allow_pickle=False) as stored:
    reference={k:stored[k].copy() for k in stored.files}
metrics=[];primary={};max_difference=0.;profiles=[];bindings={}
for entry in entries:
    folder=_verify_entry(root,entry);bindings[entry['path']]=entry['sha256']
    summary=read_json(folder/'summary.json');profiles.extend(summary['profiles'])
    seeds=sorted({r['seed'] for r in summary['rows']})
    for seed in seeds:
        for name in ('affine_flow','node_structured'):
            with np.load(folder/f'{name}__seed{seed}.npz',allow_pickle=False) as raw:
                for k in ('probe_ids','strata','horizons_ns','truth_zero','truth_command'):
                    assert np.array_equal(raw[k],reference[k]),('reference changed',seed,name,k)
                for hi,h in enumerate(raw['horizons_ns']):
                    for stratum in ('cold','warm'):
                        for component,indices in (('q',(0,1)),('v',(2,3))):
                            values=[]
                            for pi,s in enumerate(raw['strata']):
                                if s==stratum:
                                    values.append(math.sqrt(math.fsum((float(raw['model_command'][pi,hi,j])-float(raw['truth_command'][pi,hi,j]))**2 for j in indices)))
                            measured=math.fsum(values)/len(values)
                            reported=next(r['total_error_mean'] for r in summary['rows'] if r['seed']==seed and r['model_id']==name
                                          and r['h_ns']==int(h) and r['stratum']==stratum and r['component']==component)
                            difference=abs(measured-reported);max_difference=max(max_difference,difference)
                            assert difference <= 1e-12*(1+abs(measured))
                            metrics.append(dict(model=name,seed=seed,h_ns=int(h),stratum=stratum,component=component,mean=measured,n=len(values)))
for seed in freeze['seeds']:
    for name in ('affine_flow','node_structured'):
        for h in (5000000,10000000):
            for c in ('q','v'):
                primary[name,seed,h,c]=statistics.mean(r['mean'] for r in metrics if r['seed']==seed and r['model']==name and r['h_ns']==h and r['component']==c)
checks=[]
for row in result['predictions']:
    b=[primary['node_structured',s,row['h_ns'],row['component']] for s in freeze['seeds']]
    m=[primary['affine_flow',s,row['h_ns'],row['component']] for s in freeze['seeds']]
    d=[x-y for x,y in zip(b,m)];mean=statistics.mean(d);sd=statistics.stdev(d)
    assert math.isclose(mean,row['contrast']['mean'],rel_tol=1e-12,abs_tol=1e-15)
    assert math.isclose(sd,row['contrast']['sd'],rel_tol=1e-12,abs_tol=1e-15)
    assert math.isclose(1-statistics.mean(m)/statistics.mean(b),row['point_improvement'],rel_tol=1e-12,abs_tol=1e-15)
    half=t.isf(result['alpha_spent'],len(d)-1)*sd/math.sqrt(len(d))
    assert row['contrast']['lower']<=mean-half+1e-14
    checks.append(dict(h_ns=row['h_ns'],component=row['component'],scalar_mean=mean,scalar_sd=sd,analytic_t_lower=mean-half,n=len(d)))
for p in profiles:
    values=sorted(p['samples_ns']);n=len(values);middle=statistics.median(values)
    position=.95*(n-1);low=int(math.floor(position));fraction=position-low
    p95=values[low]+fraction*(values[min(low+1,n-1)]-values[low])
    assert middle==p['median_ns'] and p95==p['p95_ns']
costs=[];maximum_cost_difference=0.
batch_path=root/'reports/affine_precision_001_primary_batch.json'
if precision and batch_path.exists() and read_json(batch_path)['state']=='complete':
    batch=read_json(batch_path)
    expected={(name,step,s,task) for s in freeze['seeds'] for task in range(freeze['episodes_per_seed'])
              for name,step in (('affine_flow',5000000),('affine_flow',40000000),('node_structured',5000000),('node_structured',40000000))}
    seen=set()
    for entry in batch['completed'].values():
        folder=_verify_entry(root,entry);summary=read_json(folder/'summary.json');bindings[entry['path']]=entry['sha256']
        key=(summary['model_id'],summary['model_partition_ns'][0],summary['seed'],summary['episode_id'])
        assert key not in seen;seen.add(key)
        with np.load(folder/'true_trajectories.npz',allow_pickle=False) as raw:
            times=raw['time_ns'];states=raw['state'];target=raw['target']
            assert len(target)==2 and times[0]==0 and times[-1]==1600000000
            segments=[json.loads(line) for line in (folder/'action_segments.jsonl').read_text().splitlines()]
            assert len(segments)==40
            terms=[]
            for k,segment in enumerate(segments):
                assert segment['start_ns']==40000000*k and segment['duration_ns']==40000000
                when=40000000*(k+1);idx=int(np.searchsorted(times,when));assert times[idx]==when
                qerr=math.fsum((float(states[idx,j])-float(target[j]))**2 for j in (0,1))
                velocity=math.fsum(float(states[idx,j])**2 for j in (2,3))
                effort=math.fsum(float(u)**2 for u in segment['command'])
                terms.append(.04*(qerr+.05*velocity+.001*effort))
            terminal=5*math.fsum((float(states[-1,j])-float(target[j]))**2 for j in (0,1))
            cost=math.fsum(terms)+terminal
        difference=abs(cost-summary['realized_cost']);maximum_cost_difference=max(maximum_cost_difference,difference)
        assert difference<=1e-12*(1+abs(cost))
        costs.append(dict(cell=list(key),scalar_cost=cost,reported_cost=summary['realized_cost'],absolute_difference=difference))
    assert seen==expected,('complete control coverage',len(seen),len(expected))
output=root/('reports/affine_precision_001_independent_arithmetic.json' if precision else 'reports/affine_initial_independent_arithmetic.json')
record=dict(created_utc=utc_now(),protocol_id='PTCO-AFF-0.2',seeds=freeze['seeds'],evidence=bindings,
            prediction_statistic_rows=len(metrics),max_prediction_summary_difference=max_difference,
            contrast_checks=checks,timing_cells=len(profiles),cost_checks=costs,
            max_episode_cost_difference=maximum_cost_difference if costs else None,
            control_status='audited' if costs else 'not_run',
            result_sha256=sha256_file(root/('reports/affine_precision_001_result.json' if precision else 'reports/affine_fresh_result.json')),
            audit_source_sha256=sha256_file(Path(__file__)),passed=True)
atomic_json(output,record)
print({k:v for k,v in record.items() if k in ('prediction_statistic_rows','timing_cells','max_prediction_summary_difference','max_episode_cost_difference','passed')})
