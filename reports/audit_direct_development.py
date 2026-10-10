import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import json,math
import numpy as np
from poc.io import read_json,sha256_file,atomic_json,canonical_json,utc_now
from poc.contracts import verify_run
from poc.planner_qualification import training_recovery_compatibility
root=Path.cwd(); path=root/'reports/control_development_direct.json'; batch=read_json(path)
if batch['state']!='complete' or batch['completed_episodes']!=60: raise ValueError('Complete sixty-cell workload required')
seen=set(); dependencies={}; wheels={}; verified=[]; rows=[]
for entry in batch['episodes']:
 p=root/entry['manifest']
 if sha256_file(p)!=entry['manifest_sha256']: raise ValueError('Changed manifest')
 manifest=read_json(p); verify_run(root,manifest,check_dependencies=False)
 for mapping,target in ((manifest['job_inputs'].get('dependencies',{}),dependencies),(manifest['job_inputs'].get('wheel_hashes',{}),wheels)):
  for key,h in mapping.items():
   if key in target and target[key]!=h: raise ValueError('Conflicting dependency: '+key)
   target[key]=h
 for relative,keys in manifest['job_inputs'].get('configuration_keys',{}).items():
  current=read_json(root/relative)
  if any(current.get(k)!=v for k,v in keys.items()): raise ValueError('Changed configuration')
 summary=read_json(p.parent/'summary.json'); episode=read_json(p.parent/'episode.json')
 if any(canonical_json(summary[k])!=canonical_json(v) for k,v in episode.items()): raise ValueError('Episode summary mismatch')
 if canonical_json(summary)!=canonical_json(entry['summary']): raise ValueError('Receipt summary mismatch')
 h=summary['model_partition_ns'][0]; key=(summary['model_id'],h,summary['episode_id'])
 if key in seen: raise ValueError('Duplicate episode')
 seen.add(key)
 with np.load(p.parent/'true_trajectories.npz',allow_pickle=False) as data:
  states=data['anchors']; goal=data['target']; times=data['time_ns']; true=data['state']
  if len(states)!=40 or not np.array_equal(times,np.arange(0,1600000000+1,250000,dtype=np.int64)): raise ValueError('Clock coverage')
  if not np.array_equal(states,true[160::160]): raise ValueError('Anchors not true trajectory')
  position=.04*math.fsum(float((x-y)**2) for row in states[:,:2] for x,y in zip(row,goal))
  velocity=.04*.05*math.fsum(float(x*x) for row in states[:,2:4] for x in row)
  terminal=5*math.fsum(float((x-y)**2) for x,y in zip(states[-1,:2],goal))
  final=times>=1400000000
  success=bool(np.all(np.linalg.norm(true[final,:2]-goal,axis=1)<.1) and np.all(np.linalg.norm(true[final,2:4],axis=1)<.2))
 segments=[json.loads(s) for s in (p.parent/'action_segments.jsonl').read_text().splitlines()]
 t=0; effort=0
 for s in segments:
  if s['start_ns']!=t or s['duration_ns']!=40000000: raise ValueError('Executed hold path changed')
  t+=s['duration_ns']; effort+=s['duration_ns']/1e9*math.fsum(float(x*x) for x in s['command'])
 if t!=1600000000: raise ValueError('Incomplete executed path')
 cost=math.fsum([position,velocity,.001*effort,terminal])
 if not math.isclose(cost,summary['realized_cost'],rel_tol=1e-12,abs_tol=1e-12) or success!=summary['success']: raise ValueError('Raw cost or success mismatch')
 if summary['seed']!=9000 or summary['actuator']!='direct' or summary['phase']!='exploratory': raise ValueError('Incorrect scope')
 rows.append(summary); verified.append(dict(manifest=entry['manifest'],manifest_sha256=entry['manifest_sha256'],raw_cost=cost,reported_cost=summary['realized_cost']))
expected={(m,h,e) for m in ('mlp_dt','node_generic','power_correct') for h in (40000000,5000000) for e in range(10)}
if seen!=expected: raise ValueError('Incomplete frozen coverage')
recovery,_=training_recovery_compatibility(root)
for p,h in dependencies.items():
 if sha256_file(root/p)!=h and (p not in recovery or recovery[p]['prior_sha256']!=h):
  raise ValueError('Stale dependency: '+p)
for p,h in wheels.items():
 if sha256_file(root/'wheelhouse'/p)!=h: raise ValueError('Changed wheel: '+p)
indexed={(r['model_id'],r['model_partition_ns'][0],r['episode_id']):r for r in rows}
means=[]; contrasts=[]
for model in ('mlp_dt','node_generic','power_correct'):
 for h in (40000000,5000000):
  cells=[indexed[(model,h,e)] for e in range(10)]
  means.append(dict(model_id=model,model_step_ns=h,mean_cost=math.fsum(r['realized_cost'] for r in cells)/10,successes=sum(r['success'] for r in cells),fallbacks=sum(r['fallback_count'] for r in cells)))
for h in (40000000,5000000):
 for model in ('mlp_dt','node_generic'):
  b=[indexed[(model,h,e)] for e in range(10)]; m=[indexed[('power_correct',h,e)] for e in range(10)]
  for x,y in zip(b,m):
   if any(x[k]!=y[k] for k in ('initial_state','target','seed','actuator_interval_ns','physics_step_ns')): raise ValueError('Task pairing mismatch')
  bm=math.fsum(r['realized_cost'] for r in b)/10; mm=math.fsum(r['realized_cost'] for r in m)/10
  contrast=dict(baseline_model=model,model_step_ns=h,baseline_mean_cost=bm,method_mean_cost=mm,relative_reduction=1-mm/bm,episodes_with_lower_head_cost=sum(y['realized_cost']<x['realized_cost'] for x,y in zip(b,m)),episode_cost_differences=[x['realized_cost']-y['realized_cost'] for x,y in zip(b,m)])
  saved=next(c for c in batch['comparisons'] if c['baseline_model']==model and c['model_step_ns']==h)
  if not math.isclose(contrast['relative_reduction'],saved['relative_reduction'],rel_tol=1e-12,abs_tol=1e-12): raise ValueError('Aggregate mismatch')
  contrasts.append(contrast)
atomic_json(root/'reports/control_development_direct_audit.json',dict(schema_version=1,recorded_utc=utc_now(),batch_sha256=sha256_file(path),script_sha256=sha256_file(Path(__file__)),complete_episode_count=60,semantic_dependency_count=len(dependencies),wheel_count=len(wheels),statistical_status='descriptive_only',n_training_seeds=1,seed_ids=[9000],episodes_per_model_partition=10,verified_episodes=verified,means=means,contrasts=contrasts,scope='Fixed direct development workload; raw costs independently summed from true anchors and executed commands; no population confidence claim'))
print(json.dumps(dict(means=means,contrasts=contrasts),indent=2))
