"""One-time report export; reads complete frozen observations, runs no experiments."""
from pathlib import Path
import csv, hashlib, json, math, statistics

root=Path.cwd()
def read(path):return json.loads((root/path).read_text())
def digest(path):return hashlib.sha256((root/path).read_bytes()).hexdigest()
def write(path,value):
 p=root/path
 temporary=p.with_suffix(p.suffix+'.tmp')
 temporary.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
 temporary.replace(p)

result=read('reports/affine_precision_001_result.json')
reproduction=read('reports/affine_precision_001_reproduction.json')
audit=read('reports/affine_precision_001_independent_arithmetic.json')
assert result['look_index']==3 and result['control']['episode_count']==440
assert reproduction['pass'] is True and audit['passed'] is True
assert len(audit['cost_checks'])==440 and audit['result_sha256']==digest('reports/affine_precision_001_result.json')
assert result['control_convergence']['converged'] is True
assert len(result['control_episodes'])==440 and len(result['control_comparisons'])==3

configs=[]
for name,step in [('affine_flow',5000000),('node_structured',5000000),('affine_flow',40000000),('node_structured',40000000)]:
 rows=[r for r in result['control_episodes'] if r['model_id']==name and r['model_partition_ns'][0]==step]
 assert len(rows)==110 and all(r['complete'] for r in rows)
 configs.append(dict(model_id=name,model_step_ns=step,episodes=len(rows),
   mean_cost=statistics.fmean(r['realized_cost'] for r in rows),
   mean_q_rmse_rad=statistics.fmean(r['q_rmse_rad'] for r in rows),
   mean_terminal_q_error_rad=statistics.fmean(r['terminal_q_error_rad'] for r in rows),
   mean_velocity_rms_rad_s=statistics.fmean(r['velocity_rms_rad_s'] for r in rows),
   mean_command_effort_integral=statistics.fmean(r['command_effort_integral'] for r in rows),
   successes=sum(r['success'] for r in rows),fallbacks=sum(r['fallback_count'] for r in rows),
   mean_episode_median_planning_s=statistics.fmean(r['planner_latency_median_seconds'] for r in rows),
   mean_episode_p95_planning_s=statistics.fmean(r['planner_latency_p95_seconds'] for r in rows)))

resources=[]; bindings={}
receipts=[('corpus_'+role,f'reports/affine_fresh_data_{role}.json') for role in ('train','validation','confirmation')]
receipts += [('reference','reports/affine_fresh_references.json'),('measurement_initial','reports/affine_fresh_measurement.json'),('measurement_additional','reports/affine_precision_001_measurement.json')]
for seed in result['seeds']:
 for model in ('affine_flow','node_structured'):
  receipts.append(('original_fit',f'reports/affine_fresh_fits/{model}__seed{seed}.json'))
for seed in (result['seeds'][0],result['seeds'][-1]):
 for model in ('affine_flow','node_structured'):
  receipts.append(('retraining',f'reports/affine_fresh_replicas/{model}__seed{seed}.json'))
for category,path in receipts:
 entry=read(path);bindings[entry['path']]=entry['sha256']
 resources.append((category,entry))
for category in ('primary','replay','finer_physics'):
 batch_path=f'reports/affine_precision_001_{category}_batch.json';batch=read(batch_path)
 assert batch['state']=='complete' and len(batch['completed'])==(440 if category=='primary' else 44)
 for entry in batch['completed'].values():
  bindings[entry['path']]=entry['sha256'];resources.append((category,entry))

# Retain the compute consumed by the completed retraining whose evidence export
# failed when the local volume filled; its valid saved training state was resumed.
failed_path='evidence/M30/20261008T214822-26fe25aa/manifest.json'
failed=read(failed_path)
assert failed['completion_status']=='failed'
assert read(str(Path(failed_path).parent/'summary.json'))['failure']=='[Errno 28] No space left on device'
entry=dict(path=failed_path,sha256=digest(failed_path))
bindings[failed_path]=entry['sha256'];resources.append(('retraining_export_failed',entry))

phases={}
seen=set()
for category,entry in resources:
 if entry['path'] in seen:raise AssertionError('Repeated resource entry')
 seen.add(entry['path']);assert digest(entry['path'])==entry['sha256']
 folder=Path(entry['path']).parent
 assert read(str(folder/'COMPLETED'))['manifest_sha256']==entry['sha256']
 resource=read(str(folder/'resources.json'))
 phase=phases.setdefault(category,dict(jobs=0,cpu_seconds=0.,wall_seconds=0.,artifact_disk_bytes=0,peak_rss_bytes=0))
 phase['jobs']+=1
 for field in ('cpu_seconds','wall_seconds','artifact_disk_bytes'):
  phase[field]+=resource['disk_bytes' if field=='artifact_disk_bytes' else field]
 phase['peak_rss_bytes']=max(phase['peak_rss_bytes'],resource['peak_rss_bytes'])

output=dict(protocol_id='PTCO-AFF-0.2',control_configurations=configs,
 control_comparisons=result['control_comparisons'],resources=phases,evidence=bindings,
 reproduction_counts=dict(predictions=len(reproduction['predictions']),training=len(reproduction['training']),controls=len(reproduction['controls'])),
 reproduction_passed=True,independent_cost_checks=len(audit['cost_checks']),
 maximum_independent_cost_difference=audit['max_episode_cost_difference'],
 limitation='Conditional on one fresh train/probe corpus, eleven paired fits and110 paired tasks; no new statistical look. Resource records cover the listed fresh producers and the recorded disk-failed retrain export, not every historical project attempt. The first training replica continued from its complete state after that export failure, without extra updates. Artifact bytes are producer-reported logical files, not APFS allocation.')
write('reports/affine_practical_metrics.json',output)
with (root/'reports/affine_practical_metrics.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(configs[0]));w.writeheader();w.writerows(configs)
print('Exported four configurations, three prespecified comparisons, complete producer resource records.')
