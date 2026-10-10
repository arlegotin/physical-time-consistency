"""Supplementary paired reference context for the completed direct development."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import math
from poc.io import read_json, sha256_file, atomic_json, utc_now
from poc.planner_qualification import verify_frozen

root = Path.cwd()
verify_frozen(root)
development = root / 'reports/control_development_direct.json'
audit = read_json(root / 'reports/control_development_direct_audit.json')
if audit['batch_sha256'] != sha256_file(development) or audit['complete_episode_count'] != 60:
    raise ValueError('Require the independently audited complete development comparison')
refs, pointers = {}, []
for entry in read_json(root / 'reports/robot_calibration_batch.json')['episode_manifests']:
    path = root / entry['path']
    if sha256_file(path) != entry['sha256']:
        raise ValueError('Changed reference receipt')
    row = read_json(path.parent / 'summary.json')
    if row['actuator'] == 'direct' and row['episode_id'] < 10:
        key = row['model_id'], row['episode_id']
        if key in refs:
            raise ValueError('Duplicate reference cell')
        refs[key] = row
        pointers.append(entry)
models = ('zero_command', 'physics_fit', 'sim_oracle')
if set(refs) != {(m,e) for m in models for e in range(10)}:
    raise ValueError('Incomplete matching reference cells')
head = {(e['summary']['model_partition_ns'][0],e['summary']['episode_id']): e['summary']
        for e in read_json(development)['episodes'] if e['summary']['model_id'] == 'power_correct'}
if set(head) != {(h,e) for h in (40_000_000,5_000_000) for e in range(10)}:
    raise ValueError('Incomplete corrected-model development cells')
for h in (40_000_000,5_000_000):
    for e in range(10):
        for m in models:
            a,b = head[(h,e)],refs[(m,e)]
            for key in ('initial_state','target','actuator_interval_ns','physics_step_ns','seed','seed_namespaces'):
                if a[key] != b[key]:
                    raise ValueError('Unpaired reference task/stream: ' + key)
means = {m:math.fsum(refs[(m,e)]['realized_cost'] for e in range(10))/10 for m in models}
comparisons = []
for h in (40_000_000,5_000_000):
    cost = math.fsum(head[(h,e)]['realized_cost'] for e in range(10))/10
    comparisons.append(dict(model_step_ns=h,head_mean_cost=cost,reference_means=means,
        head_reduction_vs_zero=1-cost/means['zero_command'],head_cost_ratio_vs_physics=cost/means['physics_fit']))
atomic_json(root / 'reports/control_development_direct_references.json',dict(schema_version=1,
    recorded_utc=utc_now(),script_sha256=sha256_file(Path(__file__)),development_sha256=sha256_file(development),
    development_audit_sha256=sha256_file(root / 'reports/control_development_direct_audit.json'),
    reference_receipts=pointers,comparisons=comparisons,n_training_seeds=1,task_count=10,
    reference_successes={m:sum(refs[(m,e)]['success'] for e in range(10)) for m in models},
    statistical_status='descriptive_only',selection='Supplementary reference context after development outcomes; all ten matching task IDs retained',
    scope='Exact same task, initial state, target, hardware clocks and CEM seed streams; physics reference integrated independently of requested model partition; no population inference'))
print(comparisons)
