"""Independently sum the full intervention's raw costs and check task pairing."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import json
import math
import numpy as np
from poc.io import read_json, sha256_file, atomic_json, canonical_json, utc_now
from poc.contracts import verify_run
from poc.planner_qualification import training_recovery_compatibility

root = Path.cwd()
path = root / 'reports/control_attribution_development.json'
batch = read_json(path)
registry_path = root / 'config/control_attribution_development_jobs.json'
registry = read_json(registry_path)
if (batch['state'] != 'complete' or batch['completed_episodes'] != 160
        or batch['registry_sha256'] != sha256_file(registry_path)):
    raise ValueError('Complete registered 160-cell workload required')
requests = {j['cell_id']: read_json(root / j['path']) for j in registry['jobs']}
seen, dependencies, wheels, verified, indexed = set(), {}, {}, [], {}
for entry in batch['episodes']:
    manifest_path = root / entry['manifest']
    if sha256_file(manifest_path) != entry['manifest_sha256']:
        raise ValueError('Changed intervention manifest')
    manifest = read_json(manifest_path)
    verify_run(root, manifest, check_dependencies=False)
    for key, h in manifest['job_inputs'].get('dependencies', {}).items():
        dependencies.setdefault(key, set()).add(h)
    for key, h in manifest['job_inputs'].get('wheel_hashes', {}).items():
        if key in wheels and wheels[key] != h:
            raise ValueError('Conflicting wheel: ' + key)
        wheels[key] = h
    for relative, keys in manifest['job_inputs'].get('configuration_keys', {}).items():
        current = read_json(root / relative)
        if any(current.get(k) != v for k, v in keys.items()):
            raise ValueError('Changed configuration')
    folder = manifest_path.parent
    summary, episode = read_json(folder / 'summary.json'), read_json(folder / 'episode.json')
    if read_json(folder / 'request.json') != requests[entry['cell_id']]:
        raise ValueError('Unregistered request')
    if (any(canonical_json(summary[k]) != canonical_json(v) for k, v in episode.items())
            or canonical_json(summary) != canonical_json(entry['summary'])):
        raise ValueError('Episode summary or receipt mismatch')
    key = (summary['actuator'], summary['episode_id'], summary['model_id'],
           summary['power_mode'], summary['model_partition_ns'][0])
    if key in seen or summary['seed'] != 9000 or summary['phase'] != 'exploratory':
        raise ValueError('Duplicate cell or incorrect development scope')
    seen.add(key)
    with np.load(folder / 'true_trajectories.npz', allow_pickle=False) as data:
        anchors, target, times, truth = data['anchors'], data['target'], data['time_ns'], data['state']
        if (anchors.shape[0] != 40 or not np.array_equal(times,
                np.arange(0, 1_600_000_000 + 1, 250_000, dtype=np.int64))
                or not np.array_equal(anchors, truth[160::160])):
            raise ValueError('Physical clock or truth-anchor coverage mismatch')
        if (not np.array_equal(truth[0], np.asarray(summary['initial_state']))
                or not np.array_equal(target, np.asarray(summary['target']))):
            raise ValueError('Raw initial state or target differs from the episode/summary task')
        tracking = .04 * math.fsum(float((x-y)**2) for r in anchors[:, :2] for x,y in zip(r,target))
        velocity = .002 * math.fsum(float(x*x) for r in anchors[:, 2:4] for x in r)
        terminal = 5 * math.fsum(float((x-y)**2) for x,y in zip(anchors[-1, :2],target))
        final = times >= 1_400_000_000
        success = bool(np.all(np.linalg.norm(truth[final,:2]-target, axis=1) < .1)
                       and np.all(np.linalg.norm(truth[final,2:4], axis=1) < .2))
    clock, effort = 0, 0.
    for line in (folder / 'action_segments.jsonl').read_text().splitlines():
        segment = json.loads(line)
        if segment['start_ns'] != clock or segment['duration_ns'] != 40_000_000:
            raise ValueError('Executed hold clock mismatch')
        clock += segment['duration_ns']
        effort += .04 * math.fsum(float(x*x) for x in segment['command'])
    cost = math.fsum((tracking, velocity, terminal, .001 * effort))
    if (clock != 1_600_000_000 or success != summary['success']
            or not math.isclose(cost, summary['realized_cost'], rel_tol=1e-12, abs_tol=1e-12)):
        raise ValueError('Raw cost or success differs from reported result')
    indexed[key] = dict(summary, raw_cost=cost)
    verified.append(dict(manifest=entry['manifest'], manifest_sha256=entry['manifest_sha256'], raw_cost=cost))
expected = {(a,e,m,mode,h) for a in ('direct','lagged') for e in range(10)
            for m,mode in [('power_correct','low'),('power_correct','direct'),
                           ('power_correct','lagged'),('node_structured',None)]
            for h in (40_000_000,5_000_000)}
if seen != expected:
    raise ValueError('Incomplete original intervention coverage')
recovery, _ = training_recovery_compatibility(root)
for name, hashes in dependencies.items():
    for h in hashes:
        if sha256_file(root / name) != h and (name not in recovery or recovery[name]['prior_sha256'] != h):
            raise ValueError('Stale intervention dependency: ' + name)
for name, h in wheels.items():
    if sha256_file(root / 'wheelhouse' / name) != h:
        raise ValueError('Changed wheel: ' + name)
means, contrasts = [], []
for actuator in ('direct','lagged'):
    for model,mode in [('power_correct','low'),('power_correct','direct'),
                       ('power_correct','lagged'),('node_structured',None)]:
        for h in (40_000_000,5_000_000):
            cells = [indexed[(actuator,e,model,mode,h)] for e in range(10)]
            means.append(dict(actuator=actuator,model_id=model,power_mode=mode,model_step_ns=h,
                mean_cost=math.fsum(r['raw_cost'] for r in cells)/10,
                successes=sum(r['success'] for r in cells),fallbacks=sum(r['fallback_count'] for r in cells)))
    task_a, task_b = [], []
    for e in range(10):
        paired = [indexed[(actuator,e,m,mode,h)] for h in (40_000_000,5_000_000)
                  for m,mode in [('power_correct','low'),('power_correct','direct'),
                                 ('power_correct','lagged'),('node_structured',None)]]
        if any(any(r[k] != paired[0][k] for k in ('initial_state','target','seed',
                    'actuator_interval_ns','physics_step_ns')) for r in paired):
            raise ValueError('Task pairing mismatch across all eight intervention cells')
        cells = [indexed[(actuator,e,m,mode,h)] for h in (40_000_000,5_000_000)
                 for m,mode in [('power_correct','low'),('power_correct',actuator),('node_structured',None)]]
        low,correct,structured,low_fine,correct_fine,structured_fine = [r['raw_cost'] for r in cells]
        a = (low_fine-low)-(correct_fine-correct)
        b = (low_fine-low)-(structured_fine-structured)
        task_a.append(a); task_b.append(b)
    a,b = math.fsum(task_a)/10,math.fsum(task_b)/10
    saved = next(c for c in batch['contrasts'] if c['actuator'] == actuator)
    if not (math.isclose(a,saved['A'],rel_tol=1e-12,abs_tol=1e-12)
            and math.isclose(b,saved['B'],rel_tol=1e-12,abs_tol=1e-12)
            and saved['n_training_seeds']==1 and saved['fraction_explained'] is None):
        raise ValueError('Attribution aggregation mismatch')
    contrasts.append(dict(actuator=actuator,A=a,B=b,A_minus_half_B=a-.5*b,
        task_A=task_a,task_B=task_b,correct_improves_low_tasks=sum(x>0 for x in task_a)))
atomic_json(root / 'reports/control_attribution_development_audit.json',dict(schema_version=1,
    recorded_utc=utc_now(),batch_sha256=sha256_file(path),script_sha256=sha256_file(Path(__file__)),
    complete_episode_count=160,semantic_dependency_count=len(dependencies),wheel_count=len(wheels),
    compatible_source_variants={p:sorted(h) for p,h in dependencies.items() if len(h)>1},
    statistical_status='descriptive_only',n_training_seeds=1,tasks_per_plant=10,
    verified_episodes=verified,means=means,contrasts=contrasts,
    scope='Independent raw cost/success and task pairing audit; nominal raw identities are separately required by the runner'))
print(json.dumps(dict(means=means,contrasts=contrasts),indent=2))
