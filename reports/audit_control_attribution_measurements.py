"""Verify complete first-task intervention evidence and all nominal power modes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import json
import math
import numpy as np
from poc.io import atomic_json, read_json, sha256_file, utc_now
from poc.contracts import verify_run
from poc.control_attribution import _matching
from poc.planner_qualification import training_recovery_compatibility

root = Path.cwd()
source = root / 'reports/control_attribution_measurements.json'
batch = read_json(source)
if batch['state'] != 'complete' or batch['completed_episodes'] != 16:
    raise ValueError('Complete first-task intervention workload required')
folders, dependencies, wheels, checked = {}, {}, {}, []
for entry in batch['episodes']:
    path = root / entry['manifest']
    if sha256_file(path) != entry['manifest_sha256']:
        raise ValueError('Changed intervention manifest')
    manifest = read_json(path)
    verify_run(root, manifest, check_dependencies=False)
    for name, target in (('dependencies', dependencies), ('wheel_hashes', wheels)):
        for key, value in manifest['job_inputs'].get(name, {}).items():
            if key in target and target[key] != value:
                raise ValueError('Conflicting semantic dependency')
            target[key] = value
    row = read_json(path.parent / 'summary.json')
    if row != entry['summary']:
        raise ValueError('Changed receipt summary')
    with np.load(path.parent / 'true_trajectories.npz', allow_pickle=False) as data:
        states, target = data['anchors'], data['target']
        if states.shape[0] != 40:
            raise ValueError('Incomplete physical anchors')
        tracking = .04 * math.fsum(float((x-y)**2) for r in states[:, :2] for x,y in zip(r,target))
        velocity = .002 * math.fsum(float(x*x) for r in states[:, 2:4] for x in r)
        terminal = 5 * math.fsum(float((x-y)**2) for x,y in zip(states[-1,:2],target))
    segments = [json.loads(line) for line in (path.parent / 'action_segments.jsonl').read_text().splitlines()]
    clock, effort = 0, 0.
    for segment in segments:
        if segment['start_ns'] != clock or segment['duration_ns'] != 40_000_000:
            raise ValueError('Incorrect executed command clock')
        clock += segment['duration_ns']
        effort += .04 * math.fsum(float(x*x) for x in segment['command'])
    cost = math.fsum((tracking, velocity, terminal, .001*effort))
    if clock != 1_600_000_000 or not math.isclose(cost,row['realized_cost'],abs_tol=1e-12,rel_tol=1e-12):
        raise ValueError('Raw realized cost mismatch')
    key = (row['actuator'],row['model_id'],row['power_mode'],row['model_partition_ns'][0])
    if key in folders:
        raise ValueError('Duplicate intervention cell')
    folders[key] = path.parent
    checked.append(dict(manifest=entry['manifest'],manifest_sha256=entry['manifest_sha256'],raw_cost=cost))
recovery, _ = training_recovery_compatibility(root)
for name, expected in dependencies.items():
    if sha256_file(root/name) != expected and (name not in recovery or recovery[name]['prior_sha256'] != expected):
        raise ValueError('Changed semantic input: '+name)
for name, expected in wheels.items():
    if sha256_file(root/'wheelhouse'/name) != expected:
        raise ValueError('Changed wheel')
identities = []
for actuator in ('direct','lagged'):
    low = folders[(actuator,'power_correct','low',40_000_000)]
    for mode in ('direct','lagged'):
        other = folders[(actuator,'power_correct',mode,40_000_000)]
        a,b = read_json(low/'episode.json'),read_json(other/'episode.json')
        if not a['checkpoint_hash'] or a['checkpoint_hash'] != b['checkpoint_hash']:
            raise ValueError('Intervention must share checkpoint bytes')
        for name in ('candidates.npz','true_trajectories.npz','observations.npz'):
            with np.load(low/name,allow_pickle=False) as x,np.load(other/name,allow_pickle=False) as y:
                if set(x.files)!=set(y.files):
                    raise ValueError('Nominal array coverage differs')
                for key in x.files:
                    if x[key].dtype!=y[key].dtype or x[key].shape!=y[key].shape or not (
                        np.allclose(x[key],y[key],atol=1e-12,rtol=1e-12,equal_nan=False)
                        if x[key].dtype.kind in 'fc' else np.array_equal(x[key],y[key])):
                        raise ValueError('Nominal raw arrays differ')
        for name in ('decisions.jsonl','action_segments.jsonl','external_torque_segments.jsonl'):
            x=[json.loads(line) for line in (low/name).read_text().splitlines()]
            y=[json.loads(line) for line in (other/name).read_text().splitlines()]
            if name=='decisions.jsonl':
                x=[{k:v for k,v in r.items() if k!='planner_wall_seconds'} for r in x]
                y=[{k:v for k,v in r.items() if k!='planner_wall_seconds'} for r in y]
            if not _matching(x,y):
                raise ValueError('Nominal decisions or executed actions differ')
        identities.append(dict(actuator=actuator,modes=['low',mode],matched=True,checkpoint_sha256=a['checkpoint_hash']))
atomic_json(root/'reports/control_attribution_measurements_audit.json',dict(schema_version=1,
    recorded_utc=utc_now(),measurement_sha256=sha256_file(source),script_sha256=sha256_file(Path(__file__)),
    complete_episode_count=16,verified_episodes=checked,nominal_identities=identities,
    scope='One fixed task per plant; all three nominal power modes; raw-cost and hash audit',
    statistical_status='descriptive_only'))
print('Verified sixteen raw costs and all four low/direct/lagged nominal comparisons')
