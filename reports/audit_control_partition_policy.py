"""Compare saved hardware paths across model partitions on the fixed direct pilot."""
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from poc.io import atomic_json,read_json,sha256_file


def audit(root=ROOT):
    root=Path(root).resolve();path=root/'reports/control_development_direct.json';batch=read_json(path)
    if batch['state']!='complete' or len(batch['episodes'])!=60:
        raise ValueError('Complete original direct development batch required')
    index={};inputs={}
    for entry in batch['episodes']:
        row=entry['summary'];h=row['model_partition_ns'][0]
        coordinate=(row['model_id'],h,row['episode_id'])
        if (coordinate in index or row['seed']!=9000 or row['phase']!='exploratory'
                or row['actuator']!='direct' or row['condition']!='clean'
                or row['model_partition_ns']!=[h]*(40_000_000//h)):
            raise ValueError('Changed pilot identity or duplicate clock comparison')
        index[coordinate]=entry
    expected={(m,h,e) for m in ('mlp_dt','node_generic','power_correct')
              for h in (40_000_000,5_000_000) for e in range(10)}
    if set(index)!=expected:raise ValueError('Missing original paired pilot cells')
    models=[]
    for model in ('mlp_dt','node_generic','power_correct'):
        count=0;episodes=0;max_command=0.;max_q=0.;max_v=0.
        for episode in range(10):
            commands=[];states=[]
            for h in (40_000_000,5_000_000):
                entry=index[(model,h,episode)];folder=(root/entry['manifest']).parent
                if sha256_file(folder/'manifest.json')!=entry['manifest_sha256']:
                    raise ValueError('Changed saved episode manifest')
                record=read_json(folder/'manifest.json')
                for filename in ('action_segments.jsonl','true_trajectories.npz'):
                    full=folder/filename;relative=str(full.relative_to(root))
                    saved=next(r['sha256'] for r in record['produced_files'] if r['path']==relative)
                    if sha256_file(full)!=saved:raise ValueError('Changed raw hardware path or plant trajectory')
                    inputs[relative]=saved
                segments=[json.loads(line) for line in (folder/'action_segments.jsonl').read_text().splitlines()]
                if (len(segments)!=40 or any(s['duration_ns']!=40_000_000 or s['start_ns']!=i*40_000_000
                                            for i,s in enumerate(segments))):
                    raise ValueError('Changed physical actuator schedule')
                commands.append(np.array([r['command'] for r in segments]))
                with np.load(folder/'true_trajectories.npz',allow_pickle=False) as data:
                    states.append(data['state'].copy())
            matching=np.all(commands[0]==commands[1],axis=1)
            count+=int(matching.sum());episodes+=int(np.array_equal(states[0],states[1]))
            max_command=max(max_command,float(np.max(np.abs(commands[0]-commands[1]))))
            max_q=max(max_q,float(np.max(np.abs(states[0][:,:2]-states[1][:,:2]))))
            max_v=max(max_v,float(np.max(np.abs(states[0][:,2:4]-states[1][:,2:4]))))
        models.append(dict(model_id=model,n_episode_pairs=10,n_real_decisions=400,
            identical_held_commands=count,episodes_with_identical_raw_plant_states=episodes,
            max_absolute_command_difference_nm=max_command,max_absolute_q_difference_rad=max_q,
            max_absolute_v_difference_rad_s=max_v))
    result=dict(scope='Single development training seed9000, direct A/shared; hardware path comparison across40/5ms model clocks',
        source=str(path.relative_to(root)),source_sha256=sha256_file(path),models=models,input_hashes=inputs,
        evaluator_sha256=sha256_file(root/'reports/audit_control_partition_policy.py'),scientific_decision='not_evaluated',
        limitation='Finite fixed CEM streams/tasks; identical decisions do not certify identical continuous model flows')
    atomic_json(root/'reports/control_partition_policy_identity.json',result)
    print(json.dumps(models,allow_nan=False));return result


if __name__=='__main__':audit()
