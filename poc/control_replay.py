"""Fixed first-task fresh-process replay and physical-unit comparisons."""
from __future__ import annotations
import math
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import numpy as np

from .control_confirmation import ACTUATORS,MODELS,STEPS
from .io import atomic_json,job_lock,read_json,safe_path,sha256_file,utc_now


def replay_requests(root,frozen):
    """Preserve every primary semantic input, changing only the execution phase."""
    root=Path(root).resolve()
    jobs=frozen.get('jobs',[])
    if (frozen.get('state')!='complete' or frozen.get('expected_episodes')!=1200
            or len(jobs)!=1200 or len({j['cell_id'] for j in jobs})!=1200):
        raise ValueError('Complete frozen primary control registry required for replay')
    from .control_jobs import validate_request
    expected={(a,m,h,s,e) for a in ACTUATORS for m in MODELS for h in STEPS
              for s in range(5) for e in range(20)}
    seen=set();rows=[]
    for job in jobs:
        path=safe_path(root,job['path'])
        if sha256_file(path)!=job['sha256']:
            raise ValueError('Changed primary request before replay')
        request=validate_request(read_json(path));h=request['model_partition_ns'][0]
        coordinate=(request['actuator'],request['model_id'],h,request['seed'],request['episode_id'])
        if (coordinate not in expected or coordinate in seen or request['phase']!='confirmatory'
                or request['condition']!='clean' or request['physics_step_ns']!=250_000
                or request['actuator_interval_ns']!=40_000_000 or request['power_mode'] is not None
                or request['model_partition_ns']!=[h]*(40_000_000//h)):
            raise ValueError('Replay request differs from registered primary identity')
        seen.add(coordinate)
        if request['episode_id']==0:
            rows.append(dict(cell_id='fresh_replay__'+job['cell_id'],parent_cell_id=job['cell_id'],
                parent_request_sha256=job['sha256'],request={**request,'phase':'replication'}))
    if seen!=expected or len(rows)!=60:
        raise ValueError('Complete primary workload coverage required for replay')
    return rows


def compare_episode(original,replay):
    """Known tolerances qualify replay, never fresh training or a new effect size."""
    first,second=original['summary'],replay['summary']
    keys=('actuator','model_id','seed','episode_id','condition','model_partition_ns',
          'actuator_interval_ns','physics_step_ns','power_mode','checkpoint_hash',
          'initial_state','target','seed_namespaces','success','fallback_count')
    if (first.get('phase')!='confirmatory' or second.get('phase')!='replication'
            or first.get('complete') is not True or second.get('complete') is not True
            or any(first.get(k)!=second.get(k) for k in keys)
            or first['actuator'] not in ACTUATORS or first['model_id'] not in MODELS
            or first['seed'] not in range(5) or first['episode_id']!=0
            or first['condition']!='clean' or first['physics_step_ns']!=250_000
            or first['actuator_interval_ns']!=40_000_000 or first['power_mode'] is not None):
        raise ValueError('Replay differs from the fixed primary task/checkpoint/condition identity')
    step=first['model_partition_ns'][0]
    if step not in STEPS or first['model_partition_ns']!=[step]*(40_000_000//step):
        raise ValueError('Replay model clock differs from the registered primary partition')
    dimension=4 if first['actuator']=='direct' else 6
    times=np.arange(0,1_600_000_001,250_000,dtype=np.int64)
    states=[];commands=[]
    for row in (original,replay):
        state=np.asarray(row['state']);command=np.asarray(row['commands'])
        if (not np.array_equal(row['time_ns'],times) or state.shape!=(len(times),dimension)
                or command.shape!=(40,2) or not np.isfinite(state).all() or not np.isfinite(command).all()
                or np.any(np.abs(command)>2) or not np.array_equal(state[0],row['summary']['initial_state'])
                or len(row['decision_statuses'])!=40
                or any(s not in ('ok','all_candidates_failed') for s in row['decision_statuses'])
                or not math.isfinite(row['summary']['realized_cost']) or row['summary']['realized_cost']<0
                or row['decision_statuses'].count('all_candidates_failed')!=row['summary']['fallback_count']):
            raise ValueError('Incomplete/nonfinite replay clocks, trajectory, commands or failure accounting')
        states.append(state);commands.append(command)
    if original['decision_statuses']!=replay['decision_statuses']:
        raise ValueError('Replay changed candidate-failure/fallback classification')
    delta=np.abs(states[0]-states[1])
    q=float(delta[:,:2].max());v=float(delta[:,2:4].max())
    command=float(np.abs(commands[0]-commands[1]).max())
    absolute=abs(first['realized_cost']-second['realized_cost'])
    relative=absolute/first['realized_cost'] if first['realized_cost']>0 else (0. if absolute==0 else None)
    passed=q<=1e-6 and v<=1e-5 and command<=1e-6 and relative is not None and relative<=1e-4
    return dict(status='pass' if passed else 'outside_tolerance',diagnosis_required=not passed,
        actuator=first['actuator'],model_id=first['model_id'],seed=first['seed'],episode_id=0,
        model_step_ns=step,max_q_difference_rad=q,max_v_difference_rad_s=v,
        max_command_difference_nm=command,absolute_cost_difference=absolute,relative_cost_difference=relative,
        max_activation_difference_nm=float(delta[:,4:].max()) if dimension==6 else None,
        fresh_training=False,scope='Fresh-process replay of selected weights; not fresh-training replication')


def _read_episode(root,path,phase,expected_request):
    from .contracts import verify_run
    from .control_utility import audit_raw_episode
    path=Path(path).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Replay manifest escapes project')
    manifest=read_json(path);verify_run(root,manifest)
    summary=read_json(path.parent/'summary.json')
    request=read_json(path.parent/'request.json')
    if (manifest['milestone_id']!='M70' or manifest['phase']!=phase
            or summary['phase']!=phase or request!=expected_request):
        raise ValueError('Completed replay or parent differs from its exact registered request')
    audit_raw_episode(path.parent,summary)
    with np.load(path.parent/'true_trajectories.npz',allow_pickle=False) as data:
        times,state=data['time_ns'].copy(),data['state'].copy()
    segments=[json.loads(line) for line in (path.parent/'action_segments.jsonl').read_text().splitlines()]
    decisions=[json.loads(line) for line in (path.parent/'decisions.jsonl').read_text().splitlines()]
    return dict(summary=summary,time_ns=times,state=state,
        commands=np.asarray([s['command'] for s in segments]),decision_statuses=[d['status'] for d in decisions],
        resources=read_json(path.parent/'resources.json'))


def _verify_registry(root,registry):
    from .control_confirmation import verify_freeze
    design=read_json(root/'config/control_confirmation_design.json')
    frozen=read_json(root/'config/control_confirmation_cells.json');verify_freeze(root,design,frozen)
    for name,digest in registry['input_hashes'].items():
        if sha256_file(safe_path(root,name))!=digest:
            raise ValueError('Changed replay dependency: '+name)
    expected=replay_requests(root,frozen)
    if registry.get('state')!='complete' or len(registry.get('jobs',[]))!=60:
        raise ValueError('Complete fixed first-task replay registry required')
    for row,job in zip(expected,registry['jobs']):
        path=safe_path(root,job['path'])
        if (any(job.get(k)!=row[k] for k in ('cell_id','parent_cell_id','parent_request_sha256'))
                or sha256_file(path)!=job['sha256'] or read_json(path)!=row['request']):
            raise ValueError('Changed replay request or parent identity')
        parent=safe_path(root,job['parent_manifest'])
        if sha256_file(parent)!=job['parent_manifest_sha256']:
            raise ValueError('Changed primary replay parent manifest')
        _read_episode(root,parent,'confirmatory',{**row['request'],'phase':'confirmatory'})
    return registry


def prepare(root):
    root=Path(root).resolve();target=root/'config/control_replay_jobs.json'
    if target.exists():return _verify_registry(root,read_json(target))
    batch=read_json(root/'reports/control_confirmation_batch.json')
    if (batch.get('state')!='complete' or batch.get('expected_episodes')!=1200
            or batch.get('completed_episodes')!=1200 or len(batch.get('episodes',[]))!=1200):
        raise ValueError('Complete primary control batch required before fresh replay')
    from .control_confirmation import verify_freeze
    design=read_json(root/'config/control_confirmation_design.json')
    frozen=read_json(root/'config/control_confirmation_cells.json');verify_freeze(root,design,frozen)
    if batch['registry_sha256']!=sha256_file(root/'config/control_confirmation_cells.json'):
        raise ValueError('Primary replay input differs from its freeze')
    parents={e['cell_id']:e for e in batch['episodes']}
    if len(parents)!=1200 or set(parents)!={j['cell_id'] for j in frozen['jobs']}:
        raise ValueError('Complete unique primary episode coverage required before replay')
    from .control_jobs import SOURCES
    names=set(SOURCES)|{'poc/control_replay.py','poc/control_utility.py',
        'config/control_confirmation_cells.json','config/control_confirmation_design.json',
        'reports/control_confirmation_batch.json'}
    jobs=[]
    for row in replay_requests(root,frozen):
        parent=parents[row['parent_cell_id']];path=safe_path(root,parent['manifest'])
        if sha256_file(path)!=parent['manifest_sha256']:
            raise ValueError('Changed primary episode before replay freeze')
        _read_episode(root,path,'confirmatory',{**row['request'],'phase':'confirmatory'})
        request_path=root/'config/control_jobs'/f"{row['cell_id'].replace('/','__')}.json"
        if request_path.exists() and read_json(request_path)!=row['request']:
            raise ValueError('Conflicting existing replay request')
        atomic_json(request_path,row['request'])
        jobs.append({k:v for k,v in row.items() if k!='request'}|dict(path=str(request_path.relative_to(root)),
            sha256=sha256_file(request_path),parent_manifest=parent['manifest'],
            parent_manifest_sha256=parent['manifest_sha256']))
        names.add(parent['manifest']);names.add(str(request_path.relative_to(root)))
        names.update(read_json(path)['job_inputs']['dependencies'])
        names.update(item['path'] for item in read_json(path)['produced_files'])
    chosen=[parents[j['parent_cell_id']] for j in jobs]
    registry=dict(state='complete',jobs=jobs,input_hashes={n:sha256_file(root/n) for n in sorted(names)},
        fresh_training=False,resource_projection=dict(expected_episodes=60,
            estimated_producer_wall_seconds=sum(e['resources']['wall_seconds'] for e in chosen),
            estimated_artifact_bytes=sum(e['resources']['disk_bytes'] for e in chosen),
            interpretation='Measured original first-task proxy; fresh process startup/hash costs additional'),
        scope='All three models, both plants/clocks, first task of each training seed; selected-weight replay only')
    atomic_json(target,registry);return registry


def run_replays(root,registry,registry_hash,*,resume=False):
    root=Path(root).resolve();target=root/'reports/control_replay.json'
    jobs={j['cell_id']:j for j in registry['jobs']};episodes=[]
    if len(jobs)!=len(registry['jobs']):raise ValueError('Duplicate replay job identity')
    def observe(job,path):
        request=read_json(safe_path(root,job['path']))
        parent=safe_path(root,job['parent_manifest'])
        if (sha256_file(parent)!=job['parent_manifest_sha256']
                or sha256_file(safe_path(root,job['path']))!=job['sha256']):
            raise ValueError('Changed replay request or raw parent')
        original=_read_episode(root,parent,'confirmatory',{**request,'phase':'confirmatory'})
        fresh=_read_episode(root,path,'replication',request)
        return dict(cell_id=job['cell_id'],parent_cell_id=job['parent_cell_id'],manifest=str(path.relative_to(root)),
            manifest_sha256=sha256_file(path),comparison=compare_episode(original,fresh),resources=fresh.get('resources'))
    def save(state,**extra):
        result=dict(state=state,registry_sha256=registry_hash,expected_episodes=len(jobs),
            completed_episodes=len(episodes),episodes=episodes,fresh_training=False,**extra)
        atomic_json(target,result);return result
    key=hashlib.sha256(b'fixed first-task control replay').hexdigest()
    with job_lock(key,root):
        for name,digest in registry['input_hashes'].items():
            if sha256_file(safe_path(root,name))!=digest:
                raise ValueError('Changed frozen replay source/input')
        reused={}
        if target.exists():
            previous=read_json(target)
            if not resume or previous['registry_sha256']!=registry_hash:
                raise ValueError('Existing replay requires compatible --resume')
            for entry in previous['episodes']:
                cell=entry['cell_id'];path=safe_path(root,entry['manifest'])
                if cell not in jobs or cell in reused or sha256_file(path)!=entry['manifest_sha256']:
                    raise ValueError('Changed or duplicate completed replay')
                actual=observe(jobs[cell],path)
                if actual!=entry:raise ValueError('Replay receipt differs from its completed raw observation')
                reused[cell]=entry
            if previous['state']=='complete':
                if set(reused)!=set(jobs) or previous['completed_episodes']!=len(jobs):
                    raise ValueError('Incomplete completed replay coverage')
                return previous
        try:
            for i,job in enumerate(registry['jobs']):
                for name,digest in registry['input_hashes'].items():
                    if sha256_file(safe_path(root,name))!=digest:
                        raise ValueError('Changed frozen replay source/input')
                print(f"Fresh-process control replay {i+1}/{len(jobs)}: {job['cell_id']}",flush=True)
                if job['cell_id'] in reused:episodes.append(reused[job['cell_id']]);continue
                process=subprocess.run([sys.executable,'-m','poc.control_jobs',job['path']],cwd=root,capture_output=True,text=True)
                attempt=root/'reports/control_replay_attempts'/f'{i:04d}-{utc_now().replace(":","")}'
                attempt.mkdir(parents=True);(attempt/'stdout.log').write_text(process.stdout);(attempt/'stderr.log').write_text(process.stderr)
                if process.returncode:raise RuntimeError('Replay child incomplete: '+process.stderr[-2000:])
                path=Path(process.stdout.strip().splitlines()[-1]).resolve()
                episodes.append(observe(job,path));save('running')
            return save('complete',diagnosis_required=any(e['comparison']['diagnosis_required'] for e in episodes))
        except BaseException as error:
            save('interrupted',reason=f'{type(error).__name__}: {error}');raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv);root=args.root.resolve()
    if args.dry_run:
        print(dict(execute=False,expected_episodes=60,episode_ids=[0],fresh_training=False,
                   prerequisite='Complete primary batch and verified frozen selected models'));return 0
    import importlib.metadata
    from .study_reproduce import validate_runtime_versions
    if sys.version_info[:2]!=(3,11):raise ValueError('Replay requires the pinned Python 3.11 runtime')
    versions=read_json(root/'locks/environment.json')['versions']
    validate_runtime_versions(versions,{name:importlib.metadata.version(name) for name in versions})
    registry=prepare(root)
    if args.prepare:print(root/'config/control_replay_jobs.json');return 0
    result=run_replays(root,registry,sha256_file(root/'config/control_replay_jobs.json'),resume=args.resume)
    print({k:result[k] for k in ('state','completed_episodes','diagnosis_required','fresh_training') if k in result})
    return 0


if __name__=='__main__':
    raise SystemExit(main())
