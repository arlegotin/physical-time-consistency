"""Derive finer-plant jobs from the complete fixed primary control registry."""
from __future__ import annotations
from pathlib import Path
from .io import read_json, safe_path, sha256_file
from .control_confirmation import ACTUATORS, MODELS, STEPS


def refinement_requests(root: Path, freeze: dict, *, scope='subset') -> list[dict]:
    """Pure derivation; the caller verifies checkpoints and freezes before execution."""
    if scope not in ('subset','full') or freeze.get('state')!='complete' or freeze.get('expected_episodes')!=1200:
        raise ValueError('Complete primary checkpoint freeze required for refinement')
    from .control_jobs import validate_request
    jobs=freeze.get('jobs',[])
    if len(jobs)!=1200 or len({j['cell_id'] for j in jobs})!=1200:
        raise ValueError('Incomplete or duplicated primary refinement parents')
    seen=set();rows=[]
    for job in jobs:
        path=safe_path(root,job['path'])
        if sha256_file(path)!=job['sha256']:
            raise ValueError('Changed parent request hash')
        request=validate_request(read_json(path))
        a,m,s,e=request['actuator'],request['model_id'],request['seed'],request['episode_id']
        h=request['model_partition_ns'][0]
        coordinate=(a,m,h,s,e)
        if (coordinate in seen or a not in ACTUATORS or m not in MODELS or s not in range(5)
                or e not in range(20) or h not in STEPS or request['phase']!='confirmatory'
                or request['condition']!='clean' or request['actuator_interval_ns']!=40_000_000
                or request['physics_step_ns']!=250_000 or request['power_mode'] is not None
                or request['model_partition_ns']!=[h]*(40_000_000//h)):
            raise ValueError('Changed primary control identity while deriving refinement')
        seen.add(coordinate)
        if scope=='full' or e<5:
            rows.append(dict(cell_id=f"physics125000__{a}__{m}__seed{s}__h{h}__episode{e}",
                parent_cell_id=job['cell_id'],parent_request_sha256=job['sha256'],
                request={**request,'physics_step_ns':125_000}))
    expected={(a,m,h,s,e) for a in ACTUATORS for m in MODELS for h in STEPS for s in range(5) for e in range(20)}
    if seen!=expected:
        raise ValueError('Incomplete primary control workload coverage')
    return rows


import argparse
import hashlib
import subprocess
import sys
from .io import append_event, atomic_json, job_lock, utc_now


def _verify_registry(root, registry):
    from .control_confirmation import verify_freeze
    design=read_json(root/'config/control_confirmation_design.json')
    primary=read_json(root/'config/control_confirmation_cells.json')
    verify_freeze(root,design,primary)
    for name,expected in registry['input_hashes'].items():
        if sha256_file(safe_path(root,name))!=expected:
            raise ValueError('Changed refinement dependency: '+name)
    expected=refinement_requests(root,primary,scope=registry['scope'])
    jobs=registry['jobs']
    if registry['state']!='complete' or len(jobs)!=len(expected):
        raise ValueError('Incomplete refinement registry')
    for job,row in zip(jobs,expected):
        request_path=safe_path(root,job['path'])
        if (job['cell_id']!=row['cell_id'] or job['parent_cell_id']!=row['parent_cell_id']
                or job['parent_request_sha256']!=row['parent_request_sha256']
                or sha256_file(request_path)!=job['sha256'] or read_json(request_path)!=row['request']):
            raise ValueError('Changed finer-plant request or parent identity')
    return jobs


def prepare(root: Path, *, scope='subset') -> dict:
    root=Path(root).resolve()
    if scope not in ('subset','full'):
        raise ValueError('Unknown refinement scope')
    target=root/f'config/control_refinement_{scope}.json'
    if target.exists():
        registry=read_json(target);_verify_registry(root,registry);return registry
    batch=read_json(root/'reports/control_confirmation_batch.json')
    if (batch.get('state')!='complete' or batch.get('completed_episodes')!=1200
            or batch.get('expected_episodes')!=1200 or len(batch.get('episodes',[]))!=1200):
        raise ValueError('A complete primary batch is required before refinement')
    if (root/'reports/control_confirmation_raw_audit.json').exists():
        from .control_audit_closure import verify_primary_audit
        verify_primary_audit(root)
    else:
        from .control_utility import audit_batch
        audit_batch(root)
    primary=read_json(root/'config/control_confirmation_cells.json')
    rows=refinement_requests(root,primary,scope=scope)
    from .control_jobs import SOURCES
    from .control_confirmation import REGISTRY_SOURCES
    names=(*SOURCES,*REGISTRY_SOURCES,'poc/control_numerical.py','poc/control_refinement.py','poc/control_audit_closure.py',
        'config/control_confirmation_cells.json','config/control_confirmation_design.json',
        'reports/control_confirmation_batch.json','reports/control_confirmation_raw_audit.json')
    hashes={name:sha256_file(root/name) for name in names}
    jobs=[]
    for row in rows:
        path=root/'config/control_jobs'/f"{row['cell_id']}.json"
        if path.exists() and read_json(path)!=row['request']:
            raise ValueError('Conflicting refinement request already exists')
        atomic_json(path,row['request'])
        jobs.append({k:v for k,v in row.items() if k!='request'}|
                    dict(path=str(path.relative_to(root)),sha256=sha256_file(path)))
    registry=dict(schema_version=1,state='complete',scope=scope,created_utc=utc_now(),
        input_hashes=hashes,jobs=jobs,scientific_decision='not_evaluated',
        resource_projection=dict(primary_episode_proxy_seconds=
            sum(e['resources']['wall_seconds'] for e in batch['episodes'])/1200,
            expected_episodes=len(jobs),extra_physics_work='Twice the plant steps; unchanged MPC work',
            status='Estimate from measured primary episodes; refine after first completed finer episode'))
    atomic_json(target,registry)
    return registry


def run_refinement_jobs(root, jobs, registry_hash, *, scope, resume, frozen_sources=None):
    """Only execute the supplied validated fixed registry; retain every attempt."""
    from .control_confirmation_batch import episode_receipt
    root=Path(root).resolve()
    if scope not in ('subset','full') or len({j['cell_id'] for j in jobs})!=len(jobs):
        raise ValueError('Unknown or duplicated refinement workload')
    receipt_path=root/f'reports/control_refinement_{scope}_batch.json'
    for job in jobs:
        if sha256_file(safe_path(root,job['path']))!=job['sha256']:
            raise ValueError('Changed refinement request before launch')
    reused={};by_cell={j['cell_id']:j for j in jobs}
    if receipt_path.exists():
        if not resume:
            raise ValueError('Existing refinement receipt requires --resume')
        prior=read_json(receipt_path)
        if prior['registry_sha256']!=registry_hash:
            raise ValueError('Changed refinement registry is not a resumption')
        for entry in prior['episodes']:
            cell=entry['cell_id']
            if cell in reused or cell not in by_cell:
                raise ValueError('Unexpected prior refinement cell')
            path=safe_path(root,entry['manifest'])
            if sha256_file(path)!=entry['manifest_sha256']:
                raise ValueError('Changed prior refinement manifest')
            verified=episode_receipt(root,by_cell[cell],path)
            if any(entry.get(k)!=v for k,v in verified.items()):
                raise ValueError('Changed prior refinement observation')
            reused[cell]=verified
        if prior.get('state')=='complete':
            if (prior.get('expected_episodes')!=len(jobs) or prior.get('completed_episodes')!=len(jobs)
                    or len(reused)!=len(jobs)):
                raise ValueError('Completed refinement receipt has incomplete cell coverage')
            for name,expected in (frozen_sources or {}).items():
                if sha256_file(safe_path(root,name))!=expected:
                    raise ValueError('Changed frozen refinement dependency: '+name)
            return prior
    episodes=[]
    def save(state,**extra):
        record=dict(schema_version=1,state=state,scope=scope,registry_sha256=registry_hash,
            updated_utc=utc_now(),expected_episodes=len(jobs),completed_episodes=len(episodes),
            episodes=episodes,scientific_decision='not_evaluated',**extra)
        atomic_json(receipt_path,record);return record
    try:
        for i,job in enumerate(jobs):
            for name,expected in (frozen_sources or {}).items():
                if sha256_file(safe_path(root,name))!=expected:
                    raise ValueError('Changed frozen refinement dependency: '+name)
            if sha256_file(safe_path(root,job['path']))!=job['sha256']:
                raise ValueError('Changed refinement request during execution')
            print(f"Finer-plant control {i+1}/{len(jobs)}: {job['cell_id']}",flush=True)
            if job['cell_id'] in reused:
                episode={**reused[job['cell_id']],'reused':True}
            else:
                run=subprocess.run([sys.executable,'-m','poc.control_jobs',job['path']],cwd=root,
                                   capture_output=True,text=True)
                append_event(root/'reports/control_refinement_attempts.jsonl',dict(cell_id=job['cell_id'],
                    scope=scope,utc=utc_now(),returncode=run.returncode,stdout=run.stdout,stderr=run.stderr))
                if run.returncode:
                    raise RuntimeError('Refinement episode interrupted: '+run.stdout+'\n'+run.stderr)
                path=Path(run.stdout.strip().splitlines()[-1])
                episode={**episode_receipt(root,job,path),'reused':False}
            episodes.append(episode);save('running')
        return save('complete')
    except BaseException as error:
        save('interrupted',reason=f'{type(error).__name__}: {error}');raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--scope',choices=('subset','full'),default='subset')
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv);root=args.root.resolve()
    target=root/f'config/control_refinement_{args.scope}.json'
    if args.dry_run:
        print(dict(execute=False,scope=args.scope,expected_episodes=300 if args.scope=='subset' else 1200,
            registry_ready=target.exists(),prerequisite='Complete verified primary batch and raw audit',
            scientific_decision='not_evaluated'));return 0
    if args.prepare:
        prepare(root,scope=args.scope);print(target);return 0
    registry=read_json(target);jobs=_verify_registry(root,registry)
    key=hashlib.sha256(b'registered finer-plant control registry').hexdigest()
    with job_lock(key,root):
        result=run_refinement_jobs(root,jobs,sha256_file(target),scope=args.scope,resume=args.resume,
                                  frozen_sources=registry['input_hashes'])
    from .control_utility import audit_raw_episode
    from .control_numerical import compare_refinement
    for episode in result['episodes']:
        audit_raw_episode(safe_path(root,episode['manifest']).parent,episode['summary'],expected_tick_ns=125_000)
    primary=read_json(root/'reports/control_confirmation_batch.json')
    comparison=compare_refinement([e['summary'] for e in primary['episodes']],
        [e['summary'] for e in result['episodes']],read_json(root/'config/control_confirmation_design.json'),
        scope=args.scope)
    comparison.update(registry_sha256=sha256_file(target),
        batch_sha256=sha256_file(root/f'reports/control_refinement_{args.scope}_batch.json'))
    atomic_json(root/f'reports/control_refinement_{args.scope}_audit.json',comparison)
    print(root/f'reports/control_refinement_{args.scope}_audit.json');return 0


if __name__=='__main__':
    raise SystemExit(main())
