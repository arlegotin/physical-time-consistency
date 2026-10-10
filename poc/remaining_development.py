"""Finish only the original seven-model, ten-task development control workload."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from .control_jobs import SOURCES, validate_request
from .io import atomic_json, job_key, job_lock, read_json, safe_path, sha256_file, utc_now
from .utility_training import require_completed_development


def remaining_fit_keys() -> set:
    return {('direct',m) for m in ('gru_dt','node_structured')} | {
        ('lagged',m) for m in ('mlp_dt','gru_dt','node_generic','power_correct','node_structured')}


def validate_requests(requests: list[dict]) -> list[dict]:
    seen=[]
    for request in requests:
        r=validate_request(request)
        h=r['model_partition_ns'][0]
        if (r['phase']!='exploratory' or r['seed']!=9000 or r['condition']!='clean'
                or r['power_mode'] is not None or r['actuator_interval_ns']!=40_000_000
                or r['physics_step_ns']!=250_000 or h not in (40_000_000,5_000_000)
                or r['model_partition_ns']!=[h]*(40_000_000//h)):
            raise ValueError('Changed original development timing, task or power mode')
        seen.append((r['actuator'],r['model_id'],h,r['episode_id']))
    expected={(a,m,h,e) for a,m in remaining_fit_keys() for h in (40_000_000,5_000_000) for e in range(10)}
    if len(seen)!=140 or set(seen)!=expected:
        raise ValueError('Complete distinct original remaining development cells required')
    return requests


def prepare_registry(root: Path) -> dict:
    root=Path(root).resolve()
    require_completed_development(root)
    target=root/'config/control_development_remaining_jobs.json'
    if target.exists():
        return verify_registry(root,read_json(target))
    from .planner_qualification import verify_frozen
    verify_frozen(root)
    receipt=read_json(root/'reports/control_nominal_completion.json')
    settings=read_json(root/'config/planner.json')['settings']
    hashes={p:sha256_file(root/p) for p in (*SOURCES,'poc/remaining_development.py')}
    jobs=[];requests=[]
    projection=dict(source='reports/control_development_direct_resources.json',
        source_sha256=sha256_file(root/'reports/control_development_direct_resources.json'),
        note='Direct measured episode proxy; GRU/lagged latency unmeasured until representative jobs',
        expected_episodes=140,owner_imposed_caps=None)
    for fit in receipt['cells']:
        a,m=fit['cell']['actuator'],fit['cell']['model_id']
        checkpoint=sha256_file(safe_path(root,fit['manifest']).parent/'weights.pt')
        for h in (40_000_000,5_000_000):
            for episode in range(10):
                cell=f'{a}__A__shared__{m}__seed9000__h{h}__episode{episode}'
                request=validate_request(dict(schema_version=1,phase='exploratory',model_id=m,actuator=a,
                    seed=9000,episode_id=episode,condition='clean',model_partition_ns=[h]*(40_000_000//h),
                    actuator_interval_ns=40_000_000,physics_step_ns=250_000,power_mode=None,
                    model_manifest=fit['manifest'],manifest_sha256=fit['manifest_sha256'],
                    checkpoint_sha256=checkpoint,planner_settings=settings,resource_projection=projection))
                path=root/'config/control_jobs'/f'remaining_development__{cell}.json'
                if path.exists() and read_json(path)!=request:
                    raise ValueError('Conflicting original remaining development request')
                atomic_json(path,request);requests.append(request)
                jobs.append(dict(cell_id=cell,path=str(path.relative_to(root)),sha256=sha256_file(path)))
    validate_requests(requests)
    registry=dict(schema_version=1,phase='exploratory',expected_episodes=140,state='complete',
        created_utc=utc_now(),jobs=jobs,evaluator_hashes=hashes,
        input_hashes={p:sha256_file(root/p) for p in ('config/control_development.json',
            'config/control_development_cells.json','config/planner.json','reports/control_nominal_completion.json')})
    atomic_json(target,registry)
    return registry


def verify_registry(root: Path, registry: dict) -> dict:
    if registry['state']!='complete' or registry['expected_episodes']!=140 or registry['phase']!='exploratory':
        raise ValueError('Incomplete original development registry')
    if set(registry['evaluator_hashes'])!=set(SOURCES)|{'poc/remaining_development.py'}:
        raise ValueError('Incomplete development evaluator closure')
    if set(registry['input_hashes'])!= {'config/control_development.json','config/control_development_cells.json',
                                     'config/planner.json','reports/control_nominal_completion.json'}:
        raise ValueError('Incomplete original development input closure')
    for name,expected in {**registry['evaluator_hashes'],**registry['input_hashes']}.items():
        if sha256_file(safe_path(root,name))!=expected:
            raise ValueError('Changed remaining development dependency: '+name)
    requests=[]
    for job in registry['jobs']:
        path=safe_path(root,job['path'])
        if sha256_file(path)!=job['sha256']:
            raise ValueError('Changed remaining development request')
        requests.append(read_json(path))
    validate_requests(requests)
    fits={(x['cell']['actuator'],x['cell']['model_id']):x for x in read_json(root/'reports/control_nominal_completion.json')['cells']}
    for r in requests:
        f=fits[(r['actuator'],r['model_id'])]
        if (r['model_manifest']!=f['manifest'] or r['manifest_sha256']!=f['manifest_sha256']
                or r['checkpoint_sha256']!=sha256_file(safe_path(root,f['manifest']).parent/'weights.pt')
                or r['planner_settings']!=read_json(root/'config/planner.json')['settings']):
            raise ValueError('Development request differs from its nominally selected fit or common planner')
    return registry


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv);root=args.root.resolve()
    path=root/'config/control_development_remaining_jobs.json'
    if args.dry_run:
        print(dict(execute=False,expected_episodes=140,registry_exists=path.exists(),
                   prerequisite='Complete the original seven nominal fits; prepare their exact requests'))
        return 0
    if args.prepare:
        prepare_registry(root);return 0
    registry=verify_registry(root,read_json(path));registry_hash=sha256_file(path)
    receipt_path=root/'reports/control_development_remaining.json'
    if receipt_path.exists() and not args.resume:
        raise ValueError('Existing development batch requires --resume')
    from .contracts import verify_run
    episodes=[]
    def save(state,**extra):
        atomic_json(receipt_path,dict(state=state,registry_sha256=registry_hash,updated_utc=utc_now(),
            expected_episodes=140,completed_episodes=len(episodes),episodes=episodes,
            n_training_seeds=1,statistical_status='descriptive_only',**extra))
    with job_lock(job_key({'remaining_development_registry':registry_hash}),root):
        try:
            for index,job in enumerate(registry['jobs']):
                if sha256_file(path)!=registry_hash:
                    raise ValueError('Original remaining development registry changed')
                for name,expected in registry['evaluator_hashes'].items():
                    if sha256_file(safe_path(root,name))!=expected:
                        raise ValueError('Original development evaluator changed during execution: '+name)
                print(f"Remaining development {index+1}/140: {job['cell_id']}",flush=True)
                run=subprocess.run([sys.executable,'-m','poc.control_jobs',job['path']],cwd=root,
                                   capture_output=True,text=True)
                if run.returncode:
                    raise RuntimeError('Development episode failed: '+run.stdout+'\n'+run.stderr)
                manifest=Path(run.stdout.strip().splitlines()[-1]);record=read_json(manifest)
                verify_run(root,record)
                request=read_json(root/job['path']);summary=read_json(manifest.parent/'summary.json')
                if read_json(manifest.parent/'request.json')!=request or not summary['complete']:
                    raise ValueError('Completed episode differs from its original registered job')
                episodes.append(dict(cell_id=job['cell_id'],manifest=str(manifest.relative_to(root)),
                    manifest_sha256=sha256_file(manifest),summary=summary,
                    resources=read_json(manifest.parent/'resources.json')))
                save('running')
            save('complete');return 0
        except BaseException as error:
            save('interrupted',reason=f'{type(error).__name__}: {error}');raise


if __name__=='__main__':
    raise SystemExit(main())
