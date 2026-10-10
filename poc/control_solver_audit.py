"""Read-only NODE solver probes at fixed saved control observations."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name]='1'
import numpy as np
from .io import atomic_json, read_json, safe_path, sha256_file


def development_subset(batch: dict) -> list[dict]:
    if (batch.get('state')!='complete' or batch.get('expected_episodes')!=60
            or batch.get('completed_episodes')!=60 or len(batch.get('episodes',[]))!=60):
        raise ValueError('Complete fixed development batch required for solver diagnosis')
    seen=set();selected=[]
    for entry in batch['episodes']:
        row=entry['summary'];part=row['model_partition_ns'];h=part[0] if part else None
        coordinate=(row['model_id'],h,row['episode_id'])
        if (coordinate in seen or row['phase']!='exploratory' or row['seed']!=9000
                or row['complete'] is not True or row['actuator']!='direct' or row['condition']!='clean'
                or row['physics_step_ns']!=250_000 or row['actuator_interval_ns']!=40_000_000
                or h not in (40_000_000,5_000_000) or part!=[h]*(40_000_000//h)):
            raise ValueError('Changed development identity, clocks or phase')
        seen.add(coordinate)
        if row['model_id']=='node_generic' and row['episode_id']<5:selected.append(entry)
    expected={(m,h,e) for m in ('mlp_dt','node_generic','power_correct')
              for h in (40_000_000,5_000_000) for e in range(10)}
    if seen!=expected:
        raise ValueError('Incomplete or duplicated original development control cells')
    return sorted(selected,key=lambda e:(e['summary']['model_partition_ns'][0],e['summary']['episode_id']))



def summarize_probes(raw: list[dict]) -> list[dict]:
    summaries=[]
    for h in (5_000_000,40_000_000):
        rows=[r for r in raw if r['horizon_ns']==h]
        if not rows:
            raise ValueError('Missing solver horizon observations')
        for index in (0,1):
            differences=[r['differences'][index] for r in rows]
            row=dict(horizon_ns=h,coarse_substeps=2**index,fine_substeps=2**(index+1),
                n_probes=len(rows),maximum_state_difference=max(d['maximum_state_difference'] for d in differences))
            for component in ('q','v'):
                relative=[d['jacobian_differences'][component]['relative_difference'] for d in differences]
                unresolved=sum(v is None for v in relative)
                row['n_unresolved_'+component+'_norms']=unresolved
                row['maximum_relative_'+component+'_jacobian_difference']=None if unresolved else max(relative)
            summaries.append(row)
    return summaries


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args(argv);root=args.root.resolve()
    design_path=root/'config/control_development_solver_audit.json'
    design=read_json(design_path)
    expected_design=dict(schema_version=1,actuator='direct',model_id='node_generic',seed=9000,
        phase='exploratory',command_namespace=9103,episode_ids=list(range(5)),
        horizons_ns=[5_000_000,40_000_000],substeps=[1,2,4],
        scope='Read-only same-field numerical diagnosis from completed direct development observations')
    if design!=expected_design:
        raise ValueError('Changed registered development solver audit design')
    batch_path=root/'reports/control_development_direct.json';batch=read_json(batch_path)
    selected=development_subset(batch)
    if args.dry_run:
        print(dict(execute=False,phase='exploratory',episodes=len(selected),probes=400,
            substeps=[1,2,4],scope='Direct development NODE only; no control-confirmation sign'));return 0
    import torch
    from .execution import cpu_defaults,run_job
    from .contracts import verify_run
    from .control_jobs import SOURCES
    from .control_models import load_selected
    from .control_utility import audit_raw_episode
    from .control_numerical import solver_probe
    from .reference_jobs import tensor_hash
    cpu_defaults(9000)
    files=set(SOURCES)|{'poc/control_solver_audit.py','poc/control_numerical.py','poc/control_utility.py',
        'poc/control_confirmation.py','poc/statistics.py','reports/control_development_direct.json',
        'config/control_development_solver_audit.json'}
    requests=[]
    for entry in selected:
        path=safe_path(root,entry['manifest'])
        if sha256_file(path)!=entry['manifest_sha256']:
            raise ValueError('Changed saved control input manifest')
        record=read_json(path);verify_run(root,record,check_dependencies=False)
        summary=read_json(path.parent/'summary.json')
        if summary!=entry['summary']:
            raise ValueError('Saved input summary differs from its completed observation')
        audit_raw_episode(path.parent,summary)
        request=read_json(path.parent/'request.json');requests.append(request)
        files.add(entry['manifest'])
        for name in ('request.json','summary.json','observations.npz','true_trajectories.npz',
                     'action_segments.jsonl','decisions.jsonl'):
            files.add(str((path.parent/name).relative_to(root)))
        manifest=safe_path(root,request['model_manifest'])
        files.add(request['model_manifest'])
        for name in ('weights.pt','normalization.json'):
            files.add(str((manifest.parent/name).relative_to(root)))
    hashes={name:sha256_file(root/name) for name in files}
    expected=('control.solver_inputs','control.solver_weights','control.solver_work')
    def producer(folder):
        cache={};before={};raw=[]
        for entry,request in zip(selected,requests):
            checkpoint=request['checkpoint_sha256']
            if checkpoint not in cache:
                cache[checkpoint]=load_selected(safe_path(root,request['model_manifest']),
                    request['manifest_sha256'],checkpoint,model_id='node_generic',actuator='direct',seed=9000)
                before[checkpoint]=tensor_hash(cache[checkpoint].state_dict())
            with np.load(safe_path(root,entry['manifest']).parent/'observations.npz',allow_pickle=False) as data:
                states=data['state'].copy();times=data['time_ns'].copy()
            h=request['model_partition_ns'][0]
            for decision,(state,time_ns) in enumerate(zip(states,times)):
                entropy=[9103,0,9000,request['episode_id'],decision]
                command=np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy))).uniform(-1.5,1.5,2)
                row=solver_probe(cache[checkpoint],torch.tensor(state[None],dtype=torch.float64),
                                 torch.tensor(command[None],dtype=torch.float64),h)
                raw.append(dict(episode_id=request['episode_id'],time_ns=int(time_ns),
                    checkpoint_sha256=checkpoint,input_manifest=entry['manifest'],command_seed_entropy=entropy,
                    state=state.tolist(),command=command.tolist(),**row))
            print(f"NODE development solver audit: episode {request['episode_id']}, h={h}",flush=True)
        if (len(raw)!=400 or any(tensor_hash(cache[key].state_dict())!=value for key,value in before.items())
                or any(sha256_file(root/name)!=digest for name,digest in hashes.items())):
            raise ValueError('Solver audit changed selected weights, source inputs or fixed probe coverage')
        summaries=summarize_probes(raw)
        atomic_json(folder/'solver_probes.json',dict(probes=raw,command_namespace=9103))
        summary=dict(scope='Direct augmented/shared development NODE; one training seed, read-only field audit',
            n_training_seeds=1,seed=9000,model_id='node_generic',actuator='direct',phase='exploratory',
            n_probes=400,substeps=[1,2,4],summary=summaries,
            total_forward_vector_field_calls=sum(r['total_forward_vector_field_calls'] for r in raw),
            scientific_decision='not_evaluated',missing_cells=[],
            limitation='Local field/command sensitivity only; no refined closed-loop controller or confirmation')
        checks=[dict(check_id=name,status='pass',scope='Read-only solver diagnosis',tolerance=None,
            reason='Complete fixed input probes, unchanged canonical tensors and actual forward counts') for name in expected]
        return checks,summary
    record=run_job(root,'M40','exploratory',sorted(files),expected,[],producer,
        config_keys={'config/control_development_solver_audit.json':design},
        argv=['-m','poc.control_solver_audit'])
    path=root/'evidence/M40'/record['run_id']/'manifest.json'
    if record['completion_status']!='complete':
        raise RuntimeError('Development solver audit incomplete: '+str(path))
    atomic_json(root/'reports/control_development_node_solver.json',dict(manifest=str(path.relative_to(root)),
        manifest_sha256=sha256_file(path),summary=read_json(path.parent/'summary.json'),
        resources=read_json(path.parent/'resources.json')))
    print(path);return 0


if __name__=='__main__':
    raise SystemExit(main())
