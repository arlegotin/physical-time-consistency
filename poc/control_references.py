"""Same-task physical references for the registered amended control comparison."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
from .io import atomic_json, job_lock, read_json, safe_path, sha256_file, utc_now
from .control_confirmation import ACTUATORS, validate_design


def reference_requests(design: dict, settings: dict, selected: list[dict]) -> list[dict]:
    validate_design(design)
    if (len(selected)!=2 or {s['actuator'] for s in selected}!=set(ACTUATORS)
            or any(s['model_id']!='physics_fit' for s in selected)):
        raise ValueError('One exact calibrated physics reference per plant is required')
    from .control_jobs import validate_request
    fitted={s['actuator']:s for s in selected};rows=[]
    for a in ACTUATORS:
        for model in ('physics_fit','zero_command','sim_oracle'):
            checkpoint=fitted[a] if model=='physics_fit' else None
            for seed in design['initial_training_seed_ids']:
                episodes=range(5) if model=='sim_oracle' else design['initial_episode_ids']
                for episode in episodes:
                    request=validate_request(dict(schema_version=1,phase='confirmatory',model_id=model,
                        actuator=a,seed=seed,episode_id=episode,condition='clean',
                        model_partition_ns=[5_000_000]*8,actuator_interval_ns=40_000_000,
                        physics_step_ns=250_000,power_mode=None,
                        model_manifest=checkpoint['manifest'] if checkpoint else None,
                        manifest_sha256=checkpoint['manifest_sha256'] if checkpoint else None,
                        checkpoint_sha256=checkpoint['checkpoint_sha256'] if checkpoint else None,
                        planner_settings=settings,resource_projection=design['resource_projection']))
                    rows.append(dict(cell_id=f'reference__{a}__{model}__seed{seed}__episode{episode}',request=request))
    return rows


def _frozen_references(root):
    from .planner_qualification import verify_frozen
    from .control_models import load_selected
    qualification,files=verify_frozen(root)
    selected={}
    for entry in qualification['episode_manifests']:
        path=safe_path(root,entry['path'])
        request=read_json(path.parent/'request.json')
        if request['model_id']!='physics_fit':
            continue
        a=request['actuator']
        item=dict(actuator=a,model_id='physics_fit',manifest=request['model_manifest'],
            manifest_sha256=request['manifest_sha256'],checkpoint_sha256=request['checkpoint_sha256'])
        if a in selected and selected[a]!=item:
            raise ValueError('Qualified planner used conflicting physical reference fits')
        selected[a]=item
    if set(selected)!=set(ACTUATORS):
        raise ValueError('Qualified physical reference missing for a plant')
    for item in selected.values():
        path=safe_path(root,item['manifest'])
        record=read_json(path)
        if record['milestone_id']!='M40' or record['phase']=='unit':
            raise ValueError('Physical reference must be a nominal robot fit')
        load_selected(path,item['manifest_sha256'],item['checkpoint_sha256'],
                      model_id='physics_fit',actuator=item['actuator'],seed=0)
    return list(selected.values()),files


def _verify_registry(root,registry):
    from .control_confirmation import verify_freeze
    design=read_json(root/'config/control_confirmation_design.json')
    verify_freeze(root,design,read_json(root/'config/control_confirmation_cells.json'))
    for name,expected in registry['input_hashes'].items():
        if sha256_file(safe_path(root,name))!=expected:
            raise ValueError('Changed reference dependency: '+name)
    selected,_=_frozen_references(root)
    expected=reference_requests(design,read_json(root/'config/planner.json')['settings'],selected)
    if registry['state']!='complete' or registry['selected']!=selected or len(registry['jobs'])!=450:
        raise ValueError('Incomplete physical reference registry')
    for job,row in zip(registry['jobs'],expected):
        path=safe_path(root,job['path'])
        if job['cell_id']!=row['cell_id'] or sha256_file(path)!=job['sha256'] or read_json(path)!=row['request']:
            raise ValueError('Changed same-task physical reference request')
    return registry['jobs']


def prepare(root: Path) -> dict:
    root=Path(root).resolve();target=root/'config/control_reference_jobs.json'
    if target.exists():
        registry=read_json(target);_verify_registry(root,registry);return registry
    primary=read_json(root/'reports/control_confirmation_batch.json')
    if (primary.get('state')!='complete' or primary.get('completed_episodes')!=1200
            or primary.get('expected_episodes')!=1200 or len(primary.get('episodes',[]))!=1200):
        raise ValueError('Complete primary control batch required before reference execution')
    from .control_confirmation import verify_freeze, REGISTRY_SOURCES
    design=read_json(root/'config/control_confirmation_design.json')
    verify_freeze(root,design,read_json(root/'config/control_confirmation_cells.json'))
    selected,qualified_files=_frozen_references(root)
    rows=reference_requests(design,read_json(root/'config/planner.json')['settings'],selected)
    from .control_jobs import SOURCES
    names=(*SOURCES,*REGISTRY_SOURCES,*qualified_files,'poc/control_references.py',
           'config/control_confirmation_cells.json','config/control_confirmation_design.json',
           'reports/control_confirmation_batch.json')
    hashes={name:sha256_file(root/name) for name in names};jobs=[]
    for row in rows:
        path=root/'config/control_jobs'/f"{row['cell_id']}.json"
        if path.exists() and read_json(path)!=row['request']:
            raise ValueError('Conflicting prospective physical reference request')
        atomic_json(path,row['request'])
        jobs.append(dict(cell_id=row['cell_id'],path=str(path.relative_to(root)),sha256=sha256_file(path)))
    registry=dict(schema_version=1,state='complete',created_utc=utc_now(),selected=selected,
        input_hashes=hashes,jobs=jobs,expected_episodes=450,scientific_decision='not_evaluated',
        fitted_physics_models_per_plant=1,
        interpretation='Five paired task batches; no fabricated multiple-seed physics training replication')
    atomic_json(target,registry);return registry


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv);root=args.root.resolve();target=root/'config/control_reference_jobs.json'
    if args.dry_run:
        print(dict(execute=False,expected_episodes=450,physics_fit_episodes=200,zero_command_episodes=200,
            simulator_subset_episodes=50,registry_ready=target.exists(),
            prerequisite='Complete primary control and frozen nominal physical references',
            scientific_decision='not_evaluated'));return 0
    if args.prepare:
        prepare(root);print(target);return 0
    registry=read_json(target);jobs=_verify_registry(root,registry)
    from .control_confirmation_batch import run_jobs
    key=hashlib.sha256(b'registered same-task physical references').hexdigest()
    with job_lock(key,root):
        batch=run_jobs(root,jobs,sha256_file(target),resume=args.resume,
                       frozen_sources=registry['input_hashes'],batch_kind='reference')
    from .control_utility import audit_raw_episode
    audits=[]
    for entry in batch['episodes']:
        audits.append(dict(cell_id=entry['cell_id'],manifest=entry['manifest'],
            manifest_sha256=entry['manifest_sha256'],
            **audit_raw_episode(safe_path(root,entry['manifest']).parent,entry['summary'])))
    report=dict(state='complete',registry_sha256=sha256_file(target),
        batch_sha256=sha256_file(root/'reports/control_reference_batch.json'),audited_episodes=audits,
        fitted_physics_models_per_plant=1,scientific_decision='context_only_no_head_superiority_claim')
    atomic_json(root/'reports/control_reference_audit.json',report)
    print(root/'reports/control_reference_audit.json');return 0


if __name__=='__main__':
    raise SystemExit(main())
