"""Fixed serial numerical/reference follow-up to the registered primary batch."""
from __future__ import annotations
import argparse
import hashlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .io import atomic_json,job_lock,read_json,sha256_file,utc_now


def _complete_primary(root):
    path=root/'reports/control_confirmation_batch.json'
    if not path.exists():
        raise ValueError('Complete primary control batch required')
    batch=read_json(path)
    if (batch.get('state')!='complete' or batch.get('expected_episodes')!=1200
            or batch.get('completed_episodes')!=1200 or len(batch.get('episodes',[]))!=1200):
        raise ValueError('Complete primary control batch required')
    return batch


def run_followup(root,*,resume=False):
    """Use only existing fixed runners. Numerical branches keep their own evidence."""
    root=Path(root).resolve();batch=_complete_primary(root)
    key=hashlib.sha256(b'fixed utility numerical reference followup').hexdigest()
    target=root/'reports/utility_followup.json';attempts=[];scope='subset'
    primary_hash=sha256_file(root/'reports/control_confirmation_batch.json')
    with job_lock(key,root):
        if target.exists():
            previous=read_json(target)
            if not resume or previous['primary_batch_sha256']!=primary_hash:
                raise ValueError('Existing follow-up needs compatible --resume')
            attempts=list(previous['attempts'])
            scope=previous['scope']
            if scope not in ('subset','full'):
                raise ValueError('Unknown saved numerical follow-up scope')
        details={}
        def save(state,**extra):
            details.update(extra)
            record=dict(schema_version=1,state=state,scope=scope,observed_utc=utc_now(),
                primary_batch_sha256=primary_hash,attempts=attempts,scientific_verdict=None,**details)
            atomic_json(target,record);return record
        def stage(module,*args):
            if sha256_file(root/'reports/control_confirmation_batch.json')!=primary_hash:
                raise ValueError('Primary evidence changed during fixed follow-up')
            stamp=utc_now().replace(':','').replace('-','').replace('.','')
            folder=root/'reports/utility_followup_attempts'/f'{len(attempts):04d}-{stamp}'
            folder.mkdir(parents=True)
            command=[sys.executable,'-m',module,*args]
            print('Utility follow-up:', ' '.join(command[1:]),flush=True)
            attempt=dict(module=module,args=list(args),started_utc=utc_now(),
                stdout=str((folder/'stdout.log').relative_to(root)),
                stderr=str((folder/'stderr.log').relative_to(root)))
            attempts.append(attempt);save('running')
            completed=subprocess.run(command,cwd=root,capture_output=True,text=True)
            (folder/'stdout.log').write_text(completed.stdout)
            (folder/'stderr.log').write_text(completed.stderr)
            attempt.update(returncode=completed.returncode,ended_utc=utc_now())
            save('running')
            if completed.returncode:
                raise RuntimeError(f'{module}: {completed.stderr[-2000:]}')
        def storage_projection(n,receipt_name,registry_name,registry_field='registry_sha256'):
            receipt_path=root/receipt_name
            if receipt_path.exists():
                receipt=read_json(receipt_path)
                if (not (root/registry_name).exists()
                        or receipt.get(registry_field)!=sha256_file(root/registry_name)):
                    raise ValueError('Cached numerical/reference receipt differs from its resource registry')
                completed=receipt.get('completed_episodes')
                if (type(completed) is not int or not 0<=completed<=n
                        or receipt.get('expected_episodes')!=n or len(receipt.get('episodes',[]))!=completed):
                    raise ValueError('Invalid completed audit count for resource projection')
                n-=completed
            mean=sum(e['resources']['disk_bytes'] for e in batch['episodes'])/1200
            estimate=mean*n
            available=shutil.disk_usage(root).free
            save('running',next_stage_episode_proxy=n,estimated_new_artifact_bytes=estimate,
                 actual_free_bytes=available,projection_limitation='Primary artifact proxy; finer truth/solver artifacts can differ')
            if available<estimate:
                raise RuntimeError(f'Actual disk prerequisite: {available} free bytes versus {estimate} projected artifact bytes; preserve evidence and provision more disk')
        try:
            storage_projection(300 if scope=='subset' else 1200,
                f'reports/control_refinement_{scope}_batch.json',f'config/control_refinement_{scope}.json')
            stage('poc.control_refinement','--scope',scope,'--prepare')
            stage('poc.control_refinement','--scope',scope,'--resume')
            storage_projection(450,'reports/control_reference_batch.json','config/control_reference_jobs.json')
            stage('poc.control_references','--prepare')
            stage('poc.control_references','--resume')
            storage_projection(100,'reports/control_confirmation_solver.json',
                'config/control_confirmation_solver_inputs.json','input_registry_sha256')
            stage('poc.confirmation_solver','--prepare')
            stage('poc.confirmation_solver','--resume')
            stage('poc.control_result','--scope',scope)
            qualification=read_json(root/'reports/control_confirmation_numerical.json')
            if any(a['branch']=='physics_full' for a in qualification['required_actions']):
                scope='full';storage_projection(1200,'reports/control_refinement_full_batch.json',
                    'config/control_refinement_full.json')
                stage('poc.control_refinement','--scope','full','--prepare')
                stage('poc.control_refinement','--scope','full','--resume')
                stage('poc.control_result','--scope','full')
                qualification=read_json(root/'reports/control_confirmation_numerical.json')
            if qualification['state']!='qualified':
                return save('named_diagnosis_required',required_actions=qualification['required_actions'])
            stage('poc.control_result','--scope',scope,'--report')
            return save('limited_result_ready_for_reproduction_and_review',required_actions=[])
        except BaseException as error:
            save('interrupted',reason=f'{type(error).__name__}: {error}');raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--watch',action='store_true',help='Wait for primary completion without launching duplicate jobs')
    args=parser.parse_args(argv);root=args.root.resolve()
    if args.dry_run:
        print(dict(execute=False,primary_prerequisite=1200,initial_refined_episodes=300,
            reference_episodes=450,solver_episodes=100,full_refinement_only_if_required=True,
            scientific_verdict=None));return 0
    if args.watch:
        key=hashlib.sha256(b'fixed utility followup primary completion watcher').hexdigest()
        with job_lock(key,root):
            previous=None
            while True:
                path=root/'reports/control_confirmation_batch.json'
                record=read_json(path) if path.exists() else dict(state='pending')
                state=record['state']
                if state!=previous:
                    print('Utility audit prerequisite:',state,flush=True);previous=state
                if state=='complete':break
                if state in ('interrupted','blocked','unqualified'):
                    raise RuntimeError('Primary batch needs its recorded resumption: '+record.get('reason',state))
                time.sleep(30)
            result=run_followup(root,resume=args.resume)
    else:result=run_followup(root,resume=args.resume)
    print({k:result[k] for k in ('state','scope','required_actions','scientific_verdict') if k in result})
    return 0


if __name__=='__main__':
    raise SystemExit(main())
