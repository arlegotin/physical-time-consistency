"""Fixed serial replay, fresh-training and repeated-control continuation."""
from __future__ import annotations
import argparse
import hashlib
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .io import atomic_json,job_lock,read_json,safe_path,sha256_file,utc_now


def _ready(root):
    path=root/'reports/utility_followup.json'
    if not path.exists():raise ValueError('Complete numerical/reference follow-up required')
    followup=read_json(path)
    if (followup.get('state')!='limited_result_ready_for_reproduction_and_review'
            or followup.get('required_actions')!=[] or followup.get('scientific_verdict') is not None):
        raise ValueError('Complete numerical/reference follow-up required')
    result=read_json(root/'reports/control_confirmation_result.json')
    numerical=root/'reports/control_confirmation_numerical.json';qualification=read_json(numerical)
    if (result.get('status')!='limited_scope_results' or result.get('original_ptco_terminal_verdict') is not None
            or sha256_file(numerical)!=result['evidence_identity']['numerical_sha256']
            or qualification.get('state')!='qualified' or qualification.get('required_actions')!=[]):
        raise ValueError('Valid completed numerical qualification and limited result required')
    for name,digest in result['input_hashes'].items():
        if sha256_file(safe_path(root,name))!=digest:raise ValueError('Changed signed control-result evidence')
    return {name:sha256_file(root/name) for name in ('reports/utility_followup.json',
        'reports/control_confirmation_result.json','reports/control_confirmation_numerical.json')}


def run_followup(root,*,resume=False):
    root=Path(root).resolve();inputs=_ready(root);attempts=[]
    target=root/'reports/reproduction_followup.json'
    key=hashlib.sha256(b'fixed control reproduction followup').hexdigest()
    with job_lock(key,root):
        if target.exists():
            previous=read_json(target)
            if not resume or previous['input_hashes']!=inputs:
                raise ValueError('Existing reproduction queue requires compatible --resume')
            attempts=list(previous['attempts'])
        details={}
        def save(state,**extra):
            details.update(extra)
            record=dict(schema_version=1,state=state,observed_utc=utc_now(),input_hashes=inputs,
                attempts=attempts,scientific_verdict=None,whole_branch_review_completed=False,**details)
            atomic_json(target,record);return record
        def stage(module,*args):
            for name,digest in inputs.items():
                if sha256_file(root/name)!=digest:raise ValueError('Prerequisite changed during reproduction queue')
            stamp=utc_now().replace(':','').replace('-','').replace('.','')
            folder=root/'reports/reproduction_followup_attempts'/f'{len(attempts):04d}-{stamp}'
            folder.mkdir(parents=True)
            command=[sys.executable,'-m',module,*args]
            print('Reproduction follow-up:',' '.join(command[1:]),flush=True)
            attempt=dict(module=module,args=list(args),started_utc=utc_now(),
                stdout=str((folder/'stdout.log').relative_to(root)),stderr=str((folder/'stderr.log').relative_to(root)))
            attempts.append(attempt);save('running')
            child=subprocess.run(command,cwd=root,capture_output=True,text=True)
            (folder/'stdout.log').write_text(child.stdout);(folder/'stderr.log').write_text(child.stderr)
            attempt.update(returncode=child.returncode,ended_utc=utc_now());save('running')
            if child.returncode:raise RuntimeError(f'{module}: {child.stderr[-2000:]}')
        def storage(registry_path,field):
            projection=read_json(root/registry_path)['resource_projection'];estimate=projection[field]
            if type(estimate) not in (int,float) or not math.isfinite(estimate) or estimate<0:
                raise ValueError('Measured reproduction resource projection required')
            receipts={'config/control_replay_jobs.json':('reports/control_replay.json',60,'episodes'),
                'config/control_retraining_jobs.json':('reports/control_retraining.json',12,'fits'),
                'config/control_retrained_policy_jobs.json':('reports/control_retrained_policy_batch.json',480,'episodes')}
            name,count,kind=receipts[registry_path];path=root/name
            if path.exists():
                receipt=read_json(path)
                if receipt['registry_sha256']!=sha256_file(root/registry_path):
                    raise ValueError('Cached reproduction receipt differs from its resource registry')
                completed=receipt['completed_'+kind]
                if (type(completed) is not int or not 0<=completed<=count
                        or receipt['expected_'+kind]!=count or len(receipt[kind])!=completed):
                    raise ValueError('Invalid completed reproduction count for resource projection')
                estimate*= (count-completed)/count
            available=shutil.disk_usage(root).free
            save('running',next_registry=registry_path,actual_free_bytes=available,projected_new_bytes=estimate,
                projection_limitation='Exact context input bytes or measured original episode proxy; fresh fit artifacts remain unknown')
            if available<estimate:
                raise RuntimeError(f'Actual disk prerequisite: {available} free bytes versus {estimate} projected bytes; preserve evidence and provision disk')
        try:
            stage('poc.control_replay','--prepare')
            storage('config/control_replay_jobs.json','estimated_artifact_bytes')
            stage('poc.control_replay','--resume')
            stage('poc.control_retraining','--prepare')
            storage('config/control_retraining_jobs.json','context_copy_bytes')
            stage('poc.control_retraining','--resume')
            stage('poc.control_retrained_policy','--prepare')
            storage('config/control_retrained_policy_jobs.json','estimated_artifact_bytes')
            stage('poc.control_retrained_policy','--resume')
            result=read_json(root/'reports/control_retrained_policy_result.json')
            if (result.get('status')!='pass' or result.get('diagnosis_required') is not False
                    or result.get('criteria_unchanged') is not True or result.get('alpha_spent')!=0
                    or result.get('new_independent_training_clusters')!=0):
                raise RuntimeError('Reproduction diagnosis required; preserve repeated evidence')
            return save('reproduction_ready_for_whole_branch_review',
                reproduced_result_sha256=sha256_file(root/'reports/control_retrained_policy_result.json'))
        except BaseException as error:
            save('interrupted',reason=f'{type(error).__name__}: {error}');raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.cwd());parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--resume',action='store_true');parser.add_argument('--watch',action='store_true')
    args=parser.parse_args(argv);root=args.root.resolve()
    if args.dry_run:
        print(dict(execute=False,replay_episodes=60,fresh_fits=12,repeated_control_episodes=480,
            prerequisite='Complete numerical/reference follow-up',scientific_verdict=None));return 0
    if args.watch:
        key=hashlib.sha256(b'fixed reproduction readiness watcher').hexdigest()
        with job_lock(key,root):
            previous=None
            while True:
                path=root/'reports/utility_followup.json'
                record=read_json(path) if path.exists() else dict(state='pending')
                state=record['state']
                if state!=previous:
                    print('Reproduction prerequisite:',state,flush=True);previous=state
                if state=='limited_result_ready_for_reproduction_and_review':break
                if state in ('interrupted','blocked','named_diagnosis_required'):
                    raise RuntimeError('Numerical/reference follow-up needs its recorded diagnosis: '+record.get('reason',state))
                time.sleep(30)
            result=run_followup(root,resume=args.resume)
    else:result=run_followup(root,resume=args.resume)
    print({k:result[k] for k in ('state','scientific_verdict','whole_branch_review_completed')});return 0


if __name__=='__main__':
    raise SystemExit(main())
