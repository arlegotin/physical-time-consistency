"""Fixed fresh-weight control reproduction; no independent seeds or statistical looks."""
from __future__ import annotations

import math
import copy
import argparse
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

from .control_confirmation import ACTUATORS,MODELS,STEPS
from .io import atomic_json,job_key,job_lock,read_json,safe_path,sha256_file,utc_now


def retrained_requests(root,primary,fits):
    from .control_replay import replay_requests
    root=Path(root).resolve();replay_requests(root,primary)
    expected={f'{a}__A__shared__{m}__seed{s}' for a in ACTUATORS for m in MODELS for s in (0,4)}
    rows=fits.get('fits',[]);indexed={r['cell_id']:r for r in rows}
    if (fits.get('state')!='complete' or fits.get('expected_fits')!=12 or fits.get('completed_fits')!=12
            or fits.get('diagnosis_required') is not False or len(rows)!=12 or set(indexed)!=expected
            or any(r['comparison']['status']!='pass' or r['comparison']['diagnosis_required'] for r in rows)):
        raise ValueError('Complete qualified fixed twelve-fit reproduction required before control')
    result=[]
    for job in primary['jobs']:
        request=read_json(safe_path(root,job['path']))
        if request['seed'] not in (0,4):continue
        fit_id=f"{request['actuator']}__A__shared__{request['model_id']}__seed{request['seed']}"
        fresh=indexed[fit_id]
        result.append(dict(cell_id='fresh_training_control__'+job['cell_id'],parent_cell_id=job['cell_id'],
            parent_request_sha256=job['sha256'],fresh_fit_cell_id=fit_id,
            request=dict(request,phase='replication',model_manifest=fresh['manifest'],
                manifest_sha256=fresh['manifest_sha256'],checkpoint_sha256=fresh['checkpoint_sha256'])))
    if len(result)!=480:raise ValueError('All twenty paired tasks required for fixed repeated seeds')
    return result


def compare_control(original,fresh):
    keys=('actuator','model_id','seed','episode_id','condition','model_partition_ns','actuator_interval_ns',
          'physics_step_ns','power_mode','initial_state','target','seed_namespaces')
    if (original.get('phase')!='confirmatory' or fresh.get('phase')!='replication'
            or any(original.get(k)!=fresh.get(k) for k in keys)
            or original.get('actuator') not in ACTUATORS or original.get('model_id') not in MODELS
            or original.get('seed') not in (0,4) or original.get('episode_id') not in range(20)
            or original.get('condition')!='clean' or original.get('physics_step_ns')!=250_000
            or original.get('actuator_interval_ns')!=40_000_000 or original.get('power_mode') is not None):
        raise ValueError('Changed repeated control task, phase, model or physical clocks')
    partition=original['model_partition_ns']
    if not partition or partition[0] not in STEPS or partition!=[partition[0]]*(40_000_000//partition[0]):
        raise ValueError('Changed repeated control model clock')
    for summary in (original,fresh):
        if (summary.get('complete') is not True or type(summary.get('realized_cost')) not in (int,float)
                or not math.isfinite(summary['realized_cost']) or summary['realized_cost']<0
                or type(summary.get('fallback_count')) is not int or not 0<=summary['fallback_count']<=40
                or type(summary.get('success')) is not bool):
            raise ValueError('Incomplete/nonfinite repeated cost or failure accounting')
    absolute=abs(original['realized_cost']-fresh['realized_cost'])
    relative=absolute/original['realized_cost'] if original['realized_cost']>0 else (0. if absolute==0 else None)
    same_classification=all(original[k]==fresh[k] for k in ('success','fallback_count'))
    passed=relative is not None and relative<=1e-3 and same_classification
    return dict(status='pass' if passed else 'outside_tolerance',diagnosis_required=not passed,
        absolute_cost_difference=absolute,relative_cost_difference=relative,
        success_and_fallback_unchanged=same_classification,new_independent_training_clusters=0,
        scope='Fresh-weight control reproduction of existing paired tasks; criterion stability checked separately')


def criterion_stability(primary,repeated,design,reference_result,envelopes):
    """Substitute existing seed replications, keeping the original inference and look."""
    from .control_utility import seed_costs,utility_bounds
    costs=seed_costs(primary,design);look=reference_result['look_index']
    before=utility_bounds(costs,design,look,envelopes)
    if before['criteria']!=reference_result['criteria']:
        raise ValueError('Original raw costs do not reproduce the criterion table being audited')
    def coordinate(r):return r['actuator'],r['model_id'],r['model_partition_ns'][0],r['seed'],r['episode_id']
    originals={coordinate(r):r for r in primary}
    expected={c for c in originals if c[3] in (0,4)}
    indexed={coordinate(r):r for r in repeated}
    if len(repeated)!=480 or len(indexed)!=480 or set(indexed)!=expected:
        raise ValueError('Complete fixed 480-episode retrained-control reproduction required')
    comparisons=[compare_control(originals[c],indexed[c]) for c in sorted(expected)]
    updated=copy.deepcopy(costs)
    for row in updated['rows']:
        a,m,h=row['actuator'],row['model_id'],row['model_step_ns']
        for seed in (0,4):
            tasks=[indexed[(a,m,h,seed,e)] for e in range(20)]
            row['cost_by_seed'][seed]=math.fsum(t['realized_cost'] for t in tasks)/20
            row['fallback_by_seed'][seed]=sum(t['fallback_count'] for t in tasks)
            row['success_by_seed'][seed]=sum(t['success'] for t in tasks)
    after=utility_bounds(updated,design,look,envelopes)
    old={r['criterion_id']:r['status'] for r in before['criteria']}
    new={r['criterion_id']:r['status'] for r in after['criteria']}
    unchanged=old==new;passed=unchanged and all(r['status']=='pass' for r in comparisons)
    return dict(status='pass' if passed else 'outside_tolerance',diagnosis_required=not passed,
        criteria_unchanged=unchanged,original_statuses=old,reproduced_statuses=new,
        max_relative_episode_cost_difference=(None if any(r['relative_cost_difference'] is None for r in comparisons)
            else max(r['relative_cost_difference'] for r in comparisons)),
        unresolved_zero_denominators=sum(r['relative_cost_difference'] is None for r in comparisons),
        reproduced_episodes=480,repeated_training_seeds=[0,4],new_independent_training_clusters=0,
        look_index=look,alpha_spent=0,recomputed=after,
        scope='Fixed-seed reproducibility diagnostic using the original accepted numerical envelope; no new scientific look')


def _observe_policy(root,job,path):
    from .control_replay import _read_episode
    request=read_json(safe_path(root,job['path']));parent=safe_path(root,job['parent_manifest'])
    if (sha256_file(parent)!=job['parent_manifest_sha256']
            or sha256_file(safe_path(root,job['path']))!=job['sha256']):
        raise ValueError('Changed repeated control request or original physical episode')
    original_request=read_json(parent.parent/'request.json')
    original=_read_episode(root,parent,'confirmatory',original_request)
    fresh=_read_episode(root,path,'replication',request)
    comparison=compare_control(original['summary'],fresh['summary'])
    comparison['decision_failure_classifications_unchanged']=original['decision_statuses']==fresh['decision_statuses']
    if not comparison['decision_failure_classifications_unchanged']:
        comparison.update(status='outside_tolerance',diagnosis_required=True)
    return dict(cell_id=job['cell_id'],manifest=str(path.relative_to(root)),manifest_sha256=sha256_file(path),
        comparison=comparison,summary=fresh['summary'],resources=fresh['resources'])


def prepare(root):
    root=Path(root).resolve();fits=read_json(root/'reports/control_retraining.json')
    if (fits.get('state')!='complete' or fits.get('completed_fits')!=12
            or fits.get('expected_fits')!=12 or fits.get('diagnosis_required') is not False):
        raise ValueError('Complete qualified twelve-fit training reproduction required before control')
    from .control_retraining import run_fits,verify_registry as verify_training
    training=read_json(root/'config/control_retraining_jobs.json');verify_training(root,training)
    run_fits(root,training,sha256_file(root/'config/control_retraining_jobs.json'),resume=True)
    target=root/'config/control_retrained_policy_jobs.json'
    if target.exists():return verify_registry(root,read_json(target))
    primary=read_json(root/'config/control_confirmation_cells.json')
    from .control_confirmation import verify_freeze
    verify_freeze(root,read_json(root/'config/control_confirmation_design.json'),primary)
    batch=read_json(root/'reports/control_confirmation_batch.json')
    if (batch['state']!='complete' or batch['completed_episodes']!=1200
            or batch['registry_sha256']!=sha256_file(root/'config/control_confirmation_cells.json')):
        raise ValueError('Complete original primary control required for repeated policy')
    parents={r['cell_id']:r for r in batch['episodes']}
    if len(parents)!=1200 or set(parents)!={j['cell_id'] for j in primary['jobs']}:
        raise ValueError('Complete unique original paired control coverage required')
    result=read_json(root/'reports/control_confirmation_result.json')
    numerical=root/'reports/control_confirmation_numerical.json'
    if sha256_file(numerical)!=result['evidence_identity']['numerical_sha256']:
        raise ValueError('Changed original numerical qualification before control reproduction')
    for name,digest in result['input_hashes'].items():
        if sha256_file(safe_path(root,name))!=digest:raise ValueError('Changed original signed-result input')
    from .control_jobs import SOURCES
    from .control_replay import _read_episode
    names=set(SOURCES)|{'poc/control_retrained_policy.py','poc/control_retraining.py','poc/control_replay.py',
        'poc/control_utility.py','poc/control_audit_closure.py','poc/statistics.py','config/control_retraining_jobs.json',
        'reports/control_retraining.json','config/control_confirmation_cells.json',
        'reports/control_confirmation_batch.json','reports/control_confirmation_result.json',
        'reports/control_confirmation_numerical.json','config/control_confirmation_design.json'}
    names.update(result['input_hashes']);jobs=[]
    for row in retrained_requests(root,primary,fits):
        parent=parents[row['parent_cell_id']];path=safe_path(root,parent['manifest'])
        if sha256_file(path)!=parent['manifest_sha256']:raise ValueError('Changed original paired episode')
        _read_episode(root,path,'confirmatory',read_json(path.parent/'request.json'))
        request_path=root/'config/control_jobs'/f"{row['cell_id'].replace('/','__')}.json"
        if request_path.exists() and read_json(request_path)!=row['request']:
            raise ValueError('Conflicting repeated-policy request')
        atomic_json(request_path,row['request'])
        jobs.append({k:v for k,v in row.items() if k!='request'}|dict(path=str(request_path.relative_to(root)),
            sha256=sha256_file(request_path),parent_manifest=parent['manifest'],
            parent_manifest_sha256=parent['manifest_sha256']))
        names.add(str(request_path.relative_to(root)));names.add(parent['manifest'])
        fresh=safe_path(root,row['request']['model_manifest'])
        names.add(row['request']['model_manifest'])
        context=fresh.parents[3].relative_to(root)
        names.update(str(context/n) for n in read_json(fresh)['job_inputs']['dependencies'])
        names.update(str(context/item['path']) for item in read_json(fresh)['produced_files'])
        names.add(str((fresh.parent/'COMPLETED').relative_to(root)))
        for name in ('weights.pt','summary.json','normalization.json','selection.json'):
            names.add(str((fresh.parent/name).relative_to(root)))
        names.update(item['path'] for item in read_json(path)['produced_files'])
    chosen=[parents[j['parent_cell_id']] for j in jobs]
    registry=dict(state='complete',expected_episodes=480,jobs=jobs,input_hashes={n:sha256_file(root/n) for n in sorted(names)},
        new_independent_training_clusters=0,new_statistical_look=False,resource_projection=dict(
            estimated_producer_wall_seconds=sum(r['resources']['wall_seconds'] for r in chosen),
            estimated_artifact_bytes=sum(r['resources']['disk_bytes'] for r in chosen),
            interpretation='Original matched-episode measured proxy; fresh process and hashing costs additional'))
    atomic_json(target,registry);return registry


def verify_registry(root,registry):
    root=Path(root).resolve()
    for name,digest in registry['input_hashes'].items():
        if sha256_file(safe_path(root,name))!=digest:raise ValueError('Changed repeated-policy dependency: '+name)
    expected=retrained_requests(root,read_json(root/'config/control_confirmation_cells.json'),
        read_json(root/'reports/control_retraining.json'))
    if registry.get('state')!='complete' or registry.get('expected_episodes')!=480 or len(registry.get('jobs',[]))!=480:
        raise ValueError('Complete fixed repeated-policy registry required')
    for row,job in zip(expected,registry['jobs']):
        if (any(job.get(k)!=row[k] for k in ('cell_id','parent_cell_id','parent_request_sha256','fresh_fit_cell_id'))
                or sha256_file(safe_path(root,job['path']))!=job['sha256']
                or read_json(safe_path(root,job['path']))!=row['request']
                or sha256_file(safe_path(root,job['parent_manifest']))!=job['parent_manifest_sha256']):
            raise ValueError('Changed repeated-policy request or parent')
    return registry


def run_policies(root,registry,registry_hash,*,resume=False):
    root=Path(root).resolve();target=root/'reports/control_retrained_policy_batch.json';episodes=[]
    jobs={j['cell_id']:j for j in registry['jobs']}
    if len(jobs)!=len(registry['jobs']):raise ValueError('Duplicate repeated-policy cell')
    def save(state,**extra):
        receipt=dict(state=state,registry_sha256=registry_hash,expected_episodes=len(jobs),
            completed_episodes=len(episodes),episodes=episodes,criterion_stability_checked=False,**extra)
        atomic_json(target,receipt);return receipt
    with job_lock(job_key({'fixed_retrained_policy':'seeds0and4alltasks'}),root):
        verify_registry(root,registry);reused={}
        if target.exists():
            previous=read_json(target)
            if not resume or previous['registry_sha256']!=registry_hash:
                raise ValueError('Existing repeated policy requires compatible --resume')
            for item in previous['episodes']:
                cell=item['cell_id'];path=safe_path(root,item['manifest'])
                if cell not in jobs or cell in reused or sha256_file(path)!=item['manifest_sha256']:
                    raise ValueError('Changed or duplicate completed repeated policy')
                if _observe_policy(root,jobs[cell],path)!=item:
                    raise ValueError('Repeated-policy receipt differs from raw observation')
                reused[cell]=item
            if previous['state']=='complete':
                if set(reused)!=set(jobs) or previous['completed_episodes']!=len(jobs):
                    raise ValueError('Incomplete completed repeated-policy batch')
                return previous
        try:
            for i,job in enumerate(registry['jobs']):
                if job['cell_id'] in reused:episodes.append(reused[job['cell_id']]);continue
                for name,digest in registry['input_hashes'].items():
                    if sha256_file(safe_path(root,name))!=digest:raise ValueError('Changed frozen repeated-policy input')
                print(f"Fresh-weight control {i+1}/{len(jobs)}: {job['cell_id']}",flush=True)
                child=subprocess.run([sys.executable,'-m','poc.control_jobs',job['path']],cwd=root,capture_output=True,text=True)
                attempt=root/'reports/control_retrained_policy_attempts'/f'{i:04d}-{utc_now().replace(":","")}'
                attempt.mkdir(parents=True);(attempt/'stdout.log').write_text(child.stdout);(attempt/'stderr.log').write_text(child.stderr)
                if child.returncode:raise RuntimeError('Repeated policy incomplete: '+child.stderr[-2000:])
                path=Path(child.stdout.strip().splitlines()[-1]).resolve()
                episodes.append(_observe_policy(root,job,path));save('running')
            return save('complete',diagnosis_required=any(e['comparison']['diagnosis_required'] for e in episodes))
        except BaseException as error:
            save('interrupted',reason=f'{type(error).__name__}: {error}');raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--dry-run',action='store_true');parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv);root=args.root.resolve()
    if args.dry_run:
        print(dict(execute=False,expected_episodes=480,training_seeds=[0,4],tasks_per_seed=20,
            prerequisite='Complete twelve-fit fresh-training comparison',new_independent_training_clusters=0,
            new_statistical_look=False));return 0
    if sys.version_info[:2]!=(3,11):raise ValueError('Pinned CPython 3.11 required for repeated policy')
    from .study_reproduce import validate_runtime_versions
    versions=read_json(root/'locks/environment.json')['versions']
    validate_runtime_versions(versions,{n:importlib.metadata.version(n) for n in versions})
    authorization=read_json(root/'config/authorization.json')
    if authorization.get('local_compute_authorized') is not True or authorization.get('paid_services_authorized') is not False:
        raise ValueError('Existing authorized local compute required for repeated policy')
    path=root/'config/control_retrained_policy_jobs.json'
    registry=prepare(root) if args.prepare else verify_registry(root,read_json(path))
    if args.prepare and not args.resume:return 0
    result=run_policies(root,registry,sha256_file(path),resume=args.resume)
    from .control_audit_closure import verify_primary_audit
    primary=[entry['summary'] for entry in verify_primary_audit(root)['entries']]
    original=read_json(root/'reports/control_confirmation_result.json')
    numerics=read_json(root/'reports/control_confirmation_numerical.json')
    audit=criterion_stability(primary,[r['summary'] for r in result['episodes']],
        read_json(root/'config/control_confirmation_design.json'),original,numerics['envelopes'])
    audit['decision_failure_classifications_unchanged']=not result.get('diagnosis_required',True)
    if not audit['decision_failure_classifications_unchanged']:
        audit.update(status='outside_tolerance',diagnosis_required=True)
    audit.update(registry_sha256=sha256_file(path),batch_sha256=sha256_file(root/'reports/control_retrained_policy_batch.json'),
        input_hashes=registry['input_hashes'],original_ptco_terminal_verdict=None)
    atomic_json(root/'reports/control_retrained_policy_result.json',audit)
    return 1 if audit['diagnosis_required'] else 0


if __name__=='__main__':
    raise SystemExit(main())
