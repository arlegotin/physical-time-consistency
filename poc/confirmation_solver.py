"""Read-only same-field NODE refinement on the registered confirmation workload."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name]='1'

import numpy as np
import torch

from .io import atomic_json, job_lock, read_json, safe_path, sha256_file

AUDIT_SOURCES=('poc/confirmation_solver.py','poc/control_confirmation.py',
    'poc/control_confirmation_batch.py','poc/control_utility.py','poc/control_numerical.py','poc/statistics.py')


def validate_solver_design(design):
    expected=dict(schema_version=1,protocol_id='UTILITY-2026-10-02-control-v1',
        model_id='node_generic',phase='confirmatory',actuators=['direct','lagged'],
        training_seed_ids=list(range(5)),episode_ids=list(range(5)),
        horizons_ns=[5_000_000,40_000_000],substeps=[1,2,4],command_namespace=9104,
        candidate_decision_ids=[0,20,39])
    if design!=expected:
        raise ValueError('Changed prospectively registered confirmation solver audit')
    return design


def confirmation_subset(batch,design):
    from .control_utility import seed_costs
    if (batch.get('state')!='complete' or batch.get('expected_episodes')!=1200
            or batch.get('completed_episodes')!=1200 or len(batch.get('episodes',[]))!=1200):
        raise ValueError('Complete paired confirmation batch required for solver audit')
    seed_costs([entry['summary'] for entry in batch['episodes']],design)
    selected=[entry for entry in batch['episodes'] if entry['summary']['model_id']=='node_generic'
              and entry['summary']['episode_id']<5]
    return sorted(selected,key=lambda entry:(entry['summary']['actuator'],entry['summary']['seed'],
        entry['summary']['model_partition_ns'][0],entry['summary']['episode_id']))


class SolverResolution:
    """The exact selected field with a specified evaluation-only RK4 resolution."""
    def __init__(self,model,substeps):
        from .models.ode import NeuralODE
        if not isinstance(model,NeuralODE) or type(substeps) is not int or substeps not in (1,2,4):
            raise ValueError('Solver audit requires the selected NODE and registered RK4 resolution')
        self.model,self.substeps=model,substeps
        self.actuator=model.actuator

    def predict(self,state,command,h_s,history=None,power_mode=None):
        return self.model.predict(state,command,h_s,history=history,power_mode=power_mode,
                                  solver_substeps=self.substeps)


def candidate_elite_ids(costs,settings):
    """Replay each CEM iteration's finite elite selection and deterministic ties."""
    costs=np.asarray(costs,dtype=np.float64)
    count,iterations,elites=(settings[k] for k in ('candidate_count','iterations','elite_count'))
    if (any(type(v) is not int or v<=0 for v in (count,iterations,elites))
            or elites>count or costs.shape!=(count*iterations,)
            or np.isnan(costs).any() or np.isneginf(costs).any() or np.any(costs<0)):
        raise ValueError('Complete CEM iteration costs and valid elite settings required')
    result=[]
    for start in range(0,len(costs),count):
        ranked=sorted((i for i in range(start,start+count) if np.isfinite(costs[i])),
                      key=lambda i:(costs[i],i))
        result.append(ranked[:elites])
    return result


def maximum_solver_cost_shift(banks):
    """Every required bank comparison must resolve before a finite maximum exists."""
    values=[d['maximum_absolute_cost_difference'] for b in banks for d in b['differences']]
    if not values or any(v is not None and (type(v) not in (int,float) or not np.isfinite(v) or v<0) for v in values):
        raise ValueError('Complete finite or explicitly unresolved solver discrepancies required')
    return None if any(v is None for v in values) else max(values)


def audit_candidate_bank(model,state,target,knots,saved_costs,saved_statuses,saved_reasons,
                         selected_id,partition,planner_settings=None):
    """Rescore every original candidate without resampling or committing observations."""
    from .mpc import rollout_candidates
    bank=np.asarray(knots,dtype=np.float64);costs=np.asarray(saved_costs,dtype=np.float64)
    n=len(bank)
    if (state.dtype!=torch.float64 or costs.shape!=(n,) or len(saved_statuses)!=n
            or len(saved_reasons)!=n or np.isnan(costs).any() or np.isneginf(costs).any()):
        raise ValueError('Complete finite-status candidate rows required for solver audit')
    finite=np.isfinite(costs)
    if (np.any(costs[finite]<0) or any((status=='ok')!=bool(valid) or (not valid and not reason)
            for status,reason,valid in zip(saved_statuses,saved_reasons,finite))):
        raise ValueError('Saved solver candidates have inconsistent cost/failure classification')
    best=int(np.argmin(costs)) if finite.any() else None
    if selected_id!=best:
        raise ValueError('Saved selected candidate is not the deterministic best evaluated plan')
    runs=[rollout_candidates(SolverResolution(model,k),state,target,None,bank,partition,40_000_000)
          for k in (1,2,4)]
    first=runs[0]
    if (tuple(first['statuses'])!=tuple(saved_statuses)
            or not np.array_equal(np.isfinite(first['costs']),finite)
            or not np.allclose(first['costs'][finite],costs[finite],atol=1e-10,rtol=1e-10)):
        raise ValueError('Replayed original solver cost/status differs from the saved candidate bank')
    best_ids=[int(np.argmin(run['costs'])) if np.isfinite(run['costs']).any() else None for run in runs]
    settings=planner_settings or dict(candidate_count=n,iterations=1,elite_count=min(8,n))
    elites=[candidate_elite_ids(run['costs'],settings) for run in runs]
    differences=[]
    for coarse,fine in ((0,1),(1,2),(0,2)):
        before,after=runs[coarse],runs[fine]
        common=np.isfinite(before['costs']) & np.isfinite(after['costs'])
        changed=np.isfinite(before['costs'])!=np.isfinite(after['costs'])
        differences.append(dict(coarse_substeps=2**coarse,fine_substeps=2**fine,
            n_changed_failure_classifications=int(changed.sum()),
            maximum_absolute_cost_difference=float(np.max(np.abs(after['costs'][common]-before['costs'][common])))
                if common.any() else None,
            maximum_q_anchor_difference=float(np.max(np.abs(after['anchors'][common,:,:2]-before['anchors'][common,:,:2])))
                if common.any() else None,
            maximum_v_anchor_difference=float(np.max(np.abs(after['anchors'][common,:,2:4]-before['anchors'][common,:,2:4])))
                if common.any() else None))
    return dict(substeps=[1,2,4],best_candidate_ids=best_ids,saved_selected_id=selected_id,
        costs=[[float(v) if np.isfinite(v) else None for v in run['costs']] for run in runs],
        statuses=[list(run['statuses']) for run in runs],reasons=[list(run['reasons']) for run in runs],
        anchors=[np.where(np.isfinite(run['anchors']),run['anchors'],None).tolist() for run in runs],
        work=[run['work'] for run in runs],differences=differences,
        failed_candidate_count=[sum(status!='ok' for status in run['statuses']) for run in runs],
        elite_candidate_ids=elites,elite_settings={k:settings[k] for k in ('candidate_count','iterations','elite_count')},
        requires_closed_loop_solver_audit=len(set(best_ids))>1 or any(
            difference['n_changed_failure_classifications'] for difference in differences)
            or any(before!=after for before,after in zip(elites,elites[1:])),
        scientific_decision='not_evaluated')


def probe_saved_episode(model,folder,request,target):
    """No plant advancement, fresh sensor packets, candidate sampling or weight updates."""
    from .control_numerical import solver_probe
    folder=Path(folder)
    h=request['model_partition_ns'][0]
    if (request['model_id']!='node_generic' or request['phase']!='confirmatory'
            or request['actuator'] not in ('direct','lagged') or request['seed'] not in range(5)
            or request['episode_id'] not in range(5) or request['condition']!='clean'
            or request['physics_step_ns']!=250_000 or request['actuator_interval_ns']!=40_000_000
            or request['power_mode'] is not None or h not in (5_000_000,40_000_000)
            or request['model_partition_ns']!=[h]*(40_000_000//h)):
        raise ValueError('Changed registered confirmation solver episode identity')
    dimension=4 if request['actuator']=='direct' else 6
    expected_times=np.arange(0,1_600_000_000,40_000_000,dtype=np.int64)
    with np.load(folder/'observations.npz',allow_pickle=False) as data:
        states,times=data['state'].copy(),data['time_ns'].copy()
        if (states.shape!=(40,dimension) or not np.isfinite(states).all()
                or not np.array_equal(times,expected_times)
                or not np.array_equal(data['packet_id'],[f'obs-{t}' for t in expected_times])
                or data['noise_time_ns'].size):
            raise ValueError('Added or changed saved observation clocks/packets')
    decisions=[json.loads(line) for line in (folder/'decisions.jsonl').read_text().splitlines()]
    count=request['planner_settings']['candidate_count']*request['planner_settings']['iterations']
    with np.load(folder/'candidates.npz',allow_pickle=False) as data:
        banks={key:data[key].copy() for key in ('knots','costs','statuses','reasons','global_ids')}
    if (len(decisions)!=40 or banks['knots'].shape!=(40,count,6,2)
            or any(banks[key].shape!=(40,count) for key in ('costs','statuses','reasons','global_ids'))
            or not np.array_equal(banks['global_ids'],np.tile(np.arange(count),(40,1)))):
        raise ValueError('Incomplete or changed saved candidate banks/identities')
    for decision,time in zip(decisions,times):
        selected=decision['selected_global_id']
        index=int(time)//40_000_000
        if (decision['time_ns']!=int(time) or decision['observation_packet_id']!=f'obs-{time}'
                or decision['status'] not in ('ok','all_candidates_failed')):
            raise ValueError('Changed decision observation packet or physical clock')
        if decision['status']=='ok':
            if (type(selected) is not int or selected not in range(count)
                    or not np.array_equal(decision['selected_knots'],banks['knots'][index,selected])):
                raise ValueError('Selected plan differs from the saved candidate bank')
        elif selected is not None or not np.array_equal(decision['selected_knots'],np.zeros((6,2))):
            raise ValueError('Changed all-candidate-failure fallback plan')
    probes=[];rankings=[]
    a=0 if request['actuator']=='direct' else 1
    for index,(state,time) in enumerate(zip(states,times)):
        entropy=[9104,a,request['seed'],request['episode_id'],index]
        command=np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy))).uniform(-1.5,1.5,2)
        probe=solver_probe(model,torch.tensor(state[None],dtype=torch.float64),
                           torch.tensor(command[None],dtype=torch.float64),h)
        probes.append(dict(decision_id=index,time_ns=int(time),observation_packet_id=f'obs-{time}',
            command_seed_entropy=entropy,state=state.tolist(),command=command.tolist(),**probe))
        if index in (0,20,39):
            bank=audit_candidate_bank(model,torch.tensor(state,dtype=torch.float64),target,
                banks['knots'][index],banks['costs'][index],banks['statuses'][index],banks['reasons'][index],
                decisions[index]['selected_global_id'],tuple(request['model_partition_ns']),request['planner_settings'])
            rankings.append(dict(decision_id=index,time_ns=int(time),**bank))
    return probes,rankings


def _verify_inputs(root,registry):
    if registry.get('state')!='complete' or len(registry.get('episodes',[]))!=100:
        raise ValueError('Incomplete confirmation solver input freeze')
    for name,digest in registry['input_hashes'].items():
        if sha256_file(safe_path(root,name))!=digest:
            raise ValueError('Changed confirmation solver input: '+name)
    design=read_json(root/'config/control_confirmation_design.json')
    batch=read_json(root/'reports/control_confirmation_batch.json')
    if registry['episodes']!=confirmation_subset(batch,design):
        raise ValueError('Solver input freeze differs from its complete primary workload')
    from .control_confirmation import verify_freeze
    verify_freeze(root,design,read_json(root/'config/control_confirmation_cells.json'))
    from .control_confirmation_batch import episode_receipt
    for entry in registry['episodes']:
        job=registry['parent_jobs'][entry['cell_id']]
        path=safe_path(root,entry['manifest'])
        checked=episode_receipt(root,job,path)
        if any(entry.get(k)!=value for k,value in checked.items()):
            raise ValueError('Solver input episode differs from its completed observation')


def prepare(root):
    root=Path(root).resolve()
    validate_solver_design(read_json(root/'config/control_confirmation_solver_audit.json'))
    target=root/'config/control_confirmation_solver_inputs.json'
    if target.exists():
        registry=read_json(target);_verify_inputs(root,registry);return registry
    batch=read_json(root/'reports/control_confirmation_batch.json')
    if (batch.get('state')!='complete' or batch.get('expected_episodes')!=1200
            or batch.get('completed_episodes')!=1200 or len(batch.get('episodes',[]))!=1200):
        raise ValueError('Complete paired confirmation batch required for solver audit')
    design=read_json(root/'config/control_confirmation_design.json')
    entries=confirmation_subset(batch,design)
    from .control_confirmation import verify_freeze
    freeze=read_json(root/'config/control_confirmation_cells.json');verify_freeze(root,design,freeze)
    if batch['registry_sha256']!=sha256_file(root/'config/control_confirmation_cells.json'):
        raise ValueError('Primary solver inputs differ from their checkpoint freeze')
    jobs={job['cell_id']:job for job in freeze['jobs']}
    from .control_jobs import SOURCES
    from .control_confirmation_batch import episode_receipt
    from .control_utility import audit_raw_episode
    names=set(SOURCES)|set(AUDIT_SOURCES)|{'config/control_confirmation_solver_audit.json',
        'config/control_confirmation_design.json','config/control_confirmation_cells.json',
        'reports/control_confirmation_batch.json'}
    selected_jobs={}
    for entry in entries:
        path=safe_path(root,entry['manifest']);job=jobs[entry['cell_id']]
        checked=episode_receipt(root,job,path)
        if any(entry.get(k)!=value for k,value in checked.items()):
            raise ValueError('Saved primary episode differs from its verified input')
        audit_raw_episode(path.parent,checked['summary'])
        selected_jobs[entry['cell_id']]=job
        names.add(job['path']);names.add(entry['manifest'])
        names.update(read_json(path)['job_inputs']['dependencies'])
        for name in ('request.json','summary.json','observations.npz','candidates.npz',
                     'true_trajectories.npz','action_segments.jsonl','decisions.jsonl'):
            names.add(str((path.parent/name).relative_to(root)))
    registry=dict(schema_version=1,state='complete',episodes=entries,parent_jobs=selected_jobs,
        input_hashes={name:sha256_file(root/name) for name in sorted(names)},
        scientific_decision='not_evaluated',resource_projection=dict(
            n_episodes=100,n_local_probes=4000,n_saved_candidate_banks=300,
            local_forward_vector_field_calls=224000,
            candidate_work='Re-score each full bank at 1/2/4 RK4 substeps; measure first episode before projecting the remainder',
            training_updates=0))
    atomic_json(target,registry);return registry


def _episode_audit(root,entry,job,audit_design):
    from .execution import run_job
    from .control_jobs import SOURCES
    from .control_models import load_selected
    from .reference_jobs import tensor_hash
    path=safe_path(root,entry['manifest']);folder=path.parent
    request=read_json(folder/'request.json')
    model_path=safe_path(root,request['model_manifest'])
    files=set(SOURCES)|set(AUDIT_SOURCES)|{'config/control_confirmation_solver_inputs.json',
        'config/control_confirmation_solver_audit.json',entry['manifest'],job['path'],request['model_manifest']}
    for name in ('request.json','summary.json','observations.npz','candidates.npz',
                 'true_trajectories.npz','action_segments.jsonl','decisions.jsonl'):
        files.add(str((folder/name).relative_to(root)))
    for name in ('weights.pt','normalization.json'):
        files.add(str((model_path.parent/name).relative_to(root)))
    hashes={name:sha256_file(root/name) for name in files}
    checks=('control.solver_inputs','control.solver_weights','control.solver_work')
    def producer(output):
        model=load_selected(model_path,request['manifest_sha256'],request['checkpoint_sha256'],
            model_id='node_generic',actuator=request['actuator'],seed=request['seed'])
        before=tensor_hash(model.state_dict())
        local,banks=probe_saved_episode(model,folder,request,entry['summary']['target'])
        if (len(local)!=40 or len(banks)!=3 or tensor_hash(model.state_dict())!=before
                or any(sha256_file(root/name)!=digest for name,digest in hashes.items())):
            raise ValueError('Solver probes changed original inputs, selected weights or fixed coverage')
        atomic_json(output/'local_probes.json',dict(probes=local,command_namespace=9104))
        atomic_json(output/'candidate_banks.json',dict(banks=banks))
        summary=dict(cell_id=entry['cell_id'],actuator=request['actuator'],seed=request['seed'],
            episode_id=request['episode_id'],model_step_ns=request['model_partition_ns'][0],
            checkpoint_sha256=request['checkpoint_sha256'],n_local_probes=40,n_candidate_banks=3,
            local_forward_vector_field_calls=sum(row['total_forward_vector_field_calls'] for row in local),
            candidate_forward_vector_field_calls=sum(work['vector_field_calls'] for bank in banks for work in bank['work']),
            candidate_scalar_evaluations=sum(work['scalar_candidate_evaluations'] for bank in banks for work in bank['work']),
            n_ranking_or_failure_changes=sum(bank['requires_closed_loop_solver_audit'] for bank in banks),
            requires_closed_loop_solver_audit=any(bank['requires_closed_loop_solver_audit'] for bank in banks),
            maximum_absolute_cost_difference=maximum_solver_cost_shift(banks),
            scientific_decision='not_evaluated',missing_cells=[],
            scope='Same selected NODE field and saved input banks; no fresh observations or feedback execution')
        validity=[dict(check_id=key,status='pass',scope=summary['scope'],tolerance=None,
            reason='Complete fixed observations/banks, matching original solver and unchanged tensors/source') for key in checks]
        return validity,summary
    record=run_job(root,'M70','confirmatory',sorted(files),checks,[],producer,
        config_keys={'config/control_confirmation_solver_audit.json':audit_design},
        argv=['-m','poc.confirmation_solver'])
    if record['completion_status']!='complete':
        raise RuntimeError('Confirmation solver audit incomplete: '+record['run_id'])
    manifest=root/'evidence/M70'/record['run_id']/'manifest.json'
    return dict(parent_cell_id=entry['cell_id'],manifest=str(manifest.relative_to(root)),
        manifest_sha256=sha256_file(manifest),summary=read_json(manifest.parent/'summary.json'),
        resources=read_json(manifest.parent/'resources.json'))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv);root=args.root.resolve()
    audit_design=validate_solver_design(read_json(root/'config/control_confirmation_solver_audit.json'))
    if args.dry_run:
        print(dict(execute=False,episodes=100,local_probes=4000,candidate_banks=300,
            substeps=[1,2,4],prerequisite='Complete verified primary confirmation batch',
            scientific_decision='not_evaluated'));return 0
    registry=prepare(root)
    if args.prepare:
        print(root/'config/control_confirmation_solver_inputs.json');return 0
    report=root/'reports/control_confirmation_solver.json'
    if report.exists() and not args.resume:
        raise ValueError('Existing confirmation solver receipt requires --resume')
    from .execution import cpu_defaults
    cpu_defaults(0)
    key=hashlib.sha256(b'registered confirmation NODE solver audit').hexdigest()
    identity=sha256_file(root/'config/control_confirmation_solver_inputs.json')
    episodes=[]
    def save(state,**extra):
        result=dict(schema_version=1,state=state,input_registry_sha256=identity,expected_episodes=100,
            completed_episodes=len(episodes),episodes=episodes,scientific_decision='not_evaluated',**extra)
        atomic_json(report,result);return result
    with job_lock(key,root):
        if report.exists() and read_json(report)['input_registry_sha256']!=identity:
            raise ValueError('Changed solver input registry is not a resumption')
        try:
            for index,entry in enumerate(registry['episodes']):
                if sha256_file(root/'config/control_confirmation_solver_inputs.json')!=identity:
                    raise ValueError('Changed solver input freeze during execution')
                for name,digest in registry['input_hashes'].items():
                    if name.startswith('poc/') and sha256_file(safe_path(root,name))!=digest:
                        raise ValueError('Changed frozen solver source: '+name)
                print(f"Confirmation NODE solver audit {index+1}/100: {entry['cell_id']}",flush=True)
                episode=_episode_audit(root,entry,registry['parent_jobs'][entry['cell_id']],audit_design)
                episodes.append(episode);save('running')
            save('complete',closed_loop_solver_cells_required=[row['parent_cell_id'] for row in episodes
                if row['summary']['requires_closed_loop_solver_audit']])
        except BaseException as error:
            save('interrupted',reason=f'{type(error).__name__}: {error}');raise
    print(report);return 0


if __name__=='__main__':
    raise SystemExit(main())
