"""Paired numerical sensitivity for the registered control workload."""
from __future__ import annotations
import math
import numpy as np
from .control_confirmation import ACTUATORS, MODELS, STEPS, validate_design


def _coordinate(row):
    partition=row.get('model_partition_ns', [])
    return (row['actuator'],row['model_id'],partition[0] if partition else None,
            row['seed'],row['episode_id'])


def compare_refinement(primary: list[dict], refined: list[dict], design: dict, *,
                       scope='subset', near_boundary_actuators=()) -> dict:
    """Compare complete paired runs; this is qualification, never a statistical look."""
    validate_design(design)
    if scope not in ('subset','full') or not set(near_boundary_actuators)<=set(ACTUATORS):
        raise ValueError('Unknown numerical scope or near-boundary plant')
    seeds=design['initial_training_seed_ids']
    episode_ids=design['initial_episode_ids'] if scope=='full' else list(range(5))
    expected={(a,m,h,s,e) for a in ACTUATORS for m in MODELS for h in STEPS
              for s in seeds for e in episode_ids}
    from .control_episode import episode_conditions
    def indexed(rows,tick):
        result={}
        for row in rows:
            key=_coordinate(row)
            a,m,h,s,e=key
            if key in result:
                raise ValueError('Duplicated numerical episode identity')
            if (a not in ACTUATORS or m not in MODELS or h not in STEPS or s not in seeds
                    or e not in design['initial_episode_ids'] or row.get('complete') is not True
                    or row['phase']!='confirmatory' or row['condition']!='clean'
                    or row['physics_step_ns']!=tick or row['actuator_interval_ns']!=40_000_000
                    or row['model_partition_ns']!=[h]*(40_000_000//h) or row['power_mode'] is not None
                    or type(row['success']) is not bool or type(row['fallback_count']) is not int
                    or row['fallback_count']<0 or not math.isfinite(row['realized_cost'])
                    or row['realized_cost']<0):
                raise ValueError('Changed or invalid numerical control workload')
            task=episode_conditions(s,e,a,'confirmatory')
            if (row['initial_state']!=task['initial_state'].tolist() or row['target']!=task['target'].tolist()
                    or row['seed_namespaces']!={'cem':task['cem_seed_entropy'],
                        'disturbance':task['disturbance_seed_entropy']}):
                raise ValueError('Changed numerical workload seed namespace, initial state or target')
            result[key]=row
        return result
    p,r=indexed(primary,250_000),indexed(refined,125_000)
    if set(r)!=expected or not expected<=set(p):
        raise ValueError('Complete paired numerical episode coverage required')
    discrepancies=[]; expand=set(near_boundary_actuators); max_error={}
    for key in sorted(expected):
        original,finer=p[key],r[key]
        if original['checkpoint_hash']!=finer['checkpoint_hash']:
            raise ValueError('Numerical refinement changed the selected checkpoint')
        a,m,h,s,e=key
        delta=abs(finer['realized_cost']-original['realized_cost'])
        fraction=delta/original['realized_cost'] if original['realized_cost']>0 else (0. if delta==0 else None)
        changed_success=original['success']!=finer['success']
        changed_fallback=original['fallback_count']!=finer['fallback_count']
        if fraction is None or fraction>.02 or changed_success or changed_fallback:
            expand.add(a)
        max_error[(a,m,h)]=max(max_error.get((a,m,h),0.),delta)
        discrepancies.append(dict(actuator=a,model_id=m,model_step_ns=h,seed=s,episode_id=e,
            absolute_cost_difference=delta,relative_cost_difference=fraction,
            success_changed=changed_success,fallback_changed=changed_fallback))
    effects=[];envelopes={}
    for criterion in design['criteria']:
        a,m,h=criterion['actuator'],criterion['competitor'],criterion['model_step_ns']
        def means(index):
            return [math.fsum(index[(a,model,h,s,e)]['realized_cost'] for s in seeds for e in episode_ids)
                    /len(seeds)/len(episode_ids) for model in (m,'power_correct')]
        b,c=means(p);bf,cf=means(r)
        improvement=1-c/b if b>0 else None
        finer_improvement=1-cf/bf if bf>0 else None
        movement=abs(finer_improvement-improvement) if improvement is not None and finer_improvement is not None else None
        if movement is None or movement>.02:
            expand.add(a)
        eta=max_error[(a,m,h)]+max_error[(a,'power_correct',h)]
        envelopes[criterion['criterion_id']]=dict(lower=eta,upper=eta)
        effects.append(dict(criterion_id=criterion['criterion_id'],primary_reduction=improvement,
            finer_reduction=finer_improvement,absolute_reduction_change=movement))
    return dict(scope=scope,audited_episode_pairs=len(expected),discrepancies=discrepancies,
        effect_changes=effects,envelopes=envelopes,full_cells_required=sorted(expand),
        qualification='expand_full_affected_cells' if expand else 'subset_within_numerical_budgets',
        scientific_decision='not_evaluated',
        envelope_interpretation='empirical_discrepancy_not_certified_error_bound')


def solver_probe(model, state, command, horizon_ns: int) -> dict:
    """Audit the same NODE field at 1/2/4 RK4 substeps, without changing its weights."""
    import torch
    if (type(horizon_ns) is not int or horizon_ns not in (5_000_000,40_000_000)
            or state.dtype!=torch.float64 or command.dtype!=torch.float64
            or state.ndim!=2 or len(state)!=1 or command.shape!=(1,2)
            or not torch.isfinite(state).all() or not torch.isfinite(command).all()):
        raise ValueError('Solver audit requires one finite float64 registered state/command probe')
    next_states=[];jacobians=[];calls=[]
    for steps in (1,2,4):
        with torch.no_grad():
            predicted=model.predict(state,command,horizon_ns/1e9,solver_substeps=steps)
        jac=torch.autograd.functional.jacobian(
            lambda u:model.predict(state,u,horizon_ns/1e9,solver_substeps=steps).next_state[0],command)[:,0,:]
        if (predicted.vector_field_calls!=4*steps or not torch.isfinite(predicted.next_state).all()
                or not torch.isfinite(jac).all()):
            raise ValueError('Unresolved or invalid NODE solver probe')
        next_states.append(predicted.next_state[0].tolist())
        jacobians.append(jac.tolist());calls.append(predicted.vector_field_calls)
    differences=[]
    for i in (1,2):
        before,after=np.asarray(jacobians[i-1]),np.asarray(jacobians[i])
        deltas={}
        for component,part in [('q',slice(0,2)),('v',slice(2,4))]:
            norm=float(np.linalg.norm(after[part]))
            delta=float(np.linalg.norm(after[part]-before[part]))
            deltas[component]=dict(absolute_jacobian_difference=delta,refined_norm=norm,
                relative_difference=delta/norm if norm>1e-14 else None)
        differences.append(dict(coarse_substeps=2**(i-1),fine_substeps=2**i,
            maximum_state_difference=float(np.max(np.abs(np.asarray(next_states[i])-next_states[i-1]))),
            jacobian_differences=deltas))
    return dict(horizon_ns=horizon_ns,substeps=[1,2,4],next_states=next_states,
        command_jacobians=jacobians,vector_field_calls=calls,
        vector_field_calls_interpretation='per_prediction_forward',
        total_forward_vector_field_calls=2*sum(calls),differences=differences,
        scientific_decision='not_evaluated',scope='evaluation_only_same_learned_field')
