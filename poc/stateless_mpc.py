"""Common CEM for fully observed stateless models, with vectorized held paths.

The original controller is retained as the independently exercised reference.
This specialization removes history/path object construction inside substeps;
it changes neither candidate sampling nor physical cost anchors.
"""
import numpy as np
import torch
from .action_path import compile_knots, path_checksum
from .affine_flow import AffineFlow
from .models.ode import NeuralODE
from .mpc import _partition, _settings, WORK_KEYS


def rollout_candidates(model, state, target, history, knots, model_partition,
                       actuator_interval_ns, *, power_mode=None):
    steps = _partition(model_partition)
    if (actuator_interval_ns != 40_000_000 or power_mode is not None
            or not (isinstance(model, (AffineFlow, NeuralODE)) or getattr(model,'stateless_query',False))):
        raise ValueError('Only declared stateless models and forty-ms held actuation are supported')
    if history is not None and len(history.tokens) != 1:
        raise ValueError('One real history branch required')
    bank = np.asarray(knots, dtype=np.float64)
    if (bank.ndim != 3 or bank.shape[1:] != (6,2) or len(bank)==0
            or not np.isfinite(bank).all() or (np.abs(bank)>2).any()):
        raise ValueError('Finite bounded six-knot candidates required')
    x0 = state.reshape(-1)
    target = np.asarray(target,dtype=np.float64)
    if (x0.shape not in ((4,),(6,)) or x0.device.type!='cpu' or not torch.isfinite(x0).all()
            or target.shape!=(2,) or not np.isfinite(target).all()):
        raise ValueError('Finite CPU state and two-joint target required')
    count = len(bank); x = x0[None].repeat(count,1)
    anchors = np.full((count,6,len(x0)),np.nan); failed=np.zeros(count,dtype=bool)
    reasons=['']*count; work={key:0 for key in WORK_KEYS}
    commands=torch.as_tensor(bank,dtype=x.dtype)
    with torch.no_grad():
        for anchor in range(6):
            for h in steps:
                try:
                    packet=model.predict(x,commands[:,anchor],h/1e9,history=None)
                except (RuntimeError,ValueError) as error:
                    for i in np.flatnonzero(~failed):reasons[i]=f'{type(error).__name__}: {error}'
                    failed[:]=True;break
                if packet.next_state.shape!=x.shape or len(packet.status)!=count:
                    raise ValueError('Model violated candidate state/status contract')
                for key in WORK_KEYS:work[key]+=getattr(packet,key)
                finite=torch.isfinite(packet.next_state).all(1).numpy()
                guarded=(packet.next_state[:,:4].abs()>1000).any(1).numpy()
                for i,status in enumerate(packet.status):
                    if not failed[i] and (status!='ok' or not finite[i] or guarded[i]):
                        failed[i]=True
                        reasons[i]='imagined_state_guard' if guarded[i] else 'nonfinite_imagined_state' if not finite[i] else str(status)
                x=torch.where(torch.as_tensor(failed)[:,None],torch.zeros_like(x),packet.next_state)
            anchors[~failed,anchor]=x[~torch.as_tensor(failed)].numpy()
    error=anchors[:,:,:2]-target
    # Same six endpoint anchors and one terminal charge; no substep costs.
    costs=.04*np.sum(error**2,axis=(1,2))+.002*np.sum(anchors[:,:,2:4]**2,axis=(1,2))
    costs+=.00004*np.sum(bank**2,axis=(1,2))+5*np.sum(error[:,-1]**2,axis=1)
    costs[failed]=np.inf
    return dict(costs=costs,anchors=anchors,statuses=tuple('model_failure' if f else 'ok' for f in failed),
                reasons=tuple(reasons),path_checksums=tuple(path_checksum(compile_knots(k,40_000_000)) for k in bank),work=work)


def search(model,state,target,history,previous_knots,standard_normals,settings,
           model_partition,actuator_interval_ns,*,power_mode=None):
    cfg=_settings(settings); steps=_partition(model_partition)
    count,iters=cfg['candidate_count'],cfg['iterations']
    samples=np.asarray(standard_normals,dtype=np.float64)
    if samples.shape!=(iters,count,6,2) or not np.isfinite(samples).all():
        raise ValueError('Complete fixed CEM normal bank required')
    mean=np.zeros((6,2))
    if previous_knots is not None:
        compile_knots(previous_knots,actuator_interval_ns)
        mean=np.concatenate((previous_knots[1:],previous_knots[-1:]))
    std=np.full((6,2),cfg['initial_std']); best=None
    proposals=[]; banks=[]; costs=[]; statuses=[]; reasons=[]; checksums=[]
    work={key:0 for key in WORK_KEYS}
    for iteration in range(iters):
        bank=np.clip(mean[None]+std[None]*samples[iteration],-2,2)
        bank[0],bank[1]=np.clip(mean,-2,2),0
        out=rollout_candidates(model,state,target,history,bank,steps,actuator_interval_ns,power_mode=power_mode)
        values=out['costs']; finite=[]
        for i,(cost,status,reason) in enumerate(zip(values,out['statuses'],out['reasons'])):
            if np.isfinite(cost):
                if status!='ok':raise ValueError('Failed candidate has finite cost')
                finite.append(i);key=(float(cost),iteration,i)
                if best is None or key<best['key']:
                    best=dict(key=key,knots=bank[i].copy(),anchors=out['anchors'][i].copy(),path_checksum=out['path_checksums'][i])
            elif not (cost==np.inf and status!='ok' and reason):
                raise ValueError('Nonfinite candidate lacks explicit failure')
        finite.sort(key=lambda i:(values[i],i)); elite=finite[:cfg['elite_count']]
        old_mean,old_std=mean.copy(),std.copy()
        if elite:
            mean=bank[elite].mean(0);std=np.maximum(bank[elite].std(0,ddof=0),cfg['std_floor'])
        proposals.append(dict(iteration=iteration,mean=old_mean,std=old_std,next_mean=mean.copy(),next_std=std.copy(),
                              elite_global_ids=[iteration*count+i for i in elite]))
        banks.append(bank.copy()); costs.append(values.copy()); statuses.extend(out['statuses']);reasons.extend(out['reasons']);checksums.extend(out['path_checksums'])
        for key in work:work[key]+=out['work'][key]
    candidates=dict(knots=np.concatenate(banks),costs=np.concatenate(costs),statuses=tuple(statuses),reasons=tuple(reasons),
                    path_checksums=tuple(checksums),global_ids=np.arange(iters*count))
    common=dict(proposals=proposals,candidates=candidates,work=work)
    if best is None:
        return dict(status='all_candidates_failed',fallback='zero_command',cost=None,knots=np.zeros((6,2)),
                    selected_global_id=None,predicted_anchors=None,**common)
    cost,iteration,local=best['key']
    return dict(status='ok',fallback=None,cost=cost,knots=best['knots'],selected_global_id=iteration*count+local,
                predicted_anchors=best['anchors'],path_checksum=best['path_checksum'],**common)
