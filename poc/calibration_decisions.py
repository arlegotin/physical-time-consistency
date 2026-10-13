"""Shared finite-bank decisions on separate physical cost and feasibility clocks."""
from __future__ import annotations

import hashlib
from pathlib import Path
import time

import numpy as np
import torch
from torch import Tensor, nn

from .calibration_data import rng_for
from .calibration_predictions import (_recipe, _arrays, _save_arrays, _selected, load_evaluation)
from .calibration_records import write_records
from .calibration_store import CalibrationStore, Receipt
from .io import atomic_bytes, canonical_json
from .mpc import WORK_KEYS

MAIN_STEPS_NS = (5_000_000, 10_000_000, 20_000_000, 40_000_000)
TICK_NS = 5_000_000
KNOT_NS = 40_000_000


def decision_state_ids(arrays: dict, recipe: dict) -> list[int]:
    """Select by trajectory/primary/time identity, including reduced unit anchors."""
    selected = []
    for trajectory in range(recipe['evaluation']['trajectory_count']):
        eligible = np.flatnonzero((arrays['trajectory_id'] == trajectory) & arrays['primary'] &
                                  (arrays['time_ns'] >= recipe['decisions']['anchor_ns']))
        if not len(eligible):
            raise ValueError('Missing planned primary decision anchor')
        first_time = np.min(arrays['time_ns'][eligible])
        first = eligible[arrays['time_ns'][eligible] == first_time]
        if len(first) != 1:
            raise ValueError('Duplicate primary decision identity')
        selected.append(int(first[0]))
    return selected


def _bank(arrays, state_id, recipe):
    d, e = recipe['decisions'], recipe['evaluation']
    trajectory = int(arrays['trajectory_id'][state_id])
    tick = int(arrays['time_ns'][state_id]) // KNOT_NS
    generator = rng_for('development-evaluation', e['seed_session_id'], trajectory, d['bank_channel'])
    # One independent draw per observed clock coordinate; no fit/model coordinates.
    draws = generator.uniform(*d['random_command_bounds'], (tick+1, d['candidate_count']-2, 6, 2))
    bank = np.empty((d['candidate_count'], 6, 2), dtype=np.float64)
    bank[0] = 0.
    state = arrays['state'][state_id]
    bank[1] = np.clip(-state[:2]-.25*state[2:4], *d['random_command_bounds'])
    bank[2:] = draws[-1]
    return bank


def candidate_bank(evaluation_receipt: Receipt, state_id: int, store: CalibrationStore) -> np.ndarray:
    arrays = load_evaluation(store, evaluation_receipt)
    if type(state_id) is not int or state_id not in decision_state_ids(arrays, store.recipe):
        raise ValueError('Candidate bank requires a planned primary decision identity')
    return _bank(arrays, state_id, store.recipe)


def _cost(anchors, target, bank):
    error = anchors[..., :2]-target
    return (.04*np.sum(error**2, axis=(-2,-1)) + .002*np.sum(anchors[...,2:4]**2, axis=(-2,-1)) +
            .00004*np.sum(bank**2, axis=(-2,-1)) + 5*np.sum(error[..., -1,:]**2, axis=-1))


def _safe(states):
    return (np.isfinite(states).all(axis=-1) & (np.abs(states[..., :2]) <= 1.5).all(axis=-1) &
            (np.abs(states[..., 2:4]) <= 3.).all(axis=-1))


def _empty_score(count, status):
    return {'costs': np.full(count,np.inf), 'anchors': np.full((count,6,6),np.nan),
            'samples':np.full((count,49,6),np.nan), 'statuses':[status]*count,
            'sampled_feasible':np.zeros(count,bool), 'initial_feasible':False,
            'selected_index':None, 'all_predicted_infeasible':True,
            'query_log':[], 'work':{k:0 for k in (*WORK_KEYS,'prediction_batch_calls',
                'auxiliary_scalar_predictions','state_feasibility_checks','command_feasibility_checks')},
            'wall_s':0., 'cpu_s':0.}


def score_candidates(model: nn.Module, state: Tensor, target: np.ndarray, bank: np.ndarray,
                     *, query_step_ns: int) -> dict:
    """Query every 5 ms offset from its main-step start in a shared packed batch.

    All query endpoints participate in sampled constraints, but only a full-step
    endpoint advances the path. Six physical 40 ms endpoints incur cost.
    """
    wall, cpu = time.perf_counter(), time.process_time()
    bank, target = np.asarray(bank,dtype=np.float64), np.asarray(target,dtype=np.float64)
    if (type(query_step_ns) is not int or query_step_ns not in MAIN_STEPS_NS or bank.ndim != 3 or
            bank.shape[1:] != (6,2) or not len(bank) or not np.isfinite(bank).all() or
            target.shape != (2,) or not np.isfinite(target).all() or state.shape != (6,) or
            state.device.type != 'cpu' or state.dtype != torch.float64 or not torch.isfinite(state).all()):
        raise ValueError('Finite float64 CPU lagged state, target, six-knot bank and supported partition required')
    n = len(bank)
    out = _empty_score(n, 'ok')
    work = out['work']
    x = state.detach().clone()[None].repeat(n,1)
    out['samples'][:,0] = x.numpy()
    out['initial_feasible'] = bool(_safe(x.numpy()).all())
    feasible = np.full(n,out['initial_feasible']) & (np.abs(bank) <= 2.).all(axis=(1,2))
    work['state_feasibility_checks'] = n
    work['command_feasibility_checks'] = n*6
    failed = np.zeros(n,bool)
    offsets = np.arange(TICK_NS,query_step_ns+1,TICK_NS,dtype=np.int64)
    h = torch.from_numpy(np.repeat(offsets/1e9,n)[:,None])
    with torch.inference_mode():
        for start in range(0,6*KNOT_NS,query_step_ns):
            commands = torch.from_numpy(bank[:,start//KNOT_NS].copy())
            before = x.repeat(len(offsets),1)
            u = commands.repeat(len(offsets),1)
            immutable_x, immutable_u = before.clone(), u.clone()
            work['prediction_batch_calls'] += 1
            work['auxiliary_scalar_predictions'] += (len(offsets)-1)*n
            out['query_log'].append({'start_ns':start,'endpoint_ns':start+query_step_ns,
                'offsets_ns':offsets.tolist(),'batch_size':len(before)})
            try:
                packet = model.predict(before,u,h)
            except (RuntimeError,ValueError,FloatingPointError) as error:
                # Failed attempted queries still consumed batch/scalar work.
                work['scalar_candidate_evaluations'] += len(before)
                for i in np.flatnonzero(~failed):
                    out['statuses'][i] = 'predictor_exception:'+type(error).__name__+':'+str(error)
                failed[:] = True
                break
            if not torch.equal(before,immutable_x) or not torch.equal(u,immutable_u):
                raise ValueError('Predictor mutated immutable query state/command')
            if packet.next_state.shape != before.shape or len(packet.status) != len(before):
                raise ValueError('Invalid predictor state/status contract')
            for key in WORK_KEYS:
                work[key] += int(getattr(packet,key))
            values = packet.next_state.detach().numpy().reshape(len(offsets),n,6)
            statuses = np.asarray(packet.status).reshape(len(offsets),n)
            good = np.isfinite(values).all(axis=(0,2)) & (statuses == 'ok').all(axis=0)
            for i in np.flatnonzero(~failed & ~good):
                out['statuses'][i] = ('nonfinite_prediction' if not np.isfinite(values[:,i]).all()
                                      else str(next(s for s in statuses[:,i] if s != 'ok')))
            failed |= ~good
            out['samples'][:,start//TICK_NS+1:(start+query_step_ns)//TICK_NS+1] = values.transpose(1,0,2)
            feasible &= _safe(values).all(axis=0) & ~failed
            work['state_feasibility_checks'] += len(before)
            # Reuse the endpoint already returned in this batch.
            x = torch.from_numpy(values[-1].copy())
            x[torch.from_numpy(failed)] = 0.
            if (start+query_step_ns) % KNOT_NS == 0:
                out['anchors'][:,(start+query_step_ns)//KNOT_NS-1] = values[-1]
    costs = _cost(out['anchors'],target,bank)
    bad_cost = ~np.isfinite(costs)
    for i in np.flatnonzero(bad_cost & ~failed):
        out['statuses'][i] = 'nonfinite_cost'
    failed |= bad_cost
    costs[failed] = np.inf
    feasible &= ~failed
    valid = np.flatnonzero(feasible)
    selected = int(valid[np.argmin(costs[valid])]) if len(valid) else None
    out.update(costs=costs,sampled_feasible=feasible,selected_index=selected,
               all_predicted_infeasible=selected is None,wall_s=time.perf_counter()-wall,cpu_s=time.process_time()-cpu)
    return out


def decision_outcome(predicted: dict, truth_cost: np.ndarray, truth_feasible: np.ndarray,
                     envelope: np.ndarray, feasibility_status: np.ndarray | None = None) -> dict:
    """Use independent truth only, retaining possible best ties and regret bounds."""
    cost, feasible, envelope = np.asarray(truth_cost), np.asarray(truth_feasible,bool), np.asarray(envelope)
    if (cost.ndim != 1 or feasible.shape != cost.shape or envelope.shape != cost.shape or
            not np.isfinite(cost).all() or not np.isfinite(envelope).all() or (envelope < 0).any()):
        raise ValueError('Complete finite truth costs and nonnegative numerical envelopes required')
    ambiguous = (np.zeros(cost.shape,bool) if feasibility_status is None else
                 np.asarray(feasibility_status) != 'resolved')
    if ambiguous.shape != cost.shape:
        raise ValueError('Feasibility status shape mismatch')
    unresolved_feasibility = bool(ambiguous.any())
    selected = predicted['selected_index']
    valid = np.flatnonzero(feasible)
    empty = not len(valid)
    unsafe = selected is not None and not feasible[selected]
    failure = selected is None or unsafe or empty or unresolved_feasibility
    lower, upper = cost-envelope, cost+envelope
    possible = valid[lower[valid] <= np.min(upper[valid])] if not empty else np.array([],int)
    if len(valid) > 1:
        best = int(valid[np.argmin(cost[valid])])
        resolved = all(cost[c]-cost[best] > 100*(envelope[c]+envelope[best])
                       for c in valid if c != best)
    else:
        resolved = not empty
    return {'selected_index':selected, 'selected_unsafe':None if selected is not None and ambiguous[selected] else bool(unsafe), 'no_selection':selected is None,
            'all_predicted_infeasible':selected is None, 'no_true_feasible_bank':None if unresolved_feasibility else empty,
            'truth_feasibility_status':'unresolved' if unresolved_feasibility else 'resolved',
            'decision_failure_bound':'conservative_upper' if unresolved_feasibility else 'resolved',
            'decision_failure':bool(failure), 'prediction_failure_count':sum(s != 'ok' for s in predicted['statuses']),
            'false_feasible_count':None if unresolved_feasibility else int(np.sum(predicted['sampled_feasible'] & ~feasible)),
            'false_infeasible_count':None if unresolved_feasibility else int(np.sum(~predicted['sampled_feasible'] & feasible)),
            'truth_best_indices':[] if unresolved_feasibility else possible.tolist(), 'truth_ranking_status':'unresolved' if unresolved_feasibility else 'no_feasible_bank' if empty else 'resolved' if resolved else 'unresolved',
            'regret':None if failure else float(cost[selected]-np.min(cost[valid])),
            'regret_lower':None if failure else float(max(0.,lower[selected]-np.min(upper[valid]))),
            'regret_upper':None if failure else float(max(0.,upper[selected]-np.min(lower[valid]))),
            'zero_command_truth_cost':float(cost[0])}


def _replay(plant, state, target, bank):
    samples = np.empty((49,6),dtype=np.float64)
    samples[0] = state
    plant.reset(state)
    maxima = np.abs(state[:4]).copy()
    dense_count = int(not _safe(state))
    first_violation = 0 if dense_count else -1
    for tick in range(48):
        packet = plant.advance(bank[tick//8],TICK_NS,log_every_step=True)
        trace = packet['trajectory']['state']
        maxima = np.maximum(maxima,np.max(np.abs(trace[:,:4]),axis=0))
        violated = ~_safe(trace)
        dense_count += int(violated.sum())
        if first_violation < 0 and violated.any():
            first_violation = int(packet['trajectory']['time_ns'][np.flatnonzero(violated)[0]])
        samples[tick+1] = packet['state']
    return {'cost':float(_cost(samples[8::8],target,bank)), 'samples':samples,
            'dense_max_abs':maxima,'dense_violation_count':dense_count,'dense_first_violation_ns':first_violation,
            'sampled_feasible':bool(_safe(samples).all() and (np.abs(bank) <= 2).all())}


def _truth_request(store,evaluation,recipe):
    return {'evaluation_receipt':evaluation,'recipe_receipt':store.recipe_receipt,
            'decisions':recipe['decisions'],'resolution_ns':[250_000,125_000,62_500],
            'adaptive_resolution_ns':recipe['evaluation']['adaptive_refinement_steps_ns']}


def truth_candidates(store: CalibrationStore, evaluation_receipt: Receipt, *, root: Path) -> Receipt:
    from .envs import Plant
    recipe = _recipe(store,root)
    arrays = load_evaluation(store,evaluation_receipt)
    ids = decision_state_ids(arrays,recipe)
    d = recipe['decisions']
    def produce(folder):
        shape = (len(ids),d['candidate_count'])
        result = {'state_id':np.asarray(ids,dtype=np.int64),
            'bank':np.stack([_bank(arrays,s,recipe) for s in ids]),
            'cost':np.empty(shape), 'base_cost':np.empty(shape), 'envelope':np.empty(shape),
            'samples':np.empty((*shape,49,6)), 'base_anchors':np.empty((*shape,6,6)),
            'sampled_feasible':np.empty(shape,bool),'dense_max_abs':np.empty((*shape,4)),
            'dense_violation_count':np.empty(shape,np.int64),'dense_first_violation_ns':np.empty(shape,np.int64),
            'resolution_ns':np.full(shape,62_500,np.int64), 'refinement_status':np.full(shape,'audited',dtype='<U32'),
            'feasibility_disagreement':np.zeros(shape,bool),
            'feasibility_status':np.full(shape,'resolved',dtype='<U16'),
            'feasibility_by_resolution':np.full((*shape,5),-1,dtype=np.int8), 'cost_by_resolution':np.full((*shape,5),np.nan)}
        plants = {h:Plant('lagged',h,Path(root)/'assets') for h in [250_000,125_000,62_500,31_250,15_625]}
        for row,s in enumerate(ids):
            target = arrays['target'][int(arrays['trajectory_id'][s])]
            records = []
            for c, bank in enumerate(result['bank'][row]):
                runs = [_replay(plants[h],arrays['state'][s],target,bank) for h in [250_000,125_000,62_500]]
                records.append(runs)
                result['base_cost'][row,c] = runs[0]['cost']
                result['base_anchors'][row,c] = runs[0]['samples'][8::8]
                result['cost_by_resolution'][row,c,:3] = [r['cost'] for r in runs]
                result['feasibility_by_resolution'][row,c,:3] = [r['sampled_feasible'] for r in runs]
            # Ranking uses simulator costs only. Refine close truly feasible contenders;
            # exact ties remain tied, with bounded refinement and plateau stopping.
            active = set(range(shape[1]))
            for level,h in enumerate([None,*recipe['evaluation']['adaptive_refinement_steps_ns']]):
                for c,runs in enumerate(records):
                    last = runs[-1]
                    env = max(abs(runs[-1]['cost']-runs[-2]['cost']), abs(runs[-2]['cost']-runs[-3]['cost']))
                    result['envelope'][row,c] = env
                    result['cost'][row,c] = last['cost']
                    result['sampled_feasible'][row,c] = last['sampled_feasible']
                feasible = np.flatnonzero(result['sampled_feasible'][row])
                close = set()
                if len(feasible)>1:
                    best = int(feasible[np.argmin(result['cost'][row,feasible])])
                    for c in feasible:
                        gap = abs(result['cost'][row,c]-result['cost'][row,best])
                        if c != best and gap <= 100*(result['envelope'][row,c]+result['envelope'][row,best]):
                            close.update([int(c),best])
                ambiguous = {c for c,runs in enumerate(records) if runs[-1]['sampled_feasible'] != runs[-2]['sampled_feasible']}
                if h is None:
                    active = close
                    continue
                for c in sorted((active & close) | ambiguous):
                    runs = records[c]
                    older = abs(runs[-2]['cost']-runs[-3]['cost'])
                    newer = abs(runs[-1]['cost']-runs[-2]['cost'])
                    if newer >= older and len(runs)>3 and c not in ambiguous:
                        result['refinement_status'][row,c] = 'plateau_unresolved'
                        active.discard(c)
                        continue
                    runs.append(_replay(plants[h],arrays['state'][s],target,result['bank'][row,c]))
                    result['cost_by_resolution'][row,c,level+2] = runs[-1]['cost']
                    result['feasibility_by_resolution'][row,c,level+2] = runs[-1]['sampled_feasible']
                    result['resolution_ns'][row,c] = h
                    result['refinement_status'][row,c] = 'refined'
            for c,runs in enumerate(records):
                last = runs[-1]
                result['cost'][row,c] = last['cost']
                result['envelope'][row,c] = max(abs(runs[-1]['cost']-runs[-2]['cost']),abs(runs[-2]['cost']-runs[-3]['cost']))
                result['samples'][row,c] = last['samples']
                result['sampled_feasible'][row,c] = last['sampled_feasible']
                result['feasibility_status'][row,c] = ('resolved' if runs[-1]['sampled_feasible']==runs[-2]['sampled_feasible'] else 'unresolved')
                result['feasibility_disagreement'][row,c] = len({r['sampled_feasible'] for r in runs})>1
                for key in ['dense_max_abs','dense_violation_count','dense_first_violation_ns']:
                    result[key][row,c] = last[key]
            feasible = np.flatnonzero(result['sampled_feasible'][row])
            if len(feasible) > 1:
                best = int(feasible[np.argmin(result['cost'][row,feasible])])
                unresolved = set()
                for c in feasible:
                    if c != best and abs(result['cost'][row,c]-result['cost'][row,best]) <= 100*(result['envelope'][row,c]+result['envelope'][row,best]):
                        unresolved.update([int(c),best])
                for c in unresolved:
                    if result['refinement_status'][row,c] != 'plateau_unresolved':
                        result['refinement_status'][row,c] = 'exhausted_unresolved'
        _save_arrays(folder,'truth.npz',result)
        return {'phase':recipe['phase'],'empirical_claims_allowed':recipe['empirical_claims_allowed'],
                'model_independent':True,'state_count':len(ids),'candidate_count':shape[1],
                'dense_truth_retained':True,'execution':store.execution_settings,
                'bank_sha256':hashlib.sha256(result['bank'].tobytes()).hexdigest(),
                'arrays':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in result.items()}}
    return store.produce('decision_truth',_truth_request(store,evaluation_receipt,recipe),produce)


def load_decision_truth(store: CalibrationStore,evaluation_receipt: Receipt,truth_receipt: Receipt) -> dict:
    recipe = _recipe(store)
    arrays = load_evaluation(store,evaluation_receipt)
    folder, manifest = store.read(truth_receipt)
    if manifest['stage'] != 'decision_truth' or manifest['request'] != _truth_request(store,evaluation_receipt,recipe):
        raise ValueError('Decision truth identity mismatch')
    raw = _arrays(folder,'truth.npz')
    if {k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in raw.items()} != manifest['summary']['arrays']:
        raise ValueError('Decision truth array schema mismatch')
    ids = decision_state_ids(arrays,recipe)
    bank = np.stack([_bank(arrays,s,recipe) for s in ids])
    if not np.array_equal(raw['state_id'],ids) or not np.array_equal(raw['bank'],bank):
        raise ValueError('Decision truth candidate-bank binding mismatch')
    for key in ['cost','envelope','samples','dense_max_abs']:
        if not np.isfinite(raw[key]).all():
            raise ValueError('Nonfinite independent decision truth')
    for index in np.ndindex(raw['cost'].shape):
        history = raw['feasibility_by_resolution'][index]
        if not np.isin(history,[-1,0,1]).all():
            raise ValueError('Invalid truth feasibility history')
        observed = history[history != -1]
        if (len(observed)<3 or raw['sampled_feasible'][index] != bool(observed[-1]) or
            raw['feasibility_disagreement'][index] != bool(np.any(observed != observed[0])) or
            raw['feasibility_status'][index] != ('resolved' if observed[-1]==observed[-2] else 'unresolved')):
            raise ValueError('Truth feasibility qualification mismatch')
    return raw


def measure_decisions(store: CalibrationStore,fit_receipt: Receipt,evaluation_receipt: Receipt,
                      truth_receipt: Receipt) -> Receipt:
    recipe = _recipe(store)
    arrays = load_evaluation(store,evaluation_receipt)
    truth = load_decision_truth(store,evaluation_receipt,truth_receipt)
    model, manifest = _selected(store,fit_receipt)
    coordinate = {k:manifest['request'][k] for k in ['session_id','budget_s','model_id','initialization_id']}
    comparison = 'unfitted_reference' if manifest['stage']=='nominal' else 'own_solver_trained'
    request = {'fit_receipt':fit_receipt,'evaluation_receipt':evaluation_receipt,'truth_receipt':truth_receipt,
               'comparison':comparison,'bank_sha256':hashlib.sha256(truth['bank'].tobytes()).hexdigest(),
               'query_steps_ns':list(MAIN_STEPS_NS)}
    def produce(folder):
        rows, outcomes = [], []
        for i,s in enumerate(truth['state_id'].tolist()):
            trajectory = int(arrays['trajectory_id'][s])
            for step in MAIN_STEPS_NS:
                result = (_empty_score(len(truth['bank'][i]),'fit_failed') if model is None else
                    score_candidates(model,torch.from_numpy(arrays['state'][s]),arrays['target'][trajectory],
                                     truth['bank'][i],query_step_ns=step))
                outcome = decision_outcome(result,truth['cost'][i],truth['sampled_feasible'][i],truth['envelope'][i],truth['feasibility_status'][i])
                context = {**coordinate,'propagator':coordinate['model_id'],'comparison':comparison,
                    'own_solver_trained':comparison=='own_solver_trained','state_id':s,'trajectory_id':trajectory,
                    'anchor_ns':int(arrays['time_ns'][s]),'stratum':'primary','query_step_ns':step}
                outcomes.append({**context,**outcome,'work':result['work'],'query_log':result['query_log'],
                    'wall_s':result['wall_s'],'cpu_s':result['cpu_s']})
                for c in range(len(result['costs'])):
                    rows.append({**context,'candidate_id':c,'status':result['statuses'][c],
                        'predicted_cost':float(result['costs'][c]) if np.isfinite(result['costs'][c]) else None,
                        'predicted_sampled_feasible':bool(result['sampled_feasible'][c]),
                        'truth_cost':float(truth['cost'][i,c]),'truth_envelope':float(truth['envelope'][i,c]),
                        'truth_cost_lower':float(truth['cost'][i,c]-truth['envelope'][i,c]),
                        'truth_cost_upper':float(truth['cost'][i,c]+truth['envelope'][i,c]),
                        'truth_sampled_feasible':bool(truth['sampled_feasible'][i,c]),
                        'truth_resolution_ns':int(truth['resolution_ns'][i,c]),
                        'truth_feasibility_status':str(truth['feasibility_status'][i,c]),
                        'truth_feasibility_disagreement':bool(truth['feasibility_disagreement'][i,c]),
                        'truth_refinement_status':str(truth['refinement_status'][i,c]),
                        'truth_dense_max_abs':truth['dense_max_abs'][i,c].tolist(),
                        'truth_dense_violation_count':int(truth['dense_violation_count'][i,c]),
                        'truth_dense_first_violation_ns':int(truth['dense_first_violation_ns'][i,c]),
                        'selected':c==result['selected_index']})
        write_records(folder/'rows.jsonl.gz',rows)
        atomic_bytes(folder/'outcomes.jsonl',b''.join(canonical_json(o)+b'\n' for o in outcomes))
        return {'phase':recipe['phase'],'empirical_claims_allowed':recipe['empirical_claims_allowed'],
                'row_count':len(rows),'outcomes':outcomes,'execution':store.execution_settings,
                'canonical_tensor_sha256':manifest['summary'].get('canonical_tensor_sha256'),
                'optimization_limited':manifest['summary'].get('optimization_limited',False)}
    return store.produce('decisions',request,produce)
