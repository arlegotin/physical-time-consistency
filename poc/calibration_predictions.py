"""Independent moving-state truth and physical-unit PTCO-CAL-0.1 predictions.

Numerical truth jobs depend on evaluator coordinates and resolutions only. Raw
rows retain every predictor failure and distinguish solver diagnostics from fits.
"""
from __future__ import annotations

from collections import defaultdict
import io
from pathlib import Path

import numpy as np
import torch

from .action_path import ActionSegment, restrict_path, subdivide_path
from .calibration_data import OBSERVATION_NS, rng_for
from .calibration_models import build_model, propagation_family
from .calibration_records import write_records
from .calibration_store import CalibrationStore, Receipt
from .io import atomic_bytes, canonical_json, read_json
from .models.common import Scales


def _recipe(store, root=None):
    from .calibration_study import study_bindings
    _, manifest = store.read(store.recipe_receipt)
    recipe = manifest['summary']['recipe']
    if recipe != store.recipe:
        raise ValueError('Changed evaluation recipe')
    root = Path(store.bindings['root'] if root is None else root).resolve()
    current = study_bindings(root, recipe)
    if any(store.bindings.get(k) != v for k, v in current.items()):
        raise ValueError('Changed evaluation source/asset/runtime bindings')
    return recipe


def anchor_times(recipe: dict) -> np.ndarray:
    e = recipe['evaluation']
    if 'unit_anchor_times_ns' in e:
        if recipe['phase'] != 'unit':
            raise ValueError('Reduced anchors require an explicit unit recipe')
        return np.asarray(e['unit_anchor_times_ns'], dtype=np.int64)
    return np.arange(e['anchor_start_ns'], e['anchor_stop_ns']+1, e['anchor_tick_ns'], dtype=np.int64)


def query_partition(path, horizon_ns: int, step_ns: int):
    """Restrict before splitting, including every logged command boundary."""
    return subdivide_path(restrict_path(path, 0, horizon_ns), step_ns)


def physical_metrics(errors: np.ndarray) -> dict:
    errors = np.asarray(errors, dtype=np.float64)
    if errors.ndim != 2 or errors.shape[1] != 2 or not np.isfinite(errors).all():
        raise ValueError('Finite physical-unit two-joint endpoint residuals required')
    return {'endpoint_count': len(errors),
            'rmse': float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
            'p95': float(np.percentile(np.linalg.norm(errors, axis=1), 95)) if len(errors) else None}


def refinement_required(envelope: float, error: float, fraction: float = .01) -> bool:
    if not np.isfinite([envelope, error, fraction]).all() or min(envelope, error) < 0 or fraction <= 0:
        raise ValueError('Finite nonnegative error/envelope and positive fraction required')
    return envelope > fraction * error


def _save_arrays(folder, name, arrays):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    atomic_bytes(folder / name, buffer.getvalue())


def _arrays(folder, name):
    with np.load(folder / name, allow_pickle=False) as saved:
        return {k: saved[k].copy() for k in saved.files}


def _evaluation_request(store, recipe):
    return {'recipe_receipt': store.recipe_receipt, 'phase': recipe['phase'],
            'role': 'development-evaluation', 'evaluation': recipe['evaluation'],
            'collector': recipe['collector']}


def prepare_evaluation(store: CalibrationStore, *, root: Path) -> Receipt:
    from .envs import Plant
    recipe = _recipe(store, root)
    e, c = recipe['evaluation'], recipe['collector']
    count, ticks = e['trajectory_count'], 40
    warm = anchor_times(recipe)
    times = np.r_[warm, np.int64(0)] if e['initial_secondary'] else warm
    def produce(folder):
        plant = Plant('lagged', e['truth_step_ns'], Path(root) / 'assets')
        states = np.empty((count, ticks+1, 6), dtype=np.float64)
        commands = np.empty((count, ticks, 2), dtype=np.float64)
        probes = np.empty((count, ticks+1, 2), dtype=np.float64)
        targets = np.empty((count, 2), dtype=np.float64)
        for trajectory in range(count):
            initial = rng_for('development-evaluation', e['seed_session_id'], trajectory, c['initial_state_channel'])
            state = np.r_[initial.uniform(*c['initial_q'], 2), initial.uniform(*c['initial_v'], 2),
                          initial.uniform(*c['initial_w'], 2)]
            excitation = rng_for('development-evaluation', e['seed_session_id'], trajectory, c['excitation_channel'])
            probes[trajectory] = rng_for('development-evaluation', e['seed_session_id'], trajectory,
                                         e['held_probe_channel']).uniform(*e['held_probe_bounds'], (ticks+1, 2))
            targets[trajectory] = rng_for('development-evaluation', e['seed_session_id'], trajectory,
                                          e['target_channel']).uniform(*e['target_bounds'], 2)
            plant.reset(state)
            states[trajectory, 0] = state
            for tick in range(ticks):
                if tick % c['d_hold_ticks'] == 0:
                    d = excitation.uniform(*c['d_bounds'], 2)
                command = np.clip(-state[:2]-.25*state[2:4]+d, -1.5, 1.5)
                commands[trajectory, tick] = command
                state = plant.advance(command, OBSERVATION_NS)['state']
                states[trajectory, tick+1] = state
        ids = np.repeat(np.arange(count, dtype=np.int64), len(times))
        time_ns = np.tile(times, count)
        indices = time_ns // OBSERVATION_NS
        arrays = {'trajectory_state': states, 'trajectory_command': commands, 'target': targets,
                  'trajectory_time_ns': np.arange(ticks+1, dtype=np.int64)*OBSERVATION_NS,
                  'trajectory_id': ids, 'time_ns': time_ns, 'state': states[ids, indices].copy(),
                  'primary': time_ns != 0, 'held_command': probes[ids, indices].copy(),
                  'recorded_command': np.stack([commands[t, k:k+10] for t, k in zip(ids, indices)])}
        _save_arrays(folder, 'evaluation.npz', arrays)
        return {'phase': recipe['phase'], 'empirical_claims_allowed': recipe['empirical_claims_allowed'],
                'trajectory_count': count, 'primary_count': count*len(warm), 'state_count': len(ids),
                'evaluator_only': True, 'execution': store.execution_settings,
                'arrays': {k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in arrays.items()}}
    return store.produce('evaluation', _evaluation_request(store, recipe), produce)


def load_evaluation(store: CalibrationStore, receipt: Receipt) -> dict:
    recipe = _recipe(store)
    folder, manifest = store.read(receipt)
    if manifest['stage'] != 'evaluation' or manifest['request'] != _evaluation_request(store, recipe):
        raise ValueError('Evaluation identity/recipe mismatch')
    arrays = _arrays(folder, 'evaluation.npz')
    if {k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in arrays.items()} != manifest['summary']['arrays']:
        raise ValueError('Evaluation array schema mismatch')
    if any(not np.isfinite(v).all() for v in arrays.values()):
        raise ValueError('Nonfinite evaluation inputs')
    return arrays


def truth_cases(recipe: dict) -> list[dict]:
    e = recipe['evaluation']
    cases = []
    for horizon in e['endpoint_horizons_ns'] + e['stress_held_horizons_ns']:
        for branch in ('new', 'zero'):
            cases.append({'scenario': 'held' if horizon <= 40_000_000 else 'held_stress',
                          'horizon_ns': horizon, 'branch': branch})
    cases.extend({'scenario': 'recorded', 'horizon_ns': h, 'branch': 'recorded'}
                 for h in e['rollout_horizons_ns'])
    return cases


def case_path(arrays: dict, state_id: int, case: dict):
    """Public physical command path, relative to an immutable observed state."""
    h = case['horizon_ns']
    if case['branch'] != 'recorded':
        command = arrays['held_command'][state_id] if case['branch'] == 'new' else np.zeros(2)
        return (ActionSegment(0, h, command),)
    path = tuple(ActionSegment(i*OBSERVATION_NS, OBSERVATION_NS, u)
                 for i, u in enumerate(arrays['recorded_command'][state_id]))
    return restrict_path(path, 0, h)


def _replay(plant, state, path):
    plant.reset(state)
    for segment in path:
        endpoint = plant.advance(segment.command, segment.duration_ns)['state']
    return endpoint


def _truth_request(store, evaluation, recipe):
    e = recipe['evaluation']
    return {'recipe_receipt': store.recipe_receipt, 'evaluation_receipt': evaluation,
            'resolution_ns': [e['truth_step_ns'], *e['refinement_steps_ns']],
            'audit_trajectory_ids': [0, e['trajectory_count']-1], 'cases': truth_cases(recipe)}


def prediction_truth(store: CalibrationStore, evaluation_receipt: Receipt, *, root: Path) -> Receipt:
    from .envs import Plant
    recipe = _recipe(store, root)
    arrays = load_evaluation(store, evaluation_receipt)
    request = _truth_request(store, evaluation_receipt, recipe)
    cases, resolutions = request['cases'], request['resolution_ns']
    def produce(folder):
        n, k = len(arrays['state']), len(cases)
        endpoints = np.empty((n, k, 6), dtype=np.float64)
        refined = np.full((n, k, 2, 6), np.nan, dtype=np.float64)
        audit = np.isin(arrays['trajectory_id'], request['audit_trajectory_ids'])
        for level, step in enumerate(resolutions):
            plant = Plant('lagged', step, Path(root) / 'assets')
            for state_id in range(n):
                if level and not audit[state_id]:
                    continue
                for case_id, case in enumerate(cases):
                    value = _replay(plant, arrays['state'][state_id], case_path(arrays, state_id, case))
                    if level == 0:
                        endpoints[state_id, case_id] = value
                    else:
                        refined[state_id, case_id, level-1] = value
        envelope = np.maximum(np.abs(endpoints[audit]-refined[audit, :, 0]),
                              np.abs(refined[audit, :, 0]-refined[audit, :, 1]))
        # Unaudited cases carry the observed first/last maximum, explicitly labeled.
        bound = np.broadcast_to(envelope.max(axis=0), endpoints.shape).copy()
        bound[audit] = envelope
        _save_arrays(folder, 'truth.npz', {'endpoint': endpoints, 'refined_endpoint': refined,
                                         'envelope': bound, 'audit_mask': audit})
        return {'phase': recipe['phase'], 'evaluator_only': True, 'state_count': n,
                'case_count': k, 'audit_state_ids': np.flatnonzero(audit).tolist(),
                'execution': store.execution_settings, 'model_independent': True}
    return store.produce('prediction_truth', request, produce)


def load_prediction_truth(store, evaluation_receipt, truth_receipt):
    recipe = _recipe(store)
    folder, manifest = store.read(truth_receipt)
    if manifest['stage'] != 'prediction_truth' or manifest['request'] != _truth_request(store, evaluation_receipt, recipe):
        raise ValueError('Prediction truth evaluation/resolution identity mismatch')
    return _arrays(folder, 'truth.npz')


def supplemental_truth(store: CalibrationStore, evaluation_receipt: Receipt, truth_receipt: Receipt,
                       coordinates: list[list[int]], resolution_ns: int) -> Receipt:
    """Shared immutable extra truth, keyed solely by coordinates and resolution."""
    from .envs import Plant
    recipe = _recipe(store)
    arrays = load_evaluation(store, evaluation_receipt)
    load_prediction_truth(store, evaluation_receipt, truth_receipt)
    cases = truth_cases(recipe)
    coordinates = sorted(coordinates)
    if (resolution_ns not in (recipe['evaluation']['refinement_steps_ns'] +
                                  recipe['evaluation']['adaptive_refinement_steps_ns']) or
            not coordinates or len({tuple(c) for c in coordinates}) != len(coordinates) or
            any(len(c) != 2 or any(type(x) is not int for x in c) or not 0 <= c[0] < len(arrays['state'])
                or not 0 <= c[1] < len(cases) for c in coordinates)):
        raise ValueError('Invalid supplemental truth coordinates/resolution')
    request = {'evaluation_receipt': evaluation_receipt, 'truth_receipt': truth_receipt,
               'coordinates': coordinates, 'resolution_ns': resolution_ns}
    def produce(folder):
        plant = Plant('lagged', resolution_ns, Path(store.bindings['root']) / 'assets')
        endpoints = np.stack([_replay(plant, arrays['state'][s], case_path(arrays, s, cases[c]))
                              for s, c in coordinates])
        _save_arrays(folder, 'refinement.npz', {'coordinates': np.asarray(coordinates, dtype=np.int64),
                                              'endpoint': endpoints})
        return {'model_independent': True, 'case_count': len(coordinates), 'resolution_ns': resolution_ns}
    return store.produce('prediction_refinement', request, produce)


def _selected(store, receipt):
    from .calibration_training import _context, load_selected
    _, manifest = store.read(receipt)
    if manifest['stage'] not in {'fit', 'nominal'}:
        raise ValueError('Expected fit or nominal receipt')
    request, summary = manifest['request'], manifest['summary']
    expected, _, _, _ = _context(store, request['session_receipt'], request['budget_s'],
                                request['model_id'], request['initialization_id'])
    if expected != request or any(summary[k] != request[k] for k in ('model', 'account', 'phase')):
        raise ValueError('Prediction selected model identity mismatch')
    if manifest['stage'] == 'fit' and summary.get('fit_status') == 'failed':
        if summary.get('selection_kind') != 'fitted' or summary.get('selected') is not None:
            raise ValueError('Invalid failed-fit evidence')
        return None, manifest
    # Invalid hashes/checkpoints are evidence errors, not prediction failures.
    model = load_selected(store, receipt)
    if any(not bool(torch.isfinite(t).all()) for t in model.state_dict().values()):
        raise ValueError('Nonfinite selected checkpoint despite finite fit status')
    return model, manifest


def _predict_paths(model, states, paths, step_ns):
    n = len(states)
    if model is None:
        return None, ['fit_failed']*n
    value = torch.from_numpy(states.copy())
    status = ['ok']*n
    partitions = [subdivide_path(p, step_ns) for p in paths]
    if any([s.duration_ns for s in p] != [s.duration_ns for s in partitions[0]] for p in partitions):
        raise ValueError('Batch paths must share command-clock boundaries')
    with torch.no_grad():
        for tick, segment in enumerate(partitions[0]):
            command = torch.from_numpy(np.stack([p[tick].command for p in partitions]))
            before, command_before = value.clone(), command.clone()
            try:
                predicted = model.predict(value, command, segment.duration_ns/1e9)
            except (RuntimeError, FloatingPointError) as error:
                return None, [s if s != 'ok' else 'predictor_exception:'+type(error).__name__+':'+str(error) for s in status]
            if not torch.equal(value, before) or not torch.equal(command, command_before):
                raise ValueError('Predictor mutated immutable query state/command')
            for i, current in enumerate(predicted.status):
                if status[i] == 'ok' and current != 'ok':
                    status[i] = current
            valid = torch.as_tensor([s == 'ok' for s in status])
            value = torch.where(valid[:, None], predicted.next_state, before)
    return value.numpy(), status


def _adaptive(store, evaluation, truth_receipt, raw, predictions):
    """Refine only unresolved accuracy cases; keep the base truth artifact intact."""
    reference, envelope = raw['endpoint'].copy(), raw['envelope'].copy()
    previous = raw['refined_endpoint'][:, :, 1].copy()
    prior_delta = np.abs(raw['refined_endpoint'][:, :, 1]-raw['refined_endpoint'][:, :, 0])
    receipts, stopped = [], set()
    def needed(s, c):
        for values, statuses in predictions.get(c, []):
            if values is None or statuses[s] != 'ok':
                continue
            for cut in (slice(0, 2), slice(2, 4)):
                error = float(np.linalg.norm(values[s, cut]-reference[s, c, cut]))
                bound = float(np.linalg.norm(envelope[s, c, cut]))
                if refinement_required(bound, error):
                    return True
        held_count = 2*len(store.recipe['evaluation']['endpoint_horizons_ns'] +
                           store.recipe['evaluation']['stress_held_horizons_ns'])
        if c < held_count:
            new_case, zero_case = c-c%2, c-c%2+1
            for (new, ns), (zero, zs) in zip(predictions.get(new_case, []), predictions.get(zero_case, [])):
                if new is None or zero is None or ns[s] != 'ok' or zs[s] != 'ok':
                    continue
                for cut in (slice(0, 2), slice(2, 4)):
                    error = float(np.linalg.norm(new[s, cut]-zero[s, cut]-
                                  (reference[s, new_case, cut]-reference[s, zero_case, cut])))
                    bound = float(np.linalg.norm(envelope[s, new_case, cut]+envelope[s, zero_case, cut]))
                    if refinement_required(bound, error):
                        return True
        return False
    for step in store.recipe['evaluation']['adaptive_refinement_steps_ns']:
        coordinates = [[s, c] for c in predictions for s in range(len(reference))
                       if (s, c) not in stopped and needed(s, c)]
        if not coordinates:
            break
        missing = [c for c in coordinates if not np.isfinite(previous[c[0], c[1]]).all()]
        if missing:
            predecessors = []
            for predecessor_step in store.recipe['evaluation']['refinement_steps_ns']:
                predecessor = supplemental_truth(store, evaluation, truth_receipt, missing, predecessor_step)
                receipts.append(predecessor)
                predecessor_folder, _ = store.read(predecessor)
                predecessors.append(_arrays(predecessor_folder, 'refinement.npz')['endpoint'])
            for coordinate, coarse, fine in zip(sorted(missing), *predecessors):
                s, c = coordinate
                previous[s, c], prior_delta[s, c] = fine, np.abs(fine-coarse)
        receipt = supplemental_truth(store, evaluation, truth_receipt, coordinates, step)
        receipts.append(receipt)
        folder, _ = store.read(receipt)
        refined = _arrays(folder, 'refinement.npz')
        for (s, c), endpoint in zip(refined['coordinates'], refined['endpoint']):
            s, c = int(s), int(c)
            delta = np.abs(endpoint-previous[s, c])
            envelope[s, c] = np.maximum(prior_delta[s, c], delta)
            if np.linalg.norm(delta[:4]) >= np.linalg.norm(prior_delta[s, c, :4]):
                stopped.add((s, c))
            reference[s, c], previous[s, c], prior_delta[s, c] = endpoint, endpoint, delta
    return reference, envelope, receipts


def aggregate_prediction_rows(rows):
    """Recompute raw metrics without mixing fits, solvers, strata or partitions."""
    groups = defaultdict(list)
    keys = ('session_id', 'budget_s', 'model_id', 'initialization_id', 'comparison',
            'propagator', 'stratum', 'scenario', 'horizon_ns', 'query_step_ns', 'kind', 'component')
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    metrics = []
    for group, values in groups.items():
        good = [r for r in values if r['status'] == 'ok']
        errors = np.asarray([r['error'] for r in good], dtype=np.float64).reshape(-1, 2)
        envelope = max((r['envelope_norm'] for r in good), default=0.)
        metric = physical_metrics(errors)
        unresolved = any(r['truth_status'] == 'requires_refinement' for r in good)
        metrics.append({**dict(zip(keys, group)), **metric, 'total_count': len(values),
                        'failure_count': len(values)-len(good), 'failure_fraction': 1-len(good)/len(values),
                        'max_envelope_norm': envelope, 'interpretation_status': ('prediction_failed' if not good else 'unresolved' if unresolved else 'numerically_equal_unresolved'
                                                  if any(r['truth_status'] == 'numerically_equal' for r in good) else 'resolved'),
                        'conservative_rmse_upper': metric['rmse']+envelope if good else None,
                        'conservative_p95_upper': metric['p95']+envelope if good else None})
    return metrics


def _measure(store, fit_receipt, evaluation_receipt, truth_receipt, *, diagnostic):
    recipe = _recipe(store)
    arrays = load_evaluation(store, evaluation_receipt)
    raw = load_prediction_truth(store, evaluation_receipt, truth_receipt)
    model, manifest = _selected(store, fit_receipt)
    request = manifest['request']
    model_id = request['model_id']
    if diagnostic and not propagation_family(model_id):
        raise ValueError('Same-weight propagation diagnostic requires an f/B model')
    comparison = ('same_weight_propagation_diagnostic' if diagnostic else
                  'unfitted_reference' if manifest['stage'] == 'nominal' else 'own_solver_trained')
    models = {model_id: model}
    if diagnostic:
        models = {}
        for name in propagation_family(model_id):
            if model is None:
                models[name] = None
            else:
                paired = build_model(name, Scales(model.scale_q, model.scale_v, model.v_input), 0).double().eval()
                paired.load_state_dict(model.state_dict(), strict=True)
                models[name] = paired
    bound_request = {'fit_receipt': fit_receipt, 'evaluation_receipt': evaluation_receipt,
                     'truth_receipt': truth_receipt, 'comparison': comparison}
    def produce(folder):
        cases = truth_cases(recipe)
        computed, predictions = {}, defaultdict(list)
        for propagator, predictor in models.items():
            for case_id, case in enumerate(cases):
                paths = [case_path(arrays, s, case) for s in range(len(arrays['state']))]
                steps = recipe['evaluation']['query_steps_ns'] if case['branch'] == 'recorded' else [case['horizon_ns']]
                for step in steps:
                    result = _predict_paths(predictor, arrays['state'], paths, step)
                    computed[propagator, case_id, step] = result
                    predictions[case_id].append(result)
                if case['branch'] == 'new':
                    computed[propagator, case_id, 'half'] = _predict_paths(predictor, arrays['state'], paths, case['horizon_ns']//2)
        reference, envelope, refinement_receipts = _adaptive(store, evaluation_receipt, truth_receipt, raw, predictions)
        resolutions = np.full(reference.shape[:2], recipe['evaluation']['truth_step_ns'], dtype=np.int64)
        audited = np.broadcast_to(raw['audit_mask'][:, None], reference.shape[:2]).copy()
        for refinement in refinement_receipts:
            _, refined_manifest = store.read(refinement)
            for state_id, truth_case_id in refined_manifest['request']['coordinates']:
                resolutions[state_id, truth_case_id] = min(resolutions[state_id, truth_case_id],
                                                          refined_manifest['request']['resolution_ns'])
                audited[state_id, truth_case_id] = True
        rows = []
        for propagator in models:
            for case_id, case in enumerate(cases):
                if case['branch'] == 'zero':
                    continue
                steps = recipe['evaluation']['query_steps_ns'] if case['branch'] == 'recorded' else [case['horizon_ns']]
                for step in steps:
                    new, new_status = computed[propagator, case_id, step]
                    terms = [('endpoint', new, reference[:, case_id], new_status, envelope[:, case_id])]
                    if case['branch'] == 'new':
                        zero, zero_status = computed[propagator, case_id+1, step]
                        half, half_status = computed[propagator, case_id, 'half']
                        joint_status = lambda other: [a if a != 'ok' else b for a, b in zip(new_status, other)]
                        terms.extend([
                            ('drift', zero-arrays['state'] if zero is not None else None,
                             reference[:, case_id+1]-arrays['state'], zero_status, envelope[:, case_id+1]),
                            ('command_response', new-zero if new is not None and zero is not None else None,
                             reference[:, case_id]-reference[:, case_id+1], joint_status(zero_status),
                             envelope[:, case_id]+envelope[:, case_id+1]),
                            ('composition', new-half if new is not None and half is not None else None,
                             np.zeros_like(reference[:, case_id]), joint_status(half_status), np.zeros_like(envelope[:, case_id]))])
                    for kind, predicted, target, statuses, bounds in terms:
                        truth_ids = ([case_id+1] if kind == 'drift' else [case_id, case_id+1]
                                     if kind == 'command_response' else [] if kind == 'composition' else [case_id])
                        for s, status in enumerate(statuses):
                            for component, cut, unit in (('q', slice(0, 2), 'rad'), ('v', slice(2, 4), 'rad/s')):
                                error = predicted[s, cut]-target[s, cut] if status == 'ok' and predicted is not None else None
                                bound = float(np.linalg.norm(bounds[s, cut]))
                                truth_status = ('requires_refinement' if error is not None and refinement_required(bound, float(np.linalg.norm(error)))
                                                else 'audited' if all(audited[s, c] for c in truth_ids) else 'first_last_audit_bound')
                                if error is not None and float(np.linalg.norm(error)) == 0 and bound == 0:
                                    truth_status = 'numerically_equal'
                                if kind == 'composition':
                                    truth_status = 'not_applicable'
                                rows.append({**{k: request[k] for k in ('session_id', 'budget_s', 'model_id', 'initialization_id')},
                                    'propagator': propagator, 'comparison': comparison,
                                    'own_solver_trained': comparison == 'own_solver_trained',
                                    'state_id': s, 'trajectory_id': int(arrays['trajectory_id'][s]),
                                    'anchor_ns': int(arrays['time_ns'][s]),
                                    'stratum': 'primary' if arrays['primary'][s] else 'initial_secondary',
                                    'scenario': case['scenario'], 'horizon_ns': case['horizon_ns'], 'query_step_ns': step,
                                    'kind': kind, 'component': component, 'unit': unit, 'status': status,
                                    'prediction': predicted[s, cut].tolist() if error is not None else None,
                                    'truth': target[s, cut].tolist(), 'error': error.tolist() if error is not None else None,
                                    'envelope': bounds[s, cut].tolist(), 'envelope_norm': bound,
                                    'truth_status': truth_status,
                                    'truth_coordinates': [[s, c] for c in truth_ids],
                                    'truth_resolution_ns': [int(resolutions[s, c]) for c in truth_ids]})
        write_records(folder / 'rows.jsonl.gz', rows)
        return {'phase': recipe['phase'], 'empirical_claims_allowed': recipe['empirical_claims_allowed'],
                'comparison': comparison, 'row_count': len(rows), 'metrics': aggregate_prediction_rows(rows),
                'refinement_receipts': refinement_receipts, 'execution': store.execution_settings,
                'canonical_tensor_sha256': manifest['summary'].get('canonical_tensor_sha256'),
                'optimization_limited': manifest['summary'].get('optimization_limited', False)}
    return store.produce('propagation' if diagnostic else 'predictions', bound_request, produce)


def measure_prediction(store: CalibrationStore, fit_receipt: Receipt, evaluation_receipt: Receipt,
                       truth_receipt: Receipt) -> Receipt:
    return _measure(store, fit_receipt, evaluation_receipt, truth_receipt, diagnostic=False)


def propagation_diagnostic(store: CalibrationStore, fit_receipt: Receipt, evaluation_receipt: Receipt,
                           truth_receipt: Receipt) -> Receipt:
    return _measure(store, fit_receipt, evaluation_receipt, truth_receipt, diagnostic=True)
