"""Deterministic CEM with common-anchor rollouts and an independent box oracle."""
from __future__ import annotations

import math
import numpy as np
import torch
from scipy.optimize import lsq_linear

from .action_path import compile_knots, mean_command, path_checksum
from .analytic import transition
from .control import anchor_cost
from .history import HistoryBatch, HistoryToken


WORK_KEYS = ('network_calls', 'vector_field_calls', 'scalar_candidate_evaluations',
             'history_tokens_processed')


def _partition(value):
    steps = tuple(value)
    if (not steps or any(type(h) is not int or h <= 0 for h in steps)
            or sum(steps) != 40_000_000):
        raise ValueError('Model partition must use positive integer ticks summing to forty ms')
    return steps


def rollout_candidates(model, state: torch.Tensor, target, history: HistoryBatch | None,
                       knots: np.ndarray, model_partition: tuple[int, ...],
                       actuator_interval_ns: int, *, power_mode: str | None = None) -> dict:
    steps = _partition(model_partition)
    bank = np.asarray(knots, dtype=np.float64)
    if bank.ndim != 3 or bank.shape[1:] != (6, 2) or len(bank) == 0:
        raise ValueError('One six-knot physical plan per candidate is required')
    paths = tuple(compile_knots(k, actuator_interval_ns) for k in bank)
    x0 = state.reshape(-1)
    if x0.shape not in ((4,), (6,)) or not bool(torch.isfinite(x0).all()) or x0.device.type != 'cpu':
        raise ValueError('One finite fully observed CPU initial state is required')
    target = np.asarray(target, dtype=np.float64)
    if target.shape != (2,) or not np.isfinite(target).all():
        raise ValueError('A finite two-joint target is required')
    count = len(bank)
    if history is None:
        history = HistoryBatch(((),), np.array([0], dtype=np.int64))
    if len(history.tokens) != 1:
        raise ValueError('Real history must contain one branch before candidate expansion')
    base_time = int(history.query_time_ns[0])
    context = HistoryBatch(tuple(history.tokens[0] for _ in range(count)),
                           np.full(count, base_time, dtype=np.int64))
    x = x0.unsqueeze(0).repeat(count, 1)
    anchors = np.full((count, 6, len(x0)), np.nan, dtype=np.float64)
    failed = np.zeros(count, dtype=bool)
    reasons = [''] * count
    work = {key: 0 for key in WORK_KEYS}
    with torch.no_grad():
        for anchor in range(6):
            start, end = anchor * 40_000_000, (anchor + 1) * 40_000_000
            initial = x.detach().clone()
            model_boundaries = np.cumsum((0, *steps)) + start
            points = sorted({int(t) for t in model_boundaries}
                            | set(range(start, end + 1, actuator_interval_ns)))
            for left, right in zip(points, points[1:]):
                command = torch.as_tensor(np.stack([path[left // actuator_interval_ns].command
                                                    for path in paths]), dtype=x.dtype)
                query = context.at_time(np.full(count, base_time + left, dtype=np.int64)).eligible()
                try:
                    kwargs = {'history': query}
                    if power_mode is not None:
                        kwargs['power_mode'] = power_mode
                    packet = model.predict(x, command, (right - left) / 1e9, **kwargs)
                except (RuntimeError, ValueError) as error:
                    for i in np.flatnonzero(~failed):
                        reasons[i] = f'{type(error).__name__}: {error}'
                    failed[:] = True
                    break
                if packet.next_state.shape != x.shape or len(packet.status) != count:
                    raise ValueError('Model prediction shape/status violates the candidate interface')
                for key in WORK_KEYS:
                    work[key] += getattr(packet, key)
                finite = torch.isfinite(packet.next_state).all(1).cpu().numpy()
                guarded = (torch.abs(packet.next_state[:, :4]) > 1000).any(1).cpu().numpy()
                for i, status in enumerate(packet.status):
                    if not failed[i] and (status != 'ok' or not finite[i] or guarded[i]):
                        failed[i] = True
                        reasons[i] = 'imagined_state_guard' if guarded[i] else (
                            'nonfinite_imagined_state' if not finite[i] else str(status))
                # Retain failed status while giving subsequent batched calls safe inputs.
                mask = torch.as_tensor(failed).unsqueeze(1)
                x = torch.where(mask, torch.zeros_like(x), packet.next_state)
            anchors[~failed, anchor] = x[~torch.as_tensor(failed)].cpu().numpy()
            completed = tuple(HistoryToken(base_time + start, initial[i],
                                           mean_command(paths[i], start, end), 40_000_000,
                                           'imagined', f'candidate-{i}-anchor-{base_time + start}')
                              for i in range(count))
            context = context.at_time(np.full(count, base_time + end, dtype=np.int64))
            context = context.append_completed(completed).eligible()
    costs = np.full(count, np.inf)
    for i in np.flatnonzero(~failed):
        costs[i] = anchor_cost(anchors[i], target, paths[i])
    return {'costs': costs, 'anchors': anchors,
            'statuses': tuple('model_failure' if value else 'ok' for value in failed),
            'reasons': tuple(reasons), 'path_checksums': tuple(path_checksum(p) for p in paths),
            'work': work}


def _settings(value):
    required = {'candidate_count', 'iterations', 'elite_count', 'initial_std', 'std_floor'}
    if set(value) != required:
        raise ValueError('CEM settings must explicitly match the declared search interface')
    for key in ('candidate_count', 'iterations', 'elite_count'):
        if type(value[key]) is not int or value[key] <= 0:
            raise ValueError('Positive integer search budgets required')
    if value['candidate_count'] < 2 or value['elite_count'] > value['candidate_count']:
        raise ValueError('Search must include mean/zero candidates and a valid elite count')
    for key in ('initial_std', 'std_floor'):
        if not math.isfinite(value[key]) or value[key] <= 0:
            raise ValueError('Positive finite proposal standard deviations required')
    return value


def search(model, state: torch.Tensor, target: torch.Tensor, history: HistoryBatch | None,
           previous_knots: np.ndarray | None, standard_normals: np.ndarray, settings: dict,
           model_partition: tuple[int, ...], actuator_interval_ns: int,
           *, power_mode: str | None = None) -> dict:
    cfg, steps = _settings(settings), _partition(model_partition)
    count, iterations = cfg['candidate_count'], cfg['iterations']
    samples = np.asarray(standard_normals, dtype=np.float64)
    if samples.shape != (iterations, count, 6, 2) or not np.isfinite(samples).all():
        raise ValueError('Complete fixed CEM standard-normal bank required')
    mean = np.zeros((6, 2))
    if previous_knots is not None:
        previous = np.asarray(previous_knots, dtype=np.float64)
        compile_knots(previous, actuator_interval_ns)  # Validate before shifting.
        mean = np.concatenate((previous[1:], previous[-1:]))
    std = np.full((6, 2), cfg['initial_std'])
    proposals, banks, costs, statuses, reasons, checksums = [], [], [], [], [], []
    best = None
    work = {key: 0 for key in WORK_KEYS}
    for iteration in range(iterations):
        bank = np.clip(mean[None] + std[None] * samples[iteration], -2, 2)
        bank[0], bank[1] = np.clip(mean, -2, 2), 0
        kwargs = {} if power_mode is None else {'power_mode': power_mode}
        prediction = rollout_candidates(model, state, target, history, bank, steps, actuator_interval_ns, **kwargs)
        values = np.asarray(prediction['costs'], dtype=np.float64)
        if values.shape != (count,) or len(prediction['statuses']) != count:
            raise ValueError('Candidate scorer omitted a required result')
        finite = []
        for local, (cost, status, reason) in enumerate(zip(values, prediction['statuses'], prediction['reasons'])):
            if np.isfinite(cost):
                if status != 'ok':
                    raise ValueError('Failed candidate cannot have a finite planning cost')
                finite.append(local)
                key = (float(cost), iteration, local)
                if best is None or key < best['key']:
                    best = {'key': key, 'knots': bank[local].copy(),
                            'anchors': prediction['anchors'][local].copy(),
                            'path_checksum': prediction['path_checksums'][local]}
            elif not (cost == np.inf and status != 'ok' and reason):
                raise ValueError('Unscoreable candidate requires positive infinity and a failure reason')
        finite.sort(key=lambda i: (values[i], i))
        elite = finite[:min(cfg['elite_count'], len(finite))]
        old_mean, old_std = mean.copy(), std.copy()
        if elite:
            mean = bank[elite].mean(axis=0)
            std = np.maximum(bank[elite].std(axis=0, ddof=0), cfg['std_floor'])
        proposals.append({'iteration': iteration, 'mean': old_mean, 'std': old_std,
                          'next_mean': mean.copy(), 'next_std': std.copy(),
                          'elite_global_ids': [iteration * count + i for i in elite]})
        banks.append(bank.copy()); costs.append(values.copy())
        statuses.extend(prediction['statuses']); reasons.extend(prediction['reasons'])
        checksums.extend(prediction['path_checksums'])
        for key in work:
            work[key] += prediction['work'][key]
    candidates = {'knots': np.concatenate(banks), 'costs': np.concatenate(costs),
                  'statuses': tuple(statuses), 'reasons': tuple(reasons),
                  'path_checksums': tuple(checksums), 'global_ids': np.arange(iterations * count)}
    if best is None:
        return {'status': 'all_candidates_failed', 'fallback': 'zero_command', 'cost': None,
                'knots': np.zeros((6, 2)), 'selected_global_id': None, 'predicted_anchors': None,
                'proposals': proposals, 'candidates': candidates, 'work': work}
    cost, iteration, local = best['key']
    return {'status': 'ok', 'fallback': None, 'cost': cost, 'knots': best['knots'],
            'selected_global_id': iteration * count + local, 'predicted_anchors': best['anchors'],
            'path_checksum': best['path_checksum'], 'proposals': proposals,
            'candidates': candidates, 'work': work}


def analytic_box_optimum(state: np.ndarray, target: np.ndarray,
                         actuator_interval_ns: int = 40_000_000) -> dict:
    state, target = np.asarray(state, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if state.shape != (4,) or target.shape != (2,) or not np.isfinite(state).all() or not np.isfinite(target).all():
        raise ValueError('The analytic planner oracle requires finite unit-mass q/v and target')
    basis = [compile_knots(np.eye(12)[j].reshape(6, 2), actuator_interval_ns) for j in range(12)]
    q_map, v_map = np.zeros((2, 12)), np.zeros((2, 12))
    rows, rhs, q_maps, v_maps = [], [], [], []
    for index, segment in enumerate(basis[0]):
        h = segment.duration_ns / 1e9
        command_map = np.stack([p[index].command for p in basis], axis=1)
        q_map = q_map + h * v_map + .5 * h**2 * command_map
        v_map = v_map + h * command_map
        rows.append(math.sqrt(.001 * h) * command_map)
        rhs.append(np.zeros(2))
        time_ns = segment.start_ns + segment.duration_ns
        if time_ns % 40_000_000 == 0:
            q_maps.append(q_map.copy()); v_maps.append(v_map.copy())
            rows.extend((math.sqrt(.04) * q_map, math.sqrt(.04 * .05) * v_map))
            rhs.extend((math.sqrt(.04) * (target - state[:2] - time_ns / 1e9 * state[2:]),
                        -math.sqrt(.04 * .05) * state[2:]))
    rows.append(math.sqrt(5) * q_map)
    rhs.append(math.sqrt(5) * (target - state[:2] - .24 * state[2:]))
    design, response = np.concatenate(rows), np.concatenate(rhs)
    solution = lsq_linear(design, response, bounds=(-2, 2), tol=1e-12, max_iter=1000)
    if not solution.success or not np.isfinite(solution.x).all() or np.max(np.abs(solution.x)) > 2:
        raise RuntimeError(f'Box-constrained analytic oracle failed: {solution.message}')
    gradient = design.T @ (design @ solution.x - response)
    optimality = float(np.max(np.abs(solution.x - np.clip(solution.x - gradient, -2, 2))))
    if optimality > 1e-6:
        raise RuntimeError('Analytic optimum fails independently checked projected optimality')
    knots = solution.x.reshape(6, 2)
    def exact_cost(values):
        x, anchors = state.copy(), []
        path = compile_knots(values, actuator_interval_ns)
        for segment in path:
            x = transition(x, segment.command, segment.duration_ns / 1e9, 'direct')
            if (segment.start_ns + segment.duration_ns) % 40_000_000 == 0:
                anchors.append(x.copy())
        return anchor_cost(np.stack(anchors), target, path)
    observed = exact_cost(knots)
    if not math.isclose(observed, 2 * solution.cost, rel_tol=1e-10, abs_tol=1e-12):
        raise RuntimeError('Independent exact rollout disagrees with the stacked oracle cost')
    return {'success': True, 'knots': knots, 'cost': observed,
            'zero_cost': exact_cost(np.zeros((6, 2))), 'solver_half_squared_cost': float(solution.cost),
            'projected_optimality': optimality, 'solver_status': int(solution.status),
            'solver_message': str(solution.message), 'solver_iterations': int(solution.nit)}
