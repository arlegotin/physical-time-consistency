"""Validate and record physical episodes before any statistical aggregation."""
from __future__ import annotations

import csv
import io
import math
from pathlib import Path
import numpy as np

from .control import anchor_components
from .io import atomic_bytes, atomic_json, canonical_json


def json_value(value):
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Unclassified nonfinite control value cannot enter JSON')
    return value


def _npz(path, **arrays):
    if any(np.asarray(array).dtype.hasobject for array in arrays.values()):
        raise ValueError('Control arrays must not require pickle loading')
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    atomic_bytes(path, buffer.getvalue())


def _lines(path, rows):
    atomic_bytes(path, b''.join(canonical_json(json_value(row)) + b'\n' for row in rows))


def save_episode(folder: Path, result: dict, *, model_id='fixture', checkpoint_hash=None, run_id=None) -> dict:
    folder = Path(folder)
    dimension = 4 if result['actuator'] == 'direct' else 6
    tick = result['physics_step_ns']
    expected_times = np.arange(0, 1_600_000_000 + 1, tick, dtype=np.int64)
    observations = np.arange(0, 1_600_000_000, 40_000_000, dtype=np.int64)
    if (result['complete'] is not True or not np.array_equal(result['true_times_ns'], expected_times)
            or np.asarray(result['true_states']).shape != (len(expected_times), dimension)
            or not np.isfinite(result['true_states']).all()
            or not np.array_equal(result['observation_times_ns'], observations)
            or np.asarray(result['observations']).shape != (40, dimension)
            or not np.isfinite(result['observations']).all()
            or len(result['decisions']) != 40 or len(result['candidates']) != 40):
        raise ValueError('A partial or invalid control trajectory cannot yield complete evidence')
    anchors = np.asarray(result['true_anchors'])
    if (anchors.shape != (40, dimension)
            or not np.array_equal(anchors, result['true_states'][40_000_000 // tick::40_000_000 // tick])):
        raise ValueError('Control costs require real common physical anchors')
    components = anchor_components(anchors, result['target'], result['action_segments'])
    if not math.isclose(components['total'], result['realized_cost'], rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError('Recorded realized cost does not reconcile with raw truth and executed commands')
    count = len(result['candidates'][0]['costs'])
    for index, bank in enumerate(result['candidates']):
        costs, statuses, reasons = (np.asarray(bank[key]) for key in ('costs', 'statuses', 'reasons'))
        knots, identifiers = np.asarray(bank['knots']), np.asarray(bank['global_ids'])
        if (knots.shape != (count, 6, 2) or costs.shape != (count,) or statuses.shape != (count,)
                or reasons.shape != (count,) or not np.array_equal(identifiers, np.arange(count))
                or not np.isfinite(knots).all() or np.any(np.abs(knots) > 2)):
            raise ValueError('Missing or invalid candidate rows in a completed episode')
        finite, failed = np.isfinite(costs), statuses != 'ok'
        if (not np.array_equal(~finite, failed) or np.isnan(costs).any() or np.isneginf(costs).any()
                or np.any(failed & (reasons == ''))):
            raise ValueError('Only explicitly failed candidates may have positive infinite cost')
        decision = result['decisions'][index]
        if decision['time_ns'] != int(observations[index]):
            raise ValueError('Decision and observation clocks disagree')
        if decision['status'] == 'all_candidates_failed':
            if finite.any() or decision['selected_global_id'] is not None:
                raise ValueError('Fallback status conflicts with candidate results')
        else:
            selected = decision['selected_global_id']
            if selected is None or selected not in range(count) or not finite[selected]:
                raise ValueError('Executed plan must select a scored finite candidate')
            if not np.array_equal(decision['selected_knots'], knots[selected]):
                raise ValueError('Executed knots differ from the selected evaluated candidate')
    # Validate JSON before writing any complete episode summary.
    json_value(result['decisions'])
    times, states, target = result['true_times_ns'], result['true_states'], result['target']
    error = states[:, :2] - target
    final = times >= 1_400_000_000
    success = bool(np.all(np.linalg.norm(error[final], axis=1) < .1)
                   and np.all(np.linalg.norm(states[final, 2:4], axis=1) < .2))
    if success != result['success']:
        raise ValueError('Task success does not reconcile with every final-window physics step')
    summary_keys = ('complete', 'phase', 'condition', 'seed', 'episode_id', 'actuator', 'target',
                    'initial_state', 'physics_step_ns', 'actuator_interval_ns', 'model_partition_ns', 'power_mode',
                    'realized_cost', 'q_rmse_rad', 'terminal_q_error_rad', 'velocity_rms_rad_s',
                    'command_effort_integral', 'success', 'final_window_sample_count', 'fallback_count',
                    'seed_namespaces', 'work')
    summary = json_value({key: result[key] for key in summary_keys})
    summary.update({'model_id': model_id, 'checkpoint_hash': checkpoint_hash, 'run_id': run_id,
                    'experiment_id': ('E05' if result['condition'] != 'clean' else
                                      'E04' if result['actuator_interval_ns'] != 40_000_000 else 'E03'),
                    'cost_components': components, 'candidate_count_per_decision': count,
                    'planner_latency_median_seconds': float(np.median(result['planner_latency_seconds'])),
                    'planner_latency_p95_seconds': float(np.quantile(result['planner_latency_seconds'], .95, method='linear'))})
    canonical_json(summary)
    folder.mkdir(parents=True, exist_ok=True)
    _npz(folder / 'true_trajectories.npz', time_ns=times, state=states, anchors=anchors,
         anchors_with_initial=result['true_anchors_with_initial'], target=target)
    _npz(folder / 'observations.npz', packet_id=np.array([f'obs-{t}' for t in observations]),
         time_ns=observations, state=result['observations'], noise_time_ns=result['noise_packet_times_ns'])
    _npz(folder / 'candidates.npz', **{key: np.stack([bank[key] for bank in result['candidates']])
                                     for key in ('knots', 'costs', 'statuses', 'reasons', 'global_ids')})
    _lines(folder / 'decisions.jsonl', result['decisions'])
    _lines(folder / 'action_segments.jsonl', [{'start_ns': segment.start_ns, 'duration_ns': segment.duration_ns,
            'command': segment.command} for segment in result['action_segments']])
    _lines(folder / 'external_torque_segments.jsonl', result['external_torque_segments'])
    atomic_json(folder / 'coordinates.json', {'state_columns': ['q1', 'q2', 'v1', 'v2']
                + (['w1', 'w2'] if dimension == 6 else []), 'time_unit': 'ns', 'q_unit': 'rad',
                'v_unit': 'rad/s', 'command_unit': 'N*m', 'candidate_axes': ['replanning_index', 'global_candidate_id', 'knot', 'joint'],
                'candidate_positive_infinity': 'Only rows with explicit non-ok status and nonempty failure reason'})
    row = {key: summary[key] for key in ('run_id', 'experiment_id', 'phase', 'model_id', 'actuator',
            'seed', 'episode_id', 'condition', 'realized_cost', 'q_rmse_rad', 'terminal_q_error_rad',
            'velocity_rms_rad_s', 'success', 'fallback_count')}
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=list(row))
    writer.writeheader(); writer.writerow(row)
    atomic_bytes(folder / 'episodes.csv', buffer.getvalue().encode())
    atomic_json(folder / 'episode.json', summary)
    return summary
