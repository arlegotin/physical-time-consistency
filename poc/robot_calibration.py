"""Complete fixed-workload robot headroom and common-bank ranking checks."""
from __future__ import annotations

import numpy as np


def cost_envelope(refinement_costs) -> np.ndarray:
    """Empirical envelope from two successive solver refinements plus roundoff."""
    values = np.asarray(refinement_costs, float)
    if values.ndim != 2 or values.shape[0] != 3 or not np.isfinite(values).all():
        raise ValueError('Three complete finite resolution levels are required')
    return np.max(abs(np.diff(values, axis=0)), axis=0) + 64 * np.finfo(float).eps * (1 + abs(values[0]))


def summarize(rows: list[dict]) -> dict:
    """Pure qualification calculation; caller first verifies each run's hashes."""
    expected = {(a, e, m) for a in ('direct', 'lagged') for e in range(16)
                for m in ('zero_command', 'physics_fit', 'sim_oracle')}
    keys = [(r['actuator'], r['episode_id'], r['model_id']) for r in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError('Robot calibration requires all 96 fixed reference episodes exactly once')
    indexed = dict(zip(keys, rows))
    for row in rows:
        if (row['phase'] != 'calibration' or row['seed'] != 9000 or row['condition'] != 'clean'
                or row['complete'] is not True or row['fallback_count'] != 0
                or row['actuator_interval_ns'] != 40_000_000
                or row['model_partition_ns'] != [40_000_000] or row['physics_step_ns'] != 250_000
                or not np.isfinite(row['realized_cost']) or row['realized_cost'] < 0):
            raise ValueError('Invalid or mismatched robot calibration episode')
        zero = indexed[(row['actuator'], row['episode_id'], 'zero_command')]
        if (row['initial_state'] != zero['initial_state'] or row['target'] != zero['target']
                or zero['realized_cost'] <= 0):
            raise ValueError('Paired robot calibration must retain the same informative task')
    comparisons = []
    for actuator in ('direct', 'lagged'):
        zero = np.array([indexed[actuator, e, 'zero_command']['realized_cost'] for e in range(16)])
        for model in ('physics_fit', 'sim_oracle'):
            costs = np.array([indexed[actuator, e, model]['realized_cost'] for e in range(16)])
            reduction = float(1 - costs.mean() / zero.mean())
            comparisons.append({'actuator': actuator, 'model_id': model,
                'zero_mean_cost': float(zero.mean()), 'reference_mean_cost': float(costs.mean()),
                'mean_reduction': reduction, 'qualification_floor': .20, 'qualified': reduction >= .20})
    return {'qualified': all(c['qualified'] for c in comparisons), 'episode_count': len(rows),
            'comparisons': comparisons,
            'scope': 'Model-independent planner headroom; no learned-model conclusion',
            'common_bank_ranking': 'Separately required before freezing the planner'}


def ranking_check(oracle_costs, physics_costs, oracle_envelope, physics_envelope) -> dict:
    """Count inversions resolved outside independently measured cost envelopes."""
    oracle, physics = np.asarray(oracle_costs, float), np.asarray(physics_costs, float)
    if (oracle.ndim != 1 or not len(oracle) or physics.shape != oracle.shape
            or not np.isfinite(oracle).all() or not np.isfinite(physics).all()):
        raise ValueError('Complete finite scores from the identical bank are required')
    oe, pe = np.broadcast_to(oracle_envelope, oracle.shape), np.broadcast_to(physics_envelope, physics.shape)
    if not np.isfinite(oe).all() or not np.isfinite(pe).all() or (oe < 0).any() or (pe < 0).any():
        raise ValueError('Numerical envelopes must be finite and nonnegative')
    i, j = np.triu_indices(len(oracle), 1)
    od, pd = oracle[i] - oracle[j], physics[i] - physics[j]
    inversions = od * pd < 0
    resolved = inversions & (abs(od) > oe[i] + oe[j]) & (abs(pd) > pe[i] + pe[j])
    return {'qualified': not bool(resolved.any()), 'candidate_count': len(oracle),
            'resolved_inversions': int(resolved.sum()),
            'uncertain_inversions': int((inversions & ~resolved).sum()),
            'maximum_absolute_cost_disagreement': float(abs(oracle - physics).max()),
            'oracle_best_id': int(np.argmin(oracle)), 'physics_best_id': int(np.argmin(physics))}
