"""Pair complete physical episode costs without treating episodes as training seeds."""
from __future__ import annotations
import numpy as np


def paired_cost_summary(method_rows, baseline_rows, expected_pairs):
    expected = [tuple(pair) for pair in expected_pairs]
    if (not expected or len(set(expected)) != len(expected)
            or any(len(pair) != 2 or any(type(i) is not int or i < 0 for i in pair) for pair in expected)):
        raise ValueError('An explicit unique seed/episode workload is required')
    seeds = sorted({pair[0] for pair in expected})
    counts = [sum(pair[0] == seed for pair in expected) for seed in seeds]
    if len(set(counts)) != 1:
        raise ValueError('Each independent training seed must have the same paired episode coverage')

    def indexed(rows):
        keys = [(row['seed'], row['episode_id']) for row in rows]
        if len(keys) != len(expected) or set(keys) != set(expected):
            raise ValueError('Complete paired workload required; duplicate or omitted episodes cannot be averaged')
        for row in rows:
            if (row['complete'] is not True or not np.isfinite(row['realized_cost'])
                    or row['realized_cost'] < 0 or type(row['fallback_count']) is not int
                    or row['fallback_count'] < 0):
                raise ValueError('Only complete finite observed episode costs may enter the comparison')
        return dict(zip(keys, rows))

    method, baseline = indexed(method_rows), indexed(baseline_rows)
    pairing_keys = ('phase', 'actuator', 'condition', 'actuator_interval_ns', 'model_partition_ns',
                    'physics_step_ns', 'initial_state', 'target')
    for pair in expected:
        if any(method[pair][key] != baseline[pair][key] for key in pairing_keys):
            raise ValueError('A paired comparison must share task, physical condition and clock schedules')
    if len({method[p]['phase'] for p in expected}) != 1:
        raise ValueError('Exploratory and confirmatory episodes cannot be pooled')
    baseline_means = [float(np.mean([baseline[p]['realized_cost'] for p in expected if p[0] == seed]))
                      for seed in seeds]
    method_means = [float(np.mean([method[p]['realized_cost'] for p in expected if p[0] == seed]))
                    for seed in seeds]
    b, m = float(np.mean(baseline_means)), float(np.mean(method_means))
    if not np.isfinite([b, m]).all():
        raise ValueError('Cost aggregation overflowed')
    return {'phase': method[expected[0]]['phase'], 'seed_ids': seeds, 'n_training_seeds': len(seeds),
        'episodes_per_seed': counts[0], 'baseline_mean_cost': b, 'method_mean_cost': m,
        'absolute_reduction': b - m, 'relative_reduction': 1 - m / b if b > 0 else None,
        'denominator_status': 'resolved_observed_mean' if b > 0 else 'zero_baseline_mean',
        'baseline_seed_means': baseline_means, 'method_seed_means': method_means,
        'seed_contrasts': [x - y for x, y in zip(baseline_means, method_means)],
        'method_fallback_count': sum(row['fallback_count'] for row in method_rows),
        'baseline_fallback_count': sum(row['fallback_count'] for row in baseline_rows),
        'statistical_status': 'descriptive_only', 'confidence_bounds': None,
        'scope': 'Paired measured costs; seed-level inference requires its prospectively frozen contrast procedure'}
