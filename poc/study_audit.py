"""Simple independent raw-matrix aggregation for the decision-report audit."""
import numpy as np


def reference_primary_errors(rows, reference):
    n = len(rows['cell_id'])
    needed = ('q_jac', 'v_jac', 'reference_row_id', 'status', 'probe_id',
              'h_ns', 'stratum', 'cell_id')
    if any(len(rows[name]) != n for name in needed) or not n:
        raise ValueError('Incomplete primary error rows')
    if np.any(rows['status'] != 'ok'):
        raise ValueError('A model failure cannot be dropped from the primary error audit')
    horizons = reference['horizons_ns']
    jacobians = reference['jacobians']
    values = {}
    seen = set()
    for row in range(n):
        position = int(rows['reference_row_id'][row])
        probe, horizon = divmod(position, len(horizons))
        if (position < 0 or probe >= len(jacobians) or
                rows['probe_id'][row] != reference['probe_ids'][probe] or
                rows['h_ns'][row] != horizons[horizon]):
            raise ValueError('Reference coordinate mismatch')
        key = (str(rows['cell_id'][row]), str(rows['stratum'][row]), int(rows['h_ns'][row]))
        unique = (key, str(rows['probe_id'][row]))
        if unique in seen:
            raise ValueError('Duplicated primary error observation')
        seen.add(unique)
        errors = []
        for start, name in ((0, 'q_jac'), (2, 'v_jac')):
            model = rows[name][row]
            truth = jacobians[probe, horizon, start:start+2, :]
            if model.shape != (2, 2) or truth.shape != (2, 2) or not np.isfinite(model).all() or not np.isfinite(truth).all():
                raise ValueError('Invalid primary error matrices')
            denominator = float(np.sqrt(np.sum(truth*truth)))
            if denominator <= 0:
                raise ValueError('Unresolved primary error denominator')
            difference = model-truth
            errors.append(float(np.sqrt(np.sum(difference*difference)))/denominator)
        values.setdefault(key, []).append(sum(errors)/2)
    return [{'cell_id': cell, 'stratum': stratum, 'h_ns': horizon, 'n_probes': len(numbers),
             'primary_error': float(np.median(numbers))}
            for (cell, stratum, horizon), numbers in sorted(values.items())]


def paired_primary_results(audit, scope, strata=('cold', 'warm'), horizons=(5_000_000, 10_000_000)):
    results = []
    for stratum in strata:
        for horizon in horizons:
            values, counts = {}, set()
            for row in audit:
                if row['role'] != 'E01' or row['stratum'] != stratum or row['h_ns'] != horizon:
                    continue
                for model in (scope['competitor'], 'power_correct'):
                    for seed in scope['seed_ids']:
                        cell = '__'.join((scope['actuator'], scope['condition'], scope['loss_mode'], model, f'seed{seed}'))
                        if row['cell_id'] != cell:
                            continue
                        key = model, seed
                        if key in values or not np.isfinite(row['primary_error']) or row['primary_error'] < 0:
                            raise ValueError('Duplicated or invalid paired primary statistic')
                        values[key] = row['primary_error']
                        counts.add(row['n_probes'])
            required = {(model, seed) for model in (scope['competitor'], 'power_correct') for seed in scope['seed_ids']}
            if set(values) != required or len(counts) != 1:
                raise ValueError('Incomplete or inconsistent paired primary statistics')
            baseline = [values[scope['competitor'], seed] for seed in scope['seed_ids']]
            method = [values['power_correct', seed] for seed in scope['seed_ids']]
            b, m = float(np.mean(baseline)), float(np.mean(method))
            results.append({'stratum': stratum, 'h_ns': horizon, 'seed_ids': scope['seed_ids'],
                'n_probes_per_seed': counts.pop(), 'baseline_by_seed': baseline,
                'method_by_seed': method, 'baseline_mean': b, 'method_mean': m,
                'relative_reduction': 1-m/b if b > 0 else None})
    return results
