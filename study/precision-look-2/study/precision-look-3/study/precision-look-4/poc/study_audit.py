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
