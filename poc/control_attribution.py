"""Descriptive same-weight scaling contrasts, separate from seed-level inference."""
import numpy as np
from pathlib import Path
from .io import read_json, _reject_constant, _unique_object


def _matching(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_matching(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_matching(a, b) for a, b in zip(left, right))
    if isinstance(left, float) or isinstance(right, float):
        return left is not None and right is not None and bool(np.isclose(left, right, atol=1e-12, rtol=1e-12))
    return left == right


def verify_nominal_power_outputs(low_folder, correct_folder, *, comparison_mode=None):
    """Check already hash-verified episode artifacts; timing measurements may differ."""
    low, correct = Path(low_folder), Path(correct_folder)
    left, right = read_json(low / 'episode.json'), read_json(correct / 'episode.json')
    mode = right['actuator'] if comparison_mode is None else comparison_mode
    if (left['model_id'] != 'power_correct' or right['model_id'] != 'power_correct'
            or mode not in {'direct', 'lagged'}
            or left['power_mode'] != 'low' or right['power_mode'] != mode
            or not left['checkpoint_hash'] or left['checkpoint_hash'] != right['checkpoint_hash']
            or not left['complete'] or not right['complete']
            or left['model_partition_ns'] != [40_000_000] or right['model_partition_ns'] != [40_000_000]):
        raise ValueError('Nominal identity requires the same complete forty-ms checkpoint under low and the declared comparison powers')
    for key in ('phase', 'actuator', 'condition', 'seed', 'episode_id', 'initial_state', 'target',
                'actuator_interval_ns', 'physics_step_ns', 'realized_cost', 'fallback_count'):
        if not _matching(left[key], right[key]):
            raise ValueError('Nominal power query/task/cost differs: ' + key)
    for filename in ('candidates.npz', 'true_trajectories.npz', 'observations.npz'):
        with np.load(low / filename, allow_pickle=False) as a, np.load(correct / filename, allow_pickle=False) as b:
            if set(a.files) != set(b.files):
                raise ValueError('Nominal raw-array coverage differs')
            for name in a.files:
                x, y = a[name], b[name]
                numeric = x.dtype.kind in 'fc'
                if (x.shape != y.shape or x.dtype != y.dtype or
                        not (np.allclose(x, y, atol=1e-12, rtol=1e-12, equal_nan=False)
                             if numeric else np.array_equal(x, y))):
                    raise ValueError('Nominal candidate/trajectory/observation differs: ' + filename + '/' + name)
    for filename in ('action_segments.jsonl', 'external_torque_segments.jsonl', 'decisions.jsonl'):
        import json
        a = [json.loads(line, parse_constant=_reject_constant, object_pairs_hook=_unique_object)
             for line in (low / filename).read_text().splitlines()]
        b = [json.loads(line, parse_constant=_reject_constant, object_pairs_hook=_unique_object)
             for line in (correct / filename).read_text().splitlines()]
        if filename == 'decisions.jsonl':
            a = [{k: v for k, v in row.items() if k != 'planner_wall_seconds'} for row in a]
            b = [{k: v for k, v in row.items() if k != 'planner_wall_seconds'} for row in b]
        if not _matching(a, b):
            raise ValueError('Nominal executed action or selected candidate differs: ' + filename)
    return {'matched': True, 'checkpoint_hash': left['checkpoint_hash'],
            'scope': 'Raw nominal shared-weight candidate/action/trajectory equality', 'tolerance': 1e-12}


def attribution_values(low_nominal, correct_nominal, structured_nominal,
                       low_fine, correct_fine, structured_fine, *, structured_lower_bound=None):
    values = [np.asarray(row, float) for row in (low_nominal, correct_nominal, structured_nominal,
                                               low_fine, correct_fine, structured_fine)]
    if (values[0].ndim != 1 or not len(values[0]) or any(row.shape != values[0].shape
            or not np.isfinite(row).all() or (row < 0).any() for row in values)):
        raise ValueError('All six complete paired seed-level cost vectors are required')
    if not np.allclose(values[0], values[1], atol=1e-12, rtol=1e-12):
        raise ValueError('Nominal shared-weight power identity failed; scaling attribution is confounded')
    dl, dp, ds = [values[fine] - values[nominal] for nominal, fine in ((0, 3), (1, 4), (2, 5))]
    a, b = dl - dp, dl - ds
    mean_a, mean_b = float(a.mean()), float(b.mean())
    if structured_lower_bound is not None and (not np.isfinite(structured_lower_bound)
                                               or structured_lower_bound > mean_b):
        raise ValueError('Structured improvement bound must be finite and consistent with its estimate')
    resolved = structured_lower_bound is not None and structured_lower_bound > 0 and mean_b > 0
    return {'D_low_by_seed': dl.tolist(), 'D_correct_by_seed': dp.tolist(),
        'D_structured_by_seed': ds.tolist(), 'A_by_seed': a.tolist(), 'B_by_seed': b.tolist(),
        'A': mean_a, 'B': mean_b, 'A_minus_half_B_by_seed': (a - .5 * b).tolist(),
        'fraction_explained': mean_a / mean_b if resolved else None,
        'fraction_status': 'positively_resolved_denominator' if resolved else 'full_improvement_not_positively_resolved',
        'structured_lower_bound': structured_lower_bound,
        'statistical_status': 'descriptive_only', 'confidence_bounds': None,
        'identity_scope': 'Cost-vector check; raw candidate/action/trajectory identity must also be verified',
        'scope': 'Fixed-data intervention; no claim that ordinary baselines naturally implement the low-order law'}
