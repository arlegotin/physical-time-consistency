"""Paired training-seed inference, fixed-tail safeguards and append-only looks."""

import hashlib
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.stats import t

from poc.io import atomic_json, canonical_json, job_lock, read_json, utc_now


def look_alpha(look: int) -> float:
    if isinstance(look, bool) or not isinstance(look, int) or look < 1:
        raise ValueError('A confirmatory look is a positive integer')
    alpha = .05 / 2**look
    if alpha == 0:
        raise ValueError('Alpha needs higher precision at this look')
    return alpha


def bootstrap_indices(n: int, look: int) -> np.ndarray:
    if n < 2:
        raise ValueError('At least two independent seed clusters are required')
    look_alpha(look)
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([9101, look])))
    return rng.integers(0, n, size=(20_000, n), dtype=np.int64)


def _sample(values):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('Finite one-dimensional independent seed values are required')
    return x


def _alpha(alpha):
    if not np.isfinite(alpha) or not 0 < alpha < .5:
        raise ValueError('One-sided alpha must lie strictly between zero and one half')


def _envelope(envelope):
    lower, upper = float(envelope.get('lower', 0.)), float(envelope.get('upper', 0.))
    if not np.isfinite([lower, upper]).all() or min(lower, upper) < 0:
        raise ValueError('Outward numerical envelopes must be finite and nonnegative')
    return lower, upper


def _t_bounds(x, alpha):
    mean, sd = float(x.mean()), float(x.std(ddof=1))
    half = float(t.ppf(1-alpha, len(x)-1)*sd/np.sqrt(len(x)))
    return mean-half, mean+half


def _fraction_bounds(x, alpha):
    if np.all(x == 1):
        return alpha**(1/len(x)), 1.
    if np.all(x == 0):
        return 0., 1-alpha**(1/len(x))
    if x.std(ddof=1) == 0 or np.count_nonzero((x <= 1/256) | (x >= 1-1/256)) >= len(x)/2:
        half = math.sqrt(math.log(1/alpha)/(2*len(x)))
        return max(0., float(x.mean())-half), min(1., float(x.mean())+half)
    low, high = _t_bounds(x, alpha)
    return max(0., low), min(1., high)


def _influence(x, alpha, full_bounds, fraction=False, boundary=0.):
    centered = np.abs(x-x.mean())
    total = centered.sum()
    share = float(centered.max()/total) if total > 0 else 0.
    full = 1 if full_bounds[0] >= boundary else -1 if full_bounds[1] < boundary else 0
    rows, changed = [], False
    for i in range(len(x)):
        subset = np.delete(x, i)
        if len(subset) < 2:
            rows.append({'omitted_seed_index': i, 'mean': float(subset.mean()), 'interpretation': None})
            continue
        bounds = _fraction_bounds(subset, alpha) if fraction else _t_bounds(subset, alpha)
        interpretation = 1 if bounds[0] >= boundary else -1 if bounds[1] < boundary else 0
        changed |= interpretation != full
        rows.append({'omitted_seed_index': i, 'mean': float(subset.mean()), 'bounds': list(bounds),
                     'interpretation': interpretation})
    return {'unstable': bool(share > .5 or changed), 'max_centered_share': share,
            'leave_one_out_changed': bool(changed), 'leave_one_out': rows}


def contrast_bounds(values, alpha, bootstrap_indices, numerical_envelope, bounded_fraction=False,
                    influence_boundary=0.) -> dict:
    """Bound E[X]. Fractions receive their bounded rules before subtracting a gate limit.

    The numerical envelope contains additive outward displacements of E[X];
    callers construct these from the full per-seed metric intervals.
    """
    x = _sample(values)
    _alpha(alpha)
    indices = np.asarray(bootstrap_indices)
    if indices.shape != (20_000, len(x)) or indices.dtype.kind not in 'iu' or np.any(indices < 0) or np.any(indices >= len(x)):
        raise ValueError('A shared 20,000-resample seed-cluster index array is required')
    lo_eta, hi_eta = _envelope(numerical_envelope)
    means = x[indices].mean(axis=1)
    central = [float(v) for v in np.quantile(means, [.025, .975], method='linear')]
    result = {'status': 'ok', 'n': len(x), 'mean': float(x.mean()), 'sd': float(x.std(ddof=1)),
              'alpha': alpha, 'quantile_method': 'linear', 'central_bootstrap': central,
              'bootstrap_indices_sha256': hashlib.sha256(indices.tobytes()).hexdigest(),
              'numerical_envelope': {'lower': lo_eta, 'upper': hi_eta}}
    if bounded_fraction:
        if np.any((x < 0) | (x > 1)):
            raise ValueError('Fractions must lie in [0,1]')
        if np.all(x == 1):
            lower, upper, rule = alpha**(1/len(x)), 1., 'bounded_all_one'
        elif np.all(x == 0):
            lower, upper, rule = 0., 1-alpha**(1/len(x)), 'bounded_all_zero'
        elif result['sd'] == 0 or np.count_nonzero((x <= 1/256) | (x >= 1-1/256)) >= len(x)/2:
            half = math.sqrt(math.log(1/alpha)/(2*len(x)))
            lower, upper, rule = max(0., result['mean']-half), min(1., result['mean']+half), 'bounded_hoeffding'
        else:
            lower, upper = _t_bounds(x, alpha)
            rule = 't_bootstrap' if 20_000*alpha >= 100 else 't_rare_tail_with_central_bootstrap'
            if 20_000*alpha >= 100:
                lower = min(lower, float(np.quantile(means, alpha, method='linear')))
                upper = max(upper, float(np.quantile(means, 1-alpha, method='linear')))
            lower, upper = max(0., lower), min(1., upper)
    elif result['sd'] == 0:
        return {**result, 'status': 'unresolved_variance', 'lower': None, 'upper': None,
                'rule': 'constant_unbounded_requires_diagnostic',
                'influence': {'unstable': True, 'max_centered_share': 0., 'leave_one_out': []}}
    else:
        lower, upper = _t_bounds(x, alpha)
        rule = 't_bootstrap' if 20_000*alpha >= 100 else 't_rare_tail_with_central_bootstrap'
        if 20_000*alpha >= 100:
            lower = min(lower, float(np.quantile(means, alpha, method='linear')))
            upper = max(upper, float(np.quantile(means, 1-alpha, method='linear')))
    result.update(lower=lower-lo_eta, upper=upper+hi_eta, rule=rule)
    result['influence'] = _influence(x, alpha, (lower, upper), fraction=bounded_fraction,
                                      boundary=influence_boundary)
    return result


def fieller_improvement(baseline, method, alpha, numerical_envelope=None, indices=None) -> dict:
    b, m = _sample(baseline), _sample(method)
    _alpha(alpha)
    if b.shape != m.shape:
        raise ValueError('Fieller pairs require identical seed clusters')
    n, bm, mm = len(b), float(b.mean()), float(m.mean())
    if bm <= 0:
        return {'status': 'unresolved_denominator', 'lower': None, 'upper': None, 'width': None}
    covariance = np.cov(b, m, ddof=1)
    if np.linalg.eigvalsh(covariance).min() < -1e-12*max(1., np.max(np.abs(covariance))):
        raise ValueError('Fieller covariance is not positive semidefinite')
    critical = float(t.ppf(1-alpha/2, n-1))
    vb, vm, c = covariance[0, 0], covariance[1, 1], covariance[0, 1]
    a = bm*bm-critical**2*vb/n
    d = -2*bm*mm+2*critical**2*c/n
    e = mm*mm-critical**2*vm/n
    disc = d*d-4*a*e
    if a <= 0:
        return {'status': 'unresolved_denominator', 'lower': None, 'upper': None, 'width': None,
                'quadratic': [float(a), float(d), float(e)]}
    if disc < -1e-12*max(1., d*d, abs(4*a*e)):
        raise ValueError('Negative Fieller discriminant with positive-definite denominator')
    disc = max(0., disc)
    roots = sorted(((-d-math.sqrt(disc))/(2*a), (-d+math.sqrt(disc))/(2*a)))
    eta_l, eta_u = _envelope(numerical_envelope or {})
    lower, upper = 1-roots[1]-eta_l, 1-roots[0]+eta_u
    result = {'status': 'ok', 'estimate': 1-mm/bm, 'lower': lower, 'upper': upper,
              'width': upper-lower, 'quadratic': [float(a), float(d), float(e)],
              'alpha': alpha, 'assumption': 'approximately joint-normal seed means'}
    if indices is not None:
        boot_b, boot_m = b[indices].mean(axis=1), m[indices].mean(axis=1)
        if np.any(boot_b <= 0):
            result['central_ratio_bootstrap'] = None
        else:
            result['central_ratio_bootstrap'] = np.quantile(1-boot_m/boot_b, [.025, .975], method='linear').tolist()
    return result


def plan_precision(summary, pilot, look):
    n = int(summary['n'])
    desired = float(summary['desired_half_width'])
    sd = 1.25*max(float(summary['sd']), float(pilot.get('sd', 0.)))
    if n < 5 or desired <= 0 or sd < 0 or not np.isfinite([desired, sd]).all():
        raise ValueError('Invalid independent-seed precision request')
    alpha = look_alpha(look)
    minimum = n+5
    # Monotone bracketing/bisection avoids an artificial sample cap.
    def half(total):
        return float(t.ppf(1-alpha, total-1)*sd/math.sqrt(total))
    high = max(2*n, minimum)
    while half(high) > desired:
        high *= 2
    low = minimum
    while low < high:
        mid = (low+high)//2
        if half(mid) <= desired:
            high = mid
        else:
            low = mid+1
    last = int(summary.get('last_seed', n-1))
    return {'current_seeds': n, 'total_seeds': high, 'new_seed_ids': list(range(last+1, last+1+high-n)),
            'planning_sd': sd, 'desired_half_width': desired, 'predicted_half_width': half(high),
            'look_index': look, 'alpha': alpha, 'reason': 'additional independent training-seed clusters'}


def reserve_look(root: Path, evidence_identity: dict) -> dict:
    """Commit a look before aggregating its outcomes; repeated evidence is idempotent."""
    root = Path(root)
    key = hashlib.sha256(canonical_json(evidence_identity)).hexdigest()
    folder = root/'decisions'/'statistical_looks'
    folder.mkdir(parents=True, exist_ok=True)
    lock_key = hashlib.sha256(b'PTCO-1.0 statistical-look-ledger').hexdigest()
    with job_lock(lock_key, root):
        records = [read_json(path) for path in sorted(folder.glob('look-*.json'))]
        for i, record in enumerate(records, 1):
            if record['look_index'] != i or record['alpha'] != look_alpha(i):
                raise ValueError('Statistical look ledger is inconsistent')
            if record['evidence_sha256'] != hashlib.sha256(canonical_json(record['evidence_identity'])).hexdigest():
                raise ValueError('Statistical look evidence identity hash mismatch')
            if record['evidence_sha256'] == key:
                return record
        look = len(records)+1
        record = {'schema_version': 1, 'protocol_version': 'PTCO-1.0',
                  'look_index': look, 'alpha': look_alpha(look), 'rejection_alpha': look_alpha(look)/280,
                  'family_size': 280, 'cumulative_alpha': float(Fraction(1,20)*(1-Fraction(1,2)**look)),
                  'evidence_identity': evidence_identity, 'evidence_sha256': key,
                  'bootstrap_seed_entropy': [9101, look], 'created_utc': utc_now()}
        atomic_json(folder/f'look-{look:05d}.json', record)
        return record
