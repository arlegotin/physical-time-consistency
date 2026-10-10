"""Atomic contrast gates. Evidence producers supply hash-validated seed summaries."""

import numpy as np

from poc.statistics import bootstrap_indices, contrast_bounds, fieller_improvement, look_alpha


def evaluate_criterion(criterion, summary, look):
    identifier = criterion['criterion_id']
    result = {'criterion_id': identifier, 'status': 'unresolved', 'reason': '',
              'estimate': None, 'bounds': None, 'threshold': criterion['threshold'],
              'operator': criterion['direction']}
    if summary.get('validity') != 'valid' or 'values' not in summary:
        return {**result, 'status': 'invalid', 'reason': 'Missing or invalid seed evidence before arithmetic'}
    if summary.get('qualification_status') == 'unresolved':
        return {**result,'reason':summary['qualification_reason'],'needs_branch':summary['needs_branch']}
    if not criterion['statistical']:
        return {**result, 'status': 'invalid', 'reason': 'A deterministic invariant needs its independent validity evaluator'}
    try:
        values = np.asarray(summary['values'], dtype=np.float64)
        if len(values) < 5:
            return {**result, 'reason': 'At least five independent confirmation seeds are required'}
        indices = bootstrap_indices(len(values), look)
        alpha = look_alpha(look)
        fraction = identifier.startswith('K02.gain/')
        boundary = criterion['threshold'] if fraction else 0.
        envelope = summary.get('numerical_envelope', {})
        passed = contrast_bounds(values, alpha, indices, envelope, fraction, boundary)
        rejected = contrast_bounds(values, alpha/280, indices, envelope, fraction, boundary)
        result.update(pass_bounds=passed, rejection_bounds=rejected,
                      estimate=float(values.mean()), look_index=look, alpha_spent=alpha)
        if passed['status'] != 'ok' or rejected['status'] != 'ok':
            return {**result, 'reason': 'Constant unbounded contrast requires a variance/implementation diagnostic'}
        strict = criterion['direction'] == '>'
        low = passed['lower']
        point_ok = values.mean() > boundary if strict else values.mean() >= boundary
        bound_ok = low > boundary if strict else low >= boundary
        rare_pass_ok = ('rare_tail' not in passed['rule'] or passed['central_bootstrap'][0] > boundary)
        influence = passed['influence']['unstable'] or rejected['influence']['unstable']
        if point_ok and bound_ok and rare_pass_ok and not influence:
            return {**result, 'status': 'pass', 'bounds': [passed['lower'], passed['upper']],
                    'reason': 'Point estimate and preregistered lower bound clear the boundary'}
        rare_reject_ok = ('rare_tail' not in rejected['rule'] or rejected['central_bootstrap'][1] < boundary)
        if rejected['upper'] < boundary and rare_reject_ok and not influence:
            required = criterion['required_precision']
            if identifier.startswith(('K02.relative/', 'K02.actuator/', 'K03.jacobian/', 'K04.', 'K05.')):
                if 'baseline' not in summary or 'method' not in summary:
                    return {**result, 'reason': 'Rejection needs paired Fieller percentage precision'}
                if 'ratio_numerical_envelope' in summary and summary['ratio_numerical_envelope'] is None:
                    return {**result,'reason':'Numerical uncertainty leaves the baseline error denominator unresolved'}
                ratio = fieller_improvement(summary['baseline'], summary['method'], alpha/280,
                                            summary.get('ratio_numerical_envelope', {}), indices)
                result['ratio_interval'] = ratio
                precise = ratio['status'] == 'ok' and (required is None or ratio['width'] <= required)
            else:
                # Two-sided interval uses alpha/(2K), including fraction safeguards.
                two_sided = contrast_bounds(values, alpha/(2*280), indices, envelope, fraction, boundary)
                result['precision_bounds'] = two_sided
                precise = two_sided['status'] == 'ok' and (required is None or two_sided['upper']-two_sided['lower'] <= required)
            if precise:
                return {**result, 'status': 'fail', 'bounds': [rejected['lower'], rejected['upper']],
                        'reason': 'Family-corrected upper bound rejects the boundary with required precision'}
        return {**result, 'reason': 'Unresolved boundary, precision, bootstrap robustness or seed influence'}
    except (ValueError, TypeError, KeyError, OverflowError) as error:
        return {**result, 'status': 'invalid', 'reason': str(error)}
