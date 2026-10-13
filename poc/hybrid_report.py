"""Fixed prospective hybrid component decision, derived from verified raw evidence."""
from __future__ import annotations

from collections import defaultdict
import math

import numpy as np

from .io import canonical_json

PAIR_IDENTITY = ('session_id', 'budget_s', 'initialization_id', 'state_id',
                 'trajectory_id', 'anchor_ns', 'query_step_ns', 'truth_receipt')
ERROR_FIELDS = tuple(c+'_'+m for c in ('q', 'v')
                     for m in ('conservative_rmse_upper', 'conservative_p95_upper'))


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def paired_cost_comparison(candidate_outcomes: list[dict], anchor_outcomes: list[dict]) -> dict:
    """Cancel identical selected commands; otherwise retain independent cost bounds.

    Model identity is deliberately excluded; all state, budget, initialization and
    shared truth identities must match. Ranking ties remain disclosed and usable.
    """
    result = {'status': 'unresolved', 'reasons': [], 'pair_count': 0,
              'same_command_count': 0, 'mean_cost_difference': None,
              'mean_joint_envelope': None, 'ranking_ties_disclosed': False,
              'ranking_statuses': [], 'pairs': []}
    def index(rows):
        indexed = {}
        for row in rows:
            if any(k not in row or row[k] is None for k in PAIR_IDENTITY):
                raise ValueError('missing_paired_identity')
            key = canonical_json({k: row[k] for k in PAIR_IDENTITY})
            if key in indexed:
                raise ValueError('duplicate_paired_identity')
            indexed[key] = row
        return indexed
    try:
        candidate, anchor = index(candidate_outcomes), index(anchor_outcomes)
    except (ValueError, TypeError) as error:
        result['reasons'] = [str(error)]
        return result
    if not candidate or set(candidate) != set(anchor):
        result['reasons'] = ['missing_or_unmatched_paired_decisions']
        return result
    for key, row in sorted(candidate.items()):
        other = anchor[key]
        rankings = {'candidate': row.get('truth_ranking_status'),
                    'anchor': other.get('truth_ranking_status')}
        result['ranking_statuses'].append(rankings)
        result['ranking_ties_disclosed'] |= 'unresolved' in rankings.values()
        for selected in (row, other):
            if (selected.get('decision_failure') is not False or
                    selected.get('selected_unsafe') is not False or
                    selected.get('no_selection') is not False or
                    selected.get('no_true_feasible_bank') is not False or
                    selected.get('truth_feasibility_status') != 'resolved' or
                    selected.get('selected_index') is None or
                    not selected.get('selected_command_sha256') or
                    not _finite(selected.get('selected_truth_cost')) or
                    not _finite(selected.get('selected_truth_envelope')) or
                    selected['selected_truth_envelope'] < 0):
                result['reasons'] = ['invalid_selected_cost_or_sampled_feasibility']
                return result
        same = row['selected_command_sha256'] == other['selected_command_sha256']
        difference = 0. if same else row['selected_truth_cost'] - other['selected_truth_cost']
        envelope = 0. if same else row['selected_truth_envelope'] + other['selected_truth_envelope']
        result['pairs'].append({'identity': {k: row[k] for k in PAIR_IDENTITY},
                                'same_command': same, 'cost_difference': difference,
                                'joint_envelope': envelope, 'ranking_status': rankings})
        result['same_command_count'] += int(same)
    result['pair_count'] = len(result['pairs'])
    result['mean_cost_difference'] = float(np.mean([p['cost_difference'] for p in result['pairs']]))
    result['mean_joint_envelope'] = float(np.mean([p['joint_envelope'] for p in result['pairs']]))
    # Subtraction roundoff must not turn an exactly declared boundary into a loss.
    difference, envelope = result['mean_cost_difference'], result['mean_joint_envelope']
    passes = difference <= envelope or math.isclose(difference, envelope, rel_tol=1e-14, abs_tol=0.)
    result['status'] = 'pass' if passes else 'fail'
    result['reasons'] = [] if passes else ['paired_mean_cost_exceeds_joint_numerical_envelope']
    return result


def _scorer_pool(result, model, budget, step, declaration):
    """Pool 50 coordinate-bound actual calls per session, never group medians."""
    samples, reasons = [], []
    per_session = []
    for session in declaration['session_ids']:
        groups = [g for g in result['timing'] if all(g['coordinate'].get(k) == v for k, v in
                  dict(model_id=model, budget_s=budget, query_step_ns=step, session_id=session,
                       initialization_id=0, category='complete_scorer').items())]
        valid = len(groups) == 1
        rows = groups[0].get('samples', []) if valid else []
        valid = (valid and len(rows) == 50 and
                 {r.get('repetition') for r in rows} == set(range(50)) and
                 all(r.get('coordinate') == groups[0]['coordinate'] and r.get('status') == 'ok'
                     and all(_finite(r.get(k)) and r[k] >= 0 for k in ('wall_s', 'cpu_s')) for r in rows))
        per_session.append({'session_id': session, 'status': 'pass' if valid else 'unresolved',
                            'sample_count': len(rows), 'coordinate': groups[0]['coordinate'] if len(groups)==1 else None})
        if not valid:
            reasons.append('missing_or_failed_50_coordinate_bound_scorer_calls_session_'+str(session))
        else:
            samples.extend(rows)
    usable = not reasons
    return {'status': 'pass' if usable else 'unresolved', 'reasons': reasons,
            'sample_count': len(samples), 'per_session': per_session,
            **{k: {'median': float(np.median([r[k] for r in samples])) if usable else None,
                   'p95': float(np.percentile([r[k] for r in samples], 95)) if usable else None}
               for k in ('wall_s', 'cpu_s')},
            'pooling': '50 actual complete-scorer calls per session; rotated-order raw provenance retained'}


def hybrid_gate(result: dict) -> dict:
    """Evaluate every session/configuration, then apply the fixed 30s/40ms bar."""
    recipe = result['recipe']
    declaration = recipe['hybrid_component']
    models = recipe['models']
    flow, anchor = declaration['primary_model_id'], declaration['anchor_model_id']
    primary = {'budget_s': declaration['primary_budget_s'],
               'query_step_ns': declaration['primary_query_step_ns']}
    rows = defaultdict(list)
    for row in result['replica_configurations']:
        rows[row['model_id'], row['budget_s'], row['query_step_ns'], row['session_id']].append(row)
    configurations = []
    failed = result['coverage'].get('failed', [])
    for budget in recipe['budgets_s']:
        for step in recipe['evaluation']['query_steps_ns']:
            for model in models:
                sessions = []
                for session in declaration['session_ids']:
                    candidates = rows[model, budget, step, session]
                    anchors = rows[anchor, budget, step, session]
                    record = {'session_id': session, 'status': 'unresolved', 'reasons': [],
                              'optimization_limited': any(c.get('optimization_limited', False) for c in candidates)}
                    if len(candidates) != 1 or len(anchors) != 1:
                        record['reasons'] = ['missing_or_duplicate_session_configuration_or_anchor']
                        sessions.append(record)
                        continue
                    candidate, reference = candidates[0], anchors[0]
                    anchor_failed = any(c['model_id']==anchor and c['budget_s']==budget and
                                        c['session_id']==session for c in failed)
                    if anchor_failed or reference['screen']['prediction_failure_count']:
                        record['reasons'] = ['missing_or_failed_anchor']
                        sessions.append(record)
                        continue
                    def decisions(method):
                        return [o for o in result['decision_outcomes'] if all(o.get(k)==v for k,v in
                                dict(model_id=method, session_id=session, budget_s=budget,
                                     query_step_ns=step, initialization_id=0).items())]
                    paired = paired_cost_comparison(decisions(model), decisions(anchor))
                    record['paired_cost_comparison'] = paired
                    reasons, unresolved = [], []
                    screen = candidate['screen']['status']
                    if screen != 'pass':
                        (unresolved if screen=='unresolved' else reasons).append('provisional_simulator_screen_'+screen)
                    comparisons = {}
                    for field in ERROR_FIELDS:
                        value, bound = candidate.get(field), reference.get(field)
                        comparisons[field] = {'candidate': value, 'anchor': bound,
                                              'limit': declaration['error_ratio_limit']*bound if _finite(bound) else None}
                        if not _finite(value) or not _finite(bound) or value < 0 or bound < 0:
                            unresolved.append(field+'_undefined')
                        elif value > declaration['error_ratio_limit']*bound:
                            reasons.append(field+'_over_5_percent_band')
                    for field in ('prediction_failure_count', 'decision_failure_fraction'):
                        value, bound = candidate['screen'].get(field), reference['screen'].get(field)
                        comparisons[field] = {'candidate': value, 'anchor': bound}
                        if not _finite(value) or not _finite(bound):
                            unresolved.append(field+'_undefined')
                        elif value > bound:
                            reasons.append(field+'_worse_than_anchor')
                    if paired['status'] != 'pass':
                        (unresolved if paired['status']=='unresolved' else reasons).extend(paired['reasons'])
                    # A demonstrated quality loss remains a negative even if its
                    # conditional regret is undefined; undefined alone cannot pass.
                    record.update(status='fail' if reasons else 'unresolved' if unresolved else 'pass',
                                  reasons=reasons+unresolved, comparisons=comparisons)
                    sessions.append(record)
                quality = ('fail' if any(s['status']=='fail' for s in sessions) else
                           'unresolved' if any(s['status']=='unresolved' for s in sessions) else 'pass')
                pool = _scorer_pool(result, model, budget, step, declaration)
                status = 'unresolved' if quality=='pass' and pool['status']!='pass' else quality
                configurations.append({'model_id': model, 'budget_s': budget, 'query_step_ns': step,
                    'status': status, 'quality_status': quality, 'sessions': sessions, 'scorer_pool': pool})
    selected = [c for c in configurations if all(c[k]==v for k,v in primary.items())]
    by_model = {c['model_id']: c for c in selected}
    gate = {'decision': 'unresolved', 'reasons': [], 'primary': primary,
            'anchor_model_id': anchor, 'primary_model_id': flow,
            'configurations': configurations,
            'primary_pool_stats': {m: c['scorer_pool'] for m,c in by_model.items()},
            'fastest_acceptable_alternative': None, 'median_ratio': None,
            'median_saving_fraction': None, 'p95_ratio': None,
            'stage_c_launched': False,
            'scope': 'exploratory simulator component screen; no physical deadline or commissioning claim'}
    def decide(name, reason):
        gate.update(decision=name, reasons=[reason])
        return gate
    if not result['coverage']['complete']:
        return decide('unresolved', 'incomplete_or_invalid_evidence')
    if not result.get('empirical_claims_allowed', False):
        return decide('unresolved', 'unit_fixture_ineligible_for_empirical_claims')
    candidate = by_model.get(flow)
    if candidate is None or candidate['status']=='unresolved':
        return decide('unresolved', 'primary_flow_or_anchor_unresolved')
    if candidate['status']=='fail':
        return decide('no_component_advantage', 'primary_flow_not_acceptable_in_every_session')
    alternatives = [c for c in selected if c['model_id'] != flow and c['status']=='pass']
    if not alternatives:
        return decide('unresolved', 'no_resolved_acceptable_alternative')
    fastest = min(alternatives, key=lambda c: (c['scorer_pool']['wall_s']['median'], c['model_id']))
    baseline = fastest['scorer_pool']['wall_s']; measured = candidate['scorer_pool']['wall_s']
    gate['fastest_acceptable_alternative'] = fastest['model_id']
    if baseline['median'] <= 0 or baseline['p95'] <= 0:
        return decide('unresolved', 'zero_comparator_duration')
    ratio = measured['median']/baseline['median']
    gate.update(median_ratio=ratio, median_saving_fraction=1.-ratio,
                p95_ratio=measured['p95']/baseline['p95'])
    gain = ratio <= declaration['median_ratio_limit'] and gate['p95_ratio'] <= declaration['p95_ratio_limit']
    cheap = [c for c in alternatives if c['model_id'] in {'hybrid_affine_aware','hybrid_accel_aware'}]
    # A cheap median can prevent the saving bar even when another model is
    # faster. The p95 guard is declared only against the fastest comparator.
    prevents = any(measured['median'] > declaration['median_ratio_limit']*c['scorer_pool']['wall_s']['median']
                   for c in cheap) or (fastest in cheap and gate['p95_ratio'] > declaration['p95_ratio_limit'])
    if not gain and prevents:
        return decide('cheap_solver_matches', 'acceptable_two_evaluation_baseline_prevents_declared_advantage')
    if any(c['status']=='unresolved' for c in selected if c['model_id']!=flow):
        return decide('unresolved', 'alternative_quality_or_pooled_timing_unresolved')
    return decide('component_advantage' if gain else 'no_component_advantage',
                  'fixed_primary_component_bar_passed' if gain else 'fixed_primary_median_or_p95_bar_not_met')


def component_findings(result: dict) -> str:
    gate = result['gate']
    lines = ['# Hybrid AffineFlow component findings', '',
             f'Gate: **{gate["decision"]}**. Reasons: '+', '.join(gate['reasons'])+'.', '',
             'The fixed primary configuration is 30 seconds / 40 ms. Budget 120 seconds and other partitions are diagnostics; they cannot switch the headline.', '',
             'Every one of the three calibration sessions must pass the provisional simulator screen and all four conservative error bands at ≤ 1.05 times its newly trained hybrid_accel_rk4 anchor, with no worse prediction or decision failures. Paired selected truth costs must differ by no more than their mean joint numerical envelope. Identical selected commands cancel exactly. Ranking ties alone remain usable; no unique oracle optimum is claimed.', '',
             'The component bar is ≥ 25% pooled complete-scorer median saving versus the fastest acceptable other method at this configuration, with no worse p95. Pooling uses 50 actual calls in each session (150 calls), never group medians. The two acceleration residual references have 1,410 parameters; the three coefficient networks have 1,478 and share the nominal physics and seeded initial tensors.', '',
             f'Fastest acceptable alternative: {gate["fastest_acceptable_alternative"]}. Median saving fraction: {gate["median_saving_fraction"]}; p95 ratio: {gate["p95_ratio"]}.', '',
             '| Model | Budget (s) | Partition (ms) | Eligibility | Session statuses | Pooled median / p95 (s) |',
             '|---|---:|---:|---|---|---|']
    for c in gate['configurations']:
        pool = c['scorer_pool']['wall_s']
        statuses = ', '.join(str(s['session_id'])+': '+s['status'] for s in c['sessions'])
        lines.append(f'| {c["model_id"]} | {c["budget_s"]} | {c["query_step_ns"]/1e6:g} | {c["status"]} | {statuses} | {pool["median"]} / {pool["p95"]} |')
    lines += ['', 'All eligible and ineligible alternatives, session reasons, paired state differences and retained ranking statuses are in component_gate.json. Raw prediction/decision outcomes, timing samples and rotated-order provenance, fit failures and optimization limits remain in short_calibration_result.json and the immutable external store.', '',
              'These 5% margins are provisional engineering choices. Three independent sessions are exploratory, not powered confirmation. Unit fixtures are ineligible for empirical claims. A positive component result supports a later independent confirmation design; a valid nonpositive primary result ends active investment in this line. No physical deadline, hardware or total commissioning saving is established.', '']
    return '\n'.join(lines)
