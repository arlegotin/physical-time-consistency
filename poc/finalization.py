"""Verdict and claim-table primitives; callers must independently verify evidence."""
from pathlib import Path
import json

from .io import read_json


def _supported_rows(decisions):
    from .criteria import expand_criteria
    registry = {c['criterion_id']: c for c in expand_criteria()}
    rows = {}
    for decision in decisions:
        if decision.get('validity_status') != 'valid':
            continue
        for row in decision.get('criterion_evaluations', []):
            identifier = row['criterion_id']
            if identifier not in registry:
                continue
            criterion = registry[identifier]
            if criterion['statistical'] and (decision.get('phase') not in {'confirmatory', 'replication'}
                    or not decision.get('look_index') or not decision.get('alpha_spent')):
                continue
            if not criterion['statistical'] and decision.get('phase') not in {'unit', 'calibration', 'replication'}:
                continue
            status = row['status']
            if status not in {'pass', 'fail', 'unresolved', 'invalid', 'not_applicable'}:
                raise ValueError('Unknown mandatory criterion status')
            if status == 'fail' and decision['outcome'] != 'NO-GO':
                raise ValueError('Scientific rejection disagrees with its gate outcome')
            previous = rows.get(identifier)
            if previous and {previous[0]['status'], status} == {'pass', 'fail'}:
                raise ValueError('Conflicting valid mandatory criterion decisions require diagnosis')
            if previous is None or status in {'pass', 'fail'} or previous[0]['status'] == 'unresolved':
                rows[identifier] = (row, decision)
    return registry, rows


def compute_verdict(decisions, blocker=None):
    registry, rows = _supported_rows(decisions)
    if any(row['status'] == 'fail' for row, decision in rows.values()):
        return 'POC FAIL'
    replicated = any(d.get('gate_id') == 'G90' and d.get('outcome') == 'GO'
        and d.get('validity_status') == 'valid' for d in decisions)
    if replicated and set(rows) == set(registry) and all(row['status'] == 'pass' for row, decision in rows.values()):
        return 'POC PASS'
    if blocker is not None:
        if not blocker.get('required_prerequisite'):
            raise ValueError('BLOCKED requires an exact external prerequisite')
        return 'BLOCKED'
    return 'NONTERMINAL'


def measured_probe_count(root, folder, dimensions, seed_ids):
    path = root / folder / 'children.json'
    if not path.exists() or 'stratum' not in dimensions or 'h_ns' not in dimensions:
        return ''
    counts, seen = set(), set()
    for child in read_json(path)['records']:
        if child['role'] != 'E01' or child['key'][-1] not in seed_ids:
            continue
        actuator, condition, loss, model, mode, seed = child['key']
        if any(name in dimensions and dimensions[name] != value for name, value in (
                ('actuator', actuator), ('training_condition', condition), ('loss_mode', loss))):
            continue
        if 'competitor' in dimensions and model not in {dimensions['competitor'], dimensions['method']}:
            continue
        metrics = read_json(root / Path(child['path']).parent / 'summary.json')['metrics']
        for cell in metrics.values():
            metric = cell.get(dimensions['stratum'], {}).get(str(dimensions['h_ns']))
            if metric is None:
                raise ValueError('Missing measured probe count for a required cell')
            count = metric['n_probes']
            if type(count) is not int or count <= 0:
                raise ValueError('Invalid measured probe count')
            counts.add(count)
            seen.add(seed)
    if len(counts) != 1 or seen != set(seed_ids):
        raise ValueError('Inconsistent or incomplete measured probe counts')
    return counts.pop()


def claim_rows(root: Path, decisions):
    registry_file = read_json(root / 'config/criteria.json')
    from .criteria import validate_registry
    validate_registry(registry_file['criteria'])
    registry, supported = _supported_rows(decisions)
    result = []
    for identifier, criterion in registry.items():
        observed, decision = supported.get(identifier, ({}, {}))
        bounds = observed.get('bounds')
        if isinstance(bounds, dict):
            bounds = [bounds['lower'], bounds['upper']]
        references = decision.get('evidence', [])
        dimensions = criterion['dimensions']
        contrast = criterion['statistical'] and not identifier.startswith('K02.gain/')
        threshold = ({'contrast_boundary': 0., 'practical_threshold': criterion['threshold'],
                      'formula': criterion['metric']} if contrast else criterion['threshold'])
        reason = observed.get('reason') or 'Mandatory evidence not obtained; no numeric default.'
        if contrast:
            reason += ' Estimate and bounds concern the preregistered linear contrast, whose acceptance boundary is zero; the practical threshold is recorded separately.'
        n_seeds, n_probes, envelope = '', '', ''
        for reference in references:
            path = root / reference['path']
            summary_path = path.parent / 'summary.json'
            if summary_path.exists():
                summary = read_json(summary_path)
                contrast_summary = summary.get('contrast_summaries', {}).get(identifier)
                if contrast_summary:
                    seed_ids = contrast_summary.get('seed_ids', [])
                    n_seeds = len(seed_ids)
                    n_probes = measured_probe_count(root, path.parent, dimensions, seed_ids)
                    envelope = json.dumps(contrast_summary.get('numerical_envelope', {}), sort_keys=True)
                    if bounds is None and observed.get('status') == 'unresolved':
                        detail = next((r for r in summary.get('criterion_evaluations', [])
                                       if r['criterion_id'] == identifier), {})
                        display = detail.get('pass_bounds', {})
                        if display.get('lower') is not None and display.get('upper') is not None:
                            bounds = [display['lower'], display['upper']]
                            reason += ' Displayed bounds use the preregistered acceptance-level interval; rejection-level and precision checks remain in the source evidence.'
        result.append({'protocol_version': 'PTCO-1.0', 'claim_id': criterion['claim_id'],
            'criterion_id': identifier, 'cell_id': json.dumps(dimensions, sort_keys=True),
            'status': observed.get('status', 'not_tested'), 'threshold': json.dumps(threshold, sort_keys=True),
            'point_estimate': observed.get('estimate') if observed.get('estimate') is not None else '',
            'lower_bound': bounds[0] if bounds else '', 'upper_bound': bounds[1] if bounds else '',
            'numerical_envelope': envelope, 'n_seeds': n_seeds, 'n_episodes': '', 'n_probes': n_probes,
            'look_id': decision.get('look_index') or '', 'validity': decision.get('validity_status', 'not_run'),
            'evidence_paths': json.dumps([r['path'] for r in references]),
            'evidence_hashes': json.dumps([r['sha256'] for r in references]),
            'gate_id': decision.get('gate_id', ''), 'reason': reason})
    return result
