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


def claim_rows(root: Path, decisions):
    registry_file = read_json(root / 'config/criteria.json')
    from .criteria import validate_registry
    validate_registry(registry_file['criteria'])
    registry, supported = _supported_rows(decisions)
    result = []
    for identifier, criterion in registry.items():
        observed, decision = supported.get(identifier, ({}, {}))
        bounds = observed.get('bounds')
        references = decision.get('evidence', [])
        dimensions = criterion['dimensions']
        contrast = criterion['statistical'] and not identifier.startswith('K02.gain/')
        threshold = ({'contrast_boundary': 0., 'practical_threshold': criterion['threshold'],
                      'formula': criterion['metric']} if contrast else criterion['threshold'])
        reason = observed.get('reason') or 'Mandatory evidence not obtained; no numeric default.'
        if contrast:
            reason += ' Estimate and bounds concern the preregistered linear contrast, whose acceptance boundary is zero; the practical threshold is recorded separately.'
        n_seeds, envelope = '', ''
        for reference in references:
            path = root / reference['path']
            summary_path = path.parent / 'summary.json'
            if summary_path.exists():
                summary = read_json(summary_path)
                contrast_summary = summary.get('contrast_summaries', {}).get(identifier)
                if contrast_summary:
                    n_seeds = len(contrast_summary.get('seed_ids', []))
                    envelope = json.dumps(contrast_summary.get('numerical_envelope', {}), sort_keys=True)
        result.append({'protocol_version': 'PTCO-1.0', 'claim_id': criterion['claim_id'],
            'criterion_id': identifier, 'cell_id': json.dumps(dimensions, sort_keys=True),
            'status': observed.get('status', 'not_tested'), 'threshold': json.dumps(threshold, sort_keys=True),
            'point_estimate': observed.get('estimate') if observed.get('estimate') is not None else '',
            'lower_bound': bounds[0] if bounds else '', 'upper_bound': bounds[1] if bounds else '',
            'numerical_envelope': envelope, 'n_seeds': n_seeds, 'n_episodes': '', 'n_probes': '',
            'look_id': decision.get('look_index') or '', 'validity': decision.get('validity_status', 'not_run'),
            'evidence_paths': json.dumps([r['path'] for r in references]),
            'evidence_hashes': json.dumps([r['sha256'] for r in references]),
            'gate_id': decision.get('gate_id', ''), 'reason': reason})
    return result
