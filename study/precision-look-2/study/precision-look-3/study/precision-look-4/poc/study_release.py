"""Reconcile report tables with compatible decisions before release."""
from pathlib import Path
import csv

from .contracts import scan_runs, verify_decision
from .finalization import claim_rows, compute_verdict
from .io import read_json


def validate_report_tables(output: Path, study: Path, decisions):
    output, study = Path(output).resolve(), Path(study).resolve()
    for decision in decisions:
        verify_decision(study, decision)
    state = read_json(output / 'reports/state.json')
    blockers = []
    if any(d.get('outcome') == 'BLOCKED' for d in decisions):
        records, _ = scan_runs(study)
        for record in records:
            path = study / 'evidence' / record['milestone_id'] / record['run_id'] / 'validity.json'
            blocker = read_json(path).get('blocker')
            if blocker:
                blockers.append(blocker)
    verdict = compute_verdict(decisions, blockers[-1] if blockers else None)
    if state['status'] != verdict or state['terminal_verdict'] != (verdict if verdict != 'NONTERMINAL' else None):
        raise ValueError('Report verdict disagrees with valid mandatory evidence')
    expected = {r['criterion_id']: r for r in claim_rows(study, decisions)}
    with (output / 'reports/claims.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(expected) or {r['criterion_id'] for r in rows} != set(expected):
        raise ValueError('Report criterion coverage is incomplete or duplicated')
    for row in rows:
        actual = expected[row['criterion_id']]
        for name in ('claim_id', 'status', 'point_estimate', 'lower_bound', 'upper_bound',
                     'threshold', 'look_id', 'gate_id', 'evidence_paths', 'evidence_hashes', 'validity'):
            if row[name] != str(actual[name]):
                raise ValueError('Report criterion differs from its verified decision: ' + row['criterion_id'])
    if state['verified_decisions'] != [d['decision_id'] for d in decisions]:
        raise ValueError('Report selects different decision evidence')
    text = (output / 'reports/final_report.md').read_text(encoding='utf-8')
    if f'Status: **{verdict}**.' not in text:
        raise ValueError('Report text disagrees with the verified verdict')
    return {'status': 'valid', 'verdict': verdict, 'criteria_count': len(rows)}
