"""Terminal verdict precedence for already independently validated decisions."""
import importlib
from pathlib import Path
import pytest


def api(name):
    try:
        module = importlib.import_module('poc.finalization')
    except ImportError as error:
        pytest.fail(f'Missing finalization contract: {error}')
    assert hasattr(module, name)
    return getattr(module, name)


def criterion():
    from poc.criteria import expand_criteria
    return next(c for c in expand_criteria() if c['statistical'])


def fixture_decision(status, validity='valid', phase='confirmatory'):
    return {'gate_id': 'GR', 'outcome': 'NO-GO' if status == 'fail' else 'GO',
        'validity_status': validity, 'phase': phase, 'decision_id': 'unit-fixture',
        'terminal_verdict': 'POC FAIL' if status == 'fail' else None,
        'look_index': 1, 'alpha_spent': .025,
        'criterion_evaluations': [{'criterion_id': criterion()['criterion_id'], 'status': status}]}


def test_one_mandatory_valid_rejection_precedes_unrelated_external_blocker():
    verdict = api('compute_verdict')([fixture_decision('fail')], blocker={'required_prerequisite': 'more disk'})
    assert verdict == 'POC FAIL'


def test_subset_support_is_nonterminal_even_when_its_gate_is_green():
    assert api('compute_verdict')([fixture_decision('pass')]) == 'NONTERMINAL'


def test_invalid_or_discovery_evidence_cannot_reject_the_project():
    compute = api('compute_verdict')
    assert compute([fixture_decision('fail', validity='invalid')]) == 'NONTERMINAL'
    assert compute([fixture_decision('fail', phase='exploratory')]) == 'NONTERMINAL'


def test_conflicting_valid_decisions_require_diagnosis_instead_of_favorable_selection():
    with pytest.raises(ValueError, match='Conflicting'):
        api('compute_verdict')([fixture_decision('pass'), fixture_decision('fail')])


def test_empty_evidence_is_not_a_scientific_verdict():
    assert api('compute_verdict')([]) == 'NONTERMINAL'
    assert api('compute_verdict')([], blocker={'required_prerequisite': 'Python 3.11'}) == 'BLOCKED'
    with pytest.raises(ValueError):
        api('compute_verdict')([], blocker={'reason': 'this is hard'})


def test_report_rows_preserve_all_280_statistical_criteria_and_missingness(tmp_path):
    from poc.io import atomic_json
    from poc.criteria import expand_criteria
    registry = expand_criteria()
    atomic_json(tmp_path / 'config/criteria.json', {'criteria': registry})
    rows = api('claim_rows')(tmp_path, [fixture_decision('fail')])
    statistical_ids = {c['criterion_id'] for c in registry if c['statistical']}
    assert len([r for r in rows if r['criterion_id'] in statistical_ids]) == 280
    assert sum(r['status'] == 'fail' for r in rows) == 1
    assert sum(r['status'] == 'not_tested' for r in rows) == len(registry)-1
    assert all(r['reason'] for r in rows)


def test_claim_table_reads_schema_object_bounds_without_changing_the_values(tmp_path):
    from poc.io import atomic_json
    from poc.criteria import expand_criteria
    atomic_json(tmp_path / 'config/criteria.json', {'criteria': expand_criteria()})
    decision = fixture_decision('unresolved')
    decision['outcome'] = 'INCONCLUSIVE'
    decision['criterion_evaluations'][0].update(estimate=.02, bounds={'lower': -.03, 'upper': .07})
    row = next(r for r in api('claim_rows')(tmp_path, [decision])
               if r['criterion_id'] == criterion()['criterion_id'])
    assert row['point_estimate'] == .02
    assert row['lower_bound'] == -.03
    assert row['upper_bound'] == .07
    assert row['status'] == 'unresolved'


def test_unresolved_claim_retains_measured_probe_counts_and_acceptance_bounds(tmp_path):
    from poc.io import atomic_json
    from poc.criteria import expand_criteria
    atomic_json(tmp_path / 'config/criteria.json', {'criteria': expand_criteria()})
    decision = fixture_decision('unresolved')
    decision['outcome'] = 'INCONCLUSIVE'
    identifier = criterion()['criterion_id']
    decision['criterion_evaluations'][0].update(estimate=.01, bounds=None)
    decision['evidence'] = [{'path': 'evidence/MR/observed/manifest.json', 'sha256': 'fixture'}]
    atomic_json(tmp_path / 'evidence/MR/observed/summary.json', {
        'contrast_summaries': {identifier: {'seed_ids': [0, 1], 'numerical_envelope': {'lower': 1e-8, 'upper': 1e-8}}},
        'criterion_evaluations': [{'criterion_id': identifier, 'pass_bounds': {'lower': -.02, 'upper': .03}}]})
    children = []
    for seed in (0, 1):
        for model in ('node_generic', 'power_correct'):
            cell = f'direct__A__shared__{model}__seed{seed}'
            path = f'evidence/M30/{model}-{seed}/manifest.json'
            children.append({'role': 'E01', 'path': path, 'key': ['direct', 'A', 'shared', model, None, seed]})
            atomic_json(tmp_path / Path(path).parent / 'summary.json', {
                'metrics': {cell: {'cold': {'5000000': {'n_probes': 17}}}}})
    atomic_json(tmp_path / 'evidence/MR/observed/children.json', {'records': children})
    row = next(r for r in api('claim_rows')(tmp_path, [decision]) if r['criterion_id'] == identifier)
    assert row['n_seeds'] == 2
    assert row['n_probes'] == 17
    assert row['n_episodes'] == ''
    assert row['lower_bound'] == -.02
    assert row['upper_bound'] == .03
    assert 'acceptance-level' in row['reason']


def test_probe_count_disagreement_is_not_silently_averaged(tmp_path):
    from pathlib import Path
    from poc.io import atomic_json
    for seed, count in ((0, 17), (1, 16)):
        atomic_json(tmp_path / f'evidence/M30/seed{seed}/summary.json', {
            'metrics': {f'direct__A__shared__power_correct__seed{seed}': {
                'cold': {'5000000': {'n_probes': count}}}}})
    atomic_json(tmp_path / 'evidence/MR/observed/children.json', {'records': [
        {'role': 'E01', 'path': f'evidence/M30/seed{s}/manifest.json',
         'key': ['direct', 'A', 'shared', 'power_correct', None, s]} for s in (0, 1)]})
    with pytest.raises(ValueError, match='probe'):
        api('measured_probe_count')(tmp_path, Path('evidence/MR/observed'),
                                   {'stratum': 'cold', 'h_ns': 5000000}, [0, 1])
