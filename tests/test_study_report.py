"""Report output cannot promote missing or corrupt evidence to a result."""
import csv
import importlib
import shutil
from pathlib import Path

import pytest


def write_report(root, decisions):
    try:
        module = importlib.import_module('poc.study_report')
    except ImportError as error:
        pytest.fail(f'Missing scientific report writer: {error}')
    return module.write_scientific_report(root, decisions)


def test_no_evidence_keeps_every_atomic_criterion_explicitly_unmeasured(tmp_path):
    project = Path(__file__).resolve().parents[1]
    shutil.copytree(project / 'config', tmp_path / 'config')
    state = write_report(tmp_path, [])
    assert state['status'] == 'NONTERMINAL'
    assert state['terminal_verdict'] is None
    with (tmp_path / 'reports/claims.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 294
    assert all(row['status'] == 'not_tested' for row in rows)
    assert all(row['point_estimate'] == '' for row in rows)


def test_invalid_decision_is_rejected_before_a_terminal_report_is_written(tmp_path):
    project = Path(__file__).resolve().parents[1]
    shutil.copytree(project / 'config', tmp_path / 'config')
    with pytest.raises((ValueError, KeyError)):
        write_report(tmp_path, [{'outcome': 'NO-GO', 'terminal_verdict': 'POC FAIL'}])
    assert not (tmp_path / 'reports/state.json').exists()


def test_interrupted_attempts_remain_in_the_report_index(tmp_path):
    project = Path(__file__).resolve().parents[1]
    shutil.copytree(project / 'config', tmp_path / 'config')
    folder = tmp_path / 'evidence/M30/interrupted'
    folder.mkdir(parents=True)
    (folder / 'INTERRUPTED.json').write_text('{"reason":"operator-preserved scheduler", "status":"interrupted"}')
    write_report(tmp_path, [])
    with (tmp_path / 'reports/run_index.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]['run_id'] == 'interrupted'
    assert rows[0]['status'] == 'interrupted'
    assert rows[0]['exclusion_reason']


def test_prediction_source_report_does_not_deny_separate_control_development(tmp_path):
    project = Path(__file__).resolve().parents[1]
    shutil.copytree(project / 'config', tmp_path / 'config')
    separate = tmp_path / 'reports/control_development_direct_result.md'
    separate.parent.mkdir()
    separate.write_text('Separate measured development control; no confirmation claim.')
    state = write_report(tmp_path, [])
    assert state.get('evidence_scope') == 'selected_prediction_study'
    text = (tmp_path / 'reports/final_report.md').read_text()
    assert 'Robot-control, practical actuator-rate' not in text
    assert 'selected prediction study' in text
    assert separate.read_text() == 'Separate measured development control; no confirmation claim.'
