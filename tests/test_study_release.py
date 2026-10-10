"""Release validation rejects a report that invents or omits claim evidence."""
import csv
import importlib
import shutil
from pathlib import Path

import pytest


def validator():
    try:
        module = importlib.import_module('poc.study_release')
    except ImportError as error:
        pytest.fail(f'Missing report consistency validator: {error}')
    return module.validate_report_tables


def fixture(root):
    project = Path(__file__).resolve().parents[1]
    shutil.copytree(project / 'config', root / 'config')
    from poc.study_report import write_scientific_report
    write_scientific_report(root, [])


def test_empty_progress_report_is_consistent_and_never_a_pass(tmp_path):
    fixture(tmp_path)
    result = validator()(tmp_path, tmp_path, [])
    assert result['status'] == 'valid'
    assert result['verdict'] == 'NONTERMINAL'
    assert result['criteria_count'] == 294


def test_fabricated_terminal_state_is_rejected(tmp_path):
    fixture(tmp_path)
    from poc.io import read_json, atomic_json
    state = read_json(tmp_path / 'reports/state.json')
    state.update(status='POC PASS', terminal_verdict='POC PASS')
    atomic_json(tmp_path / 'reports/state.json', state)
    with pytest.raises(ValueError, match='verdict'):
        validator()(tmp_path, tmp_path, [])


def test_deleting_an_unfavorable_or_missing_criterion_fails_coverage(tmp_path):
    fixture(tmp_path)
    path = tmp_path / 'reports/claims.csv'
    with path.open() as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows[:-1])
    with pytest.raises(ValueError, match='criterion'):
        validator()(tmp_path, tmp_path, [])
