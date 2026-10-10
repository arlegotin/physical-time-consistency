"""Reporting uses the exact scientific source selected by its verified archive."""
import json
from pathlib import Path

import pytest


def source_fixture(root):
    from poc.io import atomic_json, sha256_file
    from poc.study_archive import create_snapshot
    (root / 'poc').mkdir()
    (root / 'poc/__init__.py').write_text('')
    (root / 'poc/__main__.py').write_text(
        'import json,sys\nfrom pathlib import Path\n'
        'print(json.dumps({"source":"frozen", "cwd":str(Path.cwd()), "argv":sys.argv[1:]}))\n')
    create_snapshot(root, 'study/qualified')
    archive = root / 'study/qualified'
    atomic_json(root / 'config/report_source.json', {
        'schema_version': 1, 'protocol_version': 'PTCO-1.0',
        'study_relative': 'study/qualified',
        'snapshot_sha256': sha256_file(archive / 'STUDY_SNAPSHOT.json')})
    return archive


def test_scientific_commands_import_the_frozen_source_in_a_fresh_process(tmp_path, monkeypatch):
    from poc import study_context
    study = source_fixture(tmp_path)
    (tmp_path / 'poc/__main__.py').write_text('raise RuntimeError("later reporting code")\n')
    monkeypatch.setenv('PYTHONPATH', str(tmp_path))
    result = study_context.call_source(tmp_path, ['status', '--verify'])
    assert result.returncode == 0
    value = json.loads(result.stdout)
    assert value['source'] == 'frozen'
    assert value['cwd'] == str(study)
    assert value['argv'] == ['--root', str(study), 'status', '--verify']


def test_source_selection_rejects_changed_archived_inputs(tmp_path):
    from poc import study_context
    study = source_fixture(tmp_path)
    (study / 'poc/__main__.py').write_text('changed scientific source\n')
    with pytest.raises(ValueError, match='hash'):
        study_context.source_context(tmp_path)


def test_source_selection_cannot_escape_the_project_archive(tmp_path):
    from poc import study_context
    from poc.io import read_json, atomic_json
    source_fixture(tmp_path)
    path = tmp_path / 'config/report_source.json'
    value = read_json(path)
    value['study_relative'] = '../outside'
    atomic_json(path, value)
    with pytest.raises(ValueError):
        study_context.source_context(tmp_path)


def test_public_status_uses_the_selected_scientific_namespace(tmp_path, capsys):
    from poc.__main__ import main
    study = source_fixture(tmp_path)
    assert main(['--root', str(tmp_path), 'status', '--verify']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['source'] == 'frozen'
    assert result['cwd'] == str(study)


def test_public_protocol_audit_checks_the_frozen_scientific_definition(tmp_path, capsys):
    from poc.__main__ import main
    study = source_fixture(tmp_path)
    assert main(['--root', str(tmp_path), 'validate', '--scope', 'protocol']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['source'] == 'frozen'
    assert result['cwd'] == str(study)
    assert result['argv'][-3:] == ['validate', '--scope', 'protocol']


def test_mf_rejects_a_selected_source_without_a_complete_gate(tmp_path, capsys):
    from poc.__main__ import main
    source_fixture(tmp_path)
    assert main(['--root', str(tmp_path), 'run', 'MF']) == 1
    assert 'completed gate' in capsys.readouterr().err
    assert not (tmp_path / 'reports/final_report.md').exists()
