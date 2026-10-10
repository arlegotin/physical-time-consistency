"""The public report commands close evidence without inventing scientific success."""
from pathlib import Path
import json
import shutil
import subprocess


def fixture(root, rejected=False):
    from poc.io import atomic_json, sha256_file
    from poc.study_archive import create_snapshot
    project = Path(__file__).resolve().parents[1]
    for name in ('poc', 'config', 'schemas'):
        shutil.copytree(project / name, root / name, ignore=shutil.ignore_patterns('__pycache__'))
    (root / 'config/report_source.json').unlink(missing_ok=True)
    shutil.copytree(project / 'locks', root / 'locks', ignore=shutil.ignore_patterns('active'))
    atomic_json(root / 'locks/wheels.json', {'wheels': []})
    subprocess.run(['git', 'init', '-q', str(root)], check=True)
    subprocess.run(['git', '-C', str(root), '-c', 'user.name=Unit Fixture',
                    '-c', 'user.email=fixture@example.invalid', 'commit',
                    '--allow-empty', '-qm', 'Report test fixture'], check=True)
    decision = {'schema_version': 1, 'protocol_version': 'PTCO-1.0',
        'milestone_id': 'M10', 'gate_id': 'G10', 'decision_id': 'fixture-only',
        'phase': 'unit', 'evidence': [], 'code_tree_sha256': 'a'*64,
        'configuration_sha256': 'b'*64, 'validity_status': 'valid',
        'criterion_evaluations': [], 'outcome': 'INCONCLUSIVE', 'terminal_verdict': None,
        'permitted_next_actions': [], 'look_index': None, 'alpha_spent': None,
        'created_utc': '2026-10-01T00:00:00Z'}
    if rejected:
        from poc.criteria import expand_criteria
        criterion = next(c for c in expand_criteria() if not c['statistical'])
        decision.update(outcome='NO-GO', criterion_evaluations=[{
            'criterion_id': criterion['criterion_id'], 'status': 'fail', 'estimate': None,
            'bounds': None, 'threshold': None, 'operator': 'counterexample',
            'reason': 'Synthetic deterministic rejection, unit fixture only'}])
    path = root / 'decisions/G10/fixture-only.json'
    atomic_json(path, decision)
    atomic_json(root / 'decisions/next_action.json', {'decision_path': str(path.relative_to(root)),
        'decision_sha256': sha256_file(path), 'permitted_next_actions': []})
    create_snapshot(root, 'study/qualified')
    atomic_json(root / 'config/report_source.json', {'schema_version': 1,
        'protocol_version': 'PTCO-1.0', 'study_relative': 'study/qualified',
        'snapshot_sha256': sha256_file(root / 'study/qualified/STUDY_SNAPSHOT.json')})


def test_mf_creates_complete_report_evidence_and_a_report_go(tmp_path, capsys):
    from poc.__main__ import main
    from poc.contracts import verify_run
    from poc.io import read_json
    fixture(tmp_path)
    assert main(['--root', str(tmp_path), 'run', 'MF']) == 0
    decision = json.loads(capsys.readouterr().out)
    assert decision['gate_id'] == 'GF'
    assert decision['outcome'] == 'GO'
    assert decision['terminal_verdict'] is None
    assert read_json(tmp_path / 'reports/state.json')['status'] == 'NONTERMINAL'
    manifest = read_json(tmp_path / decision['evidence'][0]['path'])
    verify_run(tmp_path, manifest)
    assert manifest['phase'] == 'report'
    assert (tmp_path / 'evidence/MF' / manifest['run_id'] / 'final_report.md').is_file()


def test_release_audits_report_in_the_selected_scientific_namespace(tmp_path, capsys):
    from poc.__main__ import main
    fixture(tmp_path)
    assert main(['--root', str(tmp_path), 'run', 'MF']) == 0
    capsys.readouterr()
    assert main(['--root', str(tmp_path), 'validate', '--scope', 'release']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['status'] == 'valid'
    assert result['verdict'] == 'NONTERMINAL'
    assert result['criteria_count'] == 294


def test_gf_rechecks_modified_public_tables_instead_of_reusing_a_cached_go(tmp_path, capsys):
    from poc.__main__ import main
    fixture(tmp_path)
    assert main(['--root', str(tmp_path), 'run', 'MF']) == 0
    capsys.readouterr()
    path = tmp_path / 'reports/claims.csv'
    path.write_text(path.read_text().replace('not_tested', 'pass', 1))
    assert main(['--root', str(tmp_path), 'decide', 'GF']) == 1
    assert 'criterion' in capsys.readouterr().err.lower()


def test_gf_reports_scientific_failure_separately_from_report_validity(tmp_path, capsys):
    from poc.__main__ import main
    fixture(tmp_path, rejected=True)
    assert main(['--root', str(tmp_path), 'run', 'MF']) == 0
    decision = json.loads(capsys.readouterr().out)
    assert decision['outcome'] == 'GO'
    assert decision['terminal_verdict'] == 'POC FAIL'


def test_reporting_revision_preserves_and_supersedes_the_earlier_receipt(tmp_path, capsys):
    from poc.__main__ import main
    from poc.io import read_json
    from poc.contracts import verify_run
    fixture(tmp_path)
    assert main(['--root', str(tmp_path), 'run', 'MF']) == 0
    first = json.loads(capsys.readouterr().out)
    old = read_json(tmp_path / first['evidence'][0]['path'])
    source = tmp_path / 'poc/study_reproduce.py'
    source.write_text(source.read_text() + '\n# Reporting-only revision fixture.\n')
    assert main(['--root', str(tmp_path), 'run', 'MF']) == 0
    second = json.loads(capsys.readouterr().out)
    current = read_json(tmp_path / second['evidence'][0]['path'])
    assert old['run_id'] in current['supersedes']
    verify_run(tmp_path, old, check_dependencies=False)
    verify_run(tmp_path, current)
