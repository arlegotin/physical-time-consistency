"""Verification commands cannot silently launch expensive experiments."""
import importlib
import json
from pathlib import Path
import subprocess
import sys
import shutil
import pytest


def api(name):
    try:
        module = importlib.import_module('poc.study_reproduce')
    except ImportError as error:
        pytest.fail(f'Missing study reproduction contract: {error}')
    return getattr(module, name)


def test_default_manifest_plan_contains_only_the_executed_gate_reaggregation():
    plan = api('reproduction_plan')('GR', rerun=False, authorization={}, locks=[])
    assert plan == [['validate', '--scope', 'protocol'], ['status', '--verify'], ['decide', 'GR']]


def test_reruns_require_current_allocation_and_no_live_lock():
    plan = api('reproduction_plan')
    with pytest.raises(ValueError, match='authorization'):
        plan('GR', rerun=True, authorization={'local_compute_authorized': False}, locks=[])
    with pytest.raises(ValueError, match='lock'):
        plan('GR', rerun=True, authorization={'local_compute_authorized': True, 'paid_services_authorized': False}, locks=['live'])


def test_explicit_rerun_reproduces_the_registered_experiment_without_changing_the_design():
    result = api('reproduction_plan')('GR', rerun=True,
        authorization={'local_compute_authorized': True, 'paid_services_authorized': False}, locks=[])
    assert result == [['validate', '--scope', 'protocol'], ['status', '--verify'],
                      ['run', 'MR', '--resume'], ['decide', 'GR']]
    with pytest.raises(ValueError, match='gate'):
        api('reproduction_plan')('G999', rerun=False, authorization={}, locks=[])


def test_replay_cannot_silently_use_a_different_numerical_package_version():
    validate = api('validate_runtime_versions')
    validate({'numpy': '2.2.6', 'torch': '2.8.0'}, {'numpy': '2.2.6', 'torch': '2.8.0'})
    with pytest.raises(ValueError, match='version'):
        validate({'numpy': '2.2.6', 'torch': '2.8.0'}, {'numpy': '2.1.0', 'torch': '2.8.0'})


def test_statistical_reaggregation_uses_the_schema_compatible_gate_writer(tmp_path):
    argv = api('command_argv')('python3.11', tmp_path, ['decide', 'GR'])
    assert argv == ['python3.11', '-m', 'poc.decision_wire', '--root', str(tmp_path), 'GR']
    assert api('command_argv')('python3.11', tmp_path, ['decide', 'G20']) == [
        'python3.11', '-m', 'poc', '--root', str(tmp_path), 'decide', 'G20']


def test_reproduced_inconclusive_decision_is_successful_verification_not_a_scientific_pass():
    accept = api('verified_decision_exit')
    request = {'decision_id': 'original', 'outcome': 'INCONCLUSIVE'}
    value = {'decision_id': 'original', 'outcome': 'INCONCLUSIVE', 'validity_status': 'valid'}
    assert accept(1, ['decide', 'GR'], value, request)
    assert not accept(1, ['decide', 'GR'], {**value, 'validity_status': 'invalid'}, request)
    assert not accept(1, ['decide', 'GR'], {**value, 'decision_id': 'changed'}, request)
    assert not accept(1, ['decide', 'GR'], {**value, 'outcome': 'NO-GO'}, request)
    assert not accept(1, ['status', '--verify'], value, request)


def test_explicit_mr_replay_uses_archived_seed_scope_and_schema_writer(tmp_path):
    from poc.io import atomic_json
    (tmp_path / 'poc').mkdir()
    (tmp_path / 'poc/__init__.py').write_text('')
    import poc.io
    shutil.copyfile(poc.io.__file__, tmp_path / 'poc/io.py')
    (tmp_path / 'poc/__main__.py').write_text('raise RuntimeError("old MR wire path")\n')
    (tmp_path / 'poc/rejection_subset.py').write_text(
        'def run_rejection(root,resume=False,seeds=None):\n'
        ' print("producer diagnostics")\n'
        ' assert resume is True and seeds == [0,1,2,3,4,5,6,7,8,9]\n'
        ' return {"seed_ids":seeds}\n')
    (tmp_path / 'poc/decision_wire.py').write_text(
        'def decide_rejection(root,evidence):\n'
        ' assert len(evidence[0]["seed_ids"]) == 10\n'
        ' return {"decision_id":"original","outcome":"INCONCLUSIVE","validity_status":"valid"}\n')
    atomic_json(tmp_path / 'config/rejection_subset_scope.json', {'seed_ids': list(range(10))})
    argv = api('command_argv')(sys.executable, tmp_path, ['run', 'MR', '--resume'])
    result = subprocess.run(argv, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 1
    assert result.stdout, result.stderr
    assert json.loads(result.stdout)['decision_id'] == 'original'
    assert 'producer diagnostics' in result.stderr
    assert 'old MR wire path' not in result.stderr


def test_rerun_context_preserves_archive_and_default_verification_does_not_copy(tmp_path):
    from poc.io import atomic_json, sha256_file
    from poc.study_archive import create_snapshot, verify_snapshot
    (tmp_path / 'poc').mkdir()
    (tmp_path / 'poc/example.py').write_text('original\n')
    create_snapshot(tmp_path, 'study/frozen')
    source = tmp_path / 'study/frozen'
    request = {'study_relative': 'study/frozen', 'snapshot_sha256': sha256_file(source / 'STUDY_SNAPSHOT.json'),
               'decision_id': 'fixture', 'outcome': 'INCONCLUSIVE'}
    select = api('execution_context')
    assert select(tmp_path, request, False) == source
    assert sorted((tmp_path / 'study').iterdir()) == [source]
    working = select(tmp_path, request, True)
    assert working.is_relative_to(tmp_path / 'study') and not working.is_relative_to(source)
    assert (working / 'INITIAL_CONTEXT.json').is_file()
    (working / 'poc/example.py').write_text('replay-only\n')
    verify_snapshot(source)
    assert (source / 'poc/example.py').read_text() == 'original\n'
