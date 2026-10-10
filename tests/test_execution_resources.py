"""Producer summaries must not lose observed resources or model identity."""
from pathlib import Path
from types import SimpleNamespace

from poc.io import atomic_json, read_json


def test_observed_fit_counts_and_hashes_reach_immutable_run(tmp_path, monkeypatch):
    from poc.execution import run_job
    fingerprint = {key: 'fixture' for key in ('python', 'architecture', 'os', 'kernel', 'core_count',
                    'torch_configuration', 'torch_threads', 'torch_interop_threads', 'deterministic_algorithms')}
    fingerprint['versions'] = {}
    atomic_json(tmp_path / 'locks/environment.json', fingerprint)
    atomic_json(tmp_path / 'locks/wheels.json', {'wheels': []})
    source = tmp_path / 'source.py'
    source.write_text('fixture = True\n')
    atomic_json(tmp_path / 'config/reference.json', {'epochs': 5})
    def subprocess_result(argv, **kwargs):
        return SimpleNamespace(stdout='fixture' if argv[1] == 'rev-parse' else '', returncode=0)
    monkeypatch.setattr('poc.execution.subprocess.run', subprocess_result)
    projection = {'wall_seconds': 7, 'disk_bytes': 2048}
    hashes = {'canonical_tensor_sha256': '1'*64}
    def producer(folder):
        return ([{'check_id': 'fit', 'status': 'pass', 'scope': 'fixture', 'reason': 'fixture', 'tolerance': None}],
                {'training_updates': 10, 'model_hashes': hashes, 'resource_projection': projection})
    argv = ['python', '-m', 'poc.reference_jobs', 'config/reference.json']
    result = run_job(tmp_path, 'M40', 'exploratory', ['source.py'], ('fit',), [], producer,
                     config_keys={'config/reference.json': {'epochs': 5}}, argv=argv)
    assert result['model_hashes'] == hashes
    assert result['argv'] == argv
    resources = read_json(tmp_path / 'evidence/M40' / result['run_id'] / 'resources.json')
    assert resources['training_updates'] == 10
    assert resources['original_estimate'] == projection

    # A robot calibration is a different producer, not a replacement for a fitted reference.
    def calibration(folder):
        return ([{'check_id': 'control.episode', 'status': 'pass', 'scope': 'fixture',
                  'reason': 'fixture', 'tolerance': None}], {})
    later = run_job(tmp_path, 'M40', 'calibration', ['source.py'], ('control.episode',), [], calibration,
                    config_keys={'config/reference.json': {'epochs': 5}}, argv=argv)
    assert result['run_id'] not in later.get('supersedes', [])

    from poc.control_models import PlantReferenceError
    def invalid_reference(folder):
        raise PlantReferenceError('simulator warning fixture')
    (tmp_path / 'invalid_reference.py').write_text('raise PlantReferenceError("fixture")\n')
    failed = run_job(tmp_path, 'M40', 'calibration', ['source.py', 'invalid_reference.py'], ('control.reference_failure',), [],
                     invalid_reference, config_keys={'config/reference.json': {'epochs': 5}})
    assert failed['completion_status'] == 'failed'
    validity = read_json(tmp_path / 'evidence/M40' / failed['run_id'] / 'validity.json')
    assert validity['validity_status'] == 'invalid'
    assert 'simulator warning fixture' in validity['checks'][0]['reason']
