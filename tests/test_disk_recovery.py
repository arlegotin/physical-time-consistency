"""A storage failure must resume the same fit without rewriting its failed receipt."""
from pathlib import Path
import pytest

from poc.io import atomic_json, read_json, sha256_file


def fixture(root, error='[Errno 28] No space left on device'):
    folder = root / 'evidence/M30/failed'
    folder.mkdir(parents=True)
    atomic_json(folder / 'manifest.json', {'completion_status': 'failed', 'run_id': 'failed',
                                          'milestone_id': 'M30', 'job_key': '1'*64})
    atomic_json(folder / 'summary.json', {'failure': error})
    atomic_json(folder / 'request.json', {'job_key': '1'*64, 'inputs': {'seed': 32}})
    for name in ('resume.pt', 'weights.pt', 'resume.rng.json', 'validation_history.json'):
        (folder / name).write_bytes(name.encode())
    atomic_json(folder / 'resume.complete.json', {
        'checkpoint_sha256': sha256_file(folder / 'resume.pt'),
        'rng_sha256': sha256_file(folder / 'resume.rng.json'), 'update': 70016})
    return folder


def test_disk_recovery_creates_one_hash_identical_partial_and_preserves_failure(tmp_path, monkeypatch):
    from poc.disk_recovery import recover_disk_fit
    folder = fixture(tmp_path)
    monkeypatch.setattr('poc.disk_recovery.verify_run', lambda root, record: None)
    before = {p.name: sha256_file(p) for p in folder.iterdir()}
    recovered = recover_disk_fit(tmp_path, 'failed')
    assert recovered != folder
    assert not (recovered / 'COMPLETED').exists()
    assert read_json(recovered / 'request.json') == read_json(folder / 'request.json')
    for name in ('resume.pt', 'resume.rng.json', 'resume.complete.json', 'weights.pt', 'validation_history.json'):
        assert sha256_file(recovered / name) == before[name]
    assert {p.name: sha256_file(p) for p in folder.iterdir()} == before
    assert recover_disk_fit(tmp_path, 'failed') == recovered
    assert read_json(recovered / 'RECOVERED_FROM.json')['failed_run_id'] == 'failed'


def test_corrupt_checkpoint_or_intrinsic_model_failure_cannot_be_recovered(tmp_path, monkeypatch):
    from poc.disk_recovery import recover_disk_fit
    folder = fixture(tmp_path)
    monkeypatch.setattr('poc.disk_recovery.verify_run', lambda root, record: None)
    (folder / 'resume.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checkpoint'):
        recover_disk_fit(tmp_path, 'failed')
    atomic_json(folder / 'summary.json', {'failure': 'Nonfinite gradient'})
    with pytest.raises(ValueError, match='storage'):
        recover_disk_fit(tmp_path, 'failed')
