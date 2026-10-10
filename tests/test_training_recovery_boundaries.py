import torch
import pytest

from poc.io import atomic_json, sha256_file
from poc.training_jobs import find_partial


def committed_partial(root, name, key):
    folder = root / 'evidence/M30' / name
    folder.mkdir(parents=True)
    atomic_json(folder / 'request.json', {'job_key': key})
    (folder / 'resume.pt').write_bytes(b'committed optimizer state')
    atomic_json(folder / 'resume.rng.json', {'rng': 1})
    atomic_json(folder / 'resume.complete.json', {
        'checkpoint_sha256': sha256_file(folder / 'resume.pt'),
        'rng_sha256': sha256_file(folder / 'resume.rng.json'), 'update': 128})
    return folder


@pytest.mark.parametrize('old_marked', [False, True])
def test_restart_recovers_ancestor_when_replacement_has_no_checkpoint(tmp_path, old_marked):
    key = 'a' * 64
    old = committed_partial(tmp_path, 'old', key)
    new = tmp_path / 'evidence/M30/new'
    new.mkdir()
    atomic_json(new / 'request.json', {'job_key': key, 'resume_from': 'evidence/M30/old'})
    if old_marked:
        atomic_json(old / 'RESUMED_BY.json', {'run_id': 'new', 'job_key': key})
    assert find_partial(tmp_path, key) == old


def test_best_selection_survives_interruption_between_packet_and_marker(tmp_path, monkeypatch):
    import poc.train as training
    rows = [{'epoch': 5, 'objective': 2.}]
    training.save_selection(tmp_path, rows, {'weight': torch.tensor([1.])})
    original = training.atomic_json
    def interrupted(path, value):
        if path.name == 'selection.complete.json':
            raise OSError('simulated interruption at selection commit')
        return original(path, value)
    monkeypatch.setattr(training, 'atomic_json', interrupted)
    with pytest.raises(OSError, match='selection commit'):
        training.save_selection(tmp_path, rows + [{'epoch': 10, 'objective': 1.}],
                                {'weight': torch.tensor([2.])})
    restored_rows, best, weights = training.restore_selection(tmp_path, 10)
    assert restored_rows == rows and best['epoch'] == 5
    torch.testing.assert_close(weights['weight'], torch.tensor([1.]))


def test_torn_export_does_not_override_committed_selection(tmp_path):
    import poc.train as training
    rows = [{'epoch': 5, 'objective': 2.}]
    training.save_selection(tmp_path, rows, {'weight': torch.tensor([1.])})
    atomic_json(tmp_path / 'validation_history.json', {'rows': rows + [{'epoch': 10, 'objective': 1.}]})
    restored_rows, best, weights = training.restore_selection(tmp_path, 10)
    assert restored_rows == rows and best['epoch'] == 5
    torch.testing.assert_close(weights['weight'], torch.tensor([1.]))


def test_first_uncommitted_validation_can_be_recomputed_from_optimizer_boundary(tmp_path, monkeypatch):
    import poc.train as training
    def interrupted(path, value):
        raise OSError('first selection marker not committed')
    monkeypatch.setattr(training, 'atomic_json', interrupted)
    with pytest.raises(OSError):
        training.save_selection(tmp_path, [{'epoch': 5, 'objective': 2.}], {'weight': torch.tensor([1.])})
    assert training.restore_selection(tmp_path, 5) == ([], None, None)
    with pytest.raises(ValueError, match='consistent committed'):
        training.restore_selection(tmp_path, 10)


def test_restore_selection_never_uses_a_validation_after_optimizer_rollback(tmp_path):
    import poc.train as training
    rows = [{'epoch': 5, 'objective': 2.}]
    training.save_selection(tmp_path, rows, {'weight': torch.tensor([1.])})
    training.save_selection(tmp_path, rows + [{'epoch': 10, 'objective': 1.}], {'weight': torch.tensor([2.])})
    restored, best, weights = training.restore_selection(tmp_path, 9)
    assert restored == rows and best['epoch'] == 5
    torch.testing.assert_close(weights['weight'], torch.tensor([1.]))


def test_legacy_torn_history_requires_independent_nominal_verification(tmp_path):
    import poc.train as training
    import io
    from poc.io import atomic_bytes
    atomic_json(tmp_path / 'validation_history.json', {'rows': [{'epoch': 5, 'objective': 1.}]})
    buffer = io.BytesIO(); torch.save({'weight': torch.tensor([2.])}, buffer)
    atomic_bytes(tmp_path / 'weights.pt', buffer.getvalue())
    with pytest.raises(ValueError, match='independent nominal'):
        training.restore_selection(tmp_path, 5)
    def verify(best, weights):
        if float(weights['weight'][0]) != best['objective']:
            raise ValueError('Legacy selected tensors do not match recorded nominal score')
    with pytest.raises(ValueError, match='do not match'):
        training.restore_selection(tmp_path, 5, verify_legacy=verify)


def test_independent_committed_branches_cannot_choose_a_favorable_checkpoint(tmp_path):
    committed_partial(tmp_path, 'one', 'a'*64)
    committed_partial(tmp_path, 'two', 'a'*64)
    with pytest.raises(ValueError, match='Conflicting multiple'):
        find_partial(tmp_path, 'a'*64)


def test_recovery_compatibility_rejects_a_change_to_the_optimizer():
    from pathlib import Path
    from poc.planner_qualification import unchanged_source_nodes
    prior = Path('config/revisions/training-recovery-originals/train.py').read_text()
    current = Path('poc/train.py').read_text()
    unchanged_source_nodes(prior, current, {'fit', 'save_selection', 'restore_selection'})
    with pytest.raises(ValueError, match='Unrelated source'):
        unchanged_source_nodes(prior, current.replace('weight_decay=1e-6', 'weight_decay=1e-4'),
                               {'fit', 'save_selection', 'restore_selection'})
