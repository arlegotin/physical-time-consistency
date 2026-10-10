"""A frozen study retains its actual source and rejects damaged evidence."""
import importlib
import os

import pytest


def archive_api(name):
    try:
        module = importlib.import_module('poc.study_archive')
    except ImportError as error:
        pytest.fail(f'Missing pinned-study archive: {error}')
    assert hasattr(module, name)
    return getattr(module, name)


def fixture(root):
    (root / 'poc').mkdir()
    (root / 'poc/model.py').write_text('coefficient = 2\n')
    (root / 'data').mkdir()
    (root / 'data/observations.npz').write_bytes(b'fixed observations')
    (root / '.venv').mkdir()
    (root / '.venv/private').write_text('not an archive input')


def test_later_reporting_edits_do_not_change_the_scientific_source_snapshot(tmp_path):
    fixture(tmp_path)
    archive_api('create_snapshot')(tmp_path, 'study/frozen')
    (tmp_path / 'poc/model.py').write_text('coefficient = 3\n')
    archive_api('verify_snapshot')(tmp_path / 'study/frozen')
    assert (tmp_path / 'study/frozen/poc/model.py').read_text() == 'coefficient = 2\n'
    assert not (tmp_path / 'study/frozen/.venv').exists()
    assert os.stat(tmp_path / 'data/observations.npz').st_ino == os.stat(tmp_path / 'study/frozen/data/observations.npz').st_ino


def test_altered_archived_binary_cannot_be_reported_as_verified(tmp_path):
    fixture(tmp_path)
    archive_api('create_snapshot')(tmp_path, 'study/frozen')
    (tmp_path / 'study/frozen/data/observations.npz').write_bytes(b'altered')
    with pytest.raises(ValueError, match='hash'):
        archive_api('verify_snapshot')(tmp_path / 'study/frozen')


def test_symlinked_input_and_existing_destination_are_rejected(tmp_path):
    fixture(tmp_path)
    (tmp_path / 'poc/linked.py').symlink_to(tmp_path / 'poc/model.py')
    with pytest.raises(ValueError, match='Symlink'):
        archive_api('create_snapshot')(tmp_path, 'study/frozen')
    (tmp_path / 'poc/linked.py').unlink()
    archive_api('create_snapshot')(tmp_path, 'study/frozen')
    with pytest.raises(ValueError, match='exists'):
        archive_api('create_snapshot')(tmp_path, 'study/frozen')


def test_a_live_job_cannot_be_frozen_as_a_complete_study(tmp_path):
    fixture(tmp_path)
    (tmp_path / 'locks/active').mkdir(parents=True)
    (tmp_path / 'locks/active/job.json').write_text('{"pid": 123}')
    with pytest.raises(ValueError, match='active job'):
        archive_api('create_snapshot')(tmp_path, 'study/frozen')


def test_reproduction_copy_is_a_sibling_and_cannot_write_the_frozen_source(tmp_path):
    fixture(tmp_path)
    archive_api('create_snapshot')(tmp_path, 'study/frozen')
    source = tmp_path / 'study/frozen'
    archive_api('create_snapshot')(source, 'study/replay', destination_root=tmp_path)
    (tmp_path / 'study/replay/poc/model.py').write_text('replay-only change\n')
    archive_api('verify_snapshot')(source)
    assert (source / 'poc/model.py').read_text() == 'coefficient = 2\n'


def test_reproduction_destination_must_own_the_source_and_stay_in_study(tmp_path):
    fixture(tmp_path)
    archive_api('create_snapshot')(tmp_path, 'study/frozen')
    source = tmp_path / 'study/frozen'
    with pytest.raises(ValueError, match='source'):
        archive_api('create_snapshot')(source, 'study/replay', destination_root=tmp_path / 'unrelated')
    with pytest.raises(ValueError, match='archive'):
        archive_api('create_snapshot')(source, 'outside', destination_root=tmp_path)
