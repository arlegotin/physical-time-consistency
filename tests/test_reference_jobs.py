"""Reference jobs preserve archive inputs and select only nominal endpoints."""
from pathlib import Path
import numpy as np
import pytest
import torch

from poc.io import atomic_json, read_json, sha256_file
from test_data import fixture


def inputs(tmp_path, actuator='direct'):
    archive = tmp_path / 'archive'
    train, arrays = fixture(archive, actuator=actuator, count=2)
    valid, _ = fixture(archive, actuator=actuator, role='validation', count=1)
    h = arrays['duration_ns'][:, :, None] / 1e9
    q = arrays['state'][:, 1:, :2] - arrays['state'][:, :-1, :2] - h * arrays['state'][:, :-1, 2:4]
    v = arrays['state'][:, 1:, 2:4] - arrays['state'][:, :-1, 2:4]
    normalization = {'scale_q': np.maximum(np.sqrt((q*q).mean((0, 1))), 1e-4).tolist(),
                     'scale_v': np.maximum(np.sqrt((v*v).mean((0, 1))), 1e-3).tolist(),
                     'velocity_input_rms': [1., 1.], 'training_manifest_sha256': sha256_file(train)}
    scaler = train.parent.parent / 'normalization.json'
    atomic_json(scaler, normalization)
    hashes = {str(path): sha256_file(path) for path in (train, valid, scaler,
                train.parent / 'trajectories.npz', valid.parent / 'trajectories.npz')}
    return train, valid, scaler, hashes


def test_reference_producer_leaves_archive_unchanged_and_retains_reloadable_weights(tmp_path):
    from poc.reference_jobs import produce_reference
    train, valid, scaler, hashes = inputs(tmp_path)
    folder = tmp_path / 'new_job'
    folder.mkdir()
    summary = produce_reference(folder, train, valid, scaler, model_id='physics_fit', epochs=5,
                                batch_size=32, phase='unit')
    assert summary['model_id'] == 'physics_fit'
    assert summary['training_updates'] == 10
    assert summary['selection']['best_epoch'] == 5
    assert all(sha256_file(Path(path)) == value for path, value in hashes.items())
    assert not (tmp_path / 'archive/evidence/data_access.jsonl').exists()
    opened = [read_json_line for read_json_line in (folder / 'data_access.jsonl').read_text().splitlines()]
    assert len(opened) == 2
    weights = torch.load(folder / 'weights.pt', weights_only=True, map_location='cpu')
    assert 'raw' in weights
    assert read_json(folder / 'selection.json')['target_components'] == ['q', 'v']
    assert read_json(folder / 'resume.complete.json')['checkpoint_sha256'] == sha256_file(folder / 'resume.pt')


def test_modified_normalization_and_wrong_roles_fail_before_training(tmp_path):
    from poc.reference_jobs import produce_reference
    train, valid, scaler, _ = inputs(tmp_path)
    folder = tmp_path / 'new_job'
    folder.mkdir()
    with pytest.raises(ValueError, match='role'):
        produce_reference(folder, valid, train, scaler, model_id='physics_fit', epochs=5, phase='unit')
    normalization = read_json(scaler)
    normalization['training_manifest_sha256'] = '0' * 64
    atomic_json(scaler, normalization)
    with pytest.raises(ValueError, match='normalization'):
        produce_reference(folder, train, valid, scaler, model_id='physics_fit', epochs=5, phase='unit')
    assert not (folder / 'weights.pt').exists()


def test_ridge_reference_records_single_solve_and_nominal_validation(tmp_path):
    from poc.reference_jobs import produce_reference
    train, valid, scaler, _ = inputs(tmp_path, actuator='lagged')
    folder = tmp_path / 'ridge_job'
    folder.mkdir()
    summary = produce_reference(folder, train, valid, scaler, model_id='accel_ridge', phase='unit')
    assert summary['training_updates'] == 0
    assert summary['selection']['transition_count'] == 64
    assert np.isfinite(summary['selection']['best_validation_loss'])
    assert 'coefficients' in torch.load(folder / 'weights.pt', weights_only=True)
    assert not (tmp_path / 'archive/evidence/data_access.jsonl').exists()
