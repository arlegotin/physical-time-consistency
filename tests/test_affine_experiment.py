"""Nominal-only selection, frozen inputs and actual optimizer-boundary recovery."""
from pathlib import Path

import pytest
import torch

from poc.affine_flow import AffineFlow
from poc.io import atomic_json, read_json, sha256_file
from poc.train import TrainingSession, analytic_control_arrays, array_scales, canonical_tensor_hash


def session_for(arrays, scales):
    from poc.affine_experiment import build_candidate
    model = build_candidate('lagged', scales, 9000)
    return TrainingSession(model, arrays, scales, 9000, 'shared', effective_batch=16)


def test_frozen_inputs_reject_changed_data_and_source(tmp_path):
    from poc.affine_experiment import verify_bindings
    path = tmp_path/'array.bin'
    path.write_bytes(b'initial measurements')
    hashes = {'array.bin':sha256_file(path)}
    verify_bindings(tmp_path, hashes)
    path.write_bytes(b'altered measurements')
    with pytest.raises(ValueError, match='changed'):
        verify_bindings(tmp_path, hashes)


def test_nominal_role_validation_refuses_confirmation_or_development():
    from poc.affine_experiment import validate_nominal_roles
    train = dict(role='train', actuator='lagged', condition='A')
    validation = dict(role='validation', actuator='lagged', condition='A')
    validate_nominal_roles(train, validation, 'lagged')
    for role in ('confirmation', 'development', 'validation'):
        with pytest.raises(ValueError, match='role'):
            validate_nominal_roles({**train, 'role':role}, validation, 'lagged')
    with pytest.raises(ValueError, match='role'):
        validate_nominal_roles(train, {**validation, 'role':'train'}, 'lagged')
    with pytest.raises(ValueError, match='actuator'):
        validate_nominal_roles(train, validation, 'direct')


def test_resumed_nominal_block_reproduces_weights_scores_and_target_counts(tmp_path):
    from poc.affine_experiment import train_block
    arrays = analytic_control_arrays('lagged', 'tiny')
    validation = analytic_control_arrays('lagged', 'validation')
    scales = array_scales(arrays)
    whole, first = session_for(arrays, scales), session_for(arrays, scales)
    direct_dir, resumed_dir = tmp_path/'whole', tmp_path/'resume'
    train_block(whole, validation, direct_dir, 10)
    train_block(first, validation, resumed_dir, 5)
    restored = session_for(arrays, scales)
    restored.restore_latest(resumed_dir/'resume.pt')
    result = train_block(restored, validation, resumed_dir, 10)
    assert whole.tensor_hash() == restored.tensor_hash()
    assert restored.valid_target_count == 160
    assert result['valid_target_count'] == 160
    assert read_json(direct_dir/'selection.json') == read_json(resumed_dir/'selection.json')
    a = torch.load(direct_dir/'weights.pt', weights_only=True)
    b = torch.load(resumed_dir/'weights.pt', weights_only=True)
    assert canonical_tensor_hash(a) == canonical_tensor_hash(b)


def test_resume_after_epoch_checkpoint_before_validation_recomputes_missing_score(tmp_path):
    from poc.affine_experiment import train_block
    arrays = analytic_control_arrays('lagged', 'tiny')
    validation = analytic_control_arrays('lagged', 'validation')
    scales = array_scales(arrays)
    whole = session_for(arrays, scales)
    expected = train_block(whole, validation, tmp_path/'whole', 5)
    interrupted = session_for(arrays, scales)
    folder = tmp_path/'interrupted'
    folder.mkdir()
    from poc.train import epoch_starts
    for epoch in range(1, 6):
        interrupted.advance_epoch(folder/'resume.pt')
        if epoch < 5:
            interrupted.epoch += 1
            interrupted.windows = epoch_starts(arrays, interrupted.rng)
            interrupted.window_position = 0
    restored = session_for(arrays, scales)
    restored.restore_latest(folder/'resume.pt')
    actual = train_block(restored, validation, folder, 5)
    assert actual['selected'] == expected['selected']
    assert actual['valid_target_count'] == expected['valid_target_count'] == 80


def test_nominal_selection_has_no_fine_time_score_input(tmp_path):
    from poc.affine_experiment import train_block
    arrays = analytic_control_arrays('lagged', 'tiny')
    scales = array_scales(arrays)
    result = train_block(session_for(arrays, scales), arrays, tmp_path, 5)
    selected = read_json(tmp_path/'selection.json')
    assert set(selected) == {'epoch', 'objective', 'one_step', 'four_step', 'one_count', 'four_count'}
    assert result['selected'] == selected
    assert selected['epoch'] == 5


def test_actuator_fits_have_different_content_addressed_job_inputs(tmp_path):
    from poc.affine_experiment import nominal_configuration
    from poc.io import job_key
    freeze = dict(nominal_requests={})
    for actor in ('direct', 'lagged'):
        name = f'config/affine_nominal_requests/{actor}.json'
        atomic_json(tmp_path/name, dict(protocol_id='PTCO-AFF-0.1', actuator=actor))
        freeze['nominal_requests'][actor] = name
    a = nominal_configuration(tmp_path, 'direct', freeze)
    b = nominal_configuration(tmp_path, 'lagged', freeze)
    assert job_key(a) != job_key(b)
    assert a['config/affine_nominal_requests/direct.json']['actuator'] == 'direct'
    assert b['config/affine_nominal_requests/lagged.json']['actuator'] == 'lagged'
    atomic_json(tmp_path/'config/affine_nominal_requests/lagged.json', dict(protocol_id='PTCO-AFF-0.1', actuator='direct'))
    with pytest.raises(ValueError, match='actuator'):
        nominal_configuration(tmp_path, 'lagged', freeze)
