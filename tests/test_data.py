"""Corpus reproducibility, label access, integrity and train-only scales."""
import importlib
from pathlib import Path

import numpy as np
import pytest

from poc.io import atomic_json, read_json, sha256_file


def modules():
    assert importlib.util.find_spec('poc.generate'), 'Deterministic corpus generator is not implemented'
    assert importlib.util.find_spec('poc.dataset'), 'Audited corpus loader is not implemented'
    return importlib.import_module('poc.generate'), importlib.import_module('poc.dataset')


def fixture(root, actuator='lagged', condition='A', role='train', count=4):
    generator, _ = modules()
    arrays = generator.generate_arrays(actuator, condition, role, count, physics_step_ns=250_000,
                                       asset_root=Path('assets'))
    folder = root / 'data' / actuator / condition / role
    folder.mkdir(parents=True)
    np.savez_compressed(folder / 'trajectories.npz', **arrays)
    manifest = generator.corpus_manifest(arrays, actuator, condition, role,
                                         folder / 'trajectories.npz', numerical={'physics_step_ns':250_000},
                                         parent_decision=None, root=root, phase='unit')
    atomic_json(folder / 'manifest.json', manifest)
    return folder / 'manifest.json', arrays


@pytest.mark.parametrize('actuator', ['direct', 'lagged'])
@pytest.mark.parametrize('condition', ['F', 'A'])
def test_seed_channels_and_duration_law_replay_exactly(actuator, condition):
    generator, _ = modules()
    first = generator.generate_arrays(actuator, condition, 'train', 2, physics_step_ns=250_000,
                                      asset_root=Path('assets'))
    second = generator.generate_arrays(actuator, condition, 'train', 2, physics_step_ns=250_000,
                                       asset_root=Path('assets'))
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    root_seed = 1201 if condition == 'F' else 1202
    actuator_code = 0 if actuator == 'direct' else 1
    expected = np.random.Generator(np.random.PCG64(np.random.SeedSequence([root_seed,actuator_code,0,1]))).uniform(-2,2,(32,2))
    np.testing.assert_array_equal(first['command'][0], expected)
    assert first['state'].shape == (2,33,4 if actuator == 'direct' else 6)
    assert first['state'].dtype == np.float64 and first['duration_ns'].dtype == np.int64
    np.testing.assert_array_equal(first['time_ns'][:,1:], np.cumsum(first['duration_ns'],axis=1))
    assert set(first['duration_ns'].ravel()) <= ({40_000_000} if condition == 'F' else {20_000_000,30_000_000,40_000_000,50_000_000,60_000_000})


def test_whole_trajectory_role_separation_and_explicit_access(tmp_path):
    _, loader = modules()
    manifest, training = fixture(tmp_path)
    arrays = loader.load_corpus(manifest, phase='unit', role='train', purpose='training')
    np.testing.assert_array_equal(arrays['state'],training['state'])
    validation, independent = fixture(tmp_path,role='validation')
    assert not np.array_equal(training['command'],independent['command'])
    with pytest.raises(ValueError, match='role'):
        loader.load_corpus(validation, phase='unit', role='train', purpose='training')
    with pytest.raises(ValueError, match='purpose|access'):
        loader.load_corpus(manifest,phase='unit',role='train',purpose='unspecified')
    assert (tmp_path / 'evidence/data_access.jsonl').is_file()


def test_archive_can_be_read_without_modifying_its_evidence_ledger(tmp_path):
    _, loader = modules()
    manifest, original = fixture(tmp_path / 'archive')
    audit = tmp_path / 'new_job' / 'data_access.jsonl'
    result = loader.load_corpus(manifest, phase='unit', role='train', purpose='training', audit_path=audit)
    np.testing.assert_array_equal(result['state'], original['state'])
    assert audit.is_file()
    assert not (tmp_path / 'archive' / 'evidence/data_access.jsonl').exists()
    assert sha256_file(manifest) in audit.read_text()


@pytest.mark.parametrize('purpose', ['training','selection'])
def test_confirmation_and_dense_oracle_cannot_enter_training_even_in_confirmatory_phase(tmp_path,purpose):
    _, loader = modules()
    manifest,_ = fixture(tmp_path,role='confirmation')
    with pytest.raises(ValueError,match='confirmation|access'):
        loader.load_corpus(manifest,phase='confirmatory',role='confirmation',purpose=purpose)
    data = read_json(manifest)
    data['role'] = 'oracle'
    atomic_json(manifest,data)
    with pytest.raises(ValueError,match='role|oracle|access'):
        loader.load_corpus(manifest,phase='confirmatory',role='oracle',purpose=purpose)


def test_tampering_truncation_and_nonfinite_labels_are_rejected(tmp_path):
    _, loader = modules()
    manifest, arrays = fixture(tmp_path)
    path = manifest.parent / 'trajectories.npz'
    arrays['state'][0,1,0] += .1
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError,match='hash'):
        loader.load_corpus(manifest,phase='unit',role='train',purpose='training')
    saved = read_json(manifest)
    saved['files']['trajectories.npz'] = sha256_file(path)
    atomic_json(manifest,saved)
    arrays['state'] = arrays['state'][:,:32]
    np.savez_compressed(path, **arrays)
    saved['files']['trajectories.npz'] = sha256_file(path)
    atomic_json(manifest,saved)
    with pytest.raises(ValueError,match='shape|dimension'):
        loader.load_corpus(manifest,phase='unit',role='train',purpose='training')


def test_nonfinite_duplicate_ids_and_wrong_times_are_not_silently_dropped(tmp_path):
    _, loader = modules()
    manifest, arrays = fixture(tmp_path)
    path = manifest.parent / 'trajectories.npz'
    for key in ('nonfinite','duplicates','times'):
        changed = {name:value.copy() for name,value in arrays.items()}
        if key == 'nonfinite':
            changed['state'][0,1,0] = np.nan
        elif key == 'duplicates':
            changed['trajectory_id'][1] = changed['trajectory_id'][0]
        else:
            changed['time_ns'][0,1] += 1
        np.savez_compressed(path,**changed)
        saved = read_json(manifest)
        saved['files']['trajectories.npz'] = sha256_file(path)
        atomic_json(manifest,saved)
        with pytest.raises(ValueError):
            loader.load_corpus(manifest,phase='unit',role='train',purpose='training')


def test_schema_unknown_keys_and_path_actuator_conflicts_are_rejected(tmp_path):
    _, loader = modules()
    manifest,_ = fixture(tmp_path)
    saved = read_json(manifest)
    saved['unregistered'] = True
    atomic_json(manifest,saved)
    with pytest.raises(ValueError,match='Unknown|unknown|Additional'):
        loader.load_corpus(manifest,phase='unit',role='train',purpose='training')
    del saved['unregistered']
    saved['actuator'] = 'direct'
    atomic_json(manifest,saved)
    with pytest.raises(ValueError,match='path|actuator'):
        loader.load_corpus(manifest,phase='unit',role='train',purpose='training')


def test_complete_trajectory_resumption_is_identical_and_partial_checkpoint_is_regenerated(tmp_path):
    generator,_ = modules()
    cache = tmp_path / 'partial'
    kwargs = dict(physics_step_ns=250_000,asset_root=Path('assets'),resume_dir=cache)
    first = generator.generate_arrays('lagged','A','train',16,**kwargs)
    assert first['state'].shape == (16,33,6)
    (cache / 'trajectory-15.json').unlink()  # Interrupted before completion marker.
    (cache / 'trajectory-15.npz').write_bytes(b'Partial output')
    second = generator.generate_arrays('lagged','A','train',16,**kwargs)
    for key in first:
        np.testing.assert_array_equal(first[key],second[key])
    with pytest.raises(ValueError,match='resumption'):
        generator.generate_arrays('lagged','A','train',16,**{**kwargs,'physics_step_ns':125_000})


def test_confirmation_generation_requires_actual_freeze_before_any_sampling(tmp_path):
    generator,_ = modules()
    assert hasattr(generator,'generate_corpus'), 'Qualified corpus producer is not implemented'
    with pytest.raises(ValueError,match='freeze'):
        generator.generate_corpus(tmp_path,'direct','F','confirmation',256)


def test_normalization_uses_only_training_increments(tmp_path):
    _, loader = modules()
    manifest, arrays = fixture(tmp_path)
    validation,_ = fixture(tmp_path,role='validation')
    first = loader.fit_normalization(manifest)
    h = arrays['duration_ns'][:,:,None]/1e9
    q_residual = arrays['state'][:,1:,:2]-arrays['state'][:,:-1,:2]-h*arrays['state'][:,:-1,2:4]
    v_residual = arrays['state'][:,1:,2:4]-arrays['state'][:,:-1,2:4]
    np.testing.assert_allclose(first['scale_q'], np.maximum(np.sqrt(np.mean(q_residual**2,axis=(0,1))),1e-4))
    np.testing.assert_allclose(first['scale_v'], np.maximum(np.sqrt(np.mean(v_residual**2,axis=(0,1))),1e-3))
    (validation.parent / 'trajectories.npz').write_bytes(b'Changed validation never opened for normalization')
    assert first == loader.fit_normalization(manifest)
    with pytest.raises(ValueError,match='train'):
        loader.fit_normalization(validation)
