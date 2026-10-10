import copy
import io

import numpy as np
import pytest
import torch

from poc.affine_fresh import (build_fresh_model, fresh_entropy, generate_fresh_arrays,
                              load_fresh_arrays, validate_fresh_manifest, fit_configuration)
from poc.io import atomic_json, sha256_file
from poc.models.common import Scales


def test_fresh_namespace_is_disjoint_and_explicit():
    assert fresh_entropy('train', 7, 0) == [20261008, 2, 0, 1, 7, 0]
    assert fresh_entropy('validation', 7, 0) != fresh_entropy('train', 7, 0)
    assert fresh_entropy('confirmation', 7, 3) == [20261008, 2, 2, 1, 7, 3]
    for role, trajectory, channel in [('development', 0, 0), ('train', -1, 0), ('train', 0, 4)]:
        with pytest.raises(ValueError):
            fresh_entropy(role, trajectory, channel)


def test_fresh_fit_identity_separates_model_seed_and_data():
    freeze = {'protocol_id':'PTCO-AFF-0.2', 'initial_seeds':list(range(2026100800, 2026100805))}
    data = {'train':{'sha256':'a'}, 'validation':{'sha256':'b'}}
    first = fit_configuration(freeze, 'affine_flow', 2026100800, data)
    assert first != fit_configuration(freeze, 'node_structured', 2026100800, data)
    assert first != fit_configuration(freeze, 'affine_flow', 2026100801, data)
    changed = copy.deepcopy(data); changed['train']['sha256']='c'
    assert first != fit_configuration(freeze, 'affine_flow', 2026100800, changed)
    assert first != fit_configuration(freeze,'affine_flow',2026100800,data,'fresh_process_1')


def test_fresh_comparators_share_scales_and_seeded_initialization():
    scales = Scales(torch.ones(2), torch.ones(2), torch.ones(2))
    for name, count in [('affine_flow', 18182), ('node_structured', 17922)]:
        first, second = (build_fresh_model(name, scales, 2026100800) for _ in range(2))
        assert sum(p.numel() for p in first.parameters()) == count
        assert all(torch.equal(a,b) for a,b in zip(first.parameters(), second.parameters()))
        assert first.actuator == 'lagged'


def test_fresh_generation_resumes_exact_whole_trajectories(tmp_path):
    first = generate_fresh_arrays('train', 2, 500000, 'assets', tmp_path)
    second = generate_fresh_arrays('train', 2, 500000, 'assets', tmp_path)
    assert all(np.array_equal(first[k], second[k]) for k in first)
    assert first['state'].shape == (2,33,6)
    assert set(np.unique(first['duration_ns'])) <= {20000000,30000000,40000000,50000000,60000000}
    assert np.array_equal(first['time_ns'][:,1:], np.cumsum(first['duration_ns'],axis=1))
    with pytest.raises(ValueError, match='resumption'):
        generate_fresh_arrays('validation', 2, 500000, 'assets', tmp_path)


def test_fresh_loader_rejects_leaks_modified_data_and_false_role(tmp_path):
    arrays=generate_fresh_arrays('train',1,500000,'assets')
    np.savez_compressed(tmp_path/'trajectories.npz',**arrays)
    manifest=dict(protocol_id='PTCO-AFF-0.2',role='train',actuator='lagged',condition='A',
                  trajectory_count=1,physics_step_ns=500000,
                  arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in arrays.items()},
                  files={'trajectories.npz':sha256_file(tmp_path/'trajectories.npz')})
    atomic_json(tmp_path/'corpus.json',manifest)
    assert np.array_equal(load_fresh_arrays(tmp_path/'corpus.json','training')['state'],arrays['state'])
    with pytest.raises(ValueError, match='role'):
        load_fresh_arrays(tmp_path/'corpus.json','selection')
    manifest['role']='confirmation'; atomic_json(tmp_path/'corpus.json',manifest)
    with pytest.raises(ValueError, match='role'):
        load_fresh_arrays(tmp_path/'corpus.json','training')
    manifest['role']='train'; manifest['files']['trajectories.npz']='wrong'
    atomic_json(tmp_path/'corpus.json',manifest)
    with pytest.raises(ValueError,match='hash'):
        load_fresh_arrays(tmp_path/'corpus.json','training')
    manifest['protocol_id']='PTCO-AFF-0.1'
    with pytest.raises(ValueError, match='protocol'):
        validate_fresh_manifest(manifest)


def test_fresh_job_configuration_is_readable_and_immutable(tmp_path):
    # An invented configuration label used as a filename breaks verify_run.
    from poc.affine_fresh import job_configuration
    from poc.io import read_json
    values = job_configuration(tmp_path, 'corpus', {'role':'train','count':2})
    assert len(values) == 1
    path, expected = next(iter(values.items()))
    assert read_json(tmp_path/path) == {'role':'train','count':2} == expected
    assert job_configuration(tmp_path, 'corpus', {'role':'train','count':2}) == values
    other = job_configuration(tmp_path, 'corpus', {'role':'validation','count':2})
    assert set(other).isdisjoint(values)
    atomic_json(tmp_path/path, {'role':'confirmation','count':2})
    with pytest.raises(ValueError, match='configuration'):
        job_configuration(tmp_path, 'corpus', {'role':'train','count':2})
