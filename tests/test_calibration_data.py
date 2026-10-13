"""Explicit unit fixtures for budget isolation and bounded collector replay."""
import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def data_module():
    assert importlib.util.find_spec('poc.calibration_data'), 'Study-local data loader is missing'
    from poc import calibration_data
    return calibration_data


def fixture_arrays(count=600):
    # Synthetic labels are unit evidence only; never an empirical session.
    ids = np.arange(count, dtype=np.int64)
    states = np.zeros((count, 26, 6), dtype=np.float64)
    for tick in range(26):
        states[:, tick, :2] = .001 * tick * tick
        states[:, tick, 2:4] = .01 * tick
    return {'state': states, 'command': np.zeros((count, 25, 2), dtype=np.float64),
            'duration_ns': np.full((count, 25), 40_000_000, dtype=np.int64),
            'time_ns': np.tile(np.arange(26, dtype=np.int64)*40_000_000, (count, 1)),
            'trajectory_id': ids}


def fixture_receipt(tmp_path, arrays):
    from poc.calibration_store import CalibrationStore
    store = CalibrationStore(tmp_path, {'recipe': 'explicit-unit-fixture'})
    def produce(folder):
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        (folder / 'session.npz').write_bytes(buf.getvalue())
        return {'phase': 'unit', 'session_id': 0, 'segment_count': len(arrays['trajectory_id']),
                'budget_seconds': [30, 120, 600]}
    return store, store.produce('collect', {'session_id': 0, 'phase': 'unit'}, produce)


def test_nested_budgets_count_validation_and_preserve_segment_roles(tmp_path):
    data = data_module()
    store, receipt = fixture_receipt(tmp_path, fixture_arrays())
    prior_train, prior_validation = set(), set()
    for budget, train_s, validation_s in [(30, 24, 6), (120, 96, 24), (600, 480, 120)]:
        train, validation, scales, account = data.load_budget(receipt, budget, store)
        assert train['state'].shape == (train_s, 26, 6)
        assert validation['state'].shape == (validation_s, 26, 6)
        train_ids, validation_ids = set(train['trajectory_id']), set(validation['trajectory_id'])
        assert not train_ids & validation_ids
        assert prior_train <= train_ids and prior_validation <= validation_ids
        assert all(i % 5 != 4 for i in train_ids)
        assert all(i % 5 == 4 for i in validation_ids)
        assert account['train_seconds'] == train_s
        assert account['validation_seconds'] == validation_s
        assert account['total_seconds'] == budget
        assert account['phase'] == 'unit'
        assert scales.q.dtype == torch.float32
        starts = data.four_step_starts(train)
        assert starts.shape == (train_s * 22, 2)
        assert starts[:, 1].max() == 21
        prior_train, prior_validation = train_ids, validation_ids


def test_scales_ignore_validation_and_later_budget_labels(tmp_path):
    data = data_module()
    arrays = fixture_arrays()
    store, receipt = fixture_receipt(tmp_path / 'first', arrays)
    _, _, before, _ = data.load_budget(receipt, 30, store)
    altered = {k: v.copy() for k, v in arrays.items()}
    altered['state'][30:] *= 100
    altered['state'][4:30:5] *= 100
    store2, receipt2 = fixture_receipt(tmp_path / 'second', altered)
    _, _, after, _ = data.load_budget(receipt2, 30, store2)
    for a, b in [(before.q, after.q), (before.v, after.v), (before.v_input, after.v_input)]:
        assert torch.equal(a, b)
    # Hand-derived q residual at tick t is .001 + .0016*t.
    expected_q = np.sqrt(np.mean((.001+.0016*np.arange(25))**2))
    np.testing.assert_allclose(before.q.numpy(), [expected_q]*2, rtol=1e-6)
    np.testing.assert_allclose(before.v.numpy(), [.01, .01], rtol=1e-6)


@pytest.mark.parametrize('mutation', ['duplicate', 'time', 'shape', 'nonfinite', 'command'])
def test_invalid_session_arrays_are_rejected(tmp_path, mutation):
    data = data_module()
    arrays = fixture_arrays()
    if mutation == 'duplicate':
        arrays['trajectory_id'][1] = 0
    elif mutation == 'time':
        arrays['time_ns'][0, 1] += 1
    elif mutation == 'shape':
        arrays['state'] = arrays['state'][:, :-1]
    elif mutation == 'nonfinite':
        arrays['state'][0, 0, 0] = np.nan
    else:
        arrays['command'][0, 0, 0] = 1.6
    store, receipt = fixture_receipt(tmp_path, arrays)
    with pytest.raises(ValueError):
        data.load_budget(receipt, 30, store)


def test_collector_exact_replay_namespace_ticks_and_excitation(tmp_path):
    data = data_module()
    first = data.collect_arrays(2, root=ROOT, segment_count=2, phase='unit', resume_dir=tmp_path)
    second = data.collect_arrays(2, root=ROOT, segment_count=2, phase='unit')
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    for trajectory in range(2):
        initial = np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261012, 1, 0, 2, trajectory, 0])))
        expected = np.r_[initial.uniform(-.5, .5, 2), initial.uniform(-.25, .25, 2), initial.uniform(-.3, .3, 2)]
        np.testing.assert_array_equal(first['state'][trajectory, 0], expected)
        draws = np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261012, 1, 0, 2, trajectory, 1])))
        for tick in range(25):
            if tick % 5 == 0:
                d = draws.uniform(-.75, .75, 2)
            state = first['state'][trajectory, tick]
            np.testing.assert_array_equal(first['command'][trajectory, tick], np.clip(-state[:2]-.25*state[2:4]+d, -1.5, 1.5))
    assert np.all(first['duration_ns'] == 40_000_000)
    assert np.all(np.diff(first['time_ns'], axis=1) == 40_000_000)
    assert np.max(np.abs(first['state'][:, 1:, 2:4])) > .25
    with pytest.raises(ValueError):
        data.collect_arrays(2, root=ROOT, segment_count=2)


def test_interruption_restarts_only_incomplete_segment_and_retains_failure(tmp_path, monkeypatch):
    data = data_module()
    from poc.envs import Plant
    original = Plant.advance
    calls = 0
    def interrupt(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 29:
            raise RuntimeError('unit injected numerical failure')
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Plant, 'advance', interrupt)
    with pytest.raises(RuntimeError):
        data.collect_arrays(0, root=ROOT, segment_count=2, phase='unit', resume_dir=tmp_path)
    assert (tmp_path / 'segment-0000.json').exists()
    assert not (tmp_path / 'segment-0001.json').exists()
    event = json.loads((tmp_path / 'failures.jsonl').read_text())
    assert event['trajectory_id'] == 1 and event['tick'] == 3
    monkeypatch.setattr(Plant, 'advance', original)
    resumed = data.collect_arrays(0, root=ROOT, segment_count=2, phase='unit', resume_dir=tmp_path)
    direct = data.collect_arrays(0, root=ROOT, segment_count=2, phase='unit')
    for key in direct:
        np.testing.assert_array_equal(resumed[key], direct[key])
    assert (tmp_path / 'failures.jsonl').exists()


@pytest.mark.parametrize('changed', ['plant','runtime'])
def test_standalone_resume_binds_executing_plant_and_runtime(tmp_path,monkeypatch,changed):
    import poc.calibration_data as m
    from poc import calibration_study
    resume=tmp_path/'standalone'
    m.collect_arrays(0,root=ROOT,segment_count=1,phase='unit',resume_dir=resume)
    from poc.io import read_json
    binding=read_json(resume/'collector.json')
    assert binding['plant_source_sha256']
    runtime=calibration_study._runtime_versions()
    if changed=='runtime':
        monkeypatch.setattr(calibration_study,'_runtime_versions',lambda:{**runtime,'mujoco':'changed'})
    else:
        import inspect
        source=Path(inspect.getfile(m.Plant))
        sha=m.sha256_file
        monkeypatch.setattr(m,'sha256_file',lambda path:'changed' if path==source else sha(path))
    with pytest.raises(ValueError,match='cannot resume'):
        m.collect_arrays(0,root=ROOT,segment_count=1,phase='unit',resume_dir=resume)
