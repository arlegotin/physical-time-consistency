import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from poc.attribution_batch import main, validate_jobs
from poc.io import atomic_json, read_json


def test_same_weights_measurement_covers_both_actuators_modes_and_partitions():
    root = Path.cwd()
    registry = read_json(root / 'config/control_attribution_measurement_jobs.json')
    assert len(validate_jobs(root, registry)) == 16


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'hash', 'design'])
def test_changed_or_incomplete_intervention_measurement_is_rejected(mutation):
    root = Path.cwd()
    registry = copy.deepcopy(read_json(root / 'config/control_attribution_measurement_jobs.json'))
    if mutation == 'missing':
        registry['jobs'].pop()
    if mutation == 'duplicate':
        registry['jobs'][-1] = copy.deepcopy(registry['jobs'][0])
    if mutation == 'hash':
        registry['jobs'][0]['sha256'] = '0' * 64
    if mutation == 'design':
        registry['design_sha256'] = '0' * 64
    with pytest.raises(ValueError):
        validate_jobs(root, registry)


def mocked_runner(tmp_path, monkeypatch):
    import poc.attribution_batch as batch
    import poc.contracts
    import poc.control_attribution
    import poc.planner_qualification

    (tmp_path / 'config').mkdir()
    jobs = []
    costs = {('low', 40_000_000): 10, ('correct', 40_000_000): 10,
             ('structured', 40_000_000): 8, ('low', 5_000_000): 20,
             ('correct', 5_000_000): 12, ('structured', 5_000_000): 9,
             ('mismatch', 40_000_000): 10, ('mismatch', 5_000_000): 16}
    for actuator in ('direct', 'lagged'):
        for mode in ('low', 'direct', 'lagged', None):
            for h in (40_000_000, 5_000_000):
                index = len(jobs); folder = tmp_path / 'fake_complete_runs' / str(index)
                folder.mkdir(parents=True)
                tag = 'structured' if mode is None else 'correct' if mode == actuator else 'low' if mode == 'low' else 'mismatch'
                atomic_json(folder / 'manifest.json', {'run_id': str(index)})
                atomic_json(folder / 'summary.json', dict(actuator=actuator,
                    model_id='node_structured' if mode is None else 'power_correct',
                    power_mode=mode, model_partition_ns=[h], realized_cost=costs[(tag, h)]))
                atomic_json(folder / 'resources.json', {'wall_seconds': 1})
                jobs.append({'path': str(folder / 'manifest.json'), 'cell_id': str(index)})
    atomic_json(tmp_path / 'config/control_attribution_measurement_jobs.json', {'jobs': jobs})
    monkeypatch.setattr(batch, 'validate_jobs', lambda root, registry: registry['jobs'])
    monkeypatch.setattr(poc.planner_qualification, 'verify_frozen', lambda root: ({}, []))
    verified, identities = [], []
    monkeypatch.setattr(poc.contracts, 'verify_run', lambda root, record: verified.append(record['run_id']))
    def identity(left, right):
        identities.append((left, right))
        return {'matched': True}
    monkeypatch.setattr(poc.control_attribution, 'verify_nominal_power_outputs', identity)
    monkeypatch.setattr(batch.subprocess, 'run', lambda argv, **kwargs:
                        SimpleNamespace(returncode=0, stdout=argv[-1] + '\n', stderr=''))
    return jobs, verified, identities


def test_runner_orders_nominal_and_fine_costs_and_verifies_raw_identity(tmp_path, monkeypatch):
    jobs, verified, identities = mocked_runner(tmp_path, monkeypatch)
    assert main(['--root', str(tmp_path)]) == 0
    receipt = read_json(tmp_path / 'reports/control_attribution_measurements.json')
    assert receipt['state'] == 'complete' and receipt['completed_episodes'] == 16
    assert len(verified) == 16 and len(identities) == 2
    for contrast in receipt['contrasts']:
        assert contrast['A'] == 8 and contrast['B'] == 9
        assert contrast['A_minus_half_B_by_seed'] == [3.5]
        assert contrast['fraction_explained'] is None


def test_raw_identity_failure_cannot_complete_receipt(tmp_path, monkeypatch):
    import poc.control_attribution
    mocked_runner(tmp_path, monkeypatch)
    def fail_identity(*args):
        raise ValueError('Nominal selected actions differ')
    monkeypatch.setattr(poc.control_attribution, 'verify_nominal_power_outputs', fail_identity)
    with pytest.raises(ValueError, match='Nominal selected actions'):
        main(['--root', str(tmp_path)])
    receipt = read_json(tmp_path / 'reports/control_attribution_measurements.json')
    assert receipt['state'] == 'interrupted' and 'contrasts' not in receipt


def test_interruption_keeps_verified_episodes_and_restart_uses_same_requests(tmp_path, monkeypatch):
    import poc.attribution_batch as batch
    jobs, verified, identities = mocked_runner(tmp_path, monkeypatch)
    calls = []
    def interrupted(argv, **kwargs):
        calls.append(argv[-1])
        return SimpleNamespace(returncode=1 if len(calls) == 3 else 0,
                               stdout=argv[-1] + '\n', stderr='interrupted child')
    monkeypatch.setattr(batch.subprocess, 'run', interrupted)
    with pytest.raises(RuntimeError, match='interrupted child'):
        main(['--root', str(tmp_path)])
    receipt = read_json(tmp_path / 'reports/control_attribution_measurements.json')
    assert receipt['state'] == 'interrupted' and receipt['completed_episodes'] == 2
    monkeypatch.setattr(batch.subprocess, 'run', lambda argv, **kwargs:
                        SimpleNamespace(returncode=0, stdout=argv[-1] + '\n', stderr=''))
    assert main(['--root', str(tmp_path)]) == 0
    receipt = read_json(tmp_path / 'reports/control_attribution_measurements.json')
    assert receipt['state'] == 'complete'
    assert [e['manifest'] for e in receipt['episodes']] == [str(Path(j['path']).relative_to(tmp_path)) for j in jobs]
