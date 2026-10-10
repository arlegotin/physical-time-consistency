import copy
from pathlib import Path

import pytest

from poc.io import read_json
from poc.attribution_development import validate_jobs, summarize


def test_all_original_ten_task_intervention_cells_are_required():
    root = Path.cwd()
    registry = read_json(root / 'config/control_attribution_development_jobs.json')
    assert len(validate_jobs(root, registry)) == 160
    assert sum('reuse' in job for job in registry['jobs']) == 16


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'hash', 'design', 'reuse'])
def test_partial_or_changed_intervention_workload_fails_closed(mutation):
    root = Path.cwd()
    registry = copy.deepcopy(read_json(root / 'config/control_attribution_development_jobs.json'))
    if mutation == 'missing': registry['jobs'].pop()
    if mutation == 'duplicate': registry['jobs'][-1] = copy.deepcopy(registry['jobs'][0])
    if mutation == 'hash': registry['jobs'][0]['sha256'] = '0' * 64
    if mutation == 'design': registry['design_sha256'] = '0' * 64
    if mutation == 'reuse': registry['jobs'][0]['reuse']['manifest_sha256'] = '0' * 64
    with pytest.raises(ValueError): validate_jobs(root, registry)


def test_attribution_averages_tasks_within_the_single_training_seed():
    rows = {}
    for actuator in ('direct', 'lagged'):
        for episode in range(10):
            for mode, model in [('low', 'power_correct'), ('direct', 'power_correct'),
                                ('lagged', 'power_correct'), (None, 'node_structured')]:
                for h in (40_000_000, 5_000_000):
                    nominal = 10 if model == 'power_correct' else 8
                    fine = 9 if model == 'node_structured' else 12 if mode == actuator else 20
                    rows[(actuator, episode, model, mode, h)] = {
                        'realized_cost': float(nominal if h == 40_000_000 else fine)}
    result = summarize(rows, [{'matched': True}] * 40)
    for row in result['contrasts']:
        assert row['A_by_seed'] == [8.] and row['B_by_seed'] == [9.]
        assert row['fraction_explained'] is None
        assert row['n_training_seeds'] == 1 and row['n_tasks'] == 10
    assert len(result['task_contrasts']) == 20


def test_missing_task_cannot_be_silently_averaged():
    with pytest.raises(ValueError, match='Complete'):
        summarize({}, [])


def test_completed_first_task_is_verified_and_reused_without_relaunch(tmp_path, monkeypatch):
    import poc.attribution_development as batch
    import poc.contracts
    import poc.planner_qualification
    from poc.io import atomic_json, sha256_file
    job_path = tmp_path / 'config/job.json'
    request = dict(actuator='direct', episode_id=0, model_id='power_correct', power_mode='low',
        model_partition_ns=[40_000_000], seed=9000, condition='clean', phase='exploratory',
        actuator_interval_ns=40_000_000, physics_step_ns=250_000, checkpoint_sha256='a' * 64)
    atomic_json(job_path, request)
    folder = tmp_path / 'evidence/M40/old'
    atomic_json(folder / 'manifest.json', {'run_id': 'old'})
    atomic_json(folder / 'request.json', request)
    atomic_json(folder / 'summary.json', dict(request, checkpoint_hash='a' * 64, complete=True))
    atomic_json(folder / 'resources.json', {'wall_seconds': 1})
    reuse = dict(manifest='evidence/M40/old/manifest.json', manifest_sha256=sha256_file(folder / 'manifest.json'))
    job = dict(path='config/job.json', cell_id='one', sha256=sha256_file(job_path), reuse=reuse)
    atomic_json(tmp_path / 'config/control_attribution_development_jobs.json', {'jobs': [job]})
    monkeypatch.setattr(batch, 'validate_jobs', lambda root, registry: registry['jobs'])
    monkeypatch.setattr(poc.planner_qualification, 'verify_frozen', lambda root: ({}, []))
    verified = []
    monkeypatch.setattr(batch, 'verify_completed', lambda root, path, h: verified.append((path,h)) or root / path)
    monkeypatch.setattr(batch.subprocess, 'run', lambda *a, **k: pytest.fail('Historical episode must not relaunch'))
    # Deliberately incomplete fixture stops at missing cells after preserving the verified episode.
    with pytest.raises(KeyError): batch.main(['--root', str(tmp_path)])
    receipt = read_json(tmp_path / 'reports/control_attribution_development.json')
    assert verified == [(reuse['manifest'], reuse['manifest_sha256'])]
    assert receipt['state'] == 'interrupted' and receipt['completed_episodes'] == 1
    assert receipt['episodes'][0]['reused'] is True


@pytest.mark.parametrize('fail_at', [None, 3])
def test_full_runner_preserves_complete_pairs_and_interrupted_attempts(tmp_path, monkeypatch, fail_at):
    from types import SimpleNamespace
    import poc.attribution_development as batch
    import poc.contracts
    import poc.control_attribution
    import poc.planner_qualification
    from poc.io import atomic_json, sha256_file
    jobs = []
    for a in ('direct', 'lagged'):
        for e in range(10):
            for m, mode in batch.MODES:
                for partition in batch.PARTITIONS:
                    i = len(jobs); folder = tmp_path / 'fake' / str(i)
                    r = dict(actuator=a, episode_id=e, model_id=m, power_mode=mode,
                        model_partition_ns=list(partition), seed=9000, condition='clean', phase='exploratory',
                        actuator_interval_ns=40_000_000, physics_step_ns=250_000, checkpoint_sha256='a' * 64)
                    path = tmp_path / 'config' / f'{i}.json'; atomic_json(path, r)
                    atomic_json(folder / 'manifest.json', {'run_id': str(i)})
                    atomic_json(folder / 'request.json', r)
                    cost = (8 if m == 'node_structured' else 10) if partition[0] == 40_000_000 else (
                        9 if m == 'node_structured' else 12 if mode == a else 20)
                    atomic_json(folder / 'summary.json', dict(r, complete=True, checkpoint_hash='a' * 64, realized_cost=cost))
                    atomic_json(folder / 'resources.json', {'wall_seconds': 1})
                    jobs.append(dict(path=str(path.relative_to(tmp_path)), cell_id=str(i), fake_manifest=str(folder/'manifest.json')))
    atomic_json(tmp_path / 'config/control_attribution_development_jobs.json', {'jobs': jobs})
    monkeypatch.setattr(batch, 'validate_jobs', lambda root, registry: registry['jobs'])
    monkeypatch.setattr(poc.planner_qualification, 'verify_frozen', lambda root: ({}, []))
    verified = []
    monkeypatch.setattr(poc.contracts, 'verify_run', lambda root, record: verified.append(record['run_id']))
    monkeypatch.setattr(poc.control_attribution, 'verify_nominal_power_outputs',
                        lambda left, right, **kwargs: {'matched': True})
    calls = []
    def run(argv, **kwargs):
        calls.append(argv[-1]); job = next(j for j in jobs if j['path'] == argv[-1])
        return SimpleNamespace(returncode=1 if len(calls) == fail_at else 0,
                               stdout=job['fake_manifest']+'\n', stderr='interrupted')
    monkeypatch.setattr(batch.subprocess, 'run', run)
    if fail_at is not None:
        with pytest.raises(RuntimeError, match='interrupted'):
            batch.main(['--root', str(tmp_path)])
        r = read_json(tmp_path / 'reports/control_attribution_development.json')
        assert r['state'] == 'interrupted' and r['completed_episodes'] == fail_at - 1
        assert 'contrasts' not in r
    else:
        assert batch.main(['--root', str(tmp_path)]) == 0
        r = read_json(tmp_path / 'reports/control_attribution_development.json')
        assert r['state'] == 'complete' and r['completed_episodes'] == 160
        assert len(verified) == 160 and len(r['nominal_identities']) == 40
        assert all(c['A'] == 8 and c['B'] == 9 and c['fraction_explained'] is None for c in r['contrasts'])
