"""Solver continuation preserves frozen tasks and cannot hide missing resolutions."""
import copy
from pathlib import Path

import pytest
from poc.io import atomic_json, read_json, sha256_file


def frozen_inputs(root):
    from test_control_refinement import private_freeze
    from test_confirmation_solver import full_batch
    freeze, batch = private_freeze(root), full_batch()
    cells = []
    for job, entry in zip(freeze['jobs'], batch['episodes']):
        entry['cell_id'] = job['cell_id']
        entry['summary']['checkpoint_hash'] = 'b' * 64
        path = root / entry['manifest']
        atomic_json(path, {'private_fixture': True})
        entry['manifest_sha256'] = sha256_file(path)
        row = entry['summary']
        if row['model_id'] == 'power_correct':
            row['realized_cost'] = .8
        if row['model_id'] == 'node_generic' and row['model_partition_ns'] == [40_000_000] and row['episode_id'] < 5:
            cells.append(job['cell_id'])
    return freeze, batch, sorted(cells)


def test_solver_jobs_change_only_resolution_and_keep_every_requested_parent(tmp_path):
    from poc.control_solver_continuation import derive_jobs
    freeze, batch, cells = frozen_inputs(tmp_path)
    rows = derive_jobs(tmp_path, cells, batch, freeze)
    assert len(rows) == len({row['cell_id'] for row in rows}) == 100
    assert {row['parent_cell_id'] for row in rows} == set(cells)
    assert {row['substeps'] for row in rows} == {2, 4}
    parents = {job['cell_id']: job for job in freeze['jobs']}
    for row in rows:
        parent = parents[row['parent_cell_id']]
        assert row['request'] == read_json(tmp_path / parent['path'])
        assert row['parent_request_sha256'] == parent['sha256']
        assert row['request']['model_partition_ns'] == [40_000_000]
        assert row['request']['model_id'] == 'node_generic'


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'hash', 'clock', 'checkpoint', 'partial'])
def test_solver_derivation_rejects_changed_or_incomplete_frozen_inputs(tmp_path, mutation):
    from poc.control_solver_continuation import derive_jobs
    freeze, batch, cells = frozen_inputs(tmp_path)
    if mutation == 'missing':
        cells.pop()
    elif mutation == 'duplicate':
        cells[-1] = cells[0]
    elif mutation == 'partial':
        batch['state'] = 'running'
    else:
        job = next(job for job in freeze['jobs'] if job['cell_id'] == cells[0])
        path = tmp_path / job['path']
        request = read_json(path)
        if mutation == 'clock':
            request['model_partition_ns'] = [20_000_000] * 2
        else:
            request['checkpoint_sha256'] = 'c' * 64
        atomic_json(path, request)
        if mutation != 'hash':
            job['sha256'] = sha256_file(path)
    with pytest.raises(ValueError):
        derive_jobs(tmp_path, cells, batch, freeze)


def refined_rows(batch, cells):
    parents = {entry['cell_id']: entry for entry in batch['episodes']}
    result = []
    for cell in cells:
        for steps in (2, 4):
            row = copy.deepcopy(parents[cell]['summary'])
            row.update(parent_cell_id=cell, solver_substeps=steps)
            row['realized_cost'] += .001 if steps == 2 else .0011
            result.append(row)
    return result


def test_solver_summary_retains_original_costs_and_expands_by_largest_observed_discrepancy(tmp_path):
    from poc.control_solver_continuation import summarize_solver
    freeze, batch, cells = frozen_inputs(tmp_path)
    rows = refined_rows(batch, cells)
    result = summarize_solver(batch, rows, cells, read_json(Path('config/control_confirmation_design.json')))
    assert result['state'] == 'qualified'
    assert result['required_actions'] == []
    assert result['n_original_episodes'] == 50 and result['n_refined_episodes'] == 100
    assert result['new_independent_training_clusters'] == 0
    assert result['scientific_decision'] == 'not_evaluated'
    assert result['envelopes']['utility_control/direct/node_generic/h40000000'] == pytest.approx({'lower': .0011, 'upper': .0011})
    assert result['original_mean_cost_by_plant']['direct'] == 1.
    assert len(result['episode_comparisons']) == 50


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'nonfinite', 'checkpoint', 'task', 'clock'])
def test_solver_summary_cannot_clear_request_with_partial_or_relabelled_evidence(tmp_path, mutation):
    from poc.control_solver_continuation import summarize_solver
    _, batch, cells = frozen_inputs(tmp_path)
    rows = refined_rows(batch, cells)
    if mutation == 'missing':
        rows.pop()
    elif mutation == 'duplicate':
        rows[-1] = rows[0]
    elif mutation == 'nonfinite':
        rows[0]['realized_cost'] = float('nan')
    elif mutation == 'checkpoint':
        rows[0]['checkpoint_hash'] = 'c' * 64
    elif mutation == 'task':
        rows[0]['target'][0] += .1
    else:
        rows[0]['model_partition_ns'] = [5_000_000] * 8
    with pytest.raises(ValueError):
        summarize_solver(batch, rows, cells, read_json(Path('config/control_confirmation_design.json')))


@pytest.mark.parametrize('mutation', ['cost', 'success', 'fallback'])
def test_unconverged_closed_loop_behavior_requests_more_resolution_instead_of_qualifying(tmp_path, mutation):
    from poc.control_solver_continuation import summarize_solver
    _, batch, cells = frozen_inputs(tmp_path)
    rows = refined_rows(batch, cells)
    fine = next(row for row in rows if row['solver_substeps'] == 4)
    if mutation == 'cost':
        fine['realized_cost'] += .2
    elif mutation == 'success':
        fine['success'] = True
    else:
        fine['fallback_count'] = 1
    result = summarize_solver(batch, rows, cells, read_json(Path('config/control_confirmation_design.json')))
    assert result['state'] == 'additional_solver_evidence_required'
    assert result['required_actions']
    assert fine['parent_cell_id'] in result['required_actions'][0]['cells']
    assert result['required_actions'][0]['next_substeps'] == 8


def test_real_solver_episode_preserves_observations_work_and_manifest_closure(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from test_control_confirmation_batch import completed_episode
    from test_confirmation_solver import unit_mass_node
    from poc.contracts import verify_run
    from poc.io import complete_run, job_key
    import poc.control_solver_continuation as module
    from poc.control_jobs import SOURCES

    fingerprint = {key: 'fixture' for key in ('python', 'architecture', 'os', 'kernel', 'core_count',
        'torch_configuration', 'torch_threads', 'torch_interop_threads', 'deterministic_algorithms')}
    fingerprint['versions'] = {}
    atomic_json(tmp_path / 'locks/environment.json', fingerprint)
    atomic_json(tmp_path / 'locks/wheels.json', {'wheels': []})
    for name in set(SOURCES) | set(module.SOURCES):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# Dependency-boundary fixture\n')
    assets = tmp_path / 'assets'
    assets.mkdir()
    (assets / 'two_link_direct.xml').write_bytes(Path('assets/two_link_direct.xml').read_bytes())
    job, parent = completed_episode(tmp_path, 0)
    # Enrich this private generated fixture before its final seal.
    (parent.parent / 'COMPLETED').unlink()
    original = read_json(tmp_path / job['path'])
    original.update(model_id='node_generic', model_partition_ns=[40_000_000],
        planner_settings=dict(candidate_count=4, iterations=1, elite_count=1, initial_std=1., std_floor=.1))
    model_path = tmp_path / original['model_manifest']
    atomic_json(model_path, {'fixture': True})
    atomic_json(model_path.parent / 'normalization.json', {'fixture': True})
    (model_path.parent / 'weights.pt').write_bytes(b'fixture')
    original['manifest_sha256'] = sha256_file(model_path)
    atomic_json(tmp_path / job['path'], original)
    atomic_json(parent.parent / 'request.json', original)
    summary = read_json(parent.parent / 'summary.json')
    summary.update(original)
    atomic_json(parent.parent / 'summary.json', summary)
    manifest = read_json(parent)
    manifest['job_inputs']['dependencies'][job['path']] = sha256_file(tmp_path / job['path'])
    manifest['job_key'] = job_key(manifest['job_inputs'])
    manifest['produced_files'] = [dict(path=str(p.relative_to(tmp_path)), sha256=sha256_file(p))
        for p in sorted(parent.parent.iterdir()) if p.name not in ('manifest.json', 'COMPLETED')]
    complete_run(tmp_path, manifest)
    wrapped = dict(schema_version=1, cell_id='solver2__' + job['cell_id'], parent_cell_id=job['cell_id'],
        substeps=2, request=original, parent_request=job['path'],
        parent_request_sha256=sha256_file(tmp_path / job['path']), parent_manifest=str(parent.relative_to(tmp_path)),
        parent_manifest_sha256=sha256_file(parent))
    request_path = tmp_path / 'config/control_solver_jobs/fixture.json'
    atomic_json(request_path, wrapped)
    monkeypatch.setattr('poc.control_models.load_selected', lambda *args, **kwargs: unit_mass_node())
    monkeypatch.setattr('poc.execution.subprocess.run', lambda argv, **kwargs:
        SimpleNamespace(stdout='fixture' if argv[1] == 'rev-parse' else '', returncode=0))
    monkeypatch.setattr('poc.io._process_start', lambda pid: 'fixture-start')
    observed = module.execute_job(tmp_path, request_path)
    verify_run(tmp_path, read_json(tmp_path / observed['manifest']))
    row = observed['summary']
    assert row['complete'] is True and row['solver_substeps'] == 2
    assert row['work']['vector_field_calls'] == 1920
    assert row['work']['scalar_candidate_evaluations'] == 960
    assert observed['raw_audit']['n_observations'] == observed['raw_audit']['n_real_anchors'] == 40
    assert row['parent_cell_id'] == job['cell_id']
    first_manifest = observed['manifest']
    again = module.execute_job(tmp_path, request_path)
    assert again['manifest'] == first_manifest
    (request_path).write_text('{"edited":true}')
    with pytest.raises(ValueError):
        module.observe_job(tmp_path, wrapped | {'path': str(request_path.relative_to(tmp_path)),
            'sha256': '0' * 64}, tmp_path / first_manifest)


def test_solver_prepare_requires_a_verified_trigger_before_writing_requests(tmp_path):
    from poc.control_solver_continuation import prepare
    with pytest.raises(ValueError, match='trigger'):
        prepare(tmp_path)
    assert not (tmp_path / 'config/control_solver_continuation_jobs.json').exists()
