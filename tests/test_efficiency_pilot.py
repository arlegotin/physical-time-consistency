"""Freeze new comparisons without weakening existing clocks or evidence closure."""
from copy import deepcopy
import importlib.util
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from poc.io import atomic_json, read_json, sha256_file


def module():
    assert importlib.util.find_spec('poc.efficiency_pilot'), 'Explicit efficiency pilot is missing'
    from poc import efficiency_pilot
    return efficiency_pilot


def request():
    return dict(schema_version=1, protocol_id='PTCO-EFF-0.1', phase='exploratory',
                model_id='power_correct', inference_variant='efficient', actuator='direct', seed=9000,
                episode_id=0, condition='clean', model_partition_ns=[40_000_000],
                actuator_interval_ns=10_000_000, physics_step_ns=250_000, power_mode=None,
                model_manifest='fit/manifest.json', manifest_sha256='a'*64, checkpoint_sha256='b'*64,
                planner_settings=dict(candidate_count=128, iterations=4, elite_count=16,
                                      initial_std=1., std_floor=.1),
                resource_projection=dict(reason='Measure representative job before full batch'),
                freeze_sha256='c'*64)


def freeze_for(item):
    return dict(selections={f'{item["actuator"]}:{item["model_id"]}':
                            {key: item[key] for key in ('model_manifest', 'manifest_sha256', 'checkpoint_sha256')}},
                planner_settings=item['planner_settings'])


def test_pilot_rejects_changed_checkpoint_and_confirmation_seed():
    api = module()
    item = request()
    assert api.validate_request(item, freeze_for(item)) == item
    for key, value in [('checkpoint_sha256', 'd'*64), ('seed', 0), ('phase', 'confirmatory'),
                       ('inference_variant', 'original'), ('episode_id', 10), ('physics_step_ns', 500000)]:
        changed = dict(item, **{key: value})
        with pytest.raises(ValueError):
            api.validate_request(changed, freeze_for(item))
    item['model_id'] = 'mlp_h'
    item['inference_variant'] = 'original'
    assert api.validate_request(item, freeze_for(item)) == item


def test_registry_has_exact_paired_cells_and_oracle_subset():
    api = module()
    cells = api.coordinates()
    assert len(cells) == 408 and len(set(cells)) == 408
    assert sum(model == 'sim_oracle' for _, model, _, _, _ in cells) == 8
    for actuator in ('direct', 'lagged'):
        for rate in (40_000_000, 10_000_000):
            assert {(step, task) for a, model, r, step, task in cells
                    if a == actuator and model == 'sim_oracle' and r == rate} == {(40_000_000, 0), (40_000_000, 1)}
            for model in ('power_correct', 'mlp_h', 'node_generic', 'node_structured', 'physics_fit'):
                assert {(step, task) for a, m, r, step, task in cells if (a, m, r) == (actuator, model, rate)} == {
                    (step, task) for step in (40_000_000, 5_000_000) for task in range(10)}
    api.validate_coordinates(cells)
    for broken in (cells[:-1], [*cells[:-1], cells[0]], [*cells, cells[0]]):
        with pytest.raises(ValueError, match='coverage'):
            api.validate_coordinates(broken)


def test_high_rate_coarse_partition_splits_without_extra_observations():
    # Break caught: crossing a command change, or committing per-query observations.
    from poc.mpc import rollout_candidates
    from poc.models.common import prediction
    calls = []
    class Exact:
        def predict(self, state, command, h, history=None):
            calls.append((float(h), int(history.query_time_ns[0]),
                          tuple(len(tokens) for tokens in history.tokens)))
            q, v = state[:, :2], state[:, 2:]
            return prediction(torch.cat((q+h*v+.5*h*h*command, v+h*command), 1), 0)
    result = rollout_candidates(Exact(), torch.zeros(4, dtype=torch.float64), np.ones(2), None,
                                np.zeros((2, 6, 2)), (40_000_000,), 10_000_000)
    assert len(calls) == 24 and {h for h, _, _ in calls} == {.01}
    assert result['anchors'].shape == (2, 6, 4)
    assert all(lengths == (time//40_000_000,)*2 for _, time, lengths in calls)


def test_completed_job_reuse_checks_request_and_raw_artifacts(tmp_path):
    api = module()
    from poc.execution import run_job
    repository = Path(__file__).resolve().parents[1]
    for name in ('locks/environment.json', 'locks/wheels.json', 'schemas/manifest.schema.json'):
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((repository/name).read_bytes())
    atomic_json(tmp_path/'locks/wheels.json', dict(wheels=[]))
    subprocess.run(['git', 'init', str(tmp_path)], check=True, capture_output=True)
    subprocess.run(['git', '-C', str(tmp_path), '-c', 'user.name=Fixture',
                    '-c', 'user.email=fixture@example.invalid', 'commit', '--allow-empty', '-m', 'Fixture'],
                   check=True, capture_output=True)
    (tmp_path/'source.py').write_text('fixture=1\n')
    item = request()
    atomic_json(tmp_path/'request.json', item)
    def produce(folder):
        atomic_json(folder/'request.json', item)
        atomic_json(folder/'episode.json', dict(complete=True, phase='exploratory',
                    model_id='power_correct', seed=9000, episode_id=0, actuator='direct',
                    model_partition_ns=[40_000_000], actuator_interval_ns=10_000_000,
                    physics_step_ns=250_000, checkpoint_hash='b'*64))
        return [dict(check_id='fixture', status='pass', reason='Measured deterministic fixture',
                     scope='fixture', tolerance=None)], dict(missing_cells=[])
    # No local scientific imports live beneath the isolated fixture root.
    record = run_job(tmp_path, 'M40', 'exploratory', ['source.py', 'request.json'], ('fixture',), [], produce,
                     config_keys={'request.json': item})
    manifest = tmp_path/f'evidence/M40/{record["run_id"]}/manifest.json'
    assert api.verify_completion(tmp_path, manifest, item)['run_id'] == record['run_id']
    changed = dict(item, actuator_interval_ns=40_000_000)
    with pytest.raises(ValueError, match='request'):
        api.verify_completion(tmp_path, manifest, changed)
    marker = manifest.parent/'COMPLETED'
    marker.unlink()
    with pytest.raises(ValueError, match='Partial'):
        api.verify_completion(tmp_path, manifest, item)
    atomic_json(marker, dict(manifest_sha256=sha256_file(manifest)))
    atomic_json(manifest.parent/'episode.json', dict(complete=True, realized_cost=0.))
    with pytest.raises(ValueError, match='hash'):
        api.verify_completion(tmp_path, manifest, item)


def test_nominal_context_records_historical_receipts_and_rejects_changed_training_semantics(tmp_path):
    api = module()
    from poc.training_jobs import SOURCES
    context = tmp_path/'study/control-development'
    for name in SOURCES:
        for folder in (tmp_path, context):
            path = folder/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('unchanged training fixture\n')
    (tmp_path/'poc/execution.py').write_text('later receipt instrumentation\n')
    hashes = api.nominal_sources(tmp_path)
    assert hashes['study/control-development/poc/execution.py'] == sha256_file(context/'poc/execution.py')
    (context/'poc/train.py').write_text('changed objective\n')
    with pytest.raises(ValueError, match='training.*train.py'):
        api.nominal_sources(tmp_path)
