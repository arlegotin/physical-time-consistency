"""Catch favorable omission, invalid timing, and misleading hardware comparisons."""
import importlib.util

import pytest


def module():
    assert importlib.util.find_spec('poc.efficiency_result'), 'Pilot result consumer is missing'
    from poc import efficiency_result
    return efficiency_result


def row(model, rate, task, cost, seconds, *, step=40_000_000, fallbacks=0):
    return dict(actuator='direct', model_id=model, actuator_interval_ns=rate,
                model_partition_ns=[step]*(40_000_000//step), episode_id=task, seed=9000,
                phase='exploratory', complete=True, realized_cost=cost,
                planner_latency_median_seconds=seconds, planner_latency_p95_seconds=seconds*1.2,
                fallback_count=fallbacks, success=False, q_rmse_rad=.2, terminal_q_error_rad=.1,
                neural_rows=128, work=dict(network_calls=1, vector_field_calls=0))


def coordinates(rows):
    return [(r['actuator'], r['model_id'], r['actuator_interval_ns'],
             r['model_partition_ns'][0], r['episode_id']) for r in rows]


def test_incomplete_duplicate_or_modified_episode_cannot_summarize():
    api = module()
    rows = [row('power_correct', 10_000_000, 0, 2., .1), row('power_correct', 10_000_000, 1, 1., .1)]
    cells = coordinates(rows)
    assert api.aggregate_rows(rows, cells)['conditions'][0]['mean_cost'] == 1.5
    for broken in (rows[:-1], [rows[0], rows[0]], [dict(rows[0], complete=False), rows[1]],
                   [dict(rows[0], realized_cost=float('inf')), rows[1]],
                   [dict(rows[0], planner_latency_median_seconds=0), rows[1]]):
        with pytest.raises(ValueError):
            api.aggregate_rows(broken, cells)


def test_frontier_retains_losses_and_distinct_hardware_rates():
    api = module()
    rows = [row('power_correct', 40_000_000, 0, 1., .1),
            row('node_structured', 40_000_000, 0, .9, .12),
            row('mlp_h', 40_000_000, 0, 1.1, .15, fallbacks=1),
            row('power_correct', 10_000_000, 0, .8, .3),
            row('node_structured', 10_000_000, 0, .85, .2)]
    result = api.aggregate_rows(rows, coordinates(rows))
    assert len(result['conditions']) == 5
    conditions = {(r['model_id'], r['actuator_interval_ns']): r for r in result['conditions']}
    assert conditions['mlp_h', 40_000_000]['on_descriptive_frontier'] is False
    assert conditions['mlp_h', 40_000_000]['fallback_count'] == 1
    assert conditions['power_correct', 40_000_000]['on_descriptive_frontier'] is True
    assert conditions['node_structured', 40_000_000]['on_descriptive_frontier'] is True
    contrasts = {(r['competitor'], r['actuator_interval_ns']): r for r in result['comparisons']}
    assert contrasts['node_structured', 40_000_000]['cost_improvement'] == pytest.approx(-1/9)
    assert contrasts['node_structured', 10_000_000]['cost_improvement'] == pytest.approx(1-.8/.85)
    assert contrasts['node_structured', 40_000_000]['method_lower_cost_tasks'] == 0
    assert contrasts['node_structured', 10_000_000]['method_lower_cost_tasks'] == 1
    assert all(r['inference'] == 'exploratory_descriptive' for r in result['comparisons'])


def test_profile_consumer_requires_same_freeze_and_every_measured_sample():
    api = module()
    profile = dict(protocol_id='PTCO-EFF-0.1', freeze_sha256='a'*64,
                   benchmark_rows=[dict(model_id='power_correct', variant='efficient', actuator='direct',
                   batch_size=1, h_s=.005, samples_ns=[100, 120], median_ns=110., p95_ns=119.)])
    expected = [('direct', 'power_correct', 'efficient', 1, .005)]
    api.validate_profiles(profile, 'a'*64, expected, sample_counts={1: 2})
    for changed in (dict(profile, freeze_sha256='b'*64), dict(profile, benchmark_rows=[]),
                    dict(profile, benchmark_rows=[dict(profile['benchmark_rows'][0], samples_ns=[100])])):
        with pytest.raises(ValueError):
            api.validate_profiles(changed, 'a'*64, expected, sample_counts={1: 2})


def test_demo_decimation_keeps_initial_and_terminal_physical_time():
    import numpy as np
    api = module()
    time = np.arange(0, 1_600_000_001, 250_000, dtype=np.int64)
    state = np.zeros((len(time), 4))
    state[:, 0] = time/1e9
    clip = api.clip_data(time, state, np.array([.4, -.3]))
    assert clip['time_s'][0] == 0 and clip['time_s'][-1] == 1.6
    assert clip['q'][0] == [0., 0.] and clip['q'][-1] == [1.6, 0.]
    assert clip['target'] == [.4, -.3]
    assert len(clip['time_s']) == 161
    with pytest.raises(ValueError):
        api.clip_data(time[:-1], state[:-1], np.array([.4, -.3]))


def test_repeated_profile_preserves_old_measured_bytes(tmp_path):
    from poc.io import read_json, sha256_file
    api = module()
    first = dict(samples_ns=[100, 120], source_sha256='a'*64)
    second = dict(samples_ns=[90, 110], source_sha256='a'*64)
    original = api.store_profile(tmp_path, first)
    replacement = api.store_profile(tmp_path, second)
    assert original['path'] != replacement['path']
    assert read_json(tmp_path/original['path']) == first
    assert sha256_file(tmp_path/original['path']) == original['sha256']
    assert read_json(tmp_path/replacement['path']) == second
    assert read_json(tmp_path/'reports/efficiency_pilot_profiles.json') == second
    assert api.store_profile(tmp_path, second) == replacement
