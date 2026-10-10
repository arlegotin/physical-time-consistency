"""Feedback receives real observation packets, never imagined or scoring-only states."""
import numpy as np
import pytest

from poc.analytic import transition
from poc.action_path import ActionSegment
from poc.control import anchor_cost
from poc import control_episode


class ExactPlant:
    actuator = 'direct'
    physics_step_ns = 10_000_000

    def reset(self, state, time_s=0):
        self.x = np.array(state, copy=True)
        self.time_ns = 0

    def state(self):
        return self.x.copy()

    def snapshot(self):
        return {'state': self.state().tolist(), 'time_ns': self.time_ns,
                'actuator': self.actuator, 'physics_step_ns': self.physics_step_ns}

    def advance(self, command, duration_ns, external_torque=None, log_every_step=False):
        assert duration_ns % self.physics_step_ns == 0
        times, states = [], []
        force = np.zeros(2) if external_torque is None else external_torque
        for _ in range(duration_ns // self.physics_step_ns):
            self.x = transition(self.x, command + force, .01, self.actuator)
            self.time_ns += self.physics_step_ns
            times.append(self.time_ns); states.append(self.state())
        return {'state': self.state(), 'time_ns': self.time_ns, 'status': 'ok',
                'trajectory': {'time_ns': np.asarray(times), 'state': np.stack(states)}}


def settings():
    return {'candidate_count': 8, 'iterations': 1, 'elite_count': 1,
            'initial_std': 1., 'std_floor': .1}


def fake_search(log, failure_index=None):
    def search(model, state, target, history, previous, normals, cfg, partition, interval, **kwargs):
        index = len(log)
        log.append({'state': state.numpy().copy(), 'history': history, 'normals': normals.copy()})
        failed = index == failure_index
        return {'status': 'all_candidates_failed' if failed else 'ok',
                'fallback': 'zero_command' if failed else None,
                'cost': None if failed else 1., 'knots': np.zeros((6, 2)),
                'selected_global_id': None if failed else 0,
                'predicted_anchors': None if failed else np.tile(state.numpy(), (6, 1)),
                'candidates': {'knots': np.zeros((8, 6, 2)),
                               'costs': np.full(8, np.inf if failed else 1.),
                               'statuses': ('model_failure' if failed else 'ok',) * 8,
                               'reasons': ('fixture failure' if failed else '',) * 8,
                               'global_ids': np.arange(8)},
                'proposals': [], 'work': {'network_calls': 1, 'vector_field_calls': 0,
                                        'scalar_candidate_evaluations': 8, 'history_tokens_processed': 0}}
    return search


def run(monkeypatch, condition='clean', interval=40_000_000, failure=None):
    log = []
    monkeypatch.setattr(control_episode, 'search', fake_search(log, failure))
    robot = ExactPlant()
    result = control_episode.execute_episode(robot, None, settings(), seed=9000, episode_id=2,
                                              phase='exploratory', condition=condition,
                                              model_partition=(5_000_000,) * 8,
                                              actuator_interval_ns=interval)
    return robot, result, log


@pytest.mark.parametrize('interval', [40_000_000, 20_000_000, 10_000_000])
def test_forty_fresh_feedback_decisions_one_terminal_cost_and_full_true_trajectory(monkeypatch, interval):
    plant, result, log = run(monkeypatch, interval=interval)
    assert result['complete'] and plant.time_ns == 1_600_000_000
    assert result['observation_times_ns'].tolist() == list(range(0, 1_600_000_000, 40_000_000))
    assert len(log) == 40
    assert len(result['true_times_ns']) == 161
    assert len(result['action_segments']) == 1_600_000_000 // interval
    assert result['fallback_count'] == 0
    independent = anchor_cost(result['true_anchors'], result['target'],
                              (ActionSegment(0, 1_600_000_000, np.zeros(2)),))
    assert result['realized_cost'] == pytest.approx(independent)
    for i, query in enumerate(log):
        assert int(query['history'].query_time_ns[0]) == i * 40_000_000
        assert len(query['history'].tokens[0]) == min(i, 8)
        assert all(t.provenance == 'observed' and t.duration_ns == 40_000_000
                   for t in query['history'].tokens[0])
        assert result['decisions'][i]['plant_snapshot_at_decision']['time_ns'] == i * 40_000_000


def test_noisy_packets_are_drawn_once_per_feedback_and_true_state_stays_scoring_only(monkeypatch):
    _, result, log = run(monkeypatch, condition='noise')
    assert len(result['noise_packet_times_ns']) == 40
    assert not np.array_equal(result['observations'][:, :4], result['true_anchors_with_initial'][:-1, :4])
    for i, query in enumerate(log):
        np.testing.assert_array_equal(query['state'], result['observations'][i])
        if i:
            np.testing.assert_array_equal(query['history'].tokens[0][-1].state, result['observations'][i - 1])
    _, replay, replay_log = run(monkeypatch, condition='noise')
    np.testing.assert_array_equal(result['observations'], replay['observations'])
    np.testing.assert_array_equal(log[7]['normals'], replay_log[7]['normals'])


def test_disturbance_is_external_balanced_and_applied_only_on_exact_boundaries(monkeypatch):
    _, result, _ = run(monkeypatch, condition='disturbance', interval=10_000_000)
    nonzero = [row for row in result['external_torque_segments'] if np.any(row['torque'])]
    assert nonzero[0]['start_ns'] == 640_000_000
    assert nonzero[-1]['start_ns'] + nonzero[-1]['duration_ns'] == 720_000_000
    assert sum(row['duration_ns'] for row in nonzero) == 80_000_000
    signs = [control_episode.episode_conditions(9000, i, 'direct', 'exploratory')['disturbance_sign']
             for i in range(20)]
    assert signs.count(1) == signs.count(-1) == 10
    assert all(np.array_equal(s.command, np.zeros(2)) for s in result['action_segments'])


def test_all_candidate_failure_preserves_episode_and_uses_actual_fallback_cost(monkeypatch):
    _, result, _ = run(monkeypatch, failure=7)
    assert result['complete'] and result['fallback_count'] == 1
    assert np.isfinite(result['realized_cost'])
    decision = result['decisions'][7]
    assert decision['status'] == 'all_candidates_failed'
    assert decision['predicted_anchors'] is None
    assert decision['predicted_first_interval_cost'] is None
    assert result['candidates'][7]['statuses'] == ('model_failure',) * 8
    assert np.isinf(result['candidates'][7]['costs']).all()


def test_planning_cannot_secretly_advance_physical_time(monkeypatch):
    robot = ExactPlant()
    def bad_search(*args, **kwargs):
        robot.time_ns += 1
        return fake_search([])(*args, **kwargs)
    monkeypatch.setattr(control_episode, 'search', bad_search)
    with pytest.raises(ValueError, match='physical'):
        control_episode.execute_episode(robot, None, settings(), seed=9000, episode_id=0,
                                         model_partition=(40_000_000,))


def test_an_unscored_elite_mean_cannot_replace_the_selected_evaluated_plan(monkeypatch):
    def bad_search(*args, **kwargs):
        value = fake_search([])(*args, **kwargs)
        value['knots'][:] = 1
        return value
    monkeypatch.setattr(control_episode, 'search', bad_search)
    with pytest.raises(ValueError, match='evaluated'):
        control_episode.execute_episode(ExactPlant(), None, settings(), seed=9000, episode_id=0,
                                         model_partition=(40_000_000,))


def test_development_seeds_cannot_be_relabelled_as_confirmation_and_plant_pairing_is_fixed():
    for phase in ('confirmatory', 'replication'):
        for seed in (9000, 9001, 9002):
            with pytest.raises(ValueError):
                control_episode.episode_conditions(seed, 0, 'direct', phase)
    direct = control_episode.episode_conditions(0, 0, 'direct', 'confirmatory')
    lagged = control_episode.episode_conditions(0, 0, 'lagged', 'confirmatory')
    np.testing.assert_array_equal(direct['initial_state'], lagged['initial_state'][:4])
    np.testing.assert_array_equal(direct['target'], lagged['target'])
    np.testing.assert_array_equal(direct['noise'], lagged['noise'])


@pytest.mark.parametrize('actuator', ['direct', 'lagged'])
def test_zero_policy_fixture_uses_real_plant_adapter_and_observed_activation(monkeypatch, actuator):
    from pathlib import Path
    from poc.envs import Plant
    log = []
    monkeypatch.setattr(control_episode, 'search', fake_search(log))
    robot = Plant(actuator, physics_step_ns=250_000, asset_root=Path('assets'))
    result = control_episode.execute_episode(robot, None, settings(), seed=8100, episode_id=0,
                                              condition='noise', model_partition=(5_000_000,) * 8)
    assert result['complete']
    assert len(result['true_times_ns']) == 6401
    assert result['final_window_sample_count'] == 801
    np.testing.assert_allclose(result['true_states'],
                               np.tile(result['initial_state'], (6401, 1)), atol=1e-12)
    if actuator == 'lagged':
        np.testing.assert_array_equal(result['observations'][:, 4:], 0)
    assert all(np.array_equal(x['state'], result['observations'][i]) for i, x in enumerate(log))
