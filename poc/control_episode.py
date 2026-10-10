"""Physical feedback episodes with dense truth kept separate from model inputs."""
from __future__ import annotations

import time
import numpy as np
import torch

from .action_path import ActionSegment, compile_knots, mean_command, path_checksum, restrict_path
from .clocks import split_events
from .control import anchor_cost, anchor_components
from .history import HistoryBatch, HistoryToken
from .mpc import search


def episode_conditions(seed: int, episode_id: int, actuator: str, phase: str) -> dict:
    if (type(seed) is not int or seed < 0 or type(episode_id) is not int or episode_id < 0
            or actuator not in {'direct', 'lagged'}
            or phase not in {'exploratory', 'calibration', 'confirmatory', 'replication'}):
        raise ValueError('Explicit valid episode identity, actuator and phase required')
    if phase in {'confirmatory', 'replication'} and seed in {9000, 9001, 9002}:
        raise ValueError('Development training seeds cannot be relabelled as confirmation')
    root = 6101 if phase in {'confirmatory', 'replication'} else 7101
    def rng(channel, identity=episode_id):
        return np.random.Generator(np.random.PCG64(np.random.SeedSequence([root, seed, identity, channel])))
    initial = np.r_[rng(0).uniform(-.5, .5, 2), np.zeros(2 if actuator == 'direct' else 4)]
    block, index = divmod(episode_id, 20)
    signs = rng(4, block).permutation(np.r_[np.ones(10, dtype=int), -np.ones(10, dtype=int)])
    return {'initial_state': initial, 'target': rng(1).uniform(-1, 1, 2),
            'noise': rng(3).normal(size=(40, 4)) * np.array([.005, .005, .02, .02]),
            'disturbance_sign': int(signs[index]), 'root_seed': root,
            'cem_seed_entropy': [root, seed, episode_id, 2],
            'disturbance_seed_entropy': [root, seed, block, 4]}


def execute_episode(plant, model, settings: dict, *, seed: int, episode_id: int,
                    phase: str = 'exploratory', condition: str = 'clean',
                    model_partition: tuple[int, ...], actuator_interval_ns: int = 40_000_000,
                    power_mode: str | None = None) -> dict:
    if condition not in {'clean', 'noise', 'disturbance'}:
        raise ValueError('Use the separately specified clean, noise or disturbance condition')
    conditions = episode_conditions(seed, episode_id, plant.actuator, phase)
    normals = np.random.Generator(np.random.PCG64(np.random.SeedSequence(conditions['cem_seed_entropy']))).normal(
        size=(40, settings['iterations'], settings['candidate_count'], 6, 2))
    # Validate the actual hold rate before executing any policy.
    compile_knots(np.zeros((6, 2)), actuator_interval_ns)
    if actuator_interval_ns % plant.physics_step_ns:
        raise ValueError('Actuator holds must be exactly representable by the accepted physics tick')
    plant.reset(conditions['initial_state'])
    context = HistoryBatch(((),), np.array([0], dtype=np.int64))
    previous, prior_observation = None, None
    observations, observation_times, true_anchors = [], [], []
    true_times, true_states = [0], [plant.state()]
    action_segments, external, decisions, candidates, latency = [], [], [], [], []
    fallback_count = 0
    work = {'network_calls': 0, 'vector_field_calls': 0,
            'scalar_candidate_evaluations': 0, 'history_tokens_processed': 0}
    for index in range(40):
        now = index * 40_000_000
        if plant.time_ns != now:
            raise ValueError('Feedback decision changed the physical observation clock')
        observed = plant.state()
        if condition == 'noise':
            observed[:4] += conditions['noise'][index]
        context = context.at_time(np.array([now], dtype=np.int64))
        if index:
            token = HistoryToken(now - 40_000_000, prior_observation,
                                 mean_command(action_segments, now - 40_000_000, now),
                                 40_000_000, 'observed', f'obs-{now - 40_000_000}')
            context = context.append_completed(token).eligible()
        observations.append(observed.copy()); observation_times.append(now)
        packet_id = f'obs-{now}'
        snapshot = plant.snapshot()
        kwargs = {} if power_mode is None else {'power_mode': power_mode}
        wall = time.perf_counter_ns()
        planned = search(model, torch.tensor(observed, dtype=torch.float64),
                         torch.tensor(conditions['target'], dtype=torch.float64), context,
                         previous, normals[index], settings, model_partition, actuator_interval_ns, **kwargs)
        latency.append((time.perf_counter_ns() - wall) / 1e9)
        if plant.time_ns != now:
            raise ValueError('Planning must not advance physical time or inject wall-clock delay')
        if planned['status'] not in {'ok', 'all_candidates_failed'}:
            raise ValueError('Planner returned an incomplete or unknown decision status')
        if planned['status'] == 'ok':
            selected = planned['selected_global_id']
            bank = planned['candidates']['knots']
            if (selected is None or selected not in range(len(bank))
                    or not np.array_equal(planned['knots'], bank[selected])):
                raise ValueError('Only a selected evaluated candidate can be executed')
        elif planned['fallback'] != 'zero_command' or np.any(planned['knots']):
            raise ValueError('A failed search must execute the declared zero-command fallback')
        fallback_count += int(planned['status'] == 'all_candidates_failed')
        previous = planned['knots'].copy()
        full_path = compile_knots(previous, actuator_interval_ns)
        first = restrict_path(full_path, 0, 40_000_000)
        predicted_cost = None
        if planned['predicted_anchors'] is not None:
            predicted_cost = anchor_cost(planned['predicted_anchors'][:1], conditions['target'], first, terminal_weight=0)
        for segment in first:
            start, end = now + segment.start_ns, now + segment.start_ns + segment.duration_ns
            for left, duration in split_events(start, end, (640_000_000, 720_000_000)):
                torque = np.zeros(2)
                if condition == 'disturbance' and 640_000_000 <= left < 720_000_000:
                    torque[1] = conditions['disturbance_sign'] * .1
                response = plant.advance(segment.command, duration, external_torque=torque, log_every_step=True)
                if response['status'] != 'ok' or response['time_ns'] != left + duration:
                    raise ValueError('True plant failed or advanced by a different physical duration')
                trajectory = response['trajectory']
                times = np.asarray(trajectory['time_ns'])
                states = np.asarray(trajectory['state'])
                expected_times = np.arange(left + plant.physics_step_ns, left + duration + 1, plant.physics_step_ns)
                if (not np.array_equal(times, expected_times)
                        or states.shape != (len(times), len(observed)) or not np.isfinite(states).all()):
                    raise ValueError('Dense scoring trajectory is incomplete or invalid')
                true_times.extend(times.tolist()); true_states.extend(states.copy())
                action_segments.append(ActionSegment(left, duration, segment.command))
                external.append({'start_ns': left, 'duration_ns': duration, 'torque': torque.copy()})
        realized = plant.state()
        true_anchors.append(realized.copy())
        actual_first_cost = anchor_cost(realized[None], conditions['target'], first, terminal_weight=0)
        decisions.append({'replanning_index': index, 'time_ns': now, 'observation_packet_id': packet_id,
                          'history_packet_ids': [t.packet_id for t in context.tokens[0]],
                          'status': planned['status'], 'fallback': planned['fallback'],
                          'selected_global_id': planned['selected_global_id'],
                          'selected_knots': previous.copy(), 'planned_cost': planned['cost'],
                          'compiled_plan_checksum': path_checksum(full_path),
                          'plant_snapshot_at_decision': snapshot,
                          'predicted_anchors': planned['predicted_anchors'],
                          'predicted_first_interval_cost': predicted_cost,
                          'actual_first_interval_cost': actual_first_cost,
                          'actual_first_interval_state': realized.copy(),
                          'proposal_history': planned['proposals'], 'planner_wall_seconds': latency[-1]})
        candidates.append(planned['candidates'])
        for key in work:
            work[key] += planned['work'][key]
        prior_observation = observed.copy()
    times, states, anchors = np.asarray(true_times, dtype=np.int64), np.stack(true_states), np.stack(true_anchors)
    if plant.time_ns != 1_600_000_000 or times[-1] != 1_600_000_000:
        raise ValueError('A partial physical episode cannot yield a complete primary cost')
    target = conditions['target']
    error = states[:, :2] - target
    final = times >= 1_400_000_000
    success = bool(np.all(np.linalg.norm(error[final], axis=1) < .1)
                   and np.all(np.linalg.norm(states[final, 2:4], axis=1) < .2))
    parts = anchor_components(anchors, target, action_segments)
    return {'complete': True, 'phase': phase, 'condition': condition, 'seed': seed,
            'episode_id': episode_id, 'actuator': plant.actuator, 'target': target.copy(),
            'physics_step_ns': plant.physics_step_ns, 'actuator_interval_ns': actuator_interval_ns,
            'model_partition_ns': tuple(model_partition), 'power_mode': power_mode,
            'initial_state': conditions['initial_state'].copy(), 'realized_cost': parts['total'],
            'cost_components': parts, 'q_rmse_rad': float(np.sqrt(np.mean(error**2))),
            'terminal_q_error_rad': float(np.linalg.norm(error[-1])),
            'velocity_rms_rad_s': float(np.sqrt(np.mean(states[:, 2:4]**2))),
            'command_effort_integral': parts['effort'] / .001, 'success': success,
            'final_window_sample_count': int(final.sum()), 'fallback_count': fallback_count,
            'true_times_ns': times, 'true_states': states, 'true_anchors': anchors,
            'true_anchors_with_initial': np.vstack((conditions['initial_state'], anchors)),
            'observations': np.stack(observations), 'observation_times_ns': np.asarray(observation_times, dtype=np.int64),
            'noise_packet_times_ns': np.asarray(observation_times if condition == 'noise' else [], dtype=np.int64),
            'action_segments': tuple(action_segments), 'external_torque_segments': external,
            'decisions': decisions, 'candidates': candidates, 'planner_latency_seconds': np.asarray(latency),
            'seed_namespaces': {'cem': conditions['cem_seed_entropy'], 'disturbance': conditions['disturbance_seed_entropy']},
            'work': work}
