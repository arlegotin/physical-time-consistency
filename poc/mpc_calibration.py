"""Measured analytic planner checks, independent of learned-model outcomes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import platform
import resource
import sys
import time
import uuid

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np
import torch

from .io import atomic_json, sha256_file, utc_now
from .models.common import Prediction
from .mpc import analytic_box_optimum, search


class ExactDoubleIntegrator:
    def predict(self, state, command, h_s, history=None):
        q, v = state[:, :2], state[:, 2:4]
        next_state = torch.cat((q + h_s * v + .5 * h_s**2 * command, v + h_s * command), 1)
        return Prediction(next_state, ('ok',) * len(state), 0, 0, len(state), 0)


def calibration_conditions(episode_id: int, stream_id: int, candidates: int, iterations: int) -> dict:
    if episode_id not in range(16) or stream_id not in range(5):
        raise ValueError('Analytic calibration uses the fixed sixteen tasks and five search streams')
    def rng(channel, stream=None):
        entropy = [7101, 9000, episode_id, channel]
        if stream is not None:
            entropy.append(stream)
        return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))
    return {'state': np.r_[rng(0).uniform(-.5, .5, 2), np.zeros(2)],
            'target': rng(1).uniform(-1, 1, 2),
            'standard_normals': rng(2, stream_id).normal(size=(iterations, candidates, 6, 2)),
            'seed_namespace': [7101, 9000, episode_id], 'cem_seed_entropy': [7101, 9000, episode_id, 2, stream_id]}


def qualification_summary(rows: list[dict]) -> dict:
    ids = [(row['stream_id'], row['episode_id']) for row in rows]
    expected = {(stream, episode) for stream in range(5) for episode in range(16)}
    if len(ids) != len(expected) or set(ids) != expected:
        raise ValueError('Qualification requires every task/stream exactly once')
    unresolved = [row for row in rows if row['status'] != 'ok']
    for row in rows:
        if row['status'] == 'ok':
            achieved = row['achieved_headroom_fraction']
            headroom = row['zero_cost'] - row['optimal_cost']
            if (headroom <= 0 or not math.isfinite(achieved)
                    or not math.isclose(achieved, (row['zero_cost'] - row['cem_cost']) / headroom,
                                        rel_tol=1e-12, abs_tol=1e-12)):
                raise ValueError('Headroom fraction disagrees with observed costs')
    if unresolved:
        return {'qualified': False, 'reason': 'Fixed calibration includes an uninformative or failed task',
                'unresolved_task_stream_ids': [(r['episode_id'], r['stream_id']) for r in unresolved],
                'stream_medians': None, 'stream_means': None,
                'minimum_achieved_fraction': None, 'spread_of_stream_means': None}
    by_stream = [[row['achieved_headroom_fraction'] for row in rows if row['stream_id'] == stream]
                 for stream in range(5)]
    medians = [float(np.median(values)) for values in by_stream]
    means = [float(np.mean(values)) for values in by_stream]
    minimum, spread = min(map(min, by_stream)), max(means) - min(means)
    qualified = min(medians) >= .75 and minimum >= .50 and spread < .05
    return {'qualified': qualified, 'reason': 'All search calibration requirements met' if qualified else
            'Common search budget does not yet meet the model-independent qualification',
            'stream_medians': medians, 'stream_means': means,
            'minimum_achieved_fraction': minimum, 'spread_of_stream_means': spread,
            'criteria': {'median_fraction_minimum': .75, 'each_fraction_minimum': .50,
                         'stream_mean_spread_strict_maximum': .05}}


def measure(settings: dict) -> dict:
    rows = []
    model = ExactDoubleIntegrator()
    for episode in range(16):
        conditions = calibration_conditions(episode, 0, settings['candidate_count'], settings['iterations'])
        optimum = analytic_box_optimum(conditions['state'], conditions['target'])
        headroom = optimum['zero_cost'] - optimum['cost']
        for stream in range(5):
            conditions = calibration_conditions(episode, stream, settings['candidate_count'], settings['iterations'])
            result = search(model, torch.tensor(conditions['state'], dtype=torch.float64),
                            torch.tensor(conditions['target'], dtype=torch.float64), None, None,
                            conditions['standard_normals'], settings, (40_000_000,), 40_000_000)
            resolved = result['status'] == 'ok' and headroom > 1e-14
            row = {'episode_id': episode, 'stream_id': stream,
                   'state': conditions['state'].tolist(), 'target': conditions['target'].tolist(),
                   'status': 'ok' if resolved else 'no_headroom' if headroom <= 1e-14 else result['status'],
                   'achieved_headroom_fraction': (optimum['zero_cost'] - result['cost']) / headroom if resolved else None,
                   'zero_cost': optimum['zero_cost'], 'optimal_cost': optimum['cost'], 'cem_cost': result['cost'],
                   'projected_optimality': optimum['projected_optimality'],
                   'optimal_knots': optimum['knots'].tolist(), 'selected_knots': result['knots'].tolist(),
                   'selected_global_id': result['selected_global_id'],
                   'candidate_knots': result['candidates']['knots'].tolist(),
                   'candidate_costs': result['candidates']['costs'].tolist(),
                   'candidate_statuses': result['candidates']['statuses'],
                   'candidate_global_ids': result['candidates']['global_ids'].tolist(),
                   'cem_seed_entropy': conditions['cem_seed_entropy'], 'work': result['work']}
            rows.append(row)
        print(f'Analytic CEM calibration: task {episode + 1}/16', file=sys.stderr, flush=True)
    return {'rows': rows, 'qualification': qualification_summary(rows)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--candidates', type=int, default=64)
    parser.add_argument('--iterations', type=int, default=2)
    args = parser.parse_args(argv)
    if args.candidates < 8 or args.candidates % 8 or args.iterations <= 0:
        parser.error('Use a positive common budget with one eighth of candidates as elites')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    settings = {'candidate_count': args.candidates, 'iterations': args.iterations,
                'elite_count': args.candidates // 8, 'initial_std': 1., 'std_floor': .10}
    root = args.root.resolve()
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:8]
    destination = root / 'reports/analytic_planner_calibration' / (run_id + '.json')
    sources = ('poc/mpc_calibration.py', 'poc/mpc.py', 'poc/action_path.py', 'poc/control.py',
               'poc/analytic.py', 'poc/clocks.py', 'poc/history.py', 'poc/models/common.py', 'poc/io.py')
    source_hashes = {p: sha256_file(root / p) for p in sources}
    started, wall, cpu = utc_now(), time.perf_counter(), time.process_time()
    result = measure(settings)
    if source_hashes != {p: sha256_file(root / p) for p in sources}:
        raise RuntimeError('Planner calibration source changed during measurement')
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record = {'schema_version': 1, 'artifact_kind': 'analytic_planner_calibration',
              'historical_protocol_version': 'PTCO-1.0', 'run_id': run_id,
              'argv': ['python', '-m', 'poc.mpc_calibration', '--root', str(root),
                       '--candidates', str(args.candidates), '--iterations', str(args.iterations)],
              'scope': 'Exact unit-mass double-integrator planner plumbing; no learned robot control result',
              'started_utc': started, 'completed_utc': utc_now(), 'completion_status': 'complete',
              'settings': settings, 'source_hashes': source_hashes,
              'environment_sha256': sha256_file(root / 'locks/environment.json'),
              'resources': {'wall_seconds': time.perf_counter() - wall, 'cpu_seconds': time.process_time() - cpu,
                            'peak_rss_bytes': peak if platform.system() == 'Darwin' else peak * 1024},
              **result}
    atomic_json(destination, record)
    print(str(destination.relative_to(root)), flush=True)
    print(record['qualification'], flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
