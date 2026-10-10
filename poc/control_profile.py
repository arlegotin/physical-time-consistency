"""Prediction latency on fixed inputs; no physical clock or history is advanced."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
import torch


def prediction_latency(model, state, command, h, history=None, *, warmup=50, repeats=500,
                       timer=time.perf_counter_ns):
    if (type(warmup) is not int or warmup < 0 or type(repeats) is not int or repeats <= 0):
        raise ValueError('Explicit nonnegative warm-up and positive measured call counts required')
    elapsed, work = [], []
    with torch.inference_mode():
        for index in range(warmup + repeats):
            started = timer()
            packet = model.predict(state, command, h, history=history)
            ended = timer()
            if (any(status != 'ok' for status in packet.status)
                    or not bool(torch.isfinite(packet.next_state).all())):
                raise ValueError('A failed prediction cannot be omitted from its latency profile')
            if index >= warmup:
                elapsed.append((ended - started) / 1e9)
                work.append((packet.network_calls, packet.vector_field_calls, packet.history_tokens_processed))
    if min(elapsed) < 0 or len(set(work)) != 1:
        raise ValueError('Nonmonotonic timer or changing fixed-input work')
    return {'batch_size': len(state), 'warmup_calls': warmup, 'samples': repeats,
        'elapsed_seconds': elapsed, 'median_seconds': float(np.median(elapsed)),
        'p95_seconds': float(np.quantile(elapsed, .95, method='linear')),
        'network_calls_per_prediction': work[0][0], 'vector_field_calls_per_prediction': work[0][1],
        'history_tokens_processed_per_prediction': work[0][2],
        'scope': 'Fixed-input CPU prediction computation; no new physical observations or commands'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request', type=Path, help='Exact selected-model control request; no episode is executed')
    parser.add_argument('--root', type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    from .io import read_json, safe_path, sha256_file, atomic_json
    from .control_jobs import validate_request, SOURCES
    from .control_models import load_selected
    from .execution import cpu_defaults, run_job
    from .dataset import load_corpus
    from .history import HistoryBatch, HistoryToken
    from .reference_jobs import tensor_hash
    root = args.root.resolve()
    request_path = args.request.resolve()
    request = validate_request(read_json(request_path))
    if not request['model_manifest']:
        raise ValueError('Latency profiling requires a selected learned or calibrated checkpoint')
    cpu_defaults(request['seed'])
    manifest = safe_path(root, request['model_manifest'])
    source = manifest.parents[3]
    metadata = read_json(manifest)
    if metadata['milestone_id'] == 'M30':
        data_manifest = source / metadata['job_inputs']['parameters']['train_manifest']
    else:
        training_request = read_json(manifest.parent / 'request.json')
        data_manifest = root / training_request['train_manifest']
    relative_request = str(request_path.relative_to(root))
    files = list(SOURCES) + ['poc/control_profile.py', 'poc/dataset.py', relative_request,
        request['model_manifest'], str((manifest.parent / 'weights.pt').relative_to(root)),
        str((manifest.parent / 'normalization.json').relative_to(root)),
        str(data_manifest.relative_to(root)), str((data_manifest.parent / 'trajectories.npz').relative_to(root))]
    hashes = {name: sha256_file(root / name) for name in files}
    expected = ('control.prediction_latency',)

    def producer(folder):
        model = load_selected(manifest, request['manifest_sha256'], request['checkpoint_sha256'],
            model_id=request['model_id'], actuator=request['actuator'], seed=request['seed'])
        before = tensor_hash(model.state_dict())
        corpus = load_corpus(data_manifest, phase='exploratory', role='train', purpose='evaluation',
                             audit_path=folder / 'data_access.jsonl')
        anchor = int(np.searchsorted(corpus['time_ns'][0], 320_000_000))
        query_time = int(corpus['time_ns'][0, anchor])
        tokens = tuple(HistoryToken(int(corpus['time_ns'][0, i]),
            torch.tensor(corpus['state'][0, i], dtype=torch.float64), corpus['command'][0, i],
            int(corpus['duration_ns'][0, i]), 'observed', f'train-trajectory0-{i}') for i in range(anchor))
        state = torch.tensor(corpus['state'][0, anchor], dtype=torch.float64)
        command = torch.tensor(corpus['command'][0, anchor], dtype=torch.float64)
        profiles = []
        for batch_size, repeats in ((1, 500), (64, 100)):
            history = HistoryBatch(tuple(tokens for _ in range(batch_size)),
                                   np.full(batch_size, query_time, dtype=np.int64)).eligible()
            for h in (.04, .005):
                profiles.append({'h_s': h, **prediction_latency(model,
                    state.unsqueeze(0).repeat(batch_size, 1), command.unsqueeze(0).repeat(batch_size, 1),
                    h, history, repeats=repeats)})
        overhead = []
        for _ in range(500):
            start = time.perf_counter_ns(); overhead.append(time.perf_counter_ns() - start)
        if tensor_hash(model.state_dict()) != before or any(sha256_file(root / name) != digest for name, digest in hashes.items()):
            raise ValueError('Latency profile mutated selected parameters or source inputs')
        atomic_json(folder / 'latency.json', {'profiles': profiles, 'timer_overhead_median_ns': float(np.median(overhead)),
            'input': {'trajectory_id': int(corpus['trajectory_id'][0]), 'anchor_index': anchor,
                      'query_time_ns': query_time, 'source': str(data_manifest.relative_to(root))}})
        return [{'check_id': expected[0], 'status': 'pass', 'scope': 'CPU prediction timing',
                 'reason': 'Fixed inputs and weights preserved; every measured call retained', 'tolerance': None}], {
            'model_id': request['model_id'], 'actuator': request['actuator'], 'profiles': profiles,
            'parameter_count': sum(p.numel() for p in model.parameters()), 'missing_cells': [],
            'scope': 'Computation measurement only; not a learned control result'}
    record = run_job(root, 'M40', 'calibration', files, expected, [], producer,
                     config_keys={relative_request: request})
    print(root / 'evidence/M40' / record['run_id'] / 'manifest.json', flush=True)
    return 0 if record['completion_status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
