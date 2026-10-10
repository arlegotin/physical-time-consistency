"""Measure nominal reference-fit work without changing parameters or science inputs."""
import os
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[variable] = '1'

from pathlib import Path
import resource
import sys
import time
import numpy as np
import torch

from poc.dataset import load_corpus
from poc.execution import cpu_defaults
from poc.io import atomic_json, read_json, sha256_file, utc_now
from poc.models.common import Scales
from poc.physics_fit import PhysicsFit
from poc.physics_training import _data, _integration_step, endpoint_loss
from poc.reference_jobs import tensor_hash

cpu_defaults(8201)
root = Path.cwd()
source = root / read_json(root / 'config/report_source.json')['study_relative']
for actuator in ('direct', 'lagged'):
    folder = root / 'reports/reference_profile' / actuator
    folder.mkdir(parents=True, exist_ok=True)
    base = source / 'data' / actuator / 'A'
    norm = read_json(base / 'normalization.json')
    scales = Scales(*(torch.tensor(norm[key], dtype=torch.float32)
                     for key in ('scale_q', 'scale_v', 'velocity_input_rms')))
    datasets = {}
    for role, purpose in (('train', 'training'), ('validation', 'selection')):
        arrays = load_corpus(base / role / 'manifest.json', 'exploratory', role, purpose,
                             audit_path=folder / 'data_access.jsonl')
        dimension = arrays['state'].shape[-1]
        datasets[role] = {'state': arrays['state'][:, :-1].reshape(-1, dimension),
                          'target': arrays['state'][:, 1:].reshape(-1, dimension),
                          'command': arrays['command'].reshape(-1, 2),
                          'h_s': arrays['duration_ns'].reshape(-1, 1) / 1e9}
    model = PhysicsFit(torch.tensor([.10, .02, .02, .05, .05]), actuator, scales)
    original = tensor_hash(model.state_dict())
    step, audits = _integration_step(model, _data(model, datasets['validation']))
    batch = {key: value[:256] for key, value in datasets['train'].items()}
    wall_times, cpu_times = [], []
    for index in range(4):
        model.zero_grad(set_to_none=True)
        wall, cpu = time.perf_counter(), time.process_time()
        loss = endpoint_loss(model, batch, step)
        loss.backward()
        if index:
            wall_times.append(time.perf_counter() - wall)
            cpu_times.append(time.process_time() - cpu)
    assert tensor_hash(model.state_dict()) == original
    updates = 100 * int(np.ceil(len(datasets['train']['state']) / 256))
    record = {'schema_version': 1, 'phase': 'calibration', 'actuator': actuator,
              'recorded_utc': utc_now(), 'optimizer_updates_executed': 0,
              'training_gradients_measured': 4, 'warmup_gradients': 1,
              'integration_step_s': step, 'solver_audits': audits,
              'batch_wall_seconds': wall_times, 'batch_cpu_seconds': cpu_times,
              'projected_updates': updates,
              'projected_wall_seconds': float(np.median(wall_times) * updates),
              'projection_limitations': 'Representative gradient timing only; excludes validation, checkpoint I/O and any finer solver branch.',
              'projected_disk_bytes': 8 * 1024 * 1024,
              'disk_projection_basis': '12800 update rows plus twenty validation rows, five parameters, optimizer states and manifests; estimate pending complete run.',
              'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
              'source_script_sha256': sha256_file(Path(__file__)),
              'input_hashes': {str(path.relative_to(root)): sha256_file(path) for path in
                               (base / 'train/manifest.json', base / 'validation/manifest.json', base / 'normalization.json')},
              'parameter_hash_before_and_after': original}
    atomic_json(folder / 'measurement.json', record)
    print(actuator, record['projected_wall_seconds'], 'seconds excluding overhead', flush=True)
