"""Small, content-addressed producer envelope shared by explicit milestone functions."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import uuid

from . import PROTOCOL_VERSION
from .contracts import verify_run
from .gates import decide, load_inputs, record_inputs
from .io import append_event, atomic_json, canonical_json, complete_run, job_key, job_lock, read_json, sha256_file, utc_now


def cpu_defaults(seed: int):
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = '1'
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed+1)
    torch.manual_seed(seed+2)
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def parent(root: Path, gate: str) -> dict:
    decision = decide(gate, load_inputs(root, gate), root)
    if decision['outcome'] != 'GO':
        raise RuntimeError(f'{gate} must GO before this producer: {decision["outcome"]}')
    return decision


def run_job(root: Path, milestone: str, phase: str, files: list[str], expected: tuple[str, ...],
            parents: list[dict], producer, config_keys: dict | None = None) -> dict:
    root = root.resolve()
    imported = {str(Path(module.__file__).resolve().relative_to(root))
                for module in list(sys.modules.values()) if getattr(module, '__file__', None)
                and str(module.__file__).endswith('.py') and Path(module.__file__).resolve().is_relative_to(root / 'poc')}
    undeclared = imported-set(files)
    if undeclared:
        raise ValueError(f'Undeclared local producer imports: {sorted(undeclared)}')
    environment = read_json(root / 'locks/environment.json')
    for name, expected_version in environment['versions'].items():
        if importlib.metadata.version(name) != expected_version:
            raise ValueError(f'Installed environment differs: {name}')
    fingerprint = {key: environment[key] for key in ('python', 'architecture', 'os', 'kernel', 'core_count',
                   'versions', 'torch_configuration', 'torch_threads', 'torch_interop_threads', 'deterministic_algorithms')}
    dependencies = {name: sha256_file(root / name) for name in files}
    for decision in parents:
        path = f'decisions/{decision["gate_id"]}/{decision["decision_id"]}.json'
        dependencies[path] = sha256_file(root / path)
    inputs = {'protocol_version': PROTOCOL_VERSION, 'milestone_id': milestone, 'phase': phase,
              'dependencies': dependencies, 'configuration_keys': config_keys or {'config/base.json': read_json(root / 'config/base.json')},
              'environment_fingerprint': fingerprint,
              'wheel_hashes': {wheel['filename']: wheel['sha256'] for wheel in read_json(root / 'locks/wheels.json')['wheels']}}
    key = job_key(inputs)
    with job_lock(key, root):
        for path in sorted((root / 'evidence' / milestone).glob('*/manifest.json')):
            record = read_json(path)
            if record['job_key'] == key and record['completion_status'] == 'complete':
                verify_run(root, record)
                record_inputs(root, 'G'+milestone[1:], [record])
                return record
        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8]
        folder = root / 'evidence' / milestone / run_id
        folder.mkdir(parents=True)
        for name in ('stdout.log', 'stderr.log', 'events.jsonl'):
            (folder / name).touch()
        start, wall, cpu = utc_now(), time.perf_counter_ns(), time.process_time()
        children_before = resource.getrusage(resource.RUSAGE_CHILDREN)
        append_event(folder / 'events.jsonl', {'event': 'producer_start', 'job_key': key, 'utc': start})
        try:
            checks, summary = producer(folder)
        except (ValueError, RuntimeError, OSError, AssertionError, TypeError) as exc:
            checks = [{'check_id': name, 'status': 'invalid', 'reason': str(exc), 'tolerance': None, 'scope': milestone} for name in expected]
            summary = {'measurements': {}, 'missing_cells': list(expected), 'failure': str(exc)}
            with (folder / 'stderr.log').open('a') as output:
                output.write(str(exc)+'\n')
        valid = {check['check_id'] for check in checks} == set(expected) and all(check['status'] == 'pass' for check in checks)
        atomic_json(folder / 'summary.json', summary)
        atomic_json(folder / 'validity.json', {'validity_status': 'valid' if valid else 'invalid', 'checks': checks, 'blocker': None})
        children = resource.getrusage(resource.RUSAGE_CHILDREN)
        peak = max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, children.ru_maxrss)
        atomic_json(folder / 'resources.json', {'wall_seconds': (time.perf_counter_ns()-wall)/1e9,
                    'cpu_seconds': time.process_time()-cpu+children.ru_utime+children.ru_stime-children_before.ru_utime-children_before.ru_stime,
                    'peak_rss_bytes': int(peak*(1 if platform.system() == 'Darwin' else 1024)),
                    'rss_convention': 'macOS bytes' if platform.system() == 'Darwin' else 'Linux KiB normalized to bytes',
                    'disk_bytes': sum(path.stat().st_size for path in folder.rglob('*') if path.is_file()),
                    'original_estimate': None, 'training_updates': 0})
        revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True)
        dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=root, capture_output=True, text=True)
        append_event(folder / 'events.jsonl', {'event': 'producer_end', 'valid': valid, 'utc': utc_now()})
        record = {'schema_version': 1, 'protocol_version': PROTOCOL_VERSION, 'milestone_id': milestone, 'run_id': run_id,
                  'job_key': key, 'job_inputs': inputs, 'phase': phase, 'argv': ['python', '-m', 'poc', 'run', milestone],
                  'code_revision': revision.stdout.strip(), 'dirty': bool(dirty.stdout) or dirty.returncode != 0,
                  'code_tree_sha256': hashlib.sha256(canonical_json(dependencies)).hexdigest(),
                  'configuration_sha256': hashlib.sha256(canonical_json(inputs['configuration_keys'])).hexdigest(),
                  'environment_sha256': hashlib.sha256(canonical_json(fingerprint)).hexdigest(),
                  'data_hashes': {}, 'model_hashes': {}, 'parent_decisions': parents,
                  'started_utc': start, 'ended_utc': utc_now(), 'completion_status': 'complete' if valid else 'failed',
                  'expected_cells': list(expected), 'produced_files': [{'path': str(path.relative_to(root)), 'sha256': sha256_file(path)}
                                     for path in sorted(folder.rglob('*')) if path.is_file()]}
        if valid and phase in {'unit','calibration'}:
            retired = []
            for path in sorted((root / 'evidence' / milestone).glob('*/manifest.json')):
                previous = read_json(path)
                verify_run(root,previous,check_dependencies=False)
                retired.append(previous['run_id'])
            if retired:
                record['supersedes'] = retired
                record['reason'] = 'Deterministic qualification repeated under current dependencies; original attempts remain archived'
        complete_run(root, record)
        record_inputs(root, 'G'+milestone[1:], [record])
        return record
