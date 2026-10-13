"""Fresh calibration telemetry, segment resume, and train-only budget loading."""
from __future__ import annotations

import io
import inspect
from pathlib import Path

import numpy as np

from .calibration_resources import checkpoint_resources
from .calibration_store import CalibrationStore, Receipt
from .envs import Plant
from .io import (append_event, atomic_bytes, atomic_json, job_key, read_json,
                 sha256_file, utc_now)
from .models.common import Scales
from .train import array_scales

ARRAY_KEYS = ('state', 'command', 'duration_ns', 'time_ns', 'trajectory_id')
OBSERVATION_NS = 40_000_000
PHYSICS_STEP_NS = 250_000
ROLE_CODES = {'train-session': 0, 'development-evaluation': 1, 'confirmation': 2}


def rng_for(role: str, session_id: int, trajectory_id: int, channel: int) -> np.random.Generator:
    if role not in ROLE_CODES or any(type(value) is not int or value < 0
                                     for value in (session_id, trajectory_id, channel)):
        raise ValueError('Invalid calibration seed coordinate')
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(
        [20261012, 1, ROLE_CODES[role], session_id, trajectory_id, channel])))


def validate_session_arrays(arrays: dict, segment_count: int) -> None:
    if set(arrays) != set(ARRAY_KEYS):
        raise ValueError('Missing or unknown calibration arrays')
    shapes = {'state': (segment_count, 26, 6), 'command': (segment_count, 25, 2),
              'duration_ns': (segment_count, 25), 'time_ns': (segment_count, 26),
              'trajectory_id': (segment_count,)}
    for key, value in arrays.items():
        dtype = np.float64 if key in {'state', 'command'} else np.int64
        if value.shape != shapes[key] or value.dtype != dtype or not np.isfinite(value).all():
            raise ValueError(f'Invalid calibration array shape, dtype or values: {key}')
    if not np.array_equal(arrays['trajectory_id'], np.arange(segment_count, dtype=np.int64)):
        raise ValueError('Calibration trajectory identities must be unique ordered coordinates')
    if np.any(np.abs(arrays['command']) > 1.5):
        raise ValueError('Calibration command exceeds frozen excitation bounds')
    if not np.all(arrays['duration_ns'] == OBSERVATION_NS):
        raise ValueError('Calibration must have complete 40 ms holds')
    expected = np.arange(26, dtype=np.int64)*OBSERVATION_NS
    if not np.all(arrays['time_ns'] == expected):
        raise ValueError('Calibration observation times must follow the integer tick')


def collect_arrays(session_id: int, *, root: Path, segment_count: int = 600,
                   phase: str = 'exploratory', resume_dir: Path | None = None) -> dict:
    """Collect bounded motion; reduced counts require explicit nonempirical unit mode."""
    rng_for('train-session', session_id, 0, 0)
    if (type(segment_count) is not int or segment_count <= 0 or
            phase not in {'unit', 'exploratory'} or (phase != 'unit' and segment_count != 600)):
        raise ValueError('Only labeled unit fixtures may change the 600-segment session')
    root = Path(root).resolve()
    robot = Plant('lagged', PHYSICS_STEP_NS, root / 'assets')
    arrays = {'state': np.empty((segment_count, 26, 6), dtype=np.float64),
              'command': np.empty((segment_count, 25, 2), dtype=np.float64),
              'duration_ns': np.full((segment_count, 25), OBSERVATION_NS, dtype=np.int64),
              'time_ns': np.tile(np.arange(26, dtype=np.int64)*OBSERVATION_NS, (segment_count, 1)),
              'trajectory_id': np.arange(segment_count, dtype=np.int64)}
    from .calibration_study import _runtime_versions
    identity = {'plant_source_sha256': sha256_file(Path(inspect.getfile(Plant))),
                'numerical_runtime': _runtime_versions(), 'session_id': session_id, 'segment_count': segment_count, 'phase': phase,
                'root': str(root), 'physics_step_ns': PHYSICS_STEP_NS,
                'asset_sha256': robot.xml_sha256, 'source_sha256': sha256_file(Path(__file__)),
                'protocol_id': 'PTCO-CAL-0.1'}
    if resume_dir is not None:
        resume_dir = Path(resume_dir)
        resume_dir.mkdir(parents=True, exist_ok=True)
        request_path = resume_dir / 'collector.json'
        if request_path.exists() and read_json(request_path) != identity:
            raise ValueError('Changed collector inputs cannot resume completed segments')
        atomic_json(request_path, identity)
    for trajectory_id in range(segment_count):
        path = resume_dir / f'segment-{trajectory_id:04d}.npz' if resume_dir else None
        marker = path.with_suffix('.json') if path else None
        if marker and marker.exists():
            saved = read_json(marker)
            if (saved != {'trajectory_id': trajectory_id, 'sha256': sha256_file(path),
                          'collector_identity': job_key(identity)}):
                raise ValueError('Completed calibration segment hash/identity mismatch')
            with np.load(path, allow_pickle=False) as segment:
                if set(segment.files) != set(ARRAY_KEYS)-{'trajectory_id'}:
                    raise ValueError('Incomplete resumed calibration segment')
                for key in segment.files:
                    arrays[key][trajectory_id] = segment[key]
            continue
        tick = -1
        try:
            initial = rng_for('train-session', session_id, trajectory_id, 0)
            state = np.r_[initial.uniform(-.5, .5, 2), initial.uniform(-.25, .25, 2),
                          initial.uniform(-.3, .3, 2)]
            excitation = rng_for('train-session', session_id, trajectory_id, 1)
            robot.reset(state)
            arrays['state'][trajectory_id, 0] = state
            for tick in range(25):
                if tick % 5 == 0:
                    d = excitation.uniform(-.75, .75, 2)
                command = np.clip(-state[:2]-.25*state[2:4]+d, -1.5, 1.5)
                arrays['command'][trajectory_id, tick] = command
                state = robot.advance(command, OBSERVATION_NS)['state']
                arrays['state'][trajectory_id, tick+1] = state
            if path:
                def commit_segment():
                    buffer = io.BytesIO()
                    np.savez_compressed(buffer, **{key: arrays[key][trajectory_id]
                                                   for key in ARRAY_KEYS if key != 'trajectory_id'})
                    atomic_bytes(path, buffer.getvalue())
                    atomic_json(marker, {'trajectory_id': trajectory_id, 'sha256': sha256_file(path),
                                         'collector_identity': job_key(identity)})
                checkpoint_resources(commit_segment)
        except BaseException as error:
            if resume_dir:
                append_event(resume_dir / 'failures.jsonl', {
                    'event': 'segment_failed', 'session_id': session_id, 'trajectory_id': trajectory_id,
                    'tick': tick, 'phase': phase, 'error_type': type(error).__name__,
                    'error': str(error), 'utc': utc_now()})
            raise
    validate_session_arrays(arrays, segment_count)
    return arrays


def four_step_starts(arrays: dict) -> np.ndarray:
    """Local row/start pairs with four complete command intervals in one segment."""
    return np.asarray([(row, tick) for row in range(len(arrays['trajectory_id']))
                       for tick in range(arrays['command'].shape[1]-3)], dtype=np.int64).reshape(-1, 2)


def load_budget(session_receipt: Receipt, budget_s: int, store: CalibrationStore
                ) -> tuple[dict, dict, Scales, dict]:
    folder, manifest = store.read(session_receipt)
    summary = manifest['summary']
    if manifest['stage'] != 'collect' or summary.get('phase') not in {'unit', 'exploratory'}:
        raise ValueError('Expected a labeled calibration session receipt')
    count = summary['segment_count']
    if summary['phase'] != 'unit' and count != 600:
        raise ValueError('Empirical calibration session must contain 600 complete segments')
    if (type(budget_s) is not int or budget_s not in summary['budget_seconds'] or
            budget_s <= 0 or budget_s > count or budget_s % 5):
        raise ValueError('Budget must be declared complete five-segment groups')
    with np.load(folder / 'session.npz', allow_pickle=False) as file:
        arrays = {key: file[key].copy() for key in file.files}
    validate_session_arrays(arrays, count)
    ids = arrays['trajectory_id'][:budget_s]
    train_rows, validation_rows = np.flatnonzero(ids % 5 != 4), np.flatnonzero(ids % 5 == 4)
    train = {key: values[train_rows].copy() for key, values in arrays.items()}
    validation = {key: values[validation_rows].copy() for key, values in arrays.items()}
    scales = array_scales(train)
    account = {'session_receipt': session_receipt, 'session_id': summary['session_id'],
               'phase': summary['phase'], 'budget_s': budget_s, 'total_seconds': budget_s,
               'train_seconds': len(train_rows), 'validation_seconds': len(validation_rows),
               'train_trajectory_ids': train['trajectory_id'].tolist(),
               'validation_trajectory_ids': validation['trajectory_id'].tolist(),
               'train_transitions': len(train_rows)*25, 'validation_transitions': len(validation_rows)*25,
               'observation_tick_ns': OBSERVATION_NS, 'reset_overhead_counted': False,
               'scales_source': 'budget training subset only', 'dense_labels_available': False,
               'units': {'q': 'rad', 'v': 'rad/s', 'w': 'N*m', 'command': 'N*m'}}
    return train, validation, scales, account
