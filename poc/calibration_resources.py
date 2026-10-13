"""Recoverable active-attempt spans; nested ledgers are diagnostics, never additive."""
from __future__ import annotations

from contextvars import ContextVar
from pathlib import Path
import time

from .io import atomic_json, read_json, utc_now

_ACTIVE = ContextVar('calibration_resource_trackers', default=())


class ResourceTracker:
    """Persist lower bounds at boundaries and close caught attempts explicitly.

    Producers may contain rates or nested store jobs. Aggregate producer attempts
    using their parent provenance; inclusive diagnostic spans overlap.
    Gaps between invocations and the ledger's own observation writes are excluded.
    A running record found on restart has an unknown tail, even at a zero bound.
    """
    def __init__(self, path: Path, *, clock=None, cpu_clock=None, producer_key=None):
        self.path = Path(path)
        self.producer_key = producer_key
        parent = next((t for t in reversed(_ACTIVE.get()) if t.producer_key is not None), None)
        self.clock = clock or time.perf_counter
        self.cpu_clock = cpu_clock or time.process_time
        self.attempts = read_json(self.path)['attempts'] if self.path.exists() else []
        for attempt in self.attempts:
            if attempt['status'] == 'running':
                attempt.update(status='abandoned', unknown_tail=True)
        self.current = {'attempt_id':len(self.attempts)+1, 'status':'running',
                        'parent_producer_attempt':None if parent is None else
                            {'job_key':parent.producer_key,'attempt_id':parent.current['attempt_id']},
                        'started_utc':utc_now(), 'last_boundary_utc':utc_now(),
                        'active_wall_s':0., 'active_cpu_s':0.,
                        'persistence_wall_s':0., 'persistence_cpu_s':0.,
                        'unknown_tail':True, 'boundary_count':0}
        self.attempts.append(self.current)

    def __enter__(self):
        checkpoint_resources()
        self.token = _ACTIVE.set((*_ACTIVE.get(), self))
        try:
            self._persist()
        except BaseException:
            _ACTIVE.reset(self.token)
            raise
        finally:
            for tracker in _ACTIVE.get():
                tracker._restart()
        return self

    def _restart(self):
        self.wall, self.cpu = self.clock(), self.cpu_clock()

    def _sample(self):
        wall, cpu = self.clock(), self.cpu_clock()
        self.current['active_wall_s'] += wall-self.wall
        self.current['active_cpu_s'] += cpu-self.cpu
        self.wall, self.cpu = wall, cpu
        self.current['last_boundary_utc'] = utc_now()

    def _persist(self):
        atomic_json(self.path, {'schema':'calibration-resources-v1','attempts':self.attempts})

    def __exit__(self, kind, error, traceback):
        trackers = _ACTIVE.get()
        for tracker in trackers:
            tracker._sample()
        self.current.update(status='completed' if kind is None else 'interrupted', unknown_tail=False)
        if kind is not None:
            self.current['error_type'] = kind.__name__
        try:
            for tracker in trackers:
                tracker._persist()
        finally:
            _ACTIVE.reset(self.token)
            for tracker in _ACTIVE.get():
                tracker._restart()
        return False

    def summary(self):
        totals = {k:sum(a[k] for a in self.attempts) for k in
                  ('active_wall_s','active_cpu_s','persistence_wall_s','persistence_cpu_s')}
        unknown = any(a['unknown_tail'] for a in self.attempts)
        return {**totals, 'producer_key':self.producer_key, 'wall_s':totals['active_wall_s']+totals['persistence_wall_s'],
                'cpu_s':totals['active_cpu_s']+totals['persistence_cpu_s'],
                'accounting_status':'lower_bound_unknown_tail' if unknown else 'measured',
                'attempts':self.attempts, 'inactive_gaps_included':False,
                'resource_record_write_measured':False,
                'scope':'producer active work plus measured checkpoint persistence; excludes inactive gaps, resource-ledger writes and post-producer finalization'}


def checkpoint_resources(commit=None):
    """Commit a numerical boundary without changing its payload or RNG state.

    Persist a pre-commit bound before the write, then its measured persistence
    span. An uncatchable stop anywhere leaves a running attempt with unknown tail.
    Without an active tracker this simply executes the supplied commit.
    """
    trackers = _ACTIVE.get()
    for tracker in trackers:
        tracker._sample()
        tracker.current['boundary_count'] += 1
    try:
        for tracker in trackers:
            tracker._persist()
        starts = [(tracker.clock(), tracker.cpu_clock()) for tracker in trackers]
        try:
            if commit is not None:
                return commit()
        finally:
            for tracker, (wall, cpu) in zip(trackers, starts):
                tracker.current['persistence_wall_s'] += tracker.clock()-wall
                tracker.current['persistence_cpu_s'] += tracker.cpu_clock()-cpu
            for tracker in trackers:
                tracker._persist()
    finally:
        for tracker in trackers:
            tracker._restart()
