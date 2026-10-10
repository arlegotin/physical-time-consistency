"""One integer-clock sample-and-hold path for prediction, execution and cost."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import numpy as np

from .clocks import partition, split_events
from .io import canonical_json


def _tick(value, *, positive=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or value < int(positive)):
        raise ValueError('Physical clocks require nonnegative integer nanoseconds')
    return int(value)


@dataclass(frozen=True)
class ActionSegment:
    start_ns: int
    duration_ns: int
    command: np.ndarray

    def __post_init__(self):
        start, duration = _tick(self.start_ns), _tick(self.duration_ns, positive=True)
        command = np.array(self.command, dtype=np.float64, copy=True)
        if command.shape != (2,) or not np.isfinite(command).all() or np.any(np.abs(command) > 2):
            raise ValueError('A hold requires two finite commands in [-2,2] N*m')
        command.setflags(write=False)
        object.__setattr__(self, 'start_ns', start)
        object.__setattr__(self, 'duration_ns', duration)
        object.__setattr__(self, 'command', command)


def validate_path(segments) -> tuple[ActionSegment, ...]:
    path = tuple(segments)
    if not path or any(not isinstance(s, ActionSegment) for s in path):
        raise ValueError('A nonempty sequence of physical holds is required')
    for left, right in zip(path, path[1:]):
        if left.start_ns + left.duration_ns != right.start_ns:
            raise ValueError('Physical holds must be contiguous and ordered without overlaps')
    return path


def compile_knots(knots: np.ndarray, actuator_interval_ns: int,
                  horizon_ns: int = 240_000_000) -> tuple[ActionSegment, ...]:
    values = np.asarray(knots, dtype=np.float64)
    if values.shape != (6, 2) or not np.isfinite(values).all() or np.any(np.abs(values) > 2):
        raise ValueError('Six finite bounded two-joint torque knots are required')
    interval, horizon = _tick(actuator_interval_ns, positive=True), _tick(horizon_ns, positive=True)
    if interval not in (40_000_000, 20_000_000, 10_000_000) or horizon % interval:
        raise ValueError('Use a declared actuator interval and an exactly divisible horizon')
    path = []
    for start in range(0, horizon, interval):
        index, offset = divmod(start, 40_000_000)
        if index >= 5:
            command = values[-1]
        else:
            fraction = offset / 40_000_000
            command = (1 - fraction) * values[index] + fraction * values[index + 1]
        path.append(ActionSegment(start, interval, command))
    return tuple(path)


def path_checksum(segments) -> str:
    path = validate_path(segments)
    value = {'clock_unit': 'ns', 'command_unit': 'N*m', 'segments': [
        [s.start_ns, s.duration_ns, s.command.tolist()] for s in path]}
    return hashlib.sha256(canonical_json(value)).hexdigest()


def restrict_path(segments, start_ns: int, end_ns: int) -> tuple[ActionSegment, ...]:
    path = validate_path(segments)
    start, end = _tick(start_ns), _tick(end_ns)
    if end <= start or start < path[0].start_ns or end > path[-1].start_ns + path[-1].duration_ns:
        raise ValueError('Requested positive physical interval must be fully covered')
    result = []
    for segment in path:
        left, right = max(start, segment.start_ns), min(end, segment.start_ns + segment.duration_ns)
        if right > left:
            result.append(ActionSegment(left, right - left, segment.command))
    return tuple(result)


def effort_integral(segments, start_ns: int, end_ns: int) -> float:
    return math.fsum(s.duration_ns / 1e9 * float(s.command @ s.command)
                     for s in restrict_path(segments, start_ns, end_ns))


def mean_command(segments, start_ns: int, end_ns: int) -> np.ndarray:
    path = restrict_path(segments, start_ns, end_ns)
    duration = end_ns - start_ns
    return np.array([math.fsum(s.duration_ns / duration * s.command[j] for s in path)
                     for j in range(2)])


def subdivide_path(segments, step_ns: int, boundaries=()) -> tuple[ActionSegment, ...]:
    path, step = validate_path(segments), _tick(step_ns, positive=True)
    events = tuple(_tick(value) for value in boundaries)
    result = []
    for segment in path:
        for start, duration in split_events(segment.start_ns, segment.start_ns + segment.duration_ns, events):
            cursor = start
            for substep in partition(duration, step):
                result.append(ActionSegment(cursor, substep, segment.command))
                cursor += substep
    return tuple(result)
