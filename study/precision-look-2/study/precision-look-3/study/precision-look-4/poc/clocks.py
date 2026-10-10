"""Authoritative integer physical clocks and event-safe partitions."""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import numpy as np


def to_ns(seconds: float) -> int:
    if isinstance(seconds, bool):
        raise ValueError('A boolean is not a duration')
    try:
        value = Decimal(str(seconds))*1_000_000_000
        if not value.is_finite() or value < 0 or value != value.to_integral_value():
            raise ValueError('Duration must be finite nonnegative seconds representable exactly in nanoseconds')
        return int(value)
    except InvalidOperation as exc:
        raise ValueError('Invalid seconds') from exc


def _integer(value, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < (1 if positive else 0):
        raise ValueError('Integer nonnegative clock ticks required')
    return int(value)


def partition(duration_ns: int, step_ns: int) -> tuple[int, ...]:
    duration, step = _integer(duration_ns), _integer(step_ns, positive=True)
    count, remainder = divmod(duration, step)
    return (step,)*count + ((remainder,) if remainder else ())


def jitter_partition(seed_entropy: tuple[int, ...]) -> tuple[int, ...]:
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed_entropy)))
    return tuple(int(value) for value in rng.permutation([2_500_000]*4 + [7_500_000]*4))


def split_events(start_ns: int, end_ns: int, boundaries: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    start, end = _integer(start_ns), _integer(end_ns)
    if end < start:
        raise ValueError('Physical time cannot run backwards')
    points = [start] + sorted(set(_integer(value) for value in boundaries if start < value < end)) + [end]
    return tuple((left, right-left) for left, right in zip(points, points[1:]) if right > left)

