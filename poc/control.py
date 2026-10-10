"""Common physical endpoint cost; episode execution is implemented separately."""
from __future__ import annotations

import math
import numpy as np

from .action_path import effort_integral, validate_path


def anchor_components(states: np.ndarray, target: np.ndarray, segments,
                      terminal_weight: float = 5.0) -> dict[str, float]:
    states, target = np.asarray(states, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if (states.ndim != 2 or states.shape[0] == 0 or states.shape[1] not in (4, 6)
            or target.shape != (2,) or not np.isfinite(states).all() or not np.isfinite(target).all()):
        raise ValueError('Finite q/v[/w] endpoint anchors and a two-joint target required')
    if not math.isfinite(terminal_weight) or terminal_weight < 0:
        raise ValueError('Finite nonnegative terminal weight required')
    path = validate_path(segments)
    end = states.shape[0] * 40_000_000
    if path[0].start_ns != 0 or path[-1].start_ns + path[-1].duration_ns != end:
        raise ValueError('Score one state endpoint per completed forty-ms physical interval')
    error = states[:, :2] - target
    parts = {
        'tracking': .04 * float(np.sum(error * error)),
        'velocity': .04 * .05 * float(np.sum(states[:, 2:4]**2)),
        'effort': .001 * effort_integral(path, 0, end),
        'terminal': terminal_weight * float(error[-1] @ error[-1]),
    }
    parts['total'] = math.fsum(parts.values())
    if not all(math.isfinite(value) for value in parts.values()):
        raise ValueError('Nonfinite physical cost cannot be used as a measured episode score')
    return parts


def anchor_cost(states: np.ndarray, target: np.ndarray, segments,
                terminal_weight: float = 5.0) -> float:
    return anchor_components(states, target, segments, terminal_weight)['total']
