"""Stable exact unit-mass flows and physical new-command derivatives."""
from __future__ import annotations

import math
import numpy as np


def _interval(h_s: float) -> float:
    if isinstance(h_s, bool) or not np.isscalar(h_s) or not math.isfinite(h_s) or h_s < 0:
        raise ValueError('Interval must be finite nonnegative seconds')
    return float(h_s)


def _series(r: float, start: int, positive: bool) -> float:
    term = r**start / math.factorial(start)
    term *= 1 if positive else -1
    result, order = term, start
    while True:
        order += 1
        term *= -r/order
        if abs(term) <= 1e-16*abs(result):
            return result
        result += term


def coefficients(h_s: float, tau_s: float = .08) -> tuple[float, float, float]:
    h = _interval(h_s)
    if not math.isfinite(tau_s) or tau_s <= 0:
        raise ValueError('Actuator time constant must be finite positive seconds')
    if h == 0:
        return 0., 0., 0.
    r = h/tau_s
    kw = -math.expm1(-r)
    if r < .1:
        kv, kq = tau_s*_series(r, 2, True), tau_s*tau_s*_series(r, 3, True)
    else:
        kv, kq = h-tau_s*kw, .5*h*h-tau_s*h+tau_s*tau_s*kw
    return kw, kv, kq


def transition(state: np.ndarray, command: np.ndarray, h_s: float, actuator: str) -> np.ndarray:
    h = _interval(h_s)
    state, command = np.asarray(state, dtype=np.float64), np.asarray(command, dtype=np.float64)
    if actuator not in {'direct', 'lagged'}:
        raise ValueError('Unknown actuator chain')
    components = 2 if actuator == 'direct' else 3
    if command.ndim != 1 or not len(command) or state.shape != (components*len(command),):
        raise ValueError('Expected q/v[/w] and a nonempty command vector')
    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(command)):
        raise ValueError('State and command must be finite')
    if h == 0:
        return state.copy()
    joints = len(command)
    q, v = state[:joints], state[joints:2*joints]
    if actuator == 'direct':
        return np.concatenate((q+h*v+.5*h*h*command, v+h*command))
    w = state[2*joints:]
    kw, kv, kq = coefficients(h)
    return np.concatenate((q+h*v+.5*w*h*h+(command-w)*kq, v+w*h+(command-w)*kv, w+(command-w)*kw))


def action_jacobian(h_s: float, actuator: str, n_joints: int = 2) -> np.ndarray:
    h = _interval(h_s)
    if isinstance(n_joints, bool) or not isinstance(n_joints, int) or n_joints <= 0:
        raise ValueError('Positive integer joint count required')
    if actuator == 'direct':
        gains = (.5*h*h, h)
    elif actuator == 'lagged':
        kw, kv, kq = coefficients(h)
        gains = (kq, kv, kw)
    else:
        raise ValueError('Unknown actuator chain')
    return np.concatenate([gain*np.eye(n_joints) for gain in gains])

