"""Shared physical units, scales, immutable queries and work counters."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class Scales:
    q: Tensor
    v: Tensor
    v_input: Tensor

    def __post_init__(self):
        for scale in (self.q, self.v, self.v_input):
            if scale.shape != (2,) or not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()):
                raise ValueError('Each scale must contain two finite positive physical values')


@dataclass(frozen=True)
class Prediction:
    next_state: Tensor
    status: tuple[str, ...]
    network_calls: int
    vector_field_calls: int
    scalar_candidate_evaluations: int
    history_tokens_processed: int


def require_same_state(reference, branch) -> None:
    if isinstance(reference, Tensor):
        unchanged = reference.shape == branch.shape and torch.equal(reference, branch)
    else:
        unchanged = np.array_equal(reference, branch)
    if not unchanged:
        raise ValueError('New-command intervention changed the initial state, including activation')


def interval_tensor(state: Tensor, command: Tensor, h_s, actuator: str) -> Tensor:
    dimension = 4 if actuator == 'direct' else 6
    if state.ndim != 2 or state.shape[1] != dimension or command.shape != (state.shape[0], 2):
        raise ValueError('Expected batched q/v[/w] state and two-dimensional physical commands')
    if not torch.isfinite(state).all() or not torch.isfinite(command).all():
        raise ValueError('State and command must be finite')
    h = torch.as_tensor(h_s, dtype=state.dtype, device=state.device)
    if h.ndim == 0:
        h = h.expand(state.shape[0], 1)
    elif h.shape == (1, 1):
        h = h.expand(state.shape[0], 1)
    if h.shape != (state.shape[0], 1) or not torch.isfinite(h).all() or (h < 0).any():
        raise ValueError('Interval must be finite nonnegative seconds, scalar or [batch,1]')
    return h


def features(state: Tensor, command: Tensor, h: Tensor, v_input: Tensor) -> Tensor:
    q, v = state[:, :2], state[:, 2:4]
    periodic = torch.stack((q[:, 0].sin(), q[:, 0].cos(), q[:, 1].sin(), q[:, 1].cos()), dim=1)
    values = [periodic, v/v_input]
    if state.shape[1] == 6:
        values.append(state[:, 4:6]/2)
    return torch.cat([*values, command/2, h/.04], dim=1)


def activation(state: Tensor, command: Tensor, h: Tensor) -> Tensor:
    w = state[:, 4:6]
    return w+(command-w)*(-torch.expm1(-h/.08))


def prediction(next_state: Tensor, calls: int, vector_fields: int = 0, history_tokens: int = 0) -> Prediction:
    finite = torch.isfinite(next_state).all(dim=1)
    guard = (next_state[:, :4].abs() <= 1000).all(dim=1)
    status = tuple('ok' if good and safe else 'nonfinite' if not good else 'imagined_state_guard'
                   for good, safe in zip(finite.tolist(), guard.tolist()))
    return Prediction(next_state, status, calls, vector_fields, len(status), history_tokens)

