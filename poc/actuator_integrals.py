"""Stable held-command lag integrals for a shared known actuator time constant."""
from __future__ import annotations

import math
from numbers import Real

import torch


def _ratio(h: torch.Tensor, tau_s: float) -> torch.Tensor:
    if isinstance(tau_s, bool) or not isinstance(tau_s, Real) or not math.isfinite(tau_s) or tau_s <= 0:
        raise ValueError('Actuator tau must be a finite positive scalar')
    if not isinstance(h, torch.Tensor) or not h.is_floating_point() or not bool(torch.isfinite(h).all()) or bool((h < 0).any()):
        raise ValueError('Interval must be a finite nonnegative floating tensor')
    return h/tau_s


def _series(r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Kv = h*r*pv and Kq = h²*r*pq. Twelve terms bound cancellation
    # error below float64 precision on the r < .1 branch.
    pv, pq = torch.zeros_like(r), torch.zeros_like(r)
    for k in range(12, 0, -1):
        pv = pv*r+((-1)**(k+1)/math.factorial(k+1))
        pq = pq*r+((-1)**(k+1)/math.factorial(k+2))
    return pv, pq


def lag_integrals(h: torch.Tensor, tau_s: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the new-command velocity and position integrals Kv and Kq."""
    r = _ratio(h, tau_s)
    decay_increment = torch.expm1(-r)
    kv = h+tau_s*decay_increment
    kq = h*h/2-tau_s*h-tau_s*tau_s*decay_increment
    # Clamp the inactive series input so large intervals do not overflow its
    # polynomial and contaminate interval gradients through torch.where.
    small_r = torch.where(r < .1, r, torch.zeros_like(r))
    pv, pq = _series(small_r)
    return (torch.where(r < .1, h*small_r*pv, kv),
            torch.where(r < .1, h*h*small_r*pq, kq))


def response_ratios(h: torch.Tensor, tau_s: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Kv/beta and Kq/beta, continuously extended to zero at h=0.

    beta = 1-exp(-h/(2*tau)); the small-r forms approach h and h²/3.
    No detached caches are used, so interval derivatives stay on the tensor path.
    """
    r = _ratio(h, tau_s)
    kv, kq = lag_integrals(h, tau_s)
    beta = -torch.expm1(-r/2)
    safe_beta = torch.where(r == 0, torch.ones_like(beta), beta)
    small_r = torch.where(r < .1, r, torch.zeros_like(r))
    pv, pq = _series(small_r)
    safe_r = torch.where(small_r == 0, torch.ones_like(r), small_r)
    beta_over_r = torch.where(small_r == 0, torch.full_like(r, .5),
                              -torch.expm1(-small_r/2)/safe_r)
    return (torch.where(r < .1, h*pv/beta_over_r, kv/safe_beta),
            torch.where(r < .1, h*h*pq/beta_over_r, kq/safe_beta))
