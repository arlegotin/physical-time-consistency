"""Matched small residual fields over a frozen common nominal physical prior.

Propagation methods are the existing solver maps. Coefficients depend only on
the current q/v stage; no state-dependent field values are cached across calls.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .actuator_aware_node import ActuatorAwareNODE
from .affine_flow import AffineFlow
from .affine_rk4 import AffineRK4
from .calibration_models import PhysicsResidual, _nominal
from .models.common import Scales
from .models.mlp import TransitionPredictor


def _zero_output(network: nn.Sequential) -> None:
    nn.init.zeros_(network[-1].weight)
    nn.init.zeros_(network[-1].bias)


class _HybridAffine(TransitionPredictor):
    stateless_query = True

    def __init__(self, scales: Scales):
        super().__init__('lagged', scales)
        self.nominal = _nominal(scales).requires_grad_(False)
        self.tau_s = .08
        with torch.random.fork_rng(devices=[]):
            self.network = nn.Sequential(nn.Linear(6, 32), nn.Tanh(), nn.Linear(32, 32),
                                         nn.Tanh(), nn.Linear(32, 6))
        self.network.to(device=scales.q.device, dtype=scales.q.dtype)
        _zero_output(self.network)

    def coefficients(self, qv: Tensor) -> tuple[Tensor, Tensor]:
        nominal_f, nominal_b = self.nominal.acceleration_coefficients(qv)
        residual_f, residual_b = AffineFlow.coefficients(self, qv)
        return nominal_f + residual_f, nominal_b + residual_b

    def vector_field(self, qv: Tensor, command: Tensor, w: Tensor | None = None) -> Tensor:
        f, b = self.coefficients(qv)
        acceleration = f + torch.bmm(b, w.unsqueeze(-1)).squeeze(-1)
        return torch.cat((qv[:, 2:4], acceleration), dim=1)


class HybridAffineFlow(_HybridAffine):
    """Two-stage analytic propagation of the common coefficient residual field."""
    _step = AffineFlow._step
    predict = AffineFlow.predict


class HybridAffineRK4(_HybridAffine):
    """Four-stage RK4 propagation of the common coefficient residual field."""
    predict = AffineRK4.predict


class HybridAffineAware(_HybridAffine):
    """Two-evaluation actuator-aware propagation of the common coefficient field."""
    predict = ActuatorAwareNODE.predict


class HybridAccelRK4(PhysicsResidual):
    """Existing 8/32/32/2 acceleration residual, initialized to the prior field."""
    def __init__(self, scales: Scales):
        with torch.random.fork_rng(devices=[]):
            super().__init__(scales)
        _zero_output(self.network)


class HybridAccelAware(HybridAccelRK4):
    """Matched practical acceleration residual with the two-evaluation map."""
    predict = ActuatorAwareNODE.predict

    def __init__(self, scales: Scales):
        super().__init__(scales)
        self.tau_s = .08

    def vector_field(self, qv: Tensor, command: Tensor, w: Tensor | None = None) -> Tensor:
        return torch.cat((qv[:, 2:4], self.acceleration(qv, w)), dim=1)
