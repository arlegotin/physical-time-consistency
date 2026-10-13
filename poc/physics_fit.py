"""Endpoint-calibrated references with a declared topology and no privileged plant inputs."""
from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .analytic import coefficients
from .io import read_json, sha256_file
from .models.common import activation, interval_tensor, prediction
from .models.mlp import TransitionPredictor


def _inverse_softplus(value):
    if not bool(torch.isfinite(value).all()) or bool((value <= 0).any()):
        raise ValueError('Physical initialization is outside the positive parameterization')
    return value + torch.log(-torch.expm1(-value))


class RigidReference(TransitionPredictor):
    def predict(self, state, command, h_s, history=None, power_mode=None,
                integration_step_s: float = .00125):
        h = interval_tensor(state, command, h_s, self.actuator)
        if not math.isfinite(integration_step_s) or integration_step_s <= 0:
            raise ValueError('Reference integrator needs a positive finite maximum step')
        if bool((h == 0).all()):
            return prediction(state.clone(), 0)
        count = max(1, math.ceil(float(h.max().detach()) / integration_step_s))
        step, x = h / count, state[:, :4]
        def field(qv, time):
            effort = activation(state, command, time) if self.actuator == 'lagged' else command
            return torch.cat((qv[:, 2:4], self.acceleration(qv, effort)), 1)
        for index in range(count):
            time = index * step
            k1 = field(x, time)
            k2 = field(x + step * k1 / 2, time + step / 2)
            k3 = field(x + step * k2 / 2, time + step / 2)
            k4 = field(x + step * k3, time + step)
            x = x + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        values = torch.cat((x, activation(state, command, h)), 1) if self.actuator == 'lagged' else x
        return prediction(torch.where(h == 0, state, values), 0, vector_fields=4 * count)


class PhysicsFit(RigidReference):
    def __init__(self, parameters: torch.Tensor, actuator: str, scales):
        super().__init__(actuator, scales)
        physical = torch.as_tensor(parameters).detach().clone()
        if physical.shape != (5,) or not physical.is_floating_point():
            raise ValueError('Five physical floating parameters are required')
        theta1, theta2, theta3, d1, d2 = physical.unbind()
        positive = torch.stack((theta1 - theta3**2 / theta2 - 1e-4,
                                theta2 - 1e-4, theta3, d1, d2))
        self.raw = nn.Parameter(_inverse_softplus(positive))

    def physical_parameters(self):
        a, b, c, e1, e2 = F.softplus(self.raw).unbind()
        theta2 = b + 1e-4
        return torch.stack((c**2 / theta2 + a + 1e-4, theta2, c, e1, e2))

    def mass_matrix(self, q):
        return self._mass_matrix(q, self.physical_parameters())

    @staticmethod
    def _mass_matrix(q, parameters):
        theta1, theta2, theta3 = parameters[:3].unbind()
        coupling = theta3 * torch.cos(q[:, 1])
        m11 = theta1 + theta2 + 2 * coupling + .02
        m12 = theta2 + coupling
        m22 = torch.ones_like(coupling) * (theta2 + .02)
        return torch.stack((torch.stack((m11, m12), 1), torch.stack((m12, m22), 1)), 1)

    def _mechanical_terms(self, qv):
        """Compute shared mass, Coriolis and damping without combining forces."""
        parameters = self.physical_parameters()
        v1, v2 = qv[:, 2], qv[:, 3]
        cross = parameters[2] * torch.sin(qv[:, 1])
        coriolis = torch.stack((-cross * (2 * v1 * v2 + v2**2), cross * v1**2), 1)
        damping = parameters[3:] * qv[:, 2:4]
        return self._mass_matrix(qv[:, :2], parameters), coriolis, damping

    def acceleration(self, qv, effort):
        mass, coriolis, damping = self._mechanical_terms(qv)
        # Retain the legacy subtraction order for the physical forward path.
        rhs = effort - coriolis - damping
        return torch.linalg.solve(mass, rhs.unsqueeze(2)).squeeze(2)

    def acceleration_coefficients(self, qv: Tensor) -> tuple[Tensor, Tensor]:
        """Nominal drift and dense effort gain from one three-column solve."""
        mass, coriolis, damping = self._mechanical_terms(qv)
        identity = torch.eye(2, dtype=qv.dtype, device=qv.device).expand(len(qv), -1, -1)
        rhs = torch.cat(((-coriolis - damping).unsqueeze(-1), identity), dim=2)
        solved = torch.linalg.solve(mass, rhs)
        return solved[:, :, 0], solved[:, :, 1:]


def ridge_features(state, effort):
    if isinstance(state, torch.Tensor):
        q1, q2, v1, v2 = state[:, :4].unbind(1)
        return torch.stack((torch.ones_like(q1), torch.sin(q1), torch.cos(q1),
                            torch.sin(q2), torch.cos(q2), v1, v2, effort[:, 0], effort[:, 1],
                            v1**2, v1 * v2, v2**2), 1)
    q1, q2, v1, v2 = np.asarray(state)[:, :4].T
    return np.stack((np.ones_like(q1), np.sin(q1), np.cos(q1), np.sin(q2), np.cos(q2),
                     v1, v2, effort[:, 0], effort[:, 1], v1**2, v1 * v2, v2**2), 1)


class AccelRidge(RigidReference):
    def __init__(self, coefficients: np.ndarray, actuator: str, scales):
        super().__init__(actuator, scales)
        value = torch.as_tensor(coefficients, dtype=self.scale_q.dtype).detach().clone()
        if value.shape != (12, 2) or not bool(torch.isfinite(value).all()):
            raise ValueError('Twelve finite two-joint regression coefficients required')
        self.register_buffer('coefficients', value)

    def acceleration(self, qv, effort):
        return ridge_features(qv, effort) @ self.coefficients


def fit_ridge(train_manifest: Path, *, audit_path: Path | None = None) -> dict:
    from .dataset import load_corpus
    metadata = read_json(train_manifest)
    if metadata.get('role') != 'train':
        raise ValueError('Physics references may fit only training endpoints')
    actuator = metadata['actuator']
    if actuator not in {'direct', 'lagged'}:
        raise ValueError('Unknown reference actuator chain')
    arrays = load_corpus(train_manifest, phase='exploratory', role='train', purpose='training',
                         audit_path=audit_path)
    states = arrays['state'][:, :-1].reshape(-1, arrays['state'].shape[-1])
    targets = arrays['state'][:, 1:].reshape(states.shape)
    commands = arrays['command'].reshape(-1, 2)
    h = arrays['duration_ns'].reshape(-1) / 1e9
    effort = commands
    if actuator == 'lagged':
        kv = np.array([coefficients(float(value))[1] for value in h])
        effort = states[:, 4:] + (commands - states[:, 4:]) * (kv / h)[:, None]
    design = ridge_features(states, effort)
    center, scale = design.mean(0), design.std(0)
    center[0], scale[0] = 0, 1
    scale[1:] = np.maximum(scale[1:], 1e-12)
    standardized = (design - center) / scale
    penalty = np.diag(np.r_[0., np.full(11, math.sqrt(1e-6))])
    response = (targets[:, 2:4] - states[:, 2:4]) / h[:, None]
    fitted, _, rank, singular = np.linalg.lstsq(np.r_[standardized, penalty],
                                               np.r_[response, np.zeros((12, 2))], rcond=None)
    raw = fitted / scale[:, None]
    raw[0] -= (center[:, None] * raw).sum(0)
    return {'coefficients': raw.tolist(), 'standardized_coefficients': fitted.tolist(),
            'feature_mean': center.tolist(), 'feature_scale': scale.tolist(),
            'ridge': 1e-6, 'constant_regularized': False, 'rank': int(rank),
            'singular_values': singular.tolist(), 'transition_count': len(h),
            'actuator': actuator, 'train_manifest_sha256': sha256_file(train_manifest),
            'target': 'average acceleration from observed velocity endpoints',
            'lagged_effort': 'exact interval average activation' if actuator == 'lagged' else None}
