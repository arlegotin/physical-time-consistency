"""Frozen comparator recipes for the PTCO-CAL-0.1 lagged calibration study.

Construction uses only common observed scales and the declared historical prior.
Each learned solver is trained and queried through its own prediction map.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .actuator_aware_node import ActuatorAwareNODE
from .affine_flow import AffineFlow
from .affine_rk4 import AffineRK4
from .models.common import Prediction, Scales, activation, features, interval_tensor, prediction
from .models.mlp import TransitionPredictor
from .models.ode import NeuralODE
from .physics_fit import PhysicsFit
from .quadrature_node import QuadratureNODE


MODEL_IDS: tuple[str, ...] = (
    'affine_flow', 'affine_rk4', 'node_structured', 'node_quadrature',
    'node_actuator_aware', 'physics_fit', 'physics_residual',
)
HYBRID_MODEL_IDS: tuple[str, ...] = (
    'hybrid_affine_flow', 'hybrid_affine_rk4', 'hybrid_affine_aware',
    'hybrid_accel_rk4', 'hybrid_accel_aware',
)
ALL_MODEL_IDS: tuple[str, ...] = MODEL_IDS + HYBRID_MODEL_IDS
NOMINAL_PARAMETERS: tuple[float, ...] = (.10, .02, .02, .05, .05)


def propagation_family(model_id: str) -> tuple[str, ...]:
    """Same-weight coefficient maps available for propagation diagnostics."""
    legacy = ('affine_flow', 'affine_rk4')
    if model_id in legacy:
        return legacy
    if model_id in HYBRID_MODEL_IDS[:3]:
        return HYBRID_MODEL_IDS[:3]
    return ()


def _hybrid_types() -> dict[str, type[nn.Module]]:
    # Deferred import lets the hybrid references reuse the legacy residual class.
    from .hybrid_residual import (HybridAffineFlow, HybridAffineRK4, HybridAffineAware,
                                  HybridAccelRK4, HybridAccelAware)
    return dict(zip(HYBRID_MODEL_IDS, (HybridAffineFlow, HybridAffineRK4, HybridAffineAware,
                                      HybridAccelRK4, HybridAccelAware)))


def _nominal(scales: Scales) -> PhysicsFit:
    parameters = scales.q.new_tensor(NOMINAL_PARAMETERS)
    return PhysicsFit(parameters, 'lagged', scales)


class PhysicsResidual(TransitionPredictor):
    """Frozen rigid prior plus a 32/32 tanh acceleration correction.

    One RK4 step spans each queried interval (40 ms for study training), using
    exact shared activation at each stage and four residual network calls.
    """
    stateless_query = True

    def __init__(self, scales: Scales):
        super().__init__('lagged', scales)
        self.nominal = _nominal(scales).requires_grad_(False)
        self.network = nn.Sequential(nn.Linear(8, 32), nn.Tanh(), nn.Linear(32, 32),
                                     nn.Tanh(), nn.Linear(32, 2))
        self.network.to(device=scales.q.device, dtype=scales.q.dtype)

    def acceleration(self, qv: Tensor, effort: Tensor) -> Tensor:
        inputs = features(qv, effort, torch.zeros_like(effort[:, :1]), self.v_input)[:, :-1]
        correction = self.network(inputs) * (self.scale_v / .04)
        return self.nominal.acceleration(qv, effort) + correction

    def predict(self, state, command, h_s, history=None, power_mode=None) -> Prediction:
        if power_mode is not None:
            raise ValueError('Physics residual has no selectable power mode')
        h = interval_tensor(state, command, h_s, self.actuator)
        if bool((h == 0).all()):
            return prediction(state.clone(), 0)

        def field(qv, time):
            effort = activation(state, command, time)
            return torch.cat((qv[:, 2:4], self.acceleration(qv, effort)), 1)

        x = state[:, :4]
        k1 = field(x, torch.zeros_like(h))
        k2 = field(x + h * k1 / 2, h / 2)
        k3 = field(x + h * k2 / 2, h / 2)
        k4 = field(x + h * k3, h)
        x = x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        values = torch.cat((x, activation(state, command, h)), 1)
        return prediction(torch.where(h == 0, state, values), 4, vector_fields=4)


def build_model(model_id: str, scales: Scales, seed: int) -> nn.Module:
    """Build a fresh lagged predictor without advancing the caller's CPU RNG.

    Resetting initialization within a fork pairs tensors across each architecture
    family, while keeping all model storage independent.
    """
    if model_id not in (*ALL_MODEL_IDS, 'physics_nominal'):
        raise ValueError(f'Unknown calibration model: {model_id}')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        if model_id in HYBRID_MODEL_IDS:
            model = _hybrid_types()[model_id](scales)
        elif model_id == 'affine_flow':
            model = AffineFlow('lagged', scales)
        elif model_id == 'affine_rk4':
            model = AffineRK4('lagged', scales)
        elif model_id == 'node_structured':
            model = NeuralODE('lagged', scales, structured=True)
        elif model_id == 'node_quadrature':
            model = QuadratureNODE('lagged', scales)
        elif model_id == 'node_actuator_aware':
            model = ActuatorAwareNODE('lagged', scales)
        elif model_id == 'physics_residual':
            model = PhysicsResidual(scales)
        else:
            model = _nominal(scales)
            if model_id == 'physics_nominal':
                model.requires_grad_(False)
    return model.to(device=scales.q.device, dtype=scales.q.dtype)


def model_metadata(model: nn.Module) -> dict:
    """Describe architecture, physical prior and default batched solver work.

    Work counts apply to a nonzero 40 ms query, independent of batch size.
    All-zero queries do no field work; mixed batches execute the full batched map.
    Physical references report their numerical work even without network calls.
    """
    model_type = type(model)
    recipes = {
        AffineFlow: ('affine_flow', 'affine_flow', 2),
        AffineRK4: ('affine_rk4', 'rk4', 4),
        NeuralODE: ('node_structured', 'rk4', 4),
        QuadratureNODE: ('node_quadrature', 'quadrature', 2),
        ActuatorAwareNODE: ('node_actuator_aware', 'actuator_aware_quadrature', 2),
        PhysicsResidual: ('physics_residual', 'rk4', 4),
    }
    hybrid_types = _hybrid_types()
    for model_id, solver_name, calls in zip(
            HYBRID_MODEL_IDS,
            ('affine_flow', 'rk4', 'actuator_aware_quadrature', 'rk4', 'actuator_aware_quadrature'),
            (2, 4, 2, 4, 2)):
        recipes[hybrid_types[model_id]] = (model_id, solver_name, calls)
    if getattr(model, 'actuator', None) != 'lagged':
        raise ValueError('Unsupported calibration model: expected a lagged predictor')
    if model_type is PhysicsFit:
        model_id = 'physics_fit' if model.raw.requires_grad else 'physics_nominal'
        solver = {'name': 'rk4', 'default_max_step_s': .00125,
                  'training_resolution': 'existing endpoint-fit solver qualification',
                  'steps_rule': 'ceil(max(h_s) / integration_step_s)'}
        work = {'network_calls_per_40ms_query': 0,
                'vector_field_calls_per_40ms_query': 128,
                'network_calls_rule': '0',
                'vector_field_calls_rule': '0 if all h_s == 0 else 4 * ceil(max(h_s) / integration_step_s)'}
    elif model_type in recipes and not (model_type is NeuralODE and not model.structured):
        model_id, solver_name, calls = recipes[model_type]
        solver = {'name': solver_name, 'steps_per_interval': 1,
                  'training_interval_s': .04}
        if model_type in (AffineRK4, NeuralODE, hybrid_types['hybrid_affine_rk4']):
            rule = '0 if all h_s == 0 else 4 * solver_substeps (default 1)'
        else:
            rule = f'0 if all h_s == 0 else {calls}'
        work = {'network_calls_per_40ms_query': calls,
                'vector_field_calls_per_40ms_query': calls,
                'network_calls_rule': rule, 'vector_field_calls_rule': rule}
    else:
        raise ValueError('Unsupported calibration model')

    network = getattr(model, 'network', None)
    architecture = None
    if network is not None:
        layers = [layer for layer in network if isinstance(layer, nn.Linear)]
        architecture = {'input_features': layers[0].in_features,
                        'hidden_widths': [layer.out_features for layer in layers[:-1]],
                        'output_features': layers[-1].out_features, 'activation': 'tanh'}
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {'model_id': model_id, 'actuator': model.actuator, 'tau_s': .08,
            'nominal_prior_parameters': list(NOMINAL_PARAMETERS),
            'nominal_physics_frozen': (model_type is PhysicsResidual or model_id == 'physics_nominal'
                                       or model_id in HYBRID_MODEL_IDS),
            'architecture': architecture,
            'parameter_counts': {'total': total, 'trainable': trainable, 'frozen': total - trainable},
            'solver': solver, 'field_work': work,
            'field_work_units': 'batched calls; independent of inference batch size'}
