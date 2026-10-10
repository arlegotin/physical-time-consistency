"""Exact selected-model restoration and isolated privileged control references."""
from pathlib import Path
import numpy as np
import torch

from .contracts import verify_run
from .io import read_json, sha256_file
from .models.common import Scales, activation, interval_tensor, prediction


class ConstantVelocity:
    def __init__(self, actuator):
        self.actuator = actuator

    def predict(self, state, command, h_s, history=None, power_mode=None):
        h = interval_tensor(state, command, h_s, self.actuator)
        values = [state[:, :2] + h * state[:, 2:4], state[:, 2:4]]
        if self.actuator == 'lagged':
            values.append(activation(state, command, h))
        return prediction(torch.cat(values, 1), 0)


class SimOracle:
    def __init__(self, actuator, physics_step_ns, asset_root):
        from .envs import Plant
        self.actuator = actuator
        self.plant = Plant(actuator, physics_step_ns, asset_root)
        self.physics_steps = 0

    def predict(self, state, command, h_s, history=None, power_mode=None):
        h = interval_tensor(state, command, h_s, self.actuator)
        results = []
        for row in range(len(state)):
            nanoseconds = float(h[row, 0]) * 1e9
            duration = round(nanoseconds)
            if abs(nanoseconds - duration) > 1e-3 or duration % self.plant.physics_step_ns:
                raise ValueError('Oracle interval does not divide the qualified physical tick')
            self.plant.reset(state[row].detach().cpu().numpy())
            if duration:
                self.plant.advance(command[row].detach().cpu().numpy(), duration)
                self.physics_steps += duration // self.plant.physics_step_ns
            results.append(self.plant.state())
        return prediction(torch.tensor(np.stack(results), dtype=state.dtype), 0)


def load_selected(manifest_path, manifest_sha256, checkpoint_sha256, *, model_id, actuator, seed):
    path = Path(manifest_path).resolve()
    if sha256_file(path) != manifest_sha256 or sha256_file(path.parent / 'weights.pt') != checkpoint_sha256:
        raise ValueError('Control manifest/checkpoint hash differs from its prospective freeze')
    source = path.parents[3]
    record = read_json(path)
    verify_run(source, record)
    if record['completion_status'] != 'complete':
        raise ValueError('Control cannot use a partial or failed model fit')
    if record['milestone_id'] == 'M30':
        parameters = record['job_inputs']['parameters']
        if any(parameters[key] != expected for key, expected in
               (('model_id', model_id), ('actuator', actuator), ('seed', seed))):
            raise ValueError('Control query and frozen training identity disagree')
        if record['phase'] != 'unit':
            from .train import qualify_nominal
            if not qualify_nominal(record, source)['qualified']:
                raise ValueError('Selected neural model has not qualified its nominal fit')
        # The selected field/decoder must have the same implementation at control time.
        for filename, expected in record['job_inputs'].get('dependencies', {}).items():
            if filename.startswith('poc/models/') or filename in {'poc/history.py', 'poc/analytic.py'}:
                current = Path(__file__).resolve().parents[1] / filename
                if sha256_file(current) != expected:
                    raise ValueError('Control predictor differs from the frozen trained model: ' + filename)
    elif record['milestone_id'] == 'M40':
        summary = read_json(path.parent / 'summary.json')
        if summary['model_id'] != model_id or summary['actuator'] != actuator:
            raise ValueError('Control and physical-reference identity disagree')
    else:
        raise ValueError('Unsupported selected-model producing milestone')
    norm = read_json(path.parent / 'normalization.json')
    scales = Scales(*(torch.tensor(norm[key], dtype=torch.float32)
                     for key in ('scale_q', 'scale_v', 'velocity_input_rms')))
    if model_id == 'physics_fit':
        from .physics_fit import PhysicsFit
        model = PhysicsFit(torch.tensor([.10, .02, .02, .05, .05]), actuator, scales)
    elif model_id == 'accel_ridge':
        from .physics_fit import AccelRidge
        model = AccelRidge(np.zeros((12, 2)), actuator, scales)
    else:
        from .models.factory import build_model
        model = build_model(model_id, actuator, scales, seed)
    from .reference_jobs import tensor_hash
    weights = torch.load(path.parent / 'weights.pt', weights_only=True, map_location='cpu')
    if tensor_hash(weights) != record['model_hashes']['canonical_tensor_sha256']:
        raise ValueError('Selected model canonical tensor hash differs from the fit receipt')
    model.load_state_dict(weights, strict=True)
    return model.double().eval()
