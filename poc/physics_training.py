"""Nominal endpoint optimization for the five-parameter physical reference."""
from __future__ import annotations

import copy
import math
import torch


def _data(model, data):
    if set(data) != {'state', 'command', 'h_s', 'target'}:
        raise ValueError('Physical fitting uses state, command, interval and endpoint labels only')
    values = {key: torch.as_tensor(value, dtype=model.scale_q.dtype, device='cpu').detach()
              for key, value in data.items()}
    state, command, h, target = (values[k] for k in ('state', 'command', 'h_s', 'target'))
    dimension = 4 if model.actuator == 'direct' else 6
    if (state.ndim != 2 or state.shape[1] != dimension or len(state) == 0
            or command.shape != (len(state), 2) or h.shape != (len(state), 1)
            or target.shape != state.shape or not all(bool(torch.isfinite(v).all()) for v in values.values())
            or bool((command.abs() > 2).any())):
        raise ValueError('Invalid physical fitting tensor shapes, units or values')
    allowed = torch.tensor([.02, .03, .04, .05, .06], dtype=h.dtype)
    if not bool(torch.isclose(h, allowed[None], atol=1e-8, rtol=0).any(1).all()):
        raise ValueError('Fine-time evaluation labels cannot enter physical reference fitting')
    return values


def endpoint_loss(model, data: dict, integration_step_s: float):
    values = _data(model, data)
    packet = model.predict(values['state'], values['command'], values['h_s'],
                           integration_step_s=integration_step_s)
    if any(status != 'ok' for status in packet.status):
        raise ValueError('Physical reference fitting produced a failed rollout')
    scale = torch.cat((model.scale_q, model.scale_v))
    residual = (packet.next_state[:, :4] - values['target'][:, :4]) / scale
    return (residual**2).mean()


def _integration_step(model, validation):
    step, audits = .005, []
    with torch.no_grad():
        while True:
            coarse = model.predict(validation['state'], validation['command'], validation['h_s'],
                                   integration_step_s=step)
            fine = model.predict(validation['state'], validation['command'], validation['h_s'],
                                 integration_step_s=step / 2)
            if any(s != 'ok' for packet in (coarse, fine) for s in packet.status):
                raise ValueError('Reference solver refinement has a failed nominal prediction')
            scale = torch.cat((model.scale_q, model.scale_v))
            difference = float(((coarse.next_state[:, :4] - fine.next_state[:, :4]) / scale).abs().max())
            audits.append({'coarse_step_s': step, 'fine_step_s': step / 2,
                           'max_normalized_endpoint_difference': difference})
            if difference < .005:
                return step, audits
            step /= 2
            if step == 0 or not math.isfinite(step):
                raise ValueError('Physical reference solver could not resolve nominal endpoints')


def optimize_endpoints(model, training: dict, validation: dict, *, epochs: int = 100,
                       batch_size: int = 256, seed: int = 8201, resume: dict | None = None,
                       on_epoch=None) -> dict:
    if any(type(value) is not int or value <= 0 for value in (epochs, batch_size)) or epochs % 5:
        raise ValueError('Use positive nominal epochs in five-epoch validation units and a positive batch')
    if type(seed) is not int or seed < 0:
        raise ValueError('Explicit deterministic physical-reference shuffle seed required')
    train, valid = _data(model, training), _data(model, validation)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    count = len(train['state'])
    if resume is None:
        step, solver_audits = _integration_step(model, valid)
        training_history, validation_history = [], []
        best_loss, best_epoch, best = math.inf, None, None
        updates, first_epoch = 0, 1
    else:
        if (resume['shuffle_seed'] != seed or resume['batch_size'] != batch_size
                or resume['target_count'] != count or not 0 < resume['epoch'] <= epochs):
            raise ValueError('Incompatible physical-reference optimizer resumption')
        model.load_state_dict(resume['last_state'])
        optimizer.load_state_dict(resume['optimizer_state'])
        generator.set_state(resume['rng_state'])
        step, solver_audits = resume['integration_step_s'], resume['solver_audits_before_fit']
        training_history = copy.deepcopy(resume['training_history'])
        validation_history = copy.deepcopy(resume['validation_history'])
        best_loss, best_epoch = resume['best_validation_loss'], resume['best_epoch']
        best = copy.deepcopy(resume['best_state'])
        updates, first_epoch = resume['optimizer_updates'], resume['epoch'] + 1
    for epoch in range(first_epoch, epochs + 1):
        permutation = torch.randperm(count, generator=generator)
        for start in range(0, count, batch_size):
            indices = permutation[start:start + batch_size]
            batch = {key: value[indices] for key, value in train.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = endpoint_loss(model, batch, step)
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite physical-reference training loss')
            loss.backward()
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise ValueError('Missing or nonfinite physical-parameter gradient')
            optimizer.step()
            updates += 1
            training_history.append({'epoch': epoch, 'update': updates, 'valid_targets': len(indices),
                                     'loss': float(loss.detach())})
        if epoch % 5 == 0:
            with torch.no_grad():
                score = float(endpoint_loss(model, valid, step))
            validation_history.append({'epoch': epoch, 'loss': score})
            # The earlier validation checkpoint wins relative ties.
            if best is None or score < best_loss - 1e-8 * abs(best_loss):
                best_loss, best_epoch = score, epoch
                best = copy.deepcopy(model.state_dict())
        if on_epoch is not None:
            on_epoch({'epoch': epoch, 'last_state': copy.deepcopy(model.state_dict()),
                      'optimizer_state': copy.deepcopy(optimizer.state_dict()),
                      'rng_state': generator.get_state().clone(), 'training_history': copy.deepcopy(training_history),
                      'validation_history': copy.deepcopy(validation_history),
                      'best_validation_loss': best_loss, 'best_epoch': best_epoch,
                      'best_state': copy.deepcopy(best), 'optimizer_updates': updates,
                      'integration_step_s': step, 'solver_audits_before_fit': solver_audits,
                      'shuffle_seed': seed, 'batch_size': batch_size, 'target_count': count})
    last = copy.deepcopy(model.state_dict())
    model.load_state_dict(best)
    selected_step, selected_audits = _integration_step(model, valid)
    if selected_step < step:
        raise ValueError('Selected physical field needs a finer training solver; repeat from the declared initialization')
    return {'best_state': best, 'last_state': last, 'best_epoch': best_epoch,
            'best_validation_loss': best_loss, 'training_history': training_history,
            'validation_history': validation_history, 'epochs_completed': epochs,
            'optimizer_updates': updates, 'valid_target_exposures': epochs * count,
            'optimizer_state': copy.deepcopy(optimizer.state_dict()), 'rng_state': generator.get_state().clone(),
            'integration_step_s': step, 'evaluation_step_s': min(step, .00125),
            'solver_audits_before_fit': solver_audits, 'solver_audits_selected_field': selected_audits,
            'optimizer': 'Adam', 'learning_rate': .01, 'shuffle_seed': seed,
            'selection': 'nominal endpoint validation only'}
