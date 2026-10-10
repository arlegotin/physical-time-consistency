"""M10 exact-case evidence; no robot data generation or model fitting."""
from __future__ import annotations

from decimal import Decimal, localcontext
import hashlib
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
import mujoco

from .analytic import action_jacobian, coefficients, transition
from .clocks import jitter_partition, partition, to_ns
from .execution import cpu_defaults, parent, run_job
from .io import atomic_json, canonical_json, sha256_file
from .milestones.m10 import CHECKS
from .models.common import Scales, require_same_state
from .models.power import PowerHead


def _decimal(h):
    with localcontext() as context:
        context.prec = 60
        time, tau = Decimal(str(h)), Decimal('.08')
        kw = 1-(-time/tau).exp()
        return tuple(float(x) for x in (kw, time-tau*kw, time*time/2-tau*time+tau*tau*kw))


def _analytic_evidence(folder):
    rng = np.random.Generator(np.random.PCG64(8001))
    horizons = np.array([0., 1e-8, 1e-6, .0025, .005, .01, .02, .04, .08])
    arrays = {'horizons_s': horizons}
    maximum, composition, jacobian, decimal = 0., 0., 0., 0.
    for actuator, dimension in (('direct',4), ('lagged',6)):
        states, commands = rng.uniform(-2, 2, (1000, dimension)), rng.uniform(-2, 2, (1000, 2))
        states[0], commands[:4] = 0, np.array([-2., 2., 0., -1.])[:, None]
        predicted, reference = [], []
        for h in horizons:
            kw, kv, kq = _decimal(h)
            expected_coeff = np.array([kw, kv, kq])
            actual_coeff = np.array(coefficients(h))
            if h > 0:
                decimal = max(decimal, float(np.max(np.abs(actual_coeff-expected_coeff)/expected_coeff)))
            q, v = states[:, :2], states[:, 2:4]
            if actuator == 'direct':
                expected = np.concatenate((q+h*v+.5*h*h*commands, v+h*commands), axis=1)
                gains = (.5*h*h, h)
            else:
                w = states[:, 4:]
                expected = np.concatenate((q+h*v+.5*w*h*h+(commands-w)*kq, v+w*h+(commands-w)*kv, w+(commands-w)*kw), axis=1)
                gains = (kq, kv, kw)
            actual = np.stack([transition(s, u, h, actuator) for s, u in zip(states, commands)])
            divided = np.stack([transition(transition(s, u, .25*h, actuator), u, .75*h, actuator) for s, u in zip(states, commands)])
            maximum = max(maximum, float(np.max(np.abs(actual-expected))))
            composition = max(composition, float(np.max(np.abs(divided-actual))))
            expected_j = np.concatenate([gain*np.eye(2) for gain in gains])
            if h > 0:
                jacobian = max(jacobian, float(np.linalg.norm(action_jacobian(h, actuator)-expected_j)/np.linalg.norm(expected_j)))
            predicted.append(actual)
            reference.append(expected)
        arrays['states_'+actuator], arrays['commands_'+actuator] = states, commands
        arrays['predicted_'+actuator], arrays['reference_'+actuator] = np.stack(predicted), np.stack(reference)
    np.savez_compressed(folder / 'analytic.npz', **arrays)
    return {'analytic_max_abs_error': maximum, 'decimal_max_relative_error': decimal,
            'jacobian_max_relative_error': jacobian, 'composition_max_abs_error': composition, 'exact_cases_per_actuator': 1000}


def _power_evidence(folder):
    arrays, checkpoint_hashes, maximum, detected = {}, [], 0., {}
    rng = np.random.Generator(np.random.PCG64(8002))
    for actuator in ('direct','lagged'):
        network = torch.nn.Linear(9 if actuator == 'direct' else 11, 4).double()
        scales = Scales(torch.ones(2, dtype=torch.float64), torch.ones(2, dtype=torch.float64), torch.ones(2, dtype=torch.float64))
        model = PowerHead(network, actuator, scales)
        state = torch.tensor(rng.uniform(-2, 2, (32, 4 if actuator == 'direct' else 6)), dtype=torch.float64)
        command = torch.tensor(rng.uniform(-2, 2, (32, 2)), dtype=torch.float64)
        storage = {key: [] for key in ('outputs', 'state_jac', 'command_jac', 'weight_grad', 'four_step_loss', 'four_step_grad')}
        checkpoint = folder / (actuator+'_affine_weights.pt')
        torch.save(model.state_dict(), checkpoint)
        hashes = []
        for update in range(2):
            records = {key: [] for key in storage}
            for mode in ('low', 'direct', 'lagged'):
                predict = lambda x, u: model.predict(x, u, .04, power_mode=mode).next_state
                result = predict(state, command)
                small, small_u = state[:1], command[:1]
                sj = torch.autograd.functional.jacobian(lambda x: predict(x, small_u), small)
                uj = torch.autograd.functional.jacobian(lambda u: predict(small, u), small_u)
                target = torch.tensor(np.stack([transition(s.numpy(), u.numpy(), .04, actuator) for s, u in zip(state, command)]))
                one_loss = (result[:, :4]-target[:, :4]).square().mean()
                model.zero_grad()
                one_loss.backward()
                wg = torch.cat([p.grad.reshape(-1) for p in model.parameters()]).detach().clone()
                imagined, truth, loss = state, state, torch.zeros((), dtype=torch.float64)
                for _ in range(4):
                    imagined = predict(imagined, command)
                    truth = torch.tensor(np.stack([transition(s.numpy(), u.numpy(), .04, actuator) for s, u in zip(truth, command)]))
                    loss = loss+(imagined[:, :4]-truth[:, :4]).square().mean()/4
                model.zero_grad()
                loss.backward()
                rg = torch.cat([p.grad.reshape(-1) for p in model.parameters()]).detach().clone()
                for name, value in zip(storage, (result, sj, uj, wg, loss, rg)):
                    records[name].append(value.detach().numpy())
                if update == 0:
                    hashes.append(sha256_file(checkpoint))
            for name, values in records.items():
                values = np.stack(values)
                maximum = max(maximum, float(np.max(np.abs(values-values[:1]))))
                storage[name].append(values)
            if update == 0:
                optimizer = torch.optim.AdamW(model.parameters(), lr=.001, betas=(.9,.999), eps=1e-8,
                                             weight_decay=1e-6, amsgrad=False, foreach=False, fused=False)
                optimizer.step()
        for name, values in storage.items():
            arrays[actuator+'_'+name] = np.stack(values)
        checkpoint_hashes.append(hashes)
        slopes = []
        for mode in ('low', 'direct', 'lagged'):
            norms = []
            for h in (.0025, .005, .01):
                derivative = torch.autograd.functional.jacobian(lambda u: model.predict(state[:1], u, h, power_mode=mode).next_state, command[:1])[0, :, 0, :]
                norms.append([torch.linalg.matrix_norm(derivative[:2]).item(), torch.linalg.matrix_norm(derivative[2:4]).item()])
            slopes.append([np.polyfit(np.log([.0025,.005,.01]), np.log(np.asarray(norms)[:, j]), 1)[0] for j in (0, 1)])
        arrays[actuator+'_mode_slopes'] = np.asarray(slopes)
        expected = (2, 1) if actuator == 'direct' else (3, 2)
        detected[actuator+'_low_order'] = bool(abs(slopes[0][0]-expected[0]) > .4)
        if actuator == 'lagged':
            detected['lagged_direct_mismatch'] = all(abs(slopes[1][j]-expected[j]) > .4 for j in (0, 1))
            mutated = state[:1].detach().clone()
            mutated[:, 4:] = command[:1]
            try:
                require_same_state(state[:1], mutated)
                detected['w_initialized_from_u'] = False
            except ValueError:
                detected['w_initialized_from_u'] = True
        reference = action_jacobian(.005, actuator)[:4]
        for name, matrix in (('zero', np.zeros_like(reference)), ('sign', -reference), ('columns', reference[:, ::-1])):
            error = np.linalg.norm(matrix-reference)/np.linalg.norm(reference)
            arrays[actuator+'_mutation_'+name] = matrix
            detected[actuator+'_'+name] = bool(error > .5)
    arrays['checkpoint_hashes'] = np.asarray(checkpoint_hashes)
    np.savez_compressed(folder / 'power_identity.npz', **arrays)
    atomic_json(folder / 'mutations.json', {'detected': detected, 'classification': 'Known deliberate software fixtures, not research results'})
    return {'identity_max_abs_error': maximum, 'shared_checkpoint_hashes': checkpoint_hashes, 'mutation_count': len(detected)}


def _slide_evidence(root, folder):
    horizons = np.array([.0025,.005,.01,.02,.04,.08])
    steps = np.array([.0005,.00025,.000125])
    cases = [('direct', np.array([.7,-.4]), np.array([1.9])), ('lagged', np.array([.7,-.4,.3]), np.array([1.9])),
             ('direct', np.array([-.9,1.2]), np.array([-2.])), ('lagged', np.array([-.9,1.2,-.7]), np.array([-2.]))]
    endpoints, reference = [], []
    for actuator, state, command in cases:
        ref = np.stack([transition(state, command, h, actuator) for h in horizons])
        reference.append(np.pad(ref, ((0,0),(0,3-ref.shape[1]))))
    for step in steps:
        at_step = []
        for actuator, state, command in cases:
            model = mujoco.MjModel.from_xml_path(str(root / 'assets' / ('analytic_'+actuator+'.xml')))
            model.opt.timestep = float(step)
            rows = []
            for h in horizons:
                data = mujoco.MjData(model)
                data.qpos[:] = state[:1]
                data.qvel[:] = state[1:2]
                if actuator == 'lagged':
                    data.act[:] = state[2:3]
                data.ctrl[:] = command
                mujoco.mj_forward(model, data)
                mujoco.mj_step(model, data, nstep=to_ns(h)//to_ns(step))
                if abs(data.time-h) > 1e-10 or data.ncon or any(warning.number for warning in data.warning):
                    raise ValueError('Analytic MuJoCo fixture warning/contact/time discrepancy')
                value = np.concatenate((data.qpos.copy(), data.qvel.copy(), data.act.copy()))
                rows.append(np.pad(value, (0,3-len(value))))
            at_step.append(rows)
        endpoints.append(at_step)
    endpoints, reference = np.asarray(endpoints), np.asarray(reference)
    np.savez_compressed(folder / 'slide_refinement.npz', steps_s=steps, horizons_s=horizons, endpoints=endpoints, reference=reference)
    return {'slide_accepted_max_abs_error': float(np.max(np.abs(endpoints[-1]-reference))),
            'slide_step_max_abs_errors': [float(np.max(np.abs(row-reference))) for row in endpoints], 'slide_accepted_step_s': float(steps[-1])}


def run_m10(root):
    cpu_defaults(8001)
    predecessor = parent(root, 'G00')
    files = ['poc/__init__.py', 'poc/__main__.py', 'poc/contracts.py', 'poc/gates.py', 'poc/io.py',
             'poc/execution.py', 'poc/mechanics.py', 'poc/analytic.py', 'poc/clocks.py', 'poc/models/__init__.py',
             'poc/models/common.py', 'poc/models/power.py', 'poc/milestones/__init__.py', 'poc/milestones/m10.py',
             'assets/analytic_direct.xml', 'assets/analytic_lagged.xml', 'tests/test_analytic.py', 'tests/test_power.py', 'tests/test_clocks.py']
    def produce(folder):
        measurements = {**_analytic_evidence(folder), **_power_evidence(folder), **_slide_evidence(root, folder)}
        partitions = [{'id': f'uniform-{step}', 'durations_ns': list(partition(40_000_000, step))}
                      for step in (40_000_000,20_000_000,10_000_000,5_000_000,2_500_000)]
        partitions.append({'id': 'jitter', 'durations_ns': list(jitter_partition((1201,0,0,4)))})
        atomic_json(folder / 'clock_traces.json', {'partitions': partitions, 'fresh_observation_times_ns': [0,40_000_000],
                                               'units': {'time': 'ns', 'command': 'N*m', 'q': 'rad', 'v': 'rad/s'}})
        argv = [sys.executable, '-m', 'pytest', '-q', 'tests/test_analytic.py', 'tests/test_power.py', 'tests/test_clocks.py']
        with (folder / 'stdout.log').open('ab') as output, (folder / 'stderr.log').open('ab') as errors:
            result = subprocess.run(argv, cwd=root, stdout=output, stderr=errors)
        if result.returncode:
            raise ValueError('M10 focused pytest suite failed; full logs preserved')
        budgets = {'analytic_max_abs_error': 1e-12, 'decimal_max_relative_error': 1e-12, 'jacobian_max_relative_error': 1e-10,
                   'composition_max_abs_error': 1e-12, 'identity_max_abs_error': 1e-12, 'slide_accepted_max_abs_error': 1e-12}
        for key, limit in budgets.items():
            if measurements[key] > limit:
                raise ValueError(f'{key}={measurements[key]} exceeds {limit}')
        if not all(__import__('poc.io', fromlist=['read_json']).read_json(folder / 'mutations.json')['detected'].values()):
            raise ValueError('Mechanics mutation detection failed')
        checks = [{'check_id': key, 'status': 'pass', 'reason': 'Independent exact arrays, same-weight audits, mutation fixtures and focused tests passed',
                   'tolerance': None, 'scope': 'M10'} for key in CHECKS]
        return checks, {'measurements': measurements, 'missing_cells': [], 'phase': 'unit', 'robot_experiments_run': False}
    return run_job(root, 'M10', 'unit', files, CHECKS, [predecessor], produce)
