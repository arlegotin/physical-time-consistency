"""Additive nominal-only fitting for the exploratory affine-flow refinement."""
from __future__ import annotations

import argparse
import io
import os
from pathlib import Path
import time

for _variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_variable] = '1'

import numpy as np
import torch

from .affine_flow import AffineFlow
from .io import append_event, atomic_bytes, atomic_json, job_key, read_json, safe_path, sha256_file, utc_now
from .models.common import Scales
from .train import (TrainingSession, _write_training_csv, analytic_control_arrays, array_scales,
                    batch_objective, canonical_tensor_hash, constant_velocity_error, epoch_starts,
                    nominal_convergence, restore_selection, save_selection, select_checkpoint, validate_nominal)

PROTOCOL = 'PTCO-AFF-0.1'
FREEZE = 'config/affine_refinement_freeze.json'
SPEC = 'docs/superpowers/specs/2026-10-08-affine-flow-refinement.md'
ACTUATORS = ('direct', 'lagged')
SEED = 9000


def build_candidate(actuator, scales, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return AffineFlow(actuator, scales)


def verify_bindings(root, hashes):
    for name, expected in hashes.items():
        if sha256_file(safe_path(root, name)) != expected:
            raise ValueError('Affine experiment input/source changed: '+name)


def validate_nominal_roles(train, validation, actor):
    if train.get('role') != 'train' or validation.get('role') != 'validation':
        raise ValueError('Nominal fit/selection role must be train/validation')
    if any(item.get('actuator') != actor or item.get('condition') != 'A' for item in (train, validation)):
        raise ValueError('Nominal data actuator/condition mismatch')


def _sources():
    from .training_jobs import SOURCES
    return sorted(set(SOURCES)|{'poc/affine_flow.py', 'poc/affine_experiment.py'})


def prepare(root):
    root = Path(root).resolve()
    if (root/FREEZE).exists():
        return load_freeze(root)
    authority = read_json(root/'config/authorization.json')
    if not authority['local_compute_authorized'] or authority['paid_services_authorized']:
        raise ValueError('Authorized local CPU computation required')
    diagnostic = read_json(root/'config/shared_state_diagnostic_freeze.json')
    pilot = read_json(root/'config/efficiency_pilot_freeze.json')
    verify_bindings(root, {**diagnostic['source_hashes'], **diagnostic['input_hashes']})
    inputs = {name:sha256_file(root/name) for name in
              (SPEC, 'config/authorization.json', 'config/numerical.json', 'config/planner.json',
               'config/shared_state_diagnostic_freeze.json', 'config/efficiency_pilot_freeze.json',
               'locks/environment.json', 'locks/wheels.json', 'assets/two_link_direct.xml', 'assets/two_link_lagged.xml')}
    data = {}
    context = pilot['nominal_context']
    for actor in ACTUATORS:
        data[actor] = {}
        for role in ('train', 'validation'):
            name = f'data/{actor}/A/{role}'
            original, preserved = root/name, root/context/name
            for filename in ('manifest.json', 'trajectories.npz'):
                if sha256_file(original/filename) != sha256_file(preserved/filename):
                    raise ValueError('Historical corpus bytes differ from original')
                for path in (original/filename, preserved/filename):
                    inputs[str(path.relative_to(root))] = sha256_file(path)
            data[actor][role] = str((preserved/'manifest.json').relative_to(root))
        norm = root/context/f'data/{actor}/A/normalization.json'
        data[actor]['normalization'] = str(norm.relative_to(root))
        inputs[str(norm.relative_to(root))] = sha256_file(norm)
        validate_nominal_roles(read_json(root/data[actor]['train']), read_json(root/data[actor]['validation']), actor)
    requests = {}
    for actor in ACTUATORS:
        name = f'config/affine_nominal_requests/{actor}.json'
        atomic_json(root/name, dict(protocol_id=PROTOCOL, actuator=actor, seed=SEED,
                                   model_id='affine_flow', condition='A', loss_mode='shared'))
        requests[actor] = name
        inputs[name] = sha256_file(root/name)
    freeze = dict(schema_version=1, protocol_id=PROTOCOL, phase='exploratory', seed=SEED,
                  created_utc=utc_now(), model_id='affine_flow', parameter_count=18182,
                  loss_mode='shared', learning_rates=[.001, .0003], epoch_block=100,
                  data=data, comparators=diagnostic['selections'], nominal_requests=requests,
                  source_hashes={name:sha256_file(root/name) for name in _sources()}, input_hashes=inputs,
                  evidence_role='development; no independent confirmation', alpha_spent=0,
                  work_context='study/affine-flow', resource_projection=dict(
                      new_disk_bytes_estimate=350*1024**2, additional_control_episodes=80,
                      runtime_projection='Measure nominal blocks and episode throughput; no artificial cap'))
    atomic_json(root/FREEZE, freeze)
    return freeze


def load_freeze(root):
    freeze = read_json(root/FREEZE)
    if (freeze['protocol_id'] != PROTOCOL or freeze['phase'] != 'exploratory'
            or freeze['seed'] != SEED or freeze['model_id'] != 'affine_flow'
            or freeze['loss_mode'] != 'shared' or freeze['learning_rates'] != [.001, .0003]
            or freeze['epoch_block'] != 100 or freeze['parameter_count'] != 18182):
        raise ValueError('Changed affine nominal design')
    verify_bindings(root, {**freeze['source_hashes'], **freeze['input_hashes']})
    return freeze


def nominal_configuration(root, actor, freeze):
    if actor not in ACTUATORS:
        raise ValueError('Unknown nominal actuator')
    relative = freeze['nominal_requests'][actor]
    request = read_json(safe_path(root, relative))
    if request.get('actuator') != actor or request.get('protocol_id') != PROTOCOL:
        raise ValueError('Nominal request actuator/protocol mismatch')
    return {FREEZE:freeze, relative:request}


def load_nominal_arrays(root, actor, freeze, audit_path):
    from .dataset import load_corpus
    paths = freeze['data'][actor]
    validate_nominal_roles(read_json(root/paths['train']), read_json(root/paths['validation']), actor)
    train = load_corpus(root/paths['train'], 'exploratory', 'train', 'training', audit_path=audit_path)
    validation = load_corpus(root/paths['validation'], 'exploratory', 'validation', 'selection', audit_path=audit_path)
    normalization = read_json(root/paths['normalization'])
    scales = Scales(*(torch.tensor(normalization[key], dtype=torch.float32) for key in
                     ('scale_q', 'scale_v', 'velocity_input_rms')))
    independent = array_scales(train)
    for a, b in ((scales.q, independent.q), (scales.v, independent.v), (scales.v_input, independent.v_input)):
        if not torch.equal(a, b):
            raise ValueError('Frozen normalization differs from train-only scales')
    return train, validation, scales, normalization


def train_block(session, validation, folder, end_epoch):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rows, best, best_weights = restore_selection(folder, session.epoch)
    if type(end_epoch) is not int or end_epoch < session.epoch:
        raise ValueError('Training block cannot go backward')
    while session.epoch <= end_epoch:
        epoch = session.epoch
        wall, cpu = time.perf_counter(), time.process_time()
        if session.window_position < len(session.windows):
            session.advance_epoch(folder/'resume.pt')
        if epoch % 5 == 0 and (not rows or rows[-1]['epoch'] < epoch):
            row = {'epoch':epoch, **validate_nominal(session.model, validation, session.scales, session.loss_mode)}
            rows.append(row)
            chosen = select_checkpoint(rows)
            if best is None or chosen['epoch'] != best['epoch']:
                best, best_weights = chosen, {n:v.detach().clone() for n,v in session.model.state_dict().items()}
            save_selection(folder, rows, best_weights)
            atomic_json(folder/'progress.json', dict(protocol_id=PROTOCOL, state='nominal_training', epoch=epoch,
                       selected_epoch=best['epoch'], selected_objective=best['objective'],
                       valid_target_count=session.valid_target_count, updates=session.updates, utc=utc_now()))
            print(f'{session.model.actuator} epoch {epoch}, nominal {row["objective"]:.6g}, selected {best["epoch"]}', flush=True)
        append_event(folder/'events.jsonl', dict(event='epoch_boundary', epoch=epoch,
                     valid_target_count=session.valid_target_count, cpu_seconds=time.process_time()-cpu,
                     wall_seconds=time.perf_counter()-wall, utc=utc_now()))
        if epoch == end_epoch:
            break
        session.epoch += 1
        session.windows, session.window_position = epoch_starts(session.arrays, session.rng), 0
    if best is None:
        raise ValueError('No completed nominal validation checkpoint')
    cv = constant_velocity_error(validation, session.scales)
    convergence = nominal_convergence(rows, end_epoch)
    result = dict(selected=best, convergence=convergence, constant_velocity_error=cv,
                  qualified=bool(best['one_step'] < .5*cv and convergence['stabilized']),
                  nominal_learning=bool(best['one_step'] < .5*cv), epochs=end_epoch,
                  valid_target_count=session.valid_target_count, updates=session.updates,
                  canonical_tensor_sha256=canonical_tensor_hash(best_weights))
    _write_training_csv(folder/'training.csv', session.training_rows, session.update_timing)
    _write_training_csv(folder/'validation.csv', rows)
    atomic_json(folder/'selection.json', best)
    atomic_json(folder/'block.json', result)
    return result


def learning_controls(actor, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'controls.json').exists():
        value = read_json(folder/'controls.json')
        if value['tiny_loss'] >= .001 or value['analytic_validation'] >= .05:
            raise ValueError('Stored analytic learning control did not pass')
        return value
    tiny = analytic_control_arrays(actor, 'tiny')
    scales = array_scales(tiny)
    model = build_candidate(actor, scales, 8100)
    session = TrainingSession(model, tiny, scales, 8100, effective_batch=16)
    starts = np.array([[i,0] for i in range(16)], dtype=np.int64)
    loss = float(batch_objective(model, tiny, starts, scales)['loss'].detach())
    while loss >= .001 and session.updates < 5000:
        session.advance_updates(1)
        loss = float(batch_objective(model, tiny, starts, scales)['loss'].detach())
    np.savez_compressed(folder/'tiny_arrays.npz', **tiny)
    _write_training_csv(folder/'tiny_training.csv', session.training_rows, session.update_timing)
    session.save(folder/'tiny_resume.pt')
    if loss >= .001:
        raise ValueError('Analytic tiny-fit failed; diagnose optimizer/decoder before robot training')
    train, validation = analytic_control_arrays(actor, 'train'), analytic_control_arrays(actor, 'validation')
    norm = array_scales(train)
    full = TrainingSession(build_candidate(actor, norm, 8100), train, norm, 8100)
    result = train_block(full, validation, folder/'analytic', 100)
    if result['selected']['one_step'] >= .05:
        raise ValueError('Analytic nominal learning control failed')
    np.savez_compressed(folder/'analytic_train.npz', **train)
    np.savez_compressed(folder/'analytic_validation.npz', **validation)
    value = dict(tiny_loss=loss, tiny_updates=session.updates,
                 analytic_validation=result['selected']['one_step'], robot_result=False)
    atomic_json(folder/'controls.json', value)
    return value


def fit_actor(root, actor):
    from .execution import cpu_defaults, run_job
    root = Path(root).resolve()
    freeze = load_freeze(root)
    cpu_defaults(SEED)
    files = sorted(set(freeze['source_hashes'])|set(freeze['input_hashes'])|{FREEZE})
    expected = ('affine.nominal_accounting', 'affine.nominal_selection', 'affine.learning_controls')
    work = root/freeze['work_context']/job_key(dict(freeze_sha256=sha256_file(root/FREEZE), actuator=actor))
    work.mkdir(parents=True, exist_ok=True)

    def produce(folder):
        atomic_json(folder/'request.json', dict(freeze_sha256=sha256_file(root/FREEZE), actor=actor, protocol_id=PROTOCOL))
        controls = learning_controls(actor, work/'controls')
        arrays, validation, scales, normalization = load_nominal_arrays(root, actor, freeze, folder/'data_access.jsonl')
        result, chosen_folder = None, None
        for rate in freeze['learning_rates']:
            output = work/('lr'+str(rate).replace('.', '_'))
            output.mkdir(exist_ok=True)
            model = build_candidate(actor, scales, SEED)
            session = TrainingSession(model, arrays, scales, SEED, 'shared', learning_rate=rate)
            if (output/'resume.complete.json').exists() or (output/'resume.prev.complete.json').exists():
                session.restore_latest(output/'resume.pt')
            end = max(100, ((session.epoch+99)//100)*100)
            while True:
                result = train_block(session, validation, output, end)
                if result['qualified']:
                    chosen_folder = output
                    break
                if result['convergence']['stabilized']:
                    break  # Nominal plateau without sufficient learning gets the other prespecified rate.
                end += 100
            if chosen_folder is not None:
                break
        if chosen_folder is None:
            raise ValueError('Both nominal repair rates plateaued without qualification; diagnosed optimization limit, not scientific rejection')
        expected_count = result['epochs']*arrays['command'].shape[0]*arrays['command'].shape[1]
        if result['valid_target_count'] != expected_count:
            raise ValueError('Dataset-equivalent target exposure mismatch')
        # Final export is immutable; resumable attempts and their own hashes remain in the work context.
        for name in ('weights.pt', 'selection.json', 'validation.csv', 'training.csv', 'resume.pt',
                     'resume.rng.json', 'resume.complete.json', 'selection.pt', 'selection.complete.json'):
            atomic_bytes(folder/name, (chosen_folder/name).read_bytes())
        atomic_json(folder/'normalization.json', normalization)
        atomic_json(folder/'learning_controls.json', controls)
        work_files = {str(p.relative_to(root)):sha256_file(p) for p in work.rglob('*') if p.is_file()}
        atomic_json(folder/'attempt_inventory.json', work_files)
        verify_bindings(root, {**freeze['source_hashes'], **freeze['input_hashes']})
        summary = dict(protocol_id=PROTOCOL, actuator=actor, seed=SEED, phase='exploratory',
                       validation=result['selected'], qualification=result, learning_controls=controls,
                       model_id='affine_flow', parameter_count=18182, missing_cells=[],
                       work_context=str(work.relative_to(root)), training_updates=result['updates'],
                       model_hashes={'canonical_tensor_sha256':result['canonical_tensor_sha256']},
                       resource_projection=freeze['resource_projection'])
        checks = [dict(check_id=name, status='pass', scope=actor, tolerance=None,
                       reason='Qualified nominal fit, analytic controls, complete targets and nominal-only selection') for name in expected]
        return checks, summary

    record = run_job(root, 'M30', 'exploratory', files, expected, [], produce,
                     config_keys={**nominal_configuration(root, actor, freeze),
                                  'config/base.json':read_json(root/'config/base.json')},
                     argv=['python', '-m', 'poc.affine_experiment', 'train', '--actuator', actor])
    if record['completion_status'] != 'complete':
        raise ValueError('Affine nominal fit failed: '+read_json(root/f'evidence/M30/{record["run_id"]}/summary.json').get('failure','unknown'))
    if read_json(root/f'evidence/M30/{record["run_id"]}/summary.json')['actuator'] != actor:
        raise ValueError('Completed fit actuator differs from requested actuator')
    pointer = dict(protocol_id=PROTOCOL, actuator=actor, model_manifest=f'evidence/M30/{record["run_id"]}/manifest.json',
                   manifest_sha256=sha256_file(root/f'evidence/M30/{record["run_id"]}/manifest.json'),
                   checkpoint_sha256=sha256_file(root/f'evidence/M30/{record["run_id"]}/weights.pt'))
    atomic_json(root/f'reports/affine_refinement_{actor}_selection.json', pointer)
    return pointer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'train', 'status'))
    parser.add_argument('--actuator', choices=ACTUATORS)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == 'prepare':
        value = prepare(root)
        print(dict(protocol_id=PROTOCOL, state='frozen_nominal_recipe', expected_fits=2, parameter_count=value['parameter_count']))
    elif args.command == 'train':
        for actor in (ACTUATORS if args.actuator is None else (args.actuator,)):
            print(fit_actor(root, actor), flush=True)
    else:
        freeze = load_freeze(root)
        for actor in ACTUATORS:
            pointer = root/f'reports/affine_refinement_{actor}_selection.json'
            work = root/freeze['work_context']/job_key(dict(freeze_sha256=sha256_file(root/FREEZE), actuator=actor))
            value = read_json(pointer) if pointer.exists() else dict(state='not_complete',
                    progress=[read_json(p) for p in sorted(work.glob('lr*/progress.json'))])
            print(actor, value)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
