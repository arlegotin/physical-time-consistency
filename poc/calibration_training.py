"""Budget-local, own-solver calibration and exact optimizer-boundary recovery."""
from __future__ import annotations

import copy
import functools
import hashlib
import io
from pathlib import Path
import random
import time

import numpy as np
import torch

from .calibration_data import load_budget
from .calibration_models import ALL_MODEL_IDS, build_model, model_metadata
from .calibration_resources import ResourceTracker, checkpoint_resources
from .calibration_store import CalibrationStore, Receipt
from .io import atomic_bytes, atomic_json, canonical_json, read_json, sha256_file
from .physics_training import _integration_step, optimize_endpoints
from .train import (batch_objective, canonical_tensor_hash, epoch_starts,
                    nominal_convergence, optimizer_for, select_checkpoint, validate_nominal)


def _array_hash(arrays):
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        digest.update(canonical_json({'name': name, 'dtype': str(value.dtype), 'shape': list(value.shape)}))
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _scale_hash(scales):
    return canonical_tensor_hash({'q': scales.q, 'v': scales.v, 'input': scales.v_input})


def _weights(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _packet_payload(contents):
    header, digest, payload = contents.split(b'\n', 2)
    if header != b'PTCO-CAL-PACKET-1' or digest != hashlib.sha256(payload).hexdigest().encode():
        raise ValueError('Calibration packet checksum mismatch')
    return payload


def _decode_packet(contents):
    return torch.load(io.BytesIO(_packet_payload(contents)), weights_only=True, map_location='cpu')


def _packet_exists(path):
    path = Path(path)
    previous = path.with_name(path.stem + '.prev.pt')
    return any(p.exists() for p in (path, previous, path.with_suffix('.complete.json'),
                                    previous.with_suffix('.complete.json')))


def _write_packet(path, packet):
    """Commit payload and checksum together; markers are secondary export records."""
    path = Path(path)
    marker = path.with_suffix('.complete.json')
    if path.exists():
        contents = path.read_bytes()
        try:
            _packet_payload(contents)
        except (ValueError, RuntimeError, EOFError):
            pass  # Keep the previous valid boundary if the current write was torn.
        else:
            committed = {'sha256': hashlib.sha256(contents).hexdigest()}
            if not marker.exists() or read_json(marker) != committed:
                # Preserve the exact interrupted product before subsequent updates.
                archive = path.with_name(path.stem + '.interrupted-' + committed['sha256'] + '.pt')
                if not archive.exists():
                    atomic_bytes(archive, contents)
            previous = path.with_name(path.stem + '.prev.pt')
            atomic_bytes(previous, contents)
            atomic_json(previous.with_suffix('.complete.json'), committed)
    buffer = io.BytesIO()
    torch.save(packet, buffer)
    payload = buffer.getvalue()
    contents = b'PTCO-CAL-PACKET-1\n' + hashlib.sha256(payload).hexdigest().encode() + b'\n' + payload
    atomic_bytes(path, contents)
    atomic_json(marker, {'sha256': sha256_file(path)})


def _read_packet(path):
    path = Path(path)
    for candidate in (path, path.with_name(path.stem + '.prev.pt')):
        try:
            return _decode_packet(candidate.read_bytes())
        except (OSError, ValueError, RuntimeError, EOFError):
            continue
    raise ValueError('No consistent committed calibration optimizer packet')


class CalibrationTrainingSession:
    """Flatten shuffled eight-start windows before batching, including segment tails.

    The cursor always points after a completed optimizer update. Production fitting
    commits once per epoch; callers may save any update boundary for interruption.
    Validation history and its selected tensors share the optimizer commit packet.
    """
    def __init__(self, model, arrays, scales, seed, *, learning_rate):
        metadata = model_metadata(model)
        if metadata['model_id'] in {'physics_fit', 'physics_nominal'}:
            raise ValueError('Calibration neural training requires its registered stateless solver')
        if type(seed) is not int or seed < 0 or next(model.parameters()).dtype != torch.float32:
            raise ValueError('Calibration training requires float32 and an explicit seed')
        self.model, self.arrays, self.scales = model, arrays, scales
        self.optimizer = optimizer_for(model.parameters(), learning_rate)
        self.recipe = {'seed': seed, 'learning_rate': learning_rate, 'model': metadata,
                       'effective_batch': 256, 'loss_mode': 'shared', 'dtype': 'float32',
                       'data_hash': _array_hash(arrays), 'scale_hash': _scale_hash(scales),
                       'initial_tensor_hash': canonical_tensor_hash(model.state_dict())}
        self.rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([5501, seed])))
        self.starts = np.concatenate(epoch_starts(arrays, self.rng))
        self.epoch, self.cursor, self.updates, self.valid_target_count = 1, 0, 0, 0
        self.training_rows, self.validation_rows, self.best_weights = [], [], None
        self.context = None
        self.validated_epoch = 0

    def advance_updates(self, count):
        if type(count) is not int or count < 0:
            raise ValueError('Update count must be a nonnegative integer')
        for _ in range(count):
            if self.cursor == len(self.starts):
                self.epoch += 1
                self.starts = np.concatenate(epoch_starts(self.arrays, self.rng))
                self.cursor = 0
            starts = self.starts[self.cursor:self.cursor + 256]
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            result = batch_objective(self.model, self.arrays, starts, self.scales, 'shared')
            if not bool(torch.isfinite(result['loss'])):
                raise ValueError('Nonfinite calibration supervised loss')
            result['loss'].backward()
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True, foreach=False)
            self.optimizer.step()
            if any(not bool(torch.isfinite(p).all()) for p in self.model.parameters()):
                raise ValueError('Nonfinite calibration optimizer tensors')
            self.cursor += len(starts)
            self.updates += 1
            self.valid_target_count += len(starts)
            self.training_rows.append({'epoch': self.epoch, 'update': self.updates,
                'loss': float(result['loss'].detach()), 'one_step': float(result['one_loss'].detach()),
                'four_step': float(result['four_loss'].detach()), 'one_count': result['one_count'],
                'four_count': result['four_count'], 'gradient_norm': float(norm),
                'valid_target_count': self.valid_target_count})

    def save(self, path):
        checkpoint_resources(lambda: _write_packet(path, {'recipe': self.recipe, 'context': self.context, 'weights': _weights(self.model),
            'optimizer': self.optimizer.state_dict(), 'starts': torch.from_numpy(self.starts.copy()),
            'epoch': self.epoch, 'cursor': self.cursor, 'updates': self.updates,
            'valid_target_count': self.valid_target_count, 'training_rows': self.training_rows,
            'validation_rows': self.validation_rows, 'best_weights': self.best_weights,
            'validated_epoch': self.validated_epoch, 'numpy_rng': self.rng.bit_generator.state,
            'torch_rng': torch.get_rng_state(), 'python_rng': random.getstate()}))

    def restore(self, path):
        saved = _read_packet(path)
        if saved['recipe'] != self.recipe or saved['context'] != self.context:
            raise ValueError('Changed calibration recipe/inputs/solver identity cannot resume')
        starts = saved['starts'].numpy()
        count, steps = self.arrays['command'].shape[:2]
        expected = {(i, j) for i in range(count) for j in range(steps)}
        if (starts.shape != (count*steps, 2) or set(map(tuple, starts)) != expected
                or not 0 <= saved['cursor'] <= len(starts) or saved['epoch'] < 1
                or saved['valid_target_count'] != (saved['epoch']-1)*len(starts)+saved['cursor']):
            raise ValueError('Invalid calibration sampler checkpoint')
        self.model.load_state_dict(saved['weights'], strict=True)
        self.optimizer.load_state_dict(saved['optimizer'])
        self.starts = starts.copy()
        for key in ('epoch', 'cursor', 'updates', 'valid_target_count', 'training_rows',
                    'validation_rows', 'best_weights', 'validated_epoch'):
            setattr(self, key, saved[key])
        self.rng.bit_generator.state = saved['numpy_rng']
        torch.set_rng_state(saved['torch_rng'])
        random.setstate(saved['python_rng'])


def _context(store, session_receipt, budget_s, model_id, initialization_id):
    _, recipe_manifest = store.read(store.recipe_receipt)
    recipe = recipe_manifest['summary']['recipe']
    if recipe != store.recipe:
        raise ValueError('Changed study recipe cannot fit or load selected weights')
    # A long-lived store must not bypass source/runtime checks on a resumed call.
    from .calibration_study import study_bindings
    current = study_bindings(Path(store.bindings['root']), recipe)
    if any(store.bindings.get(k) != v for k, v in current.items()):
        raise ValueError('Changed study source/runtime bindings cannot resume')
    _, collected = store.read(session_receipt)
    if collected['request'].get('recipe_receipt') != store.recipe_receipt:
        raise ValueError('Calibration data belong to another recipe')
    train, valid, scales, account = load_budget(session_receipt, budget_s, store)
    coordinate = {'session_id': account['session_id'], 'budget_s': budget_s,
                  'model_id': model_id, 'initialization_id': initialization_id}
    if model_id == 'physics_nominal':
        if initialization_id is not None or not any(c['session_id'] == account['session_id'] and
                                                   c['budget_s'] == budget_s for c in store.coordinates):
            raise ValueError('Nominal reference coordinate is outside prepared membership')
    elif coordinate not in store.coordinates or (model_id == 'physics_fit' and initialization_id is not None):
        raise ValueError('Fit coordinate or initialization is outside prepared membership')
    request = {**coordinate, 'session_receipt': session_receipt, 'recipe_receipt': store.recipe_receipt,
               'phase': recipe['phase'], 'training': recipe['training'], 'account': account,
               'training_data_hash': _array_hash(train), 'validation_data_hash': _array_hash(valid),
               'scale_hash': _scale_hash(scales), 'model': model_metadata(build_model(model_id, scales, 0))}
    return request, train, valid, scales


def _limits(store):
    training = store.recipe['training']
    maximum, physical = training['max_epochs_per_rate'], training.get('physical_epochs', 100)
    if (type(maximum) is not int or maximum <= 0 or maximum % 5 or maximum > 1000
            or type(physical) is not int or physical <= 0 or physical % 5):
        raise ValueError('Invalid calibration epoch recipe')
    if store.recipe['phase'] != 'unit' and (maximum != 1000 or physical != 100):
        raise ValueError('Reduced training controls require an explicit unit fixture')
    return maximum, physical


def _rate_attempt(folder, request, train, valid, scales, seed, rate, maximum):
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / 'attempt.json'
    if result_path.exists():
        result = read_json(result_path)
        if result['request'] != request or result['learning_rate'] != rate:
            raise ValueError('Changed rate attempt recipe cannot resume')
        return result
    with ResourceTracker(folder / 'resources.json') as tracker:
        result = _run_rate_attempt(folder, request, train, valid, scales, seed, rate, maximum)
    result['resources'] = tracker.summary()
    atomic_json(result_path, result)
    return result


def _run_rate_attempt(folder, request, train, valid, scales, seed, rate, maximum):
    fit = CalibrationTrainingSession(build_model(request['model_id'], scales, seed), train, scales,
                                     seed, learning_rate=rate)
    fit.context = request
    resume = folder / 'resume.pt'
    if _packet_exists(resume):
        fit.restore(resume)
    failure = None
    stop = False
    try:
        while True:
            while fit.cursor < len(fit.starts):
                fit.advance_updates(1)
            epoch = fit.epoch
            if fit.validated_epoch < epoch:
                if epoch % 5 == 0:
                    row = {'epoch': epoch, 'learning_rate': rate,
                           **validate_nominal(fit.model, valid, scales, 'shared')}
                    if not all(np.isfinite(row[k]) for k in ('objective', 'one_step', 'four_step')):
                        raise ValueError('Nonfinite calibration nominal validation')
                    fit.validation_rows.append(row)
                    if select_checkpoint(fit.validation_rows) == row:
                        fit.best_weights = _weights(fit.model)
                fit.validated_epoch = epoch
                fit.save(resume)
            stop = epoch % 100 == 0 and nominal_convergence(fit.validation_rows, epoch)['stabilized']
            if stop or epoch >= maximum:
                break
            # Enter the next epoch only through the resumable update method.
            fit.advance_updates(1)
    except (ValueError, RuntimeError, FloatingPointError) as error:
        failure = {'error_type': type(error).__name__, 'error': str(error),
                   'failure_coordinate': {'epoch': fit.epoch, 'cursor': fit.cursor,
                                          'optimizer_updates': fit.updates},
                   'training_history': fit.training_rows}
        # Diagnostic tensors are retained but never mistaken for resumable state.
        _write_packet(folder / 'failure-diagnostic.pt', {'request': request,
            'resumable': False, 'weights': _weights(fit.model),
            'optimizer': fit.optimizer.state_dict(), 'failure': failure})
    result = {'request': request, 'learning_rate': rate, 'status': 'failed' if failure else 'finite',
              'epochs_completed': fit.validated_epoch, 'optimizer_updates': fit.updates,
              'valid_target_exposures': fit.valid_target_count,
              'four_step_start_exposures': sum(r['four_count'] for r in fit.training_rows),
              'validation_history': fit.validation_rows, 'plateau': bool(stop),
              'ceiling_hit': not failure and fit.validated_epoch >= maximum and not stop,
              'optimization_limited': not failure and fit.validated_epoch >= maximum and not stop,
              **(failure or {})}
    if fit.best_weights is not None:
        _write_packet(folder / 'selected.pt', {'request': request, 'weights': fit.best_weights,
                                             'canonical_tensor_sha256': canonical_tensor_hash(fit.best_weights)})
        result['selected'] = select_checkpoint(fit.validation_rows)
    else:
        result['selected'] = None
    return result


def _endpoints(arrays):
    return {'state': arrays['state'][:, :-1].reshape(-1, 6),
            'target': arrays['state'][:, 1:].reshape(-1, 6),
            'command': arrays['command'].reshape(-1, 2),
            'h_s': arrays['duration_ns'].reshape(-1, 1)/1e9}


def _audit_physical(model, arrays):
    """Qualify the scored float64 field using the original observed endpoints."""
    audited = copy.deepcopy(model).double().eval().requires_grad_(False)
    validation = {k: torch.as_tensor(v, dtype=torch.float64).detach().clone()
                  for k, v in _endpoints(arrays).items()}
    return _integration_step(audited, validation)


def fit_model(store: CalibrationStore, session_receipt: Receipt, model_id: str,
              budget_s: int, initialization_id: int | None) -> Receipt:
    if model_id not in ALL_MODEL_IDS:
        raise ValueError('Unknown fitted calibration model')
    request, train, valid, scales = _context(store, session_receipt, budget_s, model_id, initialization_id)
    maximum, physical_epochs = _limits(store)
    seed = 2026101200 + 2*request['session_id'] + (initialization_id or 0)
    def produce(folder):
        base = {'phase': request['phase'], 'empirical_claims_allowed': store.recipe['empirical_claims_allowed'],
                'selection_kind': 'fitted', 'coordinate': {k: request[k] for k in
                    ('session_id', 'budget_s', 'model_id', 'initialization_id')},
                'account': request['account'], 'model': request['model'], 'seed': seed,
                'execution': store.execution_settings, 'qualification': 'finite nominal fit; no historical threshold'}
        if model_id == 'physics_fit':
            model = build_model(model_id, scales, seed)
            resume_path = folder / 'physical-resume.pt'
            saved = _read_packet(resume_path) if _packet_exists(resume_path) else None
            if saved is not None and saved['request'] != request:
                raise ValueError('Changed physical optimizer identity cannot resume')
            audits = {'solver_audit_dtype': 'float64', 'optimizer_audit_dtype': 'float32'}
            try:
                prior_step, prior_audits = _audit_physical(model, valid)
                audits.update({'prior_integration_step_s': prior_step,
                               'solver_audits_before_fit': prior_audits})
                result = optimize_endpoints(model, _endpoints(train), _endpoints(valid),
                    epochs=physical_epochs, batch_size=256, seed=seed,
                    resume=saved['resume'] if saved else None,
                    on_epoch=lambda resume: checkpoint_resources(lambda: _write_packet(resume_path, {'request': request, 'resume': resume})))
                if not np.isfinite(result['best_validation_loss']):
                    raise ValueError('Nonfinite physical nominal validation')
                selected_step, selected_audits = _audit_physical(model, valid)
                audits.update({'selected_field_integration_step_s': selected_step,
                               'solver_audits_selected_field': selected_audits,
                               'optimizer_solver_audits_before_fit': result['solver_audits_before_fit'],
                               'optimizer_solver_audits_selected_field': result['solver_audits_selected_field']})
                if min(prior_step, selected_step) < result['integration_step_s']:
                    raise ValueError('Float64 physical qualification requires a finer training solver; '
                                     'the trained solver is preserved and this fit fails qualification')
                selected = {'epoch': result['best_epoch'], 'objective': result['best_validation_loss']}
                _write_packet(folder / 'weights.pt', {'request': request, 'weights': result['best_state'],
                    'canonical_tensor_sha256': canonical_tensor_hash(result['best_state'])})
                details = {k: v for k, v in result.items() if k not in
                           {'best_state', 'last_state', 'optimizer_state', 'rng_state'}}
                return {**base, **details, **audits, 'fit_status': 'finite', 'selected': selected,
                        'canonical_tensor_sha256': canonical_tensor_hash(result['best_state']),
                        'attempts': [{'status': 'finite', 'learning_rate': .01,
                                      'epochs_completed': physical_epochs}], 'optimization_limited': False}
            except (ValueError, RuntimeError, FloatingPointError) as error:
                return {**base, **audits, 'fit_status': 'failed', 'selected': None,
                        'attempts': [{'status': 'failed', 'learning_rate': .01,
                                     'error_type': type(error).__name__, 'error': str(error)}]}
        attempts = [_rate_attempt(folder / f'rate-{index}', request, train, valid, scales,
                                  seed, rate, maximum)
                    for index, rate in enumerate(request['training']['learning_rates'])]
        candidates = [dict(a['selected'], rate_index=i) for i, a in enumerate(attempts) if a['selected'] is not None]
        if not candidates:
            return {**base, 'attempts': attempts, 'fit_status': 'failed', 'selected': None,
                    'optimization_limited': False}
        selected = select_checkpoint(candidates)
        packet = _read_packet(folder / f'rate-{selected["rate_index"]}' / 'selected.pt')
        _write_packet(folder / 'weights.pt', packet)
        return {**base, 'attempts': attempts, 'fit_status': 'finite', 'selected': selected,
                'canonical_tensor_sha256': packet['canonical_tensor_sha256'],
                'optimization_limited': any(a['optimization_limited'] for a in attempts)}
    return store.produce('fit', request, produce)


def nominal_reference(store: CalibrationStore, session_receipt: Receipt, budget_s: int) -> Receipt:
    request, _, _, scales = _context(store, session_receipt, budget_s, 'physics_nominal', None)
    def produce(folder):
        model = build_model('physics_nominal', scales, 0)
        summary = {'phase': request['phase'], 'empirical_claims_allowed': store.recipe['empirical_claims_allowed'],
                   'selection_kind': 'unfitted_reference', 'model': model_metadata(model),
                   'account': request['account'], 'scale_hash': _scale_hash(scales),
                   'canonical_tensor_sha256': canonical_tensor_hash(model.state_dict()),
                   'execution': store.execution_settings}
        atomic_json(folder / 'construction.json', summary)
        return summary
    return store.produce('nominal', request, produce)


def load_selected(store: CalibrationStore, fit_receipt: Receipt) -> torch.nn.Module:
    folder, manifest = store.read(fit_receipt)
    if manifest['stage'] not in {'fit', 'nominal'}:
        raise ValueError('Expected fitted selection or unfitted nominal reference')
    request, summary = manifest['request'], manifest['summary']
    expected, _, _, scales = _context(store, request['session_receipt'], request['budget_s'],
                                      request['model_id'], request['initialization_id'])
    if (expected != request or summary['model'] != request['model']
            or summary['account'] != request['account'] or summary['phase'] != request['phase']):
        raise ValueError('Selected model data/scale/recipe/solver identity mismatch')
    model = build_model(request['model_id'], scales, summary.get('seed', 0))
    if manifest['stage'] == 'fit':
        if summary['selection_kind'] != 'fitted' or summary['fit_status'] != 'finite':
            raise ValueError('Failed calibration fit has no selected checkpoint')
        packet = _read_packet(folder / 'weights.pt')
        if (packet['request'] != request or canonical_tensor_hash(packet['weights']) != summary['canonical_tensor_sha256']
                or packet['canonical_tensor_sha256'] != summary['canonical_tensor_sha256']):
            raise ValueError('Selected checkpoint canonical identity mismatch')
        for name, value in model.named_buffers():
            if name.rsplit('.', 1)[-1] in {'scale_q', 'scale_v', 'v_input'}:
                if name not in packet['weights'] or not torch.equal(packet['weights'][name], value):
                    raise ValueError('Selected checkpoint normalization mismatch')
        model.load_state_dict(packet['weights'], strict=True)
    elif summary['selection_kind'] != 'unfitted_reference':
        raise ValueError('Nominal construction cannot count as a completed fit')
    if canonical_tensor_hash(model.state_dict()) != summary['canonical_tensor_sha256']:
        raise ValueError('Selected canonical tensors mismatch')
    # State dictionaries carry normalization buffers: reject full-budget substitutions.
    for name, value in (('scale_q', scales.q), ('scale_v', scales.v), ('v_input', scales.v_input)):
        if not torch.equal(getattr(model, name), value):
            raise ValueError('Selected checkpoint normalization mismatch')
    if request['model_id'] == 'physics_fit':
        model.predict = functools.partial(model.predict, integration_step_s=summary['evaluation_step_s'])
        model.calibration_integration_step_s = summary['evaluation_step_s']
    return model.double().eval()
