"""Frozen shared-state drift, command-response and composition diagnosis."""
from __future__ import annotations

import argparse
import io
import os
from pathlib import Path

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'

import numpy as np
import torch

from .io import atomic_bytes, atomic_json, read_json, safe_path, sha256_file, utc_now

PROTOCOL = 'PTCO-DIAG-0.1'
ACTUATORS = ('direct', 'lagged')
MODELS = ('power_correct', 'mlp_h', 'node_generic', 'node_structured')
HORIZONS = (5_000_000, 10_000_000, 40_000_000)
FREEZE = 'config/shared_state_diagnostic_freeze.json'
PILOT_FREEZE = 'config/efficiency_pilot_freeze.json'
SPEC = 'docs/superpowers/specs/2026-10-08-shared-state-diagnostic.md'
PLAN = 'docs/superpowers/plans/2026-10-08-shared-state-diagnostic.md'
PROBE_KEYS = ('states', 'commands', 'time_ns', 'strata', 'trajectory_ids', 'anchor_ids', 'probe_ids')


def decompose(model_zero, model_command, truth_zero, truth_command, composed):
    arrays = [np.asarray(x, dtype=np.float64) for x in
              (model_zero, model_command, truth_zero, truth_command, composed)]
    if (arrays[0].ndim < 2 or arrays[0].shape[-1] not in (4, 6)
            or any(x.shape != arrays[0].shape or not np.isfinite(x).all() for x in arrays)):
        raise ValueError('Complete matching finite endpoint vectors required')
    mz, mu, tz, tu, split = arrays
    return dict(drift_error=mz-tz, command_error=(mu-mz)-(tu-tz), total_error=mu-tu,
                composition_discrepancy=split-mu, composed_error=split-tu)


def model_endpoints(model, states, commands, h_ns):
    if type(h_ns) is not int or h_ns <= 0 or h_ns % 2:
        raise ValueError('Positive even integer nanosecond horizon required')
    state = torch.tensor(np.asarray(states), dtype=torch.float64)
    command = torch.tensor(np.asarray(commands), dtype=torch.float64)
    original = state.clone()
    values = {}
    with torch.no_grad():
        for key, branch in (('zero', torch.zeros_like(command)), ('command', command)):
            output = model.predict(state.clone(), branch.clone(), h_ns/1e9)
            if any(status != 'ok' for status in output.status):
                raise ValueError('Model endpoint failure: '+str(output.status))
            values[key] = output.next_state.numpy().copy()
        half = model.predict(state.clone(), command.clone(), h_ns/2e9)
        split = model.predict(half.next_state.clone(), command.clone(), h_ns/2e9)
        if any(status != 'ok' for output in (half, split) for status in output.status):
            raise ValueError('Model composed endpoint failure')
        values['composed'] = split.next_state.numpy().copy()
    if not torch.equal(original, state):
        raise ValueError('New-command branch modified its initial state')
    return values


def select_probes(library, count=64):
    if type(count) is not int or count <= 0 or set(PROBE_KEYS)-set(library):
        raise ValueError('Declared positive probe count and all coordinates required')
    size = len(library['states'])
    if any(len(library[name]) != size for name in PROBE_KEYS):
        raise ValueError('Truncated probe coordinates')
    indices = []
    for stratum in ('cold', 'warm'):
        selected = np.flatnonzero(np.asarray(library['strata']) == stratum)[:count]
        if len(selected) != count:
            raise ValueError('Missing required '+stratum+' probes')
        indices.extend(selected)
    output = {key:np.asarray(library[key])[indices].copy() for key in PROBE_KEYS}
    if (not np.array_equal(output['trajectory_ids'][:count], output['trajectory_ids'][count:])
            or len(np.unique(output['trajectory_ids'][:count])) != count
            or len(np.unique(output['probe_ids'])) != 2*count
            or np.any(output['time_ns'][:count]) or np.any(output['time_ns'][count:] < 320_000_000)
            or not np.isfinite(output['states']).all() or not np.isfinite(output['commands']).all()):
        raise ValueError('Probe identity, finite state or physical-time stratum mismatch')
    return output


def _sources():
    from .control_jobs import SOURCES
    return sorted(set(SOURCES) | {'poc/shared_state_diagnostic.py', 'poc/dataset.py', 'poc/probes.py',
                                 'poc/generate.py', 'poc/oracle_rigidbody.py', 'poc/efficient_power.py'})


def verify_hashes(root, hashes):
    for name, expected in hashes.items():
        if sha256_file(safe_path(root, name)) != expected:
            raise ValueError('Diagnostic source/input changed: '+name)


def development_manifest(root, actor, context):
    """Use the qualified historical context without changing the corpus bytes."""
    relative = f'data/{actor}/A/development'
    source = safe_path(root,context+'/'+relative)
    original = safe_path(root,relative)
    if any(sha256_file(source/name) != sha256_file(original/name)
           for name in ('manifest.json','trajectories.npz')):
        raise ValueError('Historical development corpus must be byte-identical to the original')
    return source/'manifest.json'


def prepare(root):
    """Select coordinates/checkpoints without evaluating any model outputs."""
    from .dataset import load_corpus
    from .probes import make_probe_library
    root = Path(root).resolve()
    if (root/FREEZE).exists():
        freeze = read_json(root/FREEZE)
        verify_hashes(root, {**freeze['source_hashes'], **freeze['input_hashes']})
        return freeze
    authority = read_json(root/'config/authorization.json')
    if authority['local_compute_authorized'] is not True or authority['paid_services_authorized'] is not False:
        raise ValueError('Existing authorized local compute required')
    pilot = read_json(root/PILOT_FREEZE)
    if pilot['protocol_id'] != 'PTCO-EFF-0.1' or pilot['nominal_seed'] != 9000:
        raise ValueError('Wrong selected development model context')
    inputs = [PILOT_FREEZE, SPEC, 'config/authorization.json', 'config/numerical.json',
              'assets/two_link_direct.xml', 'assets/two_link_lagged.xml', 'locks/environment.json',
              'locks/wheels.json', 'config/revisions/MX-shared-state-0001.json']
    probes, coordinates = {}, {}
    for actor in ACTUATORS:
        path = development_manifest(root,actor,pilot['nominal_context'])
        arrays = load_corpus(path, 'exploratory', 'development', 'calibration')
        selected = select_probes(make_probe_library(arrays, actor, 'A', 'development'))
        probes.update({actor+'_'+key:value for key, value in selected.items()})
        coordinates[actor] = {key:selected[key].tolist() for key in
                              ('probe_ids', 'trajectory_ids', 'anchor_ids', 'time_ns', 'strata')}
        inputs += [str(path.relative_to(root)), str((path.parent/'trajectories.npz').relative_to(root)),
                   f'data/{actor}/A/development/manifest.json', f'data/{actor}/A/development/trajectories.npz']
    selections = {actor+':'+model:pilot['selections'][actor+':'+model]
                  for actor in ACTUATORS for model in MODELS}
    for item in selections.values():
        path = safe_path(root, item['model_manifest'])
        if sha256_file(path) != item['manifest_sha256'] or sha256_file(path.parent/'weights.pt') != item['checkpoint_sha256']:
            raise ValueError('Changed pilot-selected model')
        inputs += [item['model_manifest'], str((path.parent/'weights.pt').relative_to(root)),
                   str((path.parent/'normalization.json').relative_to(root))]
    buffer = io.BytesIO(); np.savez_compressed(buffer, **probes)
    import hashlib
    probe_path = 'data/diagnostic_inputs/'+hashlib.sha256(buffer.getvalue()).hexdigest()+'.npz'
    atomic_bytes(root/probe_path, buffer.getvalue()); inputs.append(probe_path)
    freeze = dict(schema_version=1, protocol_id=PROTOCOL, phase='exploratory', created_utc=utc_now(),
                  seed=9000, training_condition='A', loss_mode='shared', alpha_spent=0, statistical_look=None,
                  models=list(MODELS), horizons_ns=list(HORIZONS), probe_path=probe_path,
                  coordinates=coordinates, selections=selections, expected_model_rows=3072,
                  physics_steps_ns=[250_000, 125_000, 62_500], tight_probe_indices=list(range(8))+list(range(64,72)),
                  source_hashes={name:sha256_file(root/name) for name in _sources()},
                  input_hashes={name:sha256_file(root/name) for name in sorted(set(inputs))},
                  resource_projection={'new_disk_bytes_estimate':20*1024**2,
                      'runtime_projection':'Measure first reference batch; no artificial runtime cap',
                      'additional_training_updates':0, 'additional_control_episodes':0})
    atomic_json(root/FREEZE, freeze)
    return freeze


def reference_envelope(base, fine, finest, oracle):
    arrays = [np.asarray(x, dtype=np.float64) for x in (base, fine, finest, oracle)]
    if any(x.shape != arrays[0].shape or not np.isfinite(x).all() for x in arrays):
        raise ValueError('Complete finite reference endpoints required')
    estimates = []
    for component, budget in ((slice(0,2), 1e-7), (slice(2,4), 1e-6)):
        differences = [(arrays[0]-arrays[1])[...,component], (arrays[1]-arrays[2])[...,component],
                       (arrays[0]-arrays[3])[...,component]]
        if any(np.max(np.abs(d)) > budget for d in differences):
            raise ValueError('Reference endpoint exceeds frozen numerical budget')
        roundoff = 16*np.finfo(float).eps*np.maximum(1., np.linalg.norm(arrays[0][...,component], axis=-1))
        estimates.append(np.maximum.reduce([np.linalg.norm(d,axis=-1) for d in differences]+[roundoff]))
    return np.stack(estimates, axis=-1)


def normalized_error(error, reference_increment, uncertainty):
    error, reference_increment, uncertainty = map(np.asarray, (error, reference_increment, uncertainty))
    norm = np.linalg.norm(reference_increment, axis=-1)
    numerator = np.linalg.norm(error, axis=-1)
    if (error.shape != reference_increment.shape or uncertainty.shape != norm.shape
            or not all(np.isfinite(x).all() for x in (error, reference_increment, uncertainty))
            or np.any(uncertainty < 0) or np.any(norm <= np.maximum(100*uncertainty, 1e-14))):
        raise ValueError('Command increment signal is unresolved or malformed')
    return (numerator/norm, np.maximum(0.,numerator-uncertainty)/(norm+uncertainty),
            (numerator+uncertainty)/(norm-uncertainty))


def summarize_measurements(raw, freeze, actor):
    dimension = 4 if actor == 'direct' else 6
    expected = freeze['coordinates'][actor]
    if (not np.array_equal(raw['model_ids'], MODELS) or not np.array_equal(raw['horizons_ns'], HORIZONS)
            or not np.array_equal(raw['probe_ids'], expected['probe_ids'])
            or not np.array_equal(raw['strata'], expected['strata']) or len(raw['probe_ids']) != 128):
        raise ValueError('Missing or changed measurement coordinate')
    for key in ('model_zero', 'model_command', 'model_composed'):
        if raw[key].shape != (4,128,3,dimension) or not np.isfinite(raw[key]).all():
            raise ValueError('Missing/nonfinite model endpoint: '+key)
    for key, shape in (('truth_zero',(128,3,dimension)), ('truth_command',(128,3,dimension)),
                       ('uncertainty_zero',(128,3,2)), ('uncertainty_command',(128,3,2))):
        if raw[key].shape != shape or not np.isfinite(raw[key]).all():
            raise ValueError('Missing/nonfinite reference endpoint/envelope: '+key)
    if any(np.any(raw[key] < 0) for key in ('uncertainty_zero', 'uncertainty_command')):
        raise ValueError('Negative numerical uncertainty')
    errors = decompose(raw['model_zero'], raw['model_command'],
                       np.broadcast_to(raw['truth_zero'],raw['model_zero'].shape),
                       np.broadcast_to(raw['truth_command'],raw['model_zero'].shape), raw['model_composed'])
    if not np.allclose(errors['drift_error']+errors['command_error'], errors['total_error'], atol=1e-13, rtol=1e-12):
        raise ValueError('Signed decomposition failed')
    rows, comparisons = [], []
    for ci, (component, slc, unit) in enumerate((('q',slice(0,2),'rad'), ('v',slice(2,4),'rad/s'))):
        eta0, etau = raw['uncertainty_zero'][...,ci], raw['uncertainty_command'][...,ci]
        eta_command = eta0+etau
        metrics = {name:np.linalg.norm(value[...,slc],axis=-1) for name,value in errors.items()}
        normalized, normalized_low, normalized_high = normalized_error(errors['command_error'][...,slc],
            np.broadcast_to((raw['truth_command']-raw['truth_zero'])[...,slc], errors['command_error'][...,slc].shape),
            np.broadcast_to(eta_command,metrics['command_error'].shape))
        metrics['normalized_command_error'] = normalized
        envelopes = {name:np.broadcast_to(eta,metrics[name].shape) for name,eta in
                     (('drift_error',eta0), ('command_error',eta_command), ('total_error',etau), ('composed_error',etau))}
        # This is model self-consistency, so plant uncertainty is not applicable.
        envelopes['composition_discrepancy'] = 16*np.finfo(float).eps*np.maximum(1.,
            np.linalg.norm(raw['model_composed'][...,slc],axis=-1)+np.linalg.norm(raw['model_command'][...,slc],axis=-1))
        lower = {name:np.maximum(0.,value-envelopes[name]) for name,value in metrics.items() if name in envelopes}
        upper = {name:value+envelopes[name] for name,value in metrics.items() if name in envelopes}
        lower['normalized_command_error'], upper['normalized_command_error'] = normalized_low, normalized_high
        for stratum in ('cold','warm'):
            mask = raw['strata'] == stratum
            for hi,horizon in enumerate(HORIZONS):
                for mi,model in enumerate(MODELS):
                    row = dict(actuator=actor, model_id=model, stratum=stratum, h_ns=horizon,
                               component=component, unit=unit, n_probes=int(mask.sum()))
                    for metric,value in metrics.items():
                        vector = value[mi,mask,hi]
                        row.update({metric+'_mean':float(np.mean(vector)), metric+'_median':float(np.median(vector)),
                                    metric+'_p90':float(np.quantile(vector,.9,method='linear')),
                                    metric+'_mean_lower':float(np.mean(lower[metric][mi,mask,hi])),
                                    metric+'_mean_upper':float(np.mean(upper[metric][mi,mask,hi]))})
                    d, c, total = [errors[key][mi,mask,hi,slc] for key in ('drift_error','command_error','total_error')]
                    for name,energy in (('drift',np.sum(d*d,axis=-1)), ('command',np.sum(c*c,axis=-1)),
                                        ('cross',2*np.sum(d*c,axis=-1)), ('total',np.sum(total*total,axis=-1))):
                        row[name+'_'+component+'_squared_mean'] = float(np.mean(energy))
                    rows.append(row)
                for mi,model in enumerate(MODELS[1:],1):
                    for metric,value in metrics.items():
                        method, base = value[0,mask,hi], value[mi,mask,hi]
                        b, m = float(np.mean(base)), float(np.mean(method))
                        bl, bu = float(np.mean(lower[metric][mi,mask,hi])), float(np.mean(upper[metric][mi,mask,hi]))
                        ml, mu = float(np.mean(lower[metric][0,mask,hi])), float(np.mean(upper[metric][0,mask,hi]))
                        comparisons.append(dict(actuator=actor, competitor=model, stratum=stratum, h_ns=horizon,
                            component=component, metric=metric, n_probes=int(mask.sum()), method_mean=m, baseline_mean=b,
                            reduction=None if b <= 0 else 1-m/b,
                            numerical_reduction_bounds=None if bl <= 0 else [1-mu/bl, 1-ml/bu],
                            paired_wins=int(np.sum(method < base)),
                            numerically_resolved_wins=int(np.sum(upper[metric][0,mask,hi] < lower[metric][mi,mask,hi]))))
    return dict(rows=rows, comparisons=comparisons, measured_model_rows=4*128*3)


def _load_freeze(root):
    freeze = read_json(root/FREEZE)
    if (freeze['protocol_id'] != PROTOCOL or freeze['phase'] != 'exploratory' or freeze['seed'] != 9000
            or freeze['models'] != list(MODELS) or freeze['horizons_ns'] != list(HORIZONS)
            or freeze['expected_model_rows'] != 3072 or freeze['physics_steps_ns'] != [250_000,125_000,62_500]
            or freeze['tight_probe_indices'] != list(range(8))+list(range(64,72))
            or freeze['alpha_spent'] != 0 or freeze['statistical_look'] is not None):
        raise ValueError('Changed diagnostic design')
    verify_hashes(root, {**freeze['source_hashes'], **freeze['input_hashes']})
    with np.load(safe_path(root,freeze['probe_path']), allow_pickle=False) as archive:
        libraries = {actor:{key:archive[actor+'_'+key].copy() for key in PROBE_KEYS} for actor in ACTUATORS}
    for actor,library in libraries.items():
        library = select_probes(library)
        for key,expected in freeze['coordinates'][actor].items():
            if not np.array_equal(library[key],expected):
                raise ValueError('Changed frozen probe coordinates')
        if library['states'].shape != (128,4 if actor == 'direct' else 6) or library['commands'].shape != (128,2):
            raise ValueError('Changed frozen probe shapes')
    return freeze,libraries


def _reference_grid(root, actor, library, freeze):
    from .envs import Plant
    from .oracle_rigidbody import integrate, parameters_from_compiled
    import time
    steps = freeze['physics_steps_ns']
    plants = [Plant(actor,step,root/'assets') for step in steps]
    parameters = parameters_from_compiled(plants[0].compiled_manifest())
    dimension = library['states'].shape[1]
    endpoints = np.empty((3,128,3,2,dimension))
    oracle = np.empty((128,3,2,dimension))
    nfev, status = np.empty((128,3,2),dtype=np.int64), np.empty((128,3,2),dtype=np.int64)
    tight_ids = np.asarray(freeze['tight_probe_indices'],dtype=np.int64)
    tight = np.empty((len(tight_ids),3,2,dimension))
    tight_nfev = np.empty((len(tight_ids),3,2),dtype=np.int64)
    tight_map = {int(probe):index for index,probe in enumerate(tight_ids)}
    started = time.perf_counter()
    for pi,(state,command,clock) in enumerate(zip(library['states'],library['commands'],library['time_ns'])):
        for hi,horizon in enumerate(HORIZONS):
            for bi,u in enumerate((np.zeros(2),command)):
                for si,plant in enumerate(plants):
                    plant.reset(state, time_s=int(clock)/1e9)
                    endpoints[si,pi,hi,bi] = plant.advance(u,horizon)['state']
                result = integrate(state,u,horizon/1e9,parameters,actor)
                oracle[pi,hi,bi], nfev[pi,hi,bi], status[pi,hi,bi] = result['state'],result['nfev'],result['status']
                if pi in tight_map:
                    result = integrate(state,u,horizon/1e9,parameters,actor,rtol=1e-12,atol=1e-14,max_step_s=.000125)
                    tight[tight_map[pi],hi,bi], tight_nfev[tight_map[pi],hi,bi] = result['state'],result['nfev']
        if pi in (0,63,127):
            print(f'{actor} references {pi+1}/128, elapsed {time.perf_counter()-started:.1f}s',flush=True)
    uncertainty = reference_envelope(*endpoints,oracle)
    # Tightened independent solver is a separate check from plant refinement.
    tight_difference = np.abs(tight-oracle[tight_ids])
    if (np.max(tight_difference[...,:2]) > 1e-7 or np.max(tight_difference[...,2:4]) > 1e-6 or np.any(status != 0)):
        raise ValueError('Independent tightened solver did not qualify')
    for ci,component in enumerate((slice(0,2),slice(2,4))):
        uncertainty[tight_ids,...,ci] = np.maximum(uncertainty[tight_ids,...,ci],
                                                  np.linalg.norm(tight_difference[...,component],axis=-1))
    activation_error = 0.
    if actor == 'lagged':
        u = np.stack((np.zeros_like(library['commands']),library['commands']),axis=1)
        w = library['states'][:,None,None,4:]
        expected = w+(u[:,None]-w)*(-np.expm1(-np.array(HORIZONS)[None,:,None,None]/8e7))
        activation_error = float(max(np.max(np.abs(endpoints[...,4:]-expected)),np.max(np.abs(oracle[...,4:]-expected))))
        if activation_error > 1e-9:
            raise ValueError('Reference activation response disagrees with exact fixed-initial-w formula')
    return dict(physics_endpoints=endpoints, oracle_endpoints=oracle, oracle_nfev=nfev,
                oracle_status=status, tight_endpoints=tight, tight_indices=tight_ids, tight_nfev=tight_nfev,
                uncertainty=uncertainty, parameters=parameters, activation_error_max=activation_error,
                wall_seconds=time.perf_counter()-started)


def run(root):
    from .execution import cpu_defaults, run_job
    from .control_models import load_selected
    from .efficient_power import EfficientPowerHead
    root = Path(root).resolve()
    freeze,libraries = _load_freeze(root)
    cpu_defaults(9000)
    files = sorted(set(freeze['source_hashes'])|set(freeze['input_hashes'])|{FREEZE})
    expected = ('diagnostic.coordinates','diagnostic.references','diagnostic.decomposition','diagnostic.provenance')

    def producer(folder):
        atomic_json(folder/'request.json',freeze)
        summary = dict(protocol_id=PROTOCOL,phase='exploratory',alpha_spent=0,statistical_look=None,
                       rows=[],comparisons=[],measured_model_rows=0,missing_cells=[],
                       resource_projection=freeze['resource_projection'],references={})
        for actor in ACTUATORS:
            library = libraries[actor]
            reference = _reference_grid(root,actor,library,freeze)
            np.savez_compressed(folder/(actor+'_references.npz'), **{k:v for k,v in reference.items()
                                                                   if isinstance(v,np.ndarray)})
            atomic_json(folder/(actor+'_reference_info.json'), {k:v for k,v in reference.items()
                                                              if not isinstance(v,np.ndarray)})
            raw = dict(model_ids=np.asarray(MODELS),horizons_ns=np.asarray(HORIZONS,dtype=np.int64),
                       **library, truth_zero=reference['physics_endpoints'][0,:,:,0],
                       truth_command=reference['physics_endpoints'][0,:,:,1],
                       uncertainty_zero=reference['uncertainty'][:,:,0],uncertainty_command=reference['uncertainty'][:,:,1])
            for key in ('zero','command','composed'):
                raw['model_'+key] = np.empty((4,128,3,library['states'].shape[1]))
            for mi,model_id in enumerate(MODELS):
                item = freeze['selections'][actor+':'+model_id]
                model = load_selected(safe_path(root,item['model_manifest']),item['manifest_sha256'],
                                      item['checkpoint_sha256'],model_id=model_id,actuator=actor,seed=9000)
                if model_id == 'power_correct':
                    model = EfficientPowerHead.from_head(model)
                for hi,horizon in enumerate(HORIZONS):
                    values = model_endpoints(model,library['states'],library['commands'],horizon)
                    for key,value in values.items():
                        raw['model_'+key][mi,:,hi] = value
                print(actor+' '+model_id+' endpoints complete',flush=True)
            np.savez_compressed(folder/(actor+'_measurements.npz'),**raw)
            measured = summarize_measurements(raw,freeze,actor)
            summary['rows'].extend(measured['rows']); summary['comparisons'].extend(measured['comparisons'])
            summary['measured_model_rows'] += measured['measured_model_rows']
            summary['references'][actor] = dict(wall_seconds=reference['wall_seconds'],
                max_envelope_q_rad=float(np.max(reference['uncertainty'][...,0])),
                max_envelope_v_rad_s=float(np.max(reference['uncertainty'][...,1])),
                activation_error_max=reference['activation_error_max'], oracle_nfev=int(reference['oracle_nfev'].sum()))
        verify_hashes(root,{**freeze['source_hashes'],**freeze['input_hashes']})
        checks = [dict(check_id=key,status='pass',scope=PROTOCOL,tolerance=None,
                       reason='Complete frozen rows; qualified references; signed error identity; source/hash closure') for key in expected]
        return checks,summary

    record = run_job(root,'M40','exploratory',files,expected,[],producer,config_keys={FREEZE:freeze},
                     argv=['python','-m','poc.shared_state_diagnostic','run'])
    pointer = dict(protocol_id=PROTOCOL, manifest_path=f'evidence/M40/{record["run_id"]}/manifest.json',
                   manifest_sha256=sha256_file(root/f'evidence/M40/{record["run_id"]}/manifest.json'))
    atomic_json(root/'reports/shared_state_diagnostic_run.json',pointer)
    return record


def reaggregate(root):
    from .contracts import verify_run
    root = Path(root).resolve()
    freeze,_ = _load_freeze(root)
    pointer = read_json(root/'reports/shared_state_diagnostic_run.json')
    verify_hashes(root,{pointer['manifest_path']:pointer['manifest_sha256']})
    path = safe_path(root,pointer['manifest_path'])
    record = read_json(path); verify_run(root,record)
    if record['completion_status'] != 'complete' or record['phase'] != 'exploratory':
        raise ValueError('Diagnostic measurement is incomplete')
    summary = read_json(path.parent/'summary.json')
    rows,comparisons,total = [],[],0
    for actor in ACTUATORS:
        with np.load(path.parent/(actor+'_measurements.npz'),allow_pickle=False) as archive:
            raw = {key:archive[key].copy() for key in archive.files}
        values = summarize_measurements(raw,freeze,actor)
        rows.extend(values['rows']); comparisons.extend(values['comparisons']); total += values['measured_model_rows']
    if (summary['protocol_id'] != PROTOCOL or total != 3072 or summary['measured_model_rows'] != total
            or summary['rows'] != rows or summary['comparisons'] != comparisons):
        raise ValueError('Diagnostic raw-to-summary reconciliation failed')
    return {**summary,**pointer, 'freeze_sha256':sha256_file(root/FREEZE),
            'limitations':['One development training seed; fixed finite probe library; descriptive comparisons.',
                           'Numerical bounds are empirical envelopes, not statistical confidence intervals.',
                           'Shared-state errors diagnose prediction; they do not establish a causal control improvement.']}


def render_report(result):
    lines = ['# Shared-state diagnostic: measured results', '',
             'All 3,072 planned model/probe/horizon rows are complete. Four selected models use identical states and held commands, with initial activation fixed. No new training or control episodes were run.', '',
             'One development training seed, 64 cold and 64 warm probes per plant. These are descriptive prediction errors, not confirmed control gains. Numerical envelopes use three plant resolutions and independent continuous equations.', '',
             'Values are means of two-joint vector norms, in radians for q and rad/s for v. Drift means zero **new** command; it retains initial velocity and activation. Command error compares the command-induced increment with truth. Split error uses two equal half steps with the same command.', '',
             '| Plant | Stratum | ms | Component | Model | Drift error | Command error | Total error | Split truth error | Composition discrepancy |',
             '|---|---|---:|---|---|---:|---:|---:|---:|---:|']
    for row in result['rows']:
        values = [row[key+'_mean'] for key in ('drift_error','command_error','total_error','composed_error','composition_discrepancy')]
        lines.append(f'| {row["actuator"]} | {row["stratum"]} | {row["h_ns"]/1e6:g} | {row["component"]} | {row["model_id"]} | '+ ' | '.join(f'{v:.6g}' for v in values)+' |')
    lines += ['', '## Paired head comparisons', '',
              'Positive reductions favor the head. “Wins” counts paired probe norms; it is not an independent training replication count. Complete medians, p90 values, normalized command errors, numerical bounds and signed squared-error cross terms are in the JSON.', '',
              '| Plant | Stratum | ms | Component | Baseline | Drift reduction | Command reduction | Total reduction | Total probe wins |',
              '|---|---|---:|---|---|---:|---:|---:|---:|']
    table = {}
    for row in result['comparisons']:
        key = tuple(row[name] for name in ('actuator','stratum','h_ns','component','competitor'))
        table.setdefault(key,{})[row['metric']] = row
    for key,entries in table.items():
        percentages = [entries[name]['reduction'] for name in ('drift_error','command_error','total_error')]
        cells = ['unresolved' if v is None else f'{100*v:+.2f}%' for v in percentages]
        lines.append(f'| {key[0]} | {key[1]} | {key[2]/1e6:g} | {key[3]} | {key[4]} | '+' | '.join(cells)+f' | {entries["total_error"]["paired_wins"]}/64 |')
    lines += ['', '## Reproduce', '', '```bash', '.venv/bin/python -m poc.shared_state_diagnostic verify',
              '.venv/bin/python -m poc.shared_state_diagnostic report', '```', '',
              'These commands verify/reaggregate existing measured artifacts. `run` reuses a completed compatible job; changed inputs require a separately versioned design. The immutable manifest is `'+result['manifest_path']+'`.', '',
              'No statistical look or alpha was spent. Correct action orders alone need not produce accurate drift, finite-time response or composition. Do not sum drift and command error norms: the signed vectors can cancel.']
    return ('\n'.join(lines)+'\n').encode()


def report(root):
    root = Path(root).resolve()
    result = reaggregate(root)
    atomic_json(root/'reports/shared_state_diagnostic_result.json',result)
    atomic_bytes(root/'reports/shared_state_diagnostic_result.md',render_report(result))
    return result


def verify_saved_reports(root, value):
    saved = read_json(root/'reports/shared_state_diagnostic_result.json')
    if saved != value:
        raise ValueError('Saved diagnostic report differs from raw reaggregation')
    if (root/'reports/shared_state_diagnostic_result.md').read_bytes() != render_report(value):
        raise ValueError('Saved Markdown report differs from measured raw results')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','run','report','verify'))
    parser.add_argument('--root',type=Path,default=Path.cwd())
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        value = prepare(args.root)
        print({'frozen_probes':256,'selected_checkpoints':len(value['selections']), 'expected_model_rows':value['expected_model_rows']})
    elif args.command == 'run':
        record = run(args.root)
        print({'completion_status':record['completion_status'],'run_id':record['run_id']})
        return 0 if record['completion_status'] == 'complete' else 1
    elif args.command == 'report':
        print({'measured_model_rows':report(args.root)['measured_model_rows']})
    else:
        value = reaggregate(args.root)
        for name in ('reports/shared_state_diagnostic_result.json','reports/shared_state_diagnostic_result.md'):
            if not (args.root/name).is_file():
                raise ValueError('Missing saved diagnostic report: '+name)
        verify_saved_reports(args.root,value)
        receipt = dict(protocol_id=PROTOCOL,status='complete',measured_model_rows=value['measured_model_rows'],
                       manifest_path=value['manifest_path'],manifest_sha256=value['manifest_sha256'],
                       report_hashes={name:sha256_file(args.root/name) for name in
                           ('reports/shared_state_diagnostic_result.json','reports/shared_state_diagnostic_result.md')})
        atomic_json(args.root/'reports/shared_state_diagnostic_verification.json',receipt)
        print({'status':'complete','measured_model_rows':value['measured_model_rows']})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
