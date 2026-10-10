"""A fixed exploratory quality/compute pilot, without changing historical engines."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

from .io import atomic_json, job_key, job_lock, read_json, safe_path, sha256_file, utc_now

PROTOCOL = 'PTCO-EFF-0.1'
MODELS = ('power_correct', 'mlp_h', 'node_generic', 'node_structured', 'physics_fit')
SPEC = 'docs/superpowers/specs/2026-10-07-targeted-efficiency-pilot.md'
PLAN = 'docs/superpowers/plans/2026-10-07-targeted-efficiency-pilot.md'
FREEZE = 'config/efficiency_pilot_freeze.json'
REGISTRY = 'config/efficiency_pilot_cells.json'
REQUEST = 'config/revisions/MX-efficiency-pilot-0001.json'
CONTEXT = 'study/control-development'


def coordinates():
    cells = [(a, m, rate, step, task) for a in ('direct', 'lagged') for m in MODELS
             for rate in (40_000_000, 10_000_000) for step in (40_000_000, 5_000_000) for task in range(10)]
    return cells + [(a, 'sim_oracle', rate, 40_000_000, task) for a in ('direct', 'lagged')
                    for rate in (40_000_000, 10_000_000) for task in range(2)]


def validate_coordinates(cells):
    if len(cells) != 408 or len(set(map(tuple, cells))) != 408 or set(map(tuple, cells)) != set(coordinates()):
        raise ValueError('Pilot coverage must contain every paired cell and the exact oracle subset')


def validate_request(value, freeze):
    from .control_jobs import validate_request as original_validate
    extras = {'protocol_id', 'inference_variant', 'freeze_sha256'}
    if not extras <= set(value) or value['protocol_id'] != PROTOCOL:
        raise ValueError('A complete explicit efficiency request is required')
    if (value['phase'] != 'exploratory' or type(value['seed']) is not int or value['seed'] != 9000
            or value['condition'] != 'clean' or value['physics_step_ns'] != 250_000
            or value['power_mode'] is not None):
        raise ValueError('Pilot workload, phase or numerical scope changed')
    coordinate = (value['actuator'], value['model_id'], value['actuator_interval_ns'],
                  value['model_partition_ns'][0], value['episode_id'])
    step = coordinate[3]
    if coordinate not in coordinates() or value['model_partition_ns'] != [step]*(40_000_000//step):
        raise ValueError('Request is outside the frozen pilot coverage')
    variant = 'efficient' if value['model_id'] == 'power_correct' else 'original'
    if value['inference_variant'] != variant or value['planner_settings'] != freeze['planner_settings']:
        raise ValueError('Changed inference variant or common planner')
    base = {key: val for key, val in value.items() if key not in extras}
    # Reuse the existing schema validation; restoration still uses actual mlp_h.
    # This adapter does not patch a frozen producer's model allowlist.
    if base['model_id'] == 'mlp_h':
        base['model_id'] = 'mlp_dt'
    original_validate(base)
    if value['model_id'] != 'sim_oracle':
        selected = freeze['selections'][f'{value["actuator"]}:{value["model_id"]}']
        if any(value[k] != selected[k] for k in ('model_manifest', 'manifest_sha256', 'checkpoint_sha256')):
            raise ValueError('Pilot checkpoint differs from its nominal selection freeze')
    return value


def verify_hashes(root, hashes):
    for name, expected in hashes.items():
        if sha256_file(safe_path(root, name)) != expected:
            raise ValueError('Changed pilot dependency: '+name)


def _selection(root, request_path):
    request = read_json(root/request_path)
    return {key: request[key] for key in ('model_manifest', 'manifest_sha256', 'checkpoint_sha256')}


def _verify_selected(root, identity, item):
    from .control_models import load_selected
    actuator, model_id = identity.split(':')
    return load_selected(safe_path(root, item['model_manifest']), item['manifest_sha256'],
                         item['checkpoint_sha256'], model_id=model_id, actuator=actuator, seed=9000)


def nominal_sources(root):
    """Preserve the original training context, including its historical receipts."""
    from .training_jobs import SOURCES
    core = {n for n in SOURCES if n.startswith('poc/models/')}
    core.update(('poc/train.py', 'poc/history.py', 'poc/analytic.py', 'poc/clocks.py'))
    for name in core:
        if sha256_file(root/name) != sha256_file(root/CONTEXT/name):
            raise ValueError('Nominal training semantics differ: '+name)
    return {f'{CONTEXT}/{name}': sha256_file(root/CONTEXT/name) for name in SOURCES}


def prepare(root: Path):
    from .control_jobs import SOURCES
    root = root.resolve()
    authority = read_json(root/'config/authorization.json')
    if authority['local_compute_authorized'] is not True or authority['paid_services_authorized'] is not False:
        raise ValueError('Existing local authorization is required; no paid service')
    selections = {}
    for actuator in ('direct', 'lagged'):
        for model_id in ('power_correct', 'node_generic', 'node_structured'):
            if actuator == 'direct' and model_id != 'node_structured':
                name = f'development__direct__A__shared__{model_id}__default__h40000000__episode0.json'
            else:
                name = f'remaining_development__{actuator}__A__shared__{model_id}__seed9000__h40000000__episode0.json'
            selections[f'{actuator}:{model_id}'] = _selection(root, 'config/control_jobs/'+name)
        selections[f'{actuator}:physics_fit'] = _selection(root, f'config/control_jobs/reference__{actuator}__physics_fit__seed0__episode0.json')
    for identity, item in selections.items():
        _verify_selected(root, identity, item)
    source_names = (*SOURCES, 'poc/efficient_power.py', 'poc/efficiency_episode.py', 'poc/efficiency_pilot.py')
    input_names = (SPEC, PLAN, REQUEST, 'config/authorization.json', 'config/planner.json',
                   'config/numerical.json', 'assets/two_link_direct.xml', 'assets/two_link_lagged.xml')
    freeze = dict(schema_version=1, protocol_id=PROTOCOL, phase='exploratory', created_utc=utc_now(),
                  selections=selections, planner_settings=read_json(root/'config/planner.json')['settings'],
                  source_hashes={n: sha256_file(root/n) for n in source_names},
                  input_hashes={n: sha256_file(root/n) for n in input_names},
                  nominal_context=CONTEXT, nominal_seed=9000, condition='A', loss_mode='shared',
                  nominal_source_hashes=nominal_sources(root),
                  nominal_recipe=dict(learning_rate=.001, initial_epochs=100, continuation_block=100,
                                      selection='original nominal qualification; no pilot outcome access'),
                  expected_episodes=408, statistical_look=None, alpha_spent=0)
    path = root/'config/efficiency_pilot_pretraining.json'
    if path.exists():
        prior = read_json(path)
        freeze['created_utc'] = prior['created_utc']
        if freeze != prior:
            raise ValueError('Existing prospective pilot inputs changed; explicit revision required')
    else:
        atomic_json(path, freeze)
    return freeze


def _child(root, command, label, cwd=None):
    folder = root/'reports/efficiency_pilot_attempts'/(utc_now().replace(':', '')+'-'+uuid.uuid4().hex[:8])
    folder.mkdir(parents=True)
    atomic_json(folder/'attempt.json', dict(state='running', label=label, argv=command, started_utc=utc_now()))
    with (folder/'stdout.log').open('w') as out, (folder/'stderr.log').open('w') as err:
        result = subprocess.run(command, cwd=cwd or root, stdout=out, stderr=err)
    atomic_json(folder/'attempt.json', dict(state='complete' if result.returncode == 0 else 'failed',
                label=label, argv=command, returncode=result.returncode, ended_utc=utc_now()))
    if result.returncode:
        raise RuntimeError(f'Pilot child failed; preserved diagnostics: {folder.relative_to(root)}')
    return folder/'stdout.log'


def run_nominal(root):
    freeze = read_json(root/'config/efficiency_pilot_pretraining.json')
    verify_hashes(root, {**freeze['source_hashes'], **freeze['input_hashes'], **freeze['nominal_source_hashes']})
    context = root/CONTEXT
    result_path = root/'reports/efficiency_pilot_nominal.json'
    result = read_json(result_path) if result_path.exists() else dict(state='partial', selected={})
    with job_lock(job_key(dict(pilot=PROTOCOL, phase='nominal')), root):
        for actuator in ('direct', 'lagged'):
            identity = actuator+':mlp_h'
            if identity in result['selected']:
                _verify_selected(root, identity, result['selected'][identity])
                continue
            print('Nominal controls and fitting: '+identity, flush=True)
            control = context/f'config/efficiency_mlp_h_{actuator}_controls.json'
            atomic_json(control, dict(model_id='mlp_h', actuator=actuator))
            _child(root, [sys.executable, '-m', 'poc.research_jobs', 'controls', str(control),
                         '--root', str(context), '--resume'], identity+' controls', context)
            epoch, previous = 100, None
            while True:
                params = dict(model_id='mlp_h', actuator=actuator, seed=9000, condition='A',
                              loss_mode='shared', epochs=epoch, learning_rate=.001,
                              train_manifest=f'data/{actuator}/A/train/manifest.json',
                              validation_manifest=f'data/{actuator}/A/validation/manifest.json')
                if previous:
                    params['continue_from'] = str(previous.parent.relative_to(context))
                request_path = context/f'config/efficiency_mlp_h_{actuator}_{epoch}.json'
                atomic_json(request_path, dict(protocol_version='PTCO-1.0', purpose='training',
                            phase='exploratory', semantic_inputs=params))
                log = _child(root, [sys.executable, '-m', 'poc.research_jobs', 'train', str(request_path),
                            '--root', str(context), '--resume'], identity+f' epochs {epoch}', context)
                info = json.loads(log.read_text().strip().splitlines()[-1])
                previous = context/info['manifest_path']
                from .train import qualify_nominal
                qualification = qualify_nominal(read_json(previous), context)
                print(identity+f' epochs {epoch}: '+str(qualification), flush=True)
                if qualification['qualified']:
                    item = dict(model_manifest=str(previous.relative_to(root)), manifest_sha256=sha256_file(previous),
                                checkpoint_sha256=sha256_file(previous.parent/'weights.pt'))
                    _verify_selected(root, identity, item)
                    result['selected'][identity] = item
                    atomic_json(result_path, result)
                    break
                if qualification['convergence']['stabilized']:
                    raise RuntimeError('Converged mlp_h did not qualify learning; nominal optimizer diagnostic required')
                epoch += 100
        result.update(state='complete', pretraining_sha256=sha256_file(root/'config/efficiency_pilot_pretraining.json'))
        atomic_json(result_path, result)
    return result


def prepare_episodes(root, freeze):
    verify_hashes(root, {**freeze['source_hashes'], **freeze['input_hashes']})
    nominal = read_json(root/'reports/efficiency_pilot_nominal.json')
    if nominal['state'] != 'complete' or set(nominal['selected']) != {'direct:mlp_h', 'lagged:mlp_h'}:
        raise ValueError('Both nominally qualified mlp_h checkpoints are required')
    freeze = dict(freeze, selections={**freeze['selections'], **nominal['selected']})
    for identity, item in freeze['selections'].items():
        _verify_selected(root, identity, item)
    if (root/FREEZE).exists() and read_json(root/FREEZE) != freeze:
        raise ValueError('Complete checkpoint freeze already exists with different inputs')
    atomic_json(root/FREEZE, freeze)
    jobs = []
    for actuator, model_id, rate, step, task in coordinates():
        item = freeze['selections'].get(f'{actuator}:{model_id}',
                    dict(model_manifest=None, manifest_sha256=None, checkpoint_sha256=None))
        value = dict(schema_version=1, protocol_id=PROTOCOL, phase='exploratory', model_id=model_id,
            inference_variant='efficient' if model_id == 'power_correct' else 'original', actuator=actuator,
            seed=9000, episode_id=task, condition='clean', model_partition_ns=[step]*(40_000_000//step),
            actuator_interval_ns=rate, physics_step_ns=250_000, power_mode=None, **item,
            planner_settings=freeze['planner_settings'], freeze_sha256=sha256_file(root/FREEZE),
            resource_projection=dict(reason='Historical 2.5 MiB/episode; measure actual new rate/model before full batch',
                                     prior_artifact_bytes_per_episode=2_500_000, owner_imposed_caps=None))
        validate_request(value, freeze)
        cell = f'{actuator}__{model_id}__rate{rate}__step{step}__task{task}'
        path = f'config/efficiency_pilot_jobs/{cell}.json'
        atomic_json(root/path, value)
        jobs.append(dict(cell_id=cell, coordinate=[actuator, model_id, rate, step, task],
                         path=path, sha256=sha256_file(root/path)))
    registry = dict(schema_version=1, protocol_id=PROTOCOL, freeze_sha256=sha256_file(root/FREEZE), jobs=jobs)
    if (root/REGISTRY).exists() and read_json(root/REGISTRY) != registry:
        raise ValueError('Pilot registry changed after its freeze')
    atomic_json(root/REGISTRY, registry)
    return registry


def verify_registry(root, registry):
    freeze = read_json(root/FREEZE)
    if registry['protocol_id'] != PROTOCOL or registry['freeze_sha256'] != sha256_file(root/FREEZE):
        raise ValueError('Changed pilot freeze')
    verify_hashes(root, {**freeze['source_hashes'], **freeze['input_hashes']})
    validate_coordinates([job['coordinate'] for job in registry['jobs']])
    for job in registry['jobs']:
        path = safe_path(root, job['path'])
        if sha256_file(path) != job['sha256']:
            raise ValueError('Changed pilot episode request')
        value = validate_request(read_json(path), freeze)
        if [value['actuator'], value['model_id'], value['actuator_interval_ns'],
            value['model_partition_ns'][0], value['episode_id']] != job['coordinate']:
            raise ValueError('Registry coordinate and request disagree')
    return freeze


def verify_completion(root, manifest, request):
    from .contracts import verify_run
    record = read_json(manifest)
    verify_run(root, record)
    folder = Path(manifest).parent
    if record['completion_status'] != 'complete' or record['phase'] != 'exploratory':
        raise ValueError('Only complete exploratory pilot evidence can be reused')
    if read_json(folder/'request.json') != request:
        raise ValueError('Completed pilot request differs from frozen input')
    summary = read_json(folder/'episode.json')
    expected = dict(complete=True, phase='exploratory', model_id=request['model_id'], seed=request['seed'],
                    episode_id=request['episode_id'], actuator=request['actuator'],
                    model_partition_ns=request['model_partition_ns'], actuator_interval_ns=request['actuator_interval_ns'],
                    physics_step_ns=request['physics_step_ns'], checkpoint_hash=request['checkpoint_sha256'])
    if any(summary.get(k) != v for k, v in expected.items()):
        raise ValueError('Completed episode identity differs from its request')
    return record


def run_registry(root, registry):
    verify_registry(root, registry)
    path = root/'reports/efficiency_pilot_batch.json'
    result = read_json(path) if path.exists() else dict(state='partial', registry_sha256=sha256_file(root/REGISTRY), completed={})
    if result['registry_sha256'] != sha256_file(root/REGISTRY):
        raise ValueError('Existing pilot batch belongs to another registry')
    with job_lock(job_key(dict(pilot=PROTOCOL, phase='control', registry=result['registry_sha256'])), root):
        for job in registry['jobs']:
            request = read_json(root/job['path'])
            if job['cell_id'] in result['completed']:
                entry = result['completed'][job['cell_id']]
                if sha256_file(root/entry['path']) != entry['sha256']:
                    raise ValueError('Changed completed pilot manifest')
                verify_completion(root, root/entry['path'], request)
                continue
            print(f'{len(result["completed"])}/{len(registry["jobs"])}: {job["cell_id"]}', flush=True)
            log = _child(root, [sys.executable, '-m', 'poc.efficiency_episode', job['path']], job['cell_id'])
            manifest = Path(log.read_text().strip().splitlines()[-1])
            verify_completion(root, manifest, request)
            result['completed'][job['cell_id']] = dict(path=str(manifest.relative_to(root)), sha256=sha256_file(manifest))
            atomic_json(path, result)
            resources = read_json(manifest.parent/'resources.json')
            if len(result['completed']) == 1:
                remaining = len(registry['jobs'])-1
                projection = dict(remaining_episodes=remaining, representative=job['cell_id'],
                    representative_wall_seconds=resources['wall_seconds'],
                    projected_remaining_seconds=remaining*resources['wall_seconds'],
                    projected_remaining_bytes=remaining*resources['disk_bytes'], free_bytes=shutil.disk_usage(root).free,
                    limitation='Initial proxy; model and query-rate runtimes differ', owner_imposed_caps=None)
                atomic_json(root/'reports/efficiency_pilot_resource_projection.json', projection)
                print(projection, flush=True)
                if projection['projected_remaining_bytes'] > projection['free_bytes']:
                    raise RuntimeError('Projected pilot artifacts exceed actual free disk; provision space without deleting evidence')
        result['state'] = 'complete'
        atomic_json(path, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'nominal', 'episodes', 'run'))
    parser.add_argument('--root', type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == 'prepare':
        value = prepare(root)
    elif args.command == 'nominal':
        value = run_nominal(root)
    elif args.command == 'episodes':
        value = prepare_episodes(root, read_json(root/'config/efficiency_pilot_pretraining.json'))
    else:
        value = run_registry(root, read_json(root/REGISTRY))
    print(json.dumps(dict(command=args.command, state=value.get('state', 'prepared'),
                         expected_episodes=value.get('expected_episodes', len(value.get('jobs', []))))), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
