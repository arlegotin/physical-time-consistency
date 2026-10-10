"""Explicit same-field closed-loop solver continuation; original policies stay intact."""
from __future__ import annotations

import math
from pathlib import Path

from .io import read_json, safe_path, sha256_file

SOURCES = ('poc/control_solver_continuation.py', 'poc/confirmation_solver.py',
           'poc/control_confirmation.py', 'poc/control_confirmation_batch.py',
           'poc/control_utility.py', 'poc/control_numerical.py', 'poc/statistics.py')
TRIGGER = 'reports/control_solver_continuation_2026-10-05/trigger.json'
REGISTRY = 'config/control_solver_continuation_jobs.json'


def _primary(batch):
    if (batch.get('state') != 'complete' or batch.get('expected_episodes') != 1200
            or batch.get('completed_episodes') != 1200 or len(batch.get('episodes', [])) != 1200):
        raise ValueError('Complete primary control batch required')
    entries = {entry['cell_id']: entry for entry in batch['episodes']}
    if len(entries) != 1200:
        raise ValueError('Duplicate primary solver parent')
    selected = {cell: entry for cell, entry in entries.items()
                if entry['summary']['model_id'] == 'node_generic'
                and entry['summary']['model_partition_ns'] == [40_000_000]
                and entry['summary']['episode_id'] in range(5)}
    coordinates = {(entry['summary']['actuator'], entry['summary']['seed'],
                    entry['summary']['episode_id']) for entry in selected.values()}
    expected = {(actuator, seed, episode) for actuator in ('direct', 'lagged')
                for seed in range(5) for episode in range(5)}
    if len(selected) != 50 or coordinates != expected:
        raise ValueError('Complete fifty-parent nominal solver cohort required')
    return selected


def derive_jobs(root, cells, batch, freeze):
    """Keep every original task/clock/model value; resolution is an outer request."""
    root = Path(root)
    selected = _primary(batch)
    if (not isinstance(cells, list) or len(cells) != 50 or len(set(cells)) != 50
            or set(cells) != set(selected) or freeze.get('state') != 'complete'
            or freeze.get('expected_episodes') != 1200 or len(freeze.get('jobs', [])) != 1200):
        raise ValueError('Complete compatible fifty-parent solver request required')
    jobs = {job['cell_id']: job for job in freeze['jobs']}
    if len(jobs) != 1200 or not set(cells) <= set(jobs):
        raise ValueError('Duplicated or missing frozen solver parents')
    from .control_jobs import validate_request
    rows = []
    for cell in sorted(cells):
        entry, job = selected[cell], jobs[cell]
        path = safe_path(root, job['path'])
        if sha256_file(path) != job['sha256'] or sha256_file(safe_path(root, entry['manifest'])) != entry['manifest_sha256']:
            raise ValueError('Changed original solver parent request or manifest')
        request = validate_request(read_json(path))
        summary = entry['summary']
        keys = ('actuator', 'model_id', 'seed', 'episode_id', 'phase', 'condition',
                'model_partition_ns', 'actuator_interval_ns', 'physics_step_ns', 'power_mode')
        if (any(request[key] != summary[key] for key in keys)
                or request['checkpoint_sha256'] != summary['checkpoint_hash']
                or request['phase'] != 'confirmatory' or request['condition'] != 'clean'
                or request['physics_step_ns'] != 250_000 or request['actuator_interval_ns'] != 40_000_000
                or request['power_mode'] is not None):
            raise ValueError('Changed frozen NODE solver identity')
        for steps in (2, 4):
            rows.append(dict(schema_version=1, cell_id=f'solver{steps}__{cell}',
                parent_cell_id=cell, substeps=steps, request=request,
                parent_request=job['path'], parent_request_sha256=job['sha256'],
                parent_manifest=entry['manifest'], parent_manifest_sha256=entry['manifest_sha256']))
    return rows


def summarize_solver(batch, refined, cells, design):
    """Empirical original/refined discrepancies; no replacement cost or extra seeds."""
    from .control_utility import seed_costs
    costs = seed_costs([entry['summary'] for entry in batch['episodes']], design)
    parents = _primary(batch)
    if len(cells) != 50 or len(set(cells)) != 50 or set(cells) != set(parents):
        raise ValueError('Complete solver parent coverage required')
    observed = {}
    identity = ('actuator', 'model_id', 'seed', 'episode_id', 'phase', 'condition',
                'physics_step_ns', 'actuator_interval_ns', 'model_partition_ns', 'power_mode',
                'checkpoint_hash', 'target', 'initial_state', 'seed_namespaces')
    for row in refined:
        cell, steps = row.get('parent_cell_id'), row.get('solver_substeps')
        if cell not in parents or type(steps) is not int or steps not in (2, 4) or (cell, steps) in observed:
            raise ValueError('Unknown or duplicate closed-loop solver observation')
        before = parents[cell]['summary']
        if (any(row.get(key) != before[key] for key in identity) or row.get('complete') is not True
                or type(row.get('realized_cost')) not in (int, float)
                or not math.isfinite(row['realized_cost']) or row['realized_cost'] < 0
                or type(row.get('success')) is not bool or type(row.get('fallback_count')) is not int
                or not 0 <= row['fallback_count'] <= 40):
            raise ValueError('Incomplete, nonfinite or relabelled solver observation')
        observed[(cell, steps)] = row
    if set(observed) != {(cell, steps) for cell in cells for steps in (2, 4)}:
        raise ValueError('Missing closed-loop solver resolutions')
    indexed = {(row['actuator'], row['model_id'], row['model_step_ns']): row['cost_by_seed']
               for row in costs['rows']}
    comparisons, required, envelopes, originals, convergence = [], set(), {}, {}, []
    for actuator in ('direct', 'lagged'):
        cohort = [cell for cell in cells if parents[cell]['summary']['actuator'] == actuator]
        discrepancies, shifts = [], []
        for cell in sorted(cohort):
            before = parents[cell]['summary']
            two, four = observed[(cell, 2)], observed[(cell, 4)]
            values = [before['realized_cost'], two['realized_cost'], four['realized_cost']]
            differences = [abs(values[i] - values[j]) for i, j in ((0, 1), (1, 2), (0, 2))]
            relative = differences[1] / values[2] if values[2] else (0. if differences[1] == 0 else None)
            classification_changed = two['success'] != four['success'] or two['fallback_count'] != four['fallback_count']
            if relative is None or relative > .02 or classification_changed:
                required.add(cell)
            discrepancies.extend(differences)
            shifts.append(values[1] - values[2])
            comparisons.append(dict(parent_cell_id=cell, actuator=actuator,
                original_cost=values[0], two_substep_cost=values[1], four_substep_cost=values[2],
                pairwise_absolute_cost_differences=differences,
                relative_two_to_four_cost_difference=relative,
                success_by_resolution=[before['success'], two['success'], four['success']],
                fallback_by_resolution=[before['fallback_count'], two['fallback_count'], four['fallback_count']],
                original_to_refined_classification_changed=any(before[key] != row[key]
                    for row in (two, four) for key in ('success', 'fallback_count'))))
        b, m = indexed[(actuator, 'node_generic', 40_000_000)], indexed[(actuator, 'power_correct', 40_000_000)]
        effect = math.fsum(x - y for x, y in zip(b, m)) / len(b)
        movement = abs(math.fsum(shifts) / len(shifts))
        if movement > .1 * abs(effect):
            required.update(cohort)
        criterion = f'utility_control/{actuator}/node_generic/h40000000'
        envelopes[criterion] = dict(lower=max(discrepancies), upper=max(discrepancies))
        originals[actuator] = math.fsum(parents[cell]['summary']['realized_cost'] for cell in cohort) / len(cohort)
        convergence.append(dict(actuator=actuator, original_paired_mean_effect=effect,
            two_to_four_mean_cost_movement=movement, diagnostic_budget=.1 * abs(effect)))
    actions = [dict(branch='solver_resolution', cells=sorted(required), next_substeps=8)] if required else []
    return dict(schema_version=1, state='additional_solver_evidence_required' if actions else 'qualified',
        required_actions=actions, parent_cells=sorted(cells), n_original_episodes=50, n_refined_episodes=100,
        new_independent_training_clusters=0, statistical_look_reserved=False, scientific_decision='not_evaluated',
        envelopes=envelopes, original_mean_cost_by_plant=originals, convergence=convergence,
        episode_comparisons=comparisons, scope='evaluation_only_same_learned_field',
        envelope_interpretation='empirical_subset_discrepancy_not_certified_error_bound')


def _wrapper(root, value):
    from .control_jobs import validate_request
    expected = {'schema_version', 'cell_id', 'parent_cell_id', 'substeps', 'request',
                'parent_request', 'parent_request_sha256', 'parent_manifest', 'parent_manifest_sha256'}
    if (set(value) != expected or value['schema_version'] != 1
            or type(value['substeps']) is not int or value['substeps'] not in (2, 4)
            or value['cell_id'] != f"solver{value['substeps']}__{value['parent_cell_id']}"):
        raise ValueError('Invalid explicit solver resolution request')
    request = validate_request(value['request'])
    if (request['model_id'] != 'node_generic' or request['phase'] != 'confirmatory'
            or request['condition'] != 'clean' or request['model_partition_ns'] != [40_000_000]
            or request['physics_step_ns'] != 250_000 or request['actuator_interval_ns'] != 40_000_000
            or request['seed'] not in range(5) or request['episode_id'] not in range(5)
            or request['power_mode'] is not None):
        raise ValueError('Solver continuation must preserve the declared nominal NODE cohort')
    for field in ('parent_request', 'parent_manifest'):
        if sha256_file(safe_path(root, value[field])) != value[field + '_sha256']:
            raise ValueError('Changed frozen solver parent input')
    if read_json(safe_path(root, value['parent_request'])) != request:
        raise ValueError('Changed original solver request')
    return value


def observe_job(root, job, manifest_path):
    from .contracts import verify_run
    from .control_utility import audit_raw_episode
    root, path = Path(root), Path(manifest_path)
    request_path = safe_path(root, job['path'])
    if sha256_file(request_path) != job['sha256']:
        raise ValueError('Changed solver continuation request hash')
    wrapper = _wrapper(root, read_json(request_path))
    manifest = read_json(path)
    verify_run(root, manifest)
    summary = read_json(path.parent / 'summary.json')
    if (manifest['milestone_id'] != 'M70' or manifest['phase'] != 'confirmatory'
            or read_json(path.parent / 'solver_request.json') != wrapper
            or read_json(path.parent / 'request.json') != wrapper['request']
            or summary.get('parent_cell_id') != wrapper['parent_cell_id']
            or summary.get('solver_substeps') != wrapper['substeps']
            or summary.get('checkpoint_hash') != wrapper['request']['checkpoint_sha256']):
        raise ValueError('Solver episode differs from its frozen request or resolution')
    return dict(cell_id=wrapper['cell_id'], parent_cell_id=wrapper['parent_cell_id'],
        substeps=wrapper['substeps'], manifest=str(path.relative_to(root)), manifest_sha256=sha256_file(path),
        summary=summary, resources=read_json(path.parent / 'resources.json'),
        raw_audit=audit_raw_episode(path.parent, summary))


def execute_job(root, request_path):
    from .control_jobs import SOURCES as CORE
    from .contracts import verify_run
    from .execution import cpu_defaults, run_job
    from .control_models import load_selected
    from .confirmation_solver import SolverResolution
    from .reference_jobs import tensor_hash
    from .control_episode import execute_episode
    from .control_artifacts import save_episode
    from .envs import Plant
    from .io import atomic_json
    root, request_path = Path(root).resolve(), Path(request_path).resolve()
    wrapper = _wrapper(root, read_json(request_path))
    request = wrapper['request']
    parent = safe_path(root, wrapper['parent_manifest'])
    verify_run(root, read_json(parent))
    cpu_defaults(request['seed'])
    model_path = safe_path(root, request['model_manifest'])
    key_path = str(request_path.relative_to(root))
    files = set(CORE) | set(SOURCES) | {key_path, wrapper['parent_request'], wrapper['parent_manifest'],
        request['model_manifest'], str((model_path.parent / 'weights.pt').relative_to(root)),
        str((model_path.parent / 'normalization.json').relative_to(root)),
        f"assets/two_link_{request['actuator']}.xml"}
    hashes = {name: sha256_file(root / name) for name in files}
    checks = ('control.solver_episode', 'control.solver_weights', 'control.solver_clocks')
    def produce(output):
        model = load_selected(model_path, request['manifest_sha256'], request['checkpoint_sha256'],
            model_id='node_generic', actuator=request['actuator'], seed=request['seed'])
        before = tensor_hash(model.state_dict())
        plant = Plant(request['actuator'], request['physics_step_ns'], root / 'assets')
        result = execute_episode(plant, SolverResolution(model, wrapper['substeps']), request['planner_settings'],
            seed=request['seed'], episode_id=request['episode_id'], phase='confirmatory', condition='clean',
            model_partition=tuple(request['model_partition_ns']), actuator_interval_ns=40_000_000)
        if tensor_hash(model.state_dict()) != before or any(sha256_file(root / name) != digest for name, digest in hashes.items()):
            raise ValueError('Solver episode changed the selected field, source or original input')
        atomic_json(output / 'request.json', request)
        atomic_json(output / 'solver_request.json', wrapper)
        summary = save_episode(output, result, model_id='node_generic', checkpoint_hash=request['checkpoint_sha256'], run_id=output.name)
        summary.update(parent_cell_id=wrapper['parent_cell_id'], solver_substeps=wrapper['substeps'],
            cell_id=wrapper['cell_id'], scope='evaluation_only_same_learned_field',
            missing_cells=[], scientific_decision='not_evaluated',
            model_hashes={'checkpoint_sha256': request['checkpoint_sha256']})
        imported = {str(Path(module.__file__).resolve().relative_to(root))
                    for module in __import__('sys').modules.values() if getattr(module, '__file__', None)
                    and str(module.__file__).endswith('.py') and Path(module.__file__).resolve().is_relative_to(root / 'poc')}
        if imported - files:
            raise ValueError('Undeclared solver episode import: ' + repr(sorted(imported - files)))
        validity = [dict(check_id=key, status='pass', scope=summary['scope'], tolerance=None,
            reason='Complete unchanged-clock episode, fixed tensors and source hashes verified') for key in checks]
        return validity, summary
    record = run_job(root, 'M70', 'confirmatory', sorted(files), checks, [], produce,
        config_keys={key_path: wrapper}, argv=['-m', 'poc.control_solver_continuation', '--episode', key_path])
    if record['completion_status'] != 'complete':
        raise RuntimeError('Solver continuation episode incomplete: ' + record['run_id'])
    path = root / 'evidence/M70' / record['run_id'] / 'manifest.json'
    return observe_job(root, dict(path=key_path, sha256=sha256_file(request_path)), path)


def verify_registry(root, registry):
    from .control_confirmation import verify_freeze
    root = Path(root)
    for name, digest in registry['input_hashes'].items():
        if sha256_file(safe_path(root, name)) != digest:
            raise ValueError('Changed solver continuation dependency: ' + name)
    design = read_json(root / 'config/control_confirmation_design.json')
    freeze = read_json(root / 'config/control_confirmation_cells.json')
    verify_freeze(root, design, freeze)
    batch = read_json(root / 'reports/control_confirmation_batch.json')
    expected = derive_jobs(root, registry['parent_cells'], batch, freeze)
    if registry.get('state') != 'complete' or len(registry.get('jobs', [])) != 100:
        raise ValueError('Complete solver continuation registry required')
    for job, wrapper in zip(registry['jobs'], expected):
        path = safe_path(root, job['path'])
        if (job['cell_id'] != wrapper['cell_id'] or sha256_file(path) != job['sha256']
                or read_json(path) != wrapper):
            raise ValueError('Changed frozen solver resolution job')
    return registry


def prepare(root):
    from .io import atomic_json, utc_now
    from .control_jobs import SOURCES as CORE
    root = Path(root).resolve()
    target = root / REGISTRY
    if target.exists():
        return verify_registry(root, read_json(target))
    if not (root / TRIGGER).exists():
        raise ValueError('Verified numerical solver trigger required')
    trigger = read_json(root / TRIGGER)
    trigger_path = safe_path(root, trigger['trigger_report'])
    if sha256_file(trigger_path) != trigger['trigger_sha256']:
        raise ValueError('Changed archived solver trigger')
    report = read_json(trigger_path)
    actions = [action for action in report['required_actions'] if action['branch'] == 'solver_closed_loop']
    if (len(actions) != 1 or actions[0]['cells'] != trigger['required_cells']
            or report['statistical_look_reserved'] is not False or report['scientific_decision'] != 'not_evaluated'):
        raise ValueError('Invalid or conflicting solver trigger')
    batch = read_json(root / 'reports/control_confirmation_batch.json')
    freeze = read_json(root / 'config/control_confirmation_cells.json')
    wrappers = derive_jobs(root, trigger['required_cells'], batch, freeze)
    names = set(CORE) | set(SOURCES) | {TRIGGER, trigger['trigger_report'],
        'config/control_confirmation_design.json', 'config/control_confirmation_cells.json',
        'reports/control_confirmation_batch.json', 'reports/control_confirmation_solver.json',
        'config/control_confirmation_solver_inputs.json'}
    jobs = []
    for wrapper in wrappers:
        path = root / 'config/control_solver_jobs' / (wrapper['cell_id'] + '.json')
        if path.exists() and read_json(path) != wrapper:
            raise ValueError('Conflicting solver continuation request')
        atomic_json(path, wrapper)
        jobs.append(dict(cell_id=wrapper['cell_id'], path=str(path.relative_to(root)), sha256=sha256_file(path)))
        names.update((wrapper['parent_request'], wrapper['parent_manifest']))
    selected = [entry for entry in batch['episodes'] if entry['cell_id'] in trigger['required_cells']]
    registry = dict(schema_version=1, state='complete', created_utc=utc_now(), jobs=jobs,
        parent_cells=trigger['required_cells'], input_hashes={name: sha256_file(root / name) for name in sorted(names)},
        resource_projection=dict(expected_episodes=100,
            original_nominal_cpu_seconds=math.fsum(entry['resources']['cpu_seconds'] for entry in selected),
            estimated_artifact_bytes=2 * sum(entry['resources']['disk_bytes'] for entry in selected),
            interpretation='Same raw episode layout; NODE field work doubles/quadruples, controller overhead also contributes'),
        scientific_decision='not_evaluated')
    verify_registry(root, registry)
    atomic_json(target, registry)
    return registry


def run_batch(root, registry, *, resume=False):
    import hashlib
    import subprocess
    import sys
    from .io import atomic_json, append_event, job_lock, utc_now
    root = Path(root).resolve()
    verify_registry(root, registry)
    registry_hash = sha256_file(root / REGISTRY)
    target = root / 'reports/control_solver_continuation_batch.json'
    jobs = {job['cell_id']: job for job in registry['jobs']}
    with job_lock(hashlib.sha256(b'registered closed-loop solver continuation').hexdigest(), root):
        reused = {}
        if target.exists():
            previous = read_json(target)
            if not resume or previous['registry_sha256'] != registry_hash:
                raise ValueError('Solver continuation requires compatible --resume')
            for entry in previous['episodes']:
                cell = entry['cell_id']
                if cell in reused or cell not in jobs:
                    raise ValueError('Duplicated or unknown saved solver continuation episode')
                path = safe_path(root, entry['manifest'])
                actual = observe_job(root, jobs[cell], path)
                if entry != actual:
                    raise ValueError('Changed saved solver continuation observation')
                reused[cell] = entry
            if previous['state'] == 'complete':
                if set(reused) != set(jobs) or previous['completed_episodes'] != 100:
                    raise ValueError('Incomplete completed solver continuation')
                return previous
        episodes = []
        def save(state, **extra):
            record = dict(schema_version=1, state=state, registry_sha256=registry_hash, updated_utc=utc_now(),
                expected_episodes=100, completed_episodes=len(episodes), episodes=episodes,
                scientific_decision='not_evaluated', **extra)
            atomic_json(target, record)
            return record
        try:
            for index, job in enumerate(registry['jobs']):
                for name, digest in registry['input_hashes'].items():
                    if sha256_file(safe_path(root, name)) != digest:
                        raise ValueError('Changed solver dependency during execution: ' + name)
                if sha256_file(root / REGISTRY) != registry_hash:
                    raise ValueError('Changed solver registry during execution')
                print(f"Closed-loop solver {index + 1}/100: {job['cell_id']}", flush=True)
                entry = reused.get(job['cell_id'])
                if entry is None:
                    process = subprocess.run([sys.executable, '-m', 'poc.control_solver_continuation',
                        '--episode', job['path']], cwd=root, capture_output=True, text=True)
                    append_event(root / 'reports/control_solver_continuation_attempts.jsonl',
                        dict(cell_id=job['cell_id'], utc=utc_now(), returncode=process.returncode,
                            stdout=process.stdout, stderr=process.stderr))
                    if process.returncode:
                        raise RuntimeError('Solver episode interrupted: ' + process.stdout + '\n' + process.stderr)
                    path = Path(process.stdout.strip().splitlines()[-1])
                    entry = observe_job(root, job, path)
                episodes.append(entry)
                save('running')
            return save('complete')
        except BaseException as error:
            save('interrupted', reason=f'{type(error).__name__}: {error}')
            raise


def main(argv=None):
    import argparse
    from .io import atomic_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--episode', type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.dry_run:
        print(dict(execute=False, expected_episodes=100, substeps=[2, 4], trigger=TRIGGER,
            registry_ready=(root / REGISTRY).exists(), scientific_decision='not_evaluated'))
        return 0
    if args.episode:
        path = args.episode if args.episode.is_absolute() else root / args.episode
        result = execute_job(root, path)
        print(root / result['manifest'], flush=True)
        return 0
    registry = prepare(root)
    if args.prepare:
        print(root / REGISTRY)
        return 0
    batch = run_batch(root, registry, resume=args.resume)
    result = summarize_solver(read_json(root / 'reports/control_confirmation_batch.json'),
        [entry['summary'] for entry in batch['episodes']], registry['parent_cells'],
        read_json(root / 'config/control_confirmation_design.json'))
    result.update(registry_sha256=sha256_file(root / REGISTRY), batch_sha256=sha256_file(root / 'reports/control_solver_continuation_batch.json'))
    atomic_json(root / 'reports/control_solver_continuation_result.json', result)
    print({key: result[key] for key in ('state', 'required_actions')})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
