"""Registered S13.6 shortcut: one prespecified comparator can reject C02 only."""
from pathlib import Path

READINESS_CHECKS = ('negative.request', 'negative.nominal', 'negative.freeze')
REJECTION_CHECKS = ('MR.freeze', 'MR.coverage', 'MR.statistics')


def project_confirmation_resources(fit_seconds, n_seeds):
    import math
    if (set(fit_seconds) != {'node_generic', 'power_correct'} or type(n_seeds) is not int or n_seeds < 5
            or any(not math.isfinite(value) or value <= 0 for value in fit_seconds.values())):
        raise ValueError('Projection needs both complete measured fits and at least five seeds')
    return {'training_wall_seconds': n_seeds * sum(fit_seconds.values()), 'training_fits': 2*n_seeds,
        'E01_jobs': 3*n_seeds, 'measured_development_fit_wall_seconds': fit_seconds,
        'cpu_time_ceiling_hours': None, 'audit_wall_seconds': None,
        'note': 'Training estimate uses actual nominally qualified development exposure; new seeds may need further nominal blocks. Audit time remains measured separately. No artificial resource or scientific ceiling.'}


def scope_criteria():
    from .criteria import expand_criteria, validate_registry
    registry = expand_criteria()
    validate_registry(registry)
    return [c for c in registry if c['criterion_id'].startswith('K02.relative/')
            and c['dimensions']['actuator'] == 'direct'
            and c['dimensions']['competitor'] == 'node_generic']


def validate_confirmation_seeds(seeds):
    expected = [i for i in range(len(seeds)+3) if i not in {9000, 9001, 9002}][:len(seeds)]
    if (len(seeds) < 5 or any(type(seed) is not int for seed in seeds)
            or seeds != expected):
        raise ValueError('Independent confirmation seed IDs must start at zero and ascend without omissions; discovery seeds are excluded')
    return seeds


def validate_scope(scope):
    fixed = {'protocol_version': 'PTCO-1.0', 'negative_only': True, 'actuator': 'direct',
        'competitor': 'node_generic', 'condition': 'A', 'loss_mode': 'shared',
        'statistical_family_size': 280, 'positive_return_milestone': 'M30'}
    if any(scope.get(key) != value for key, value in fixed.items()) or scope.get('negative_only') is not True:
        raise ValueError('Changed rejection subset dimensions or decision scope')
    validate_confirmation_seeds(scope['seed_ids'])
    if scope.get('criterion_ids') != [c['criterion_id'] for c in scope_criteria()] or not scope.get('readiness_decision_id'):
        raise ValueError('Changed or missing rejection subset criterion/qualification coverage')
    return scope


def evaluate_subset(summaries, seeds, look):
    from .criterion_gates import evaluate_criterion
    validate_confirmation_seeds(seeds)
    criteria = scope_criteria()
    if set(summaries) != {c['criterion_id'] for c in criteria}:
        raise ValueError('Incomplete or unexpected rejection subset criterion coverage')
    for summary in summaries.values():
        if summary.get('seed_ids') != seeds:
            raise ValueError('Paired seed coverage differs from the frozen request')
    return [evaluate_criterion(c, summaries[c['criterion_id']], look) for c in criteria]


def subset_transition(outcome):
    if outcome == 'NO-GO':
        return {'permitted_next_actions': [{'milestone_id': 'MF', 'branch': None,
            'question': 'Report the valid mandatory C02 rejection against a prespecified strong comparator'}],
            'terminal_verdict': 'POC FAIL'}
    if outcome == 'GO':
        return {'permitted_next_actions': [{'milestone_id': 'M30', 'branch': None,
            'question': 'Subset support is not a full gate pass; complete the original development matrix'}],
            'terminal_verdict': None}
    return {'permitted_next_actions': [{'milestone_id': 'MX',
        'branch': 'precision' if outcome == 'INCONCLUSIVE' else 'repair',
        'question': 'Resolve the registered paired sensitivity comparison without changing its threshold'}],
        'terminal_verdict': None}


def validate_readiness_request(request):
    if request.get('branch') != 'mechanism-negative' or request.get('protocol_version') != 'PTCO-1.0':
        raise ValueError('Unknown negative mechanism extension or protocol')
    if set(request.get('contract', {})) != {str(i) for i in range(1, 13)} or not all(request['contract'].values()):
        raise ValueError('Extension requires every one of the twelve execution contract elements')
    for key in ('request_id', 'unresolved_question', 'live_explanations', 'input_evidence',
                'new_information', 'expected_artifacts', 'resource_projection', 'authorization_reference'):
        if not request.get(key):
            raise ValueError('Missing extension contract field: ' + key)
    expected = {'actuator': 'direct', 'competitor': 'node_generic', 'method': 'power_correct',
        'condition': 'A', 'loss_mode': 'shared', 'development_seed': 9000,
        'initial_confirmation_seeds': [0, 1, 2, 3, 4], 'statistical_family_size': 280}
    if request.get('action_parameters') != expected or request.get('return_milestone') != 'MR':
        raise ValueError('Extension may not change models, data, objectives, seeds, family or return scope')
    return request


def _sources(root):
    from .mechanism_run import _sources as original_sources
    return sorted(set(original_sources(root)) | {'poc/rejection_subset.py', 'poc/milestones/mx.py'})


def run_readiness(root, request, request_path, resume=False):
    from .contracts import verify_run
    from .execution import parent, run_job
    from .io import atomic_json, read_json, run_directory, sha256_file
    from .mechanism import cell, train_cell, freeze_recipes
    root = Path(root).resolve()
    validate_readiness_request(request)
    predecessor = parent(root, 'G20')
    inputs = []
    for entry in request['input_evidence']:
        path = root / entry['path']
        if sha256_file(path) != entry['sha256']:
            raise ValueError('Changed negative-extension discovery evidence')
        verify_run(root, read_json(path))
        inputs.append(entry['path'])
    registry = read_json(root / 'config/milestones.json')['milestones']['MX']['requests']
    if not any(r['request_id'] == request['request_id'] and r['path'] == str(request_path.relative_to(root))
               and r['branch'] == request['branch'] for r in registry):
        raise ValueError('Negative-extension request is not registered')

    def produce(folder):
        children, qualifications, fit_seconds = [], [], {}
        atomic_json(folder / 'request.json', request)
        training = []
        for model in ('node_generic', 'power_correct'):
            definition = cell('direct', 'A', 'shared', model, 9000)
            record = train_cell(root, folder, definition, 'exploratory', resume)
            training.append(record)
            path = run_directory(root, record) / 'manifest.json'
            summary = read_json(path.parent / 'summary.json')
            qualified = (summary['convergence']['stabilized'] and
                         summary['validation']['one_step'] < .5 * summary['constant_velocity_error'])
            qualifications.append({'model_id': model, 'cell_id': definition['cell_id'], 'qualified': qualified})
            children.append({'role': 'training', 'path': str(path.relative_to(root)), 'sha256': sha256_file(path)})
            grid = read_json(folder / (definition['cell_id'] + '-nominal_grid.json'))
            paths = {str(Path(attempt['manifest']).parent / 'resources.json') for attempt in grid['attempts']}
            fit_seconds[model] = sum(read_json(root / p)['wall_seconds'] for p in paths)
        freeze = root / 'config/core_negative_development_freeze.json'
        freeze_recipes(root, freeze, training, 'exploratory')
        atomic_json(folder / 'children.json', {'records': children})
        atomic_json(folder / 'freeze.json', read_json(freeze))
        projection = project_confirmation_resources(fit_seconds, 5)
        atomic_json(folder / 'confirmation_projection.json', projection)
        atomic_json(root / 'config/core_negative_resources.json', projection)
        return [{'check_id': name, 'status': 'pass', 'scope': 'registered-negative-readiness',
                 'tolerance': None, 'reason': 'Valid request and complete nominal-only fitting attempts; qualification evaluated separately'}
                for name in READINESS_CHECKS], {
                    'extension': 'mechanism-negative', 'request_id': request['request_id'],
                    'qualifications': qualifications, 'confirmation_run': False, 'missing_cells': [],
                    'freeze_sha256': sha256_file(freeze), 'criteria_sha256': sha256_file(root / 'config/criteria.json'),
                    'statistical_family_size': 280, 'statistical_look': None,
                    'confirmation_projection': projection}
    return run_job(root, 'MX', 'calibration',
        [*_sources(root), str(request_path.relative_to(root)), *inputs], READINESS_CHECKS,
        [predecessor], produce, config_keys={str(request_path.relative_to(root)): request,
            'config/criteria.json': read_json(root / 'config/criteria.json')})


def evaluate_readiness(root, evidence, seen):
    from .contracts import verify_run
    from .io import read_json, run_directory, sha256_file
    if len(evidence) != 1 or evidence[0]['phase'] != 'calibration' or set(seen) != set(READINESS_CHECKS):
        raise ValueError('GX requires the complete registered negative-readiness contract')
    folder = run_directory(root, evidence[0])
    request = validate_readiness_request(read_json(folder / 'request.json'))
    summary = read_json(folder / 'summary.json')
    if summary.get('extension') != 'mechanism-negative' or summary['request_id'] != request['request_id']:
        raise ValueError('Readiness scope changed')
    children = read_json(folder / 'children.json')['records']
    if len(children) != 2:
        raise ValueError('Missing nominal readiness training cells')
    qualification = {}
    for child in children:
        path = root / child['path']
        if sha256_file(path) != child['sha256']:
            raise ValueError('Changed nominal readiness evidence')
        record = read_json(path)
        verify_run(root, record)
        trained = read_json(path.parent / 'summary.json')
        parameters = trained['parameters']
        if (record['phase'] != 'exploratory' or parameters['actuator'] != 'direct'
                or parameters['condition'] != 'A' or parameters['loss_mode'] != 'shared'
                or parameters['seed'] != 9000 or parameters['model_id'] not in {'node_generic', 'power_correct'}):
            raise ValueError('Readiness must qualify the original nominal development recipes')
        model = parameters['model_id']
        if model in qualification:
            raise ValueError('Duplicate readiness model')
        qualification[model] = (trained['convergence']['stabilized'] and
                               trained['validation']['one_step'] < .5 * trained['constant_velocity_error'])
    if set(qualification) != {'node_generic', 'power_correct'}:
        raise ValueError('Readiness comparator/method coverage changed')
    if qualification != {row['model_id']: row['qualified'] for row in summary['qualifications']}:
        raise ValueError('Cached nominal qualifications disagree with their raw training receipts')
    qualified = all(qualification.values())
    rows = [{'criterion_id': name, 'status': 'pass' if qualified else 'unresolved',
        'estimate': 1. if qualified else None, 'bounds': None, 'threshold': None,
        'operator': 'qualification', 'reason': 'Qualified original recipes permit a limited independent rejection test'
        if qualified else 'Nominal optimization remains unqualified; no scientific conclusion'} for name in READINESS_CHECKS]
    return ('GO' if qualified else 'INCONCLUSIVE'), 'valid', rows


def run_rejection(root, resume=False, seeds=None):
    from .execution import parent, run_job
    from .io import atomic_json, read_json, run_directory, sha256_file, canonical_json
    from .mechanism import cell, train_cell, run_worker, freeze_recipes, sensitivity_contrasts, _metrics
    from .statistics import reserve_look
    root = Path(root).resolve()
    seeds = validate_confirmation_seeds(list(range(5)) if seeds is None else seeds)
    predecessor = parent(root, 'GX')
    receipt = read_json(root / predecessor['evidence'][0]['path'])
    if read_json(run_directory(root, receipt) / 'summary.json').get('extension') != 'mechanism-negative':
        raise ValueError('Limited rejection requires its own qualified registered GX request')
    scope = {'protocol_version': 'PTCO-1.0', 'negative_only': True,
        'actuator': 'direct', 'competitor': 'node_generic', 'condition': 'A', 'loss_mode': 'shared',
        'seed_ids': seeds, 'statistical_family_size': 280,
        'criterion_ids': [c['criterion_id'] for c in scope_criteria()],
        'positive_return_milestone': 'M30', 'readiness_decision_id': predecessor['decision_id']}
    validate_scope(scope)
    scope_path = root / 'config/rejection_subset_scope.json'
    if scope_path.exists() and canonical_json(read_json(scope_path)) != canonical_json(scope):
        raise ValueError('Changed rejection subset must be registered and versioned before new outcomes')
    atomic_json(scope_path, scope)

    def produce(folder):
        initial = root / 'config/rejection_freeze.json'
        freeze_recipes(root, initial, [], 'confirmatory')
        atomic_json(folder / 'initial_freeze.json', read_json(initial))
        atomic_json(folder / 'scope.json', scope)
        corpus = run_worker(root, folder, 'generate', {'actuator': 'direct', 'condition': 'A',
            'role': 'confirmation', 'n_trajectories': 256}, resume)
        corpus_path = run_directory(root, corpus) / 'manifest.json'
        children = [{'role': 'corpus', 'path': str(corpus_path.relative_to(root)), 'sha256': sha256_file(corpus_path)}]
        measured = {}
        for seed in seeds:
            for model in ('node_generic', 'power_correct'):
                definition = cell('direct', 'A', 'shared', model, seed)
                trained_record = train_cell(root, folder, definition, 'confirmatory', resume)
                training_path = run_directory(root, trained_record) / 'manifest.json'
                trained = read_json(training_path.parent / 'summary.json')
                qualified = (trained['convergence']['stabilized'] and
                             trained['validation']['one_step'] < .5 * trained['constant_velocity_error'])
                children.append({'role': 'training', 'path': str(training_path.relative_to(root)), 'sha256': sha256_file(training_path)})
                selected = root / 'config/rejection_checkpoint_freezes' / (definition['cell_id'] + '.json')
                freeze_recipes(root, selected, [trained_record], 'confirmatory')
                parameters = {**definition, 'training_manifest': str(training_path.relative_to(root)),
                    'corpus_manifest': 'data/direct/A/confirmation/manifest.json', 'freeze_path': str(selected.relative_to(root))}
                job = {'protocol_version': 'PTCO-1.0', 'phase': 'confirmatory', 'purpose': 'evaluation', 'semantic_inputs': parameters}
                audit = run_worker(root, folder, 'E01', job, resume)
                audit_path = run_directory(root, audit) / 'manifest.json'
                value = read_json(audit_path.parent / 'summary.json')
                value['training_qualified'] = qualified
                key = ('direct', 'A', 'shared', model, None, seed)
                child = {'role': 'E01', 'path': str(audit_path.relative_to(root)), 'sha256': sha256_file(audit_path),
                         'key': list(key), 'training_manifest': str(training_path.relative_to(root))}
                if model == 'node_generic':
                    refined = run_worker(root, folder, 'E01', {**job,
                        'semantic_inputs': {**parameters, 'solver_substeps': 2}}, resume)
                    refined_path = run_directory(root, refined) / 'manifest.json'
                    value['solver_metrics'] = _metrics(read_json(refined_path.parent / 'summary.json'))
                    child['solver_manifest'] = str(refined_path.relative_to(root))
                    children.append({'role': 'solver', 'path': str(refined_path.relative_to(root)), 'sha256': sha256_file(refined_path)})
                children.append(child)
                measured[key] = value
        atomic_json(folder / 'children.json', {'records': children})
        identity = {'run_hashes': [child['sha256'] for child in children],
            'criteria': sha256_file(root / 'config/criteria.json'), 'rejection_freeze': sha256_file(initial),
            'scope': scope, 'scope_sha256': sha256_file(scope_path)}
        look = reserve_look(root, identity)
        summaries = {c['criterion_id']: sensitivity_contrasts(c, measured, seeds) for c in scope_criteria()}
        rows = evaluate_subset(summaries, seeds, look['look_index'])
        atomic_json(folder / 'contrast_summaries.json', summaries)
        atomic_json(folder / 'criterion_evaluations.json', {'rows': rows, 'look': look})
        return [{'check_id': name, 'status': 'pass', 'scope': 'negative-only-direct-NODE', 'tolerance': None,
                 'reason': 'Complete frozen subset, independent paired seeds, original K=280, numerical and precision rules'}
                for name in REJECTION_CHECKS], {'negative_only': True, 'scope': scope, 'look': look,
                    'criterion_evaluations': rows, 'contrast_summaries': summaries, 'missing_cells': [],
                    'complete_core_coverage': False, 'full_suite_claimed': False}
    return run_job(root, 'MR', 'confirmatory', _sources(root), REJECTION_CHECKS, [predecessor], produce,
        config_keys={'config/criteria.json': read_json(root / 'config/criteria.json'),
                     str(scope_path.relative_to(root)): scope})


def evaluate_rejection(root, evidence, seen):
    from .contracts import verify_run
    from .io import canonical_json, read_json, run_directory, sha256_file
    from .mechanism import cell, sensitivity_contrasts, _metrics
    from .probes import summarize_probes
    import numpy as np
    if (len(evidence) != 1 or evidence[0]['phase'] != 'confirmatory'
            or set(seen) != set(REJECTION_CHECKS) or any(c['status'] != 'pass' for c in seen.values())):
        raise ValueError('GR needs complete registered negative evidence')
    folder = run_directory(root, evidence[0])
    summary = read_json(folder / 'summary.json')
    scope = validate_scope(summary['scope'])
    seeds = validate_confirmation_seeds(scope['seed_ids'])
    expected = {c['criterion_id'] for c in scope_criteria()}
    if (not summary.get('negative_only') or scope['statistical_family_size'] != 280
            or set(scope['criterion_ids']) != expected or scope['positive_return_milestone'] != 'M30'):
        raise ValueError('Rejection subset scope or full-family rule changed')
    children = read_json(folder / 'children.json')['records']
    if len(children) != 1 + 5 * len(seeds):
        raise ValueError('Missing rejection subset child evidence')
    roles = [child['role'] for child in children]
    if (roles.count('corpus') != 1 or roles.count('training') != 2*len(seeds)
            or roles.count('E01') != 2*len(seeds) or roles.count('solver') != len(seeds)):
        raise ValueError('Rejection subset child roles are incomplete or unexpected')
    records = {}
    for child in children:
        path = root / child['path']
        if child['path'] in records or sha256_file(path) != child['sha256']:
            raise ValueError('Duplicate or changed rejection subset child evidence')
        record = read_json(path)
        verify_run(root, record)
        if record['phase'] != 'confirmatory':
            raise ValueError('Discovery evidence cannot become confirmation')
        records[child['path']] = record

    def audit_summary(path):
        directory = run_directory(root, records[path])
        cached = read_json(directory / 'summary.json')
        with np.load(directory / 'jacobians.npz', allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        derived = summarize_probes(arrays, [str(arrays['cell_id'][0])])
        if canonical_json(derived) != canonical_json(cached['metrics']) or cached['failed_rows']:
            raise ValueError('Raw rejection Jacobians disagree with their complete cached summary')
        return cached

    measured = {}
    expected_keys = {('direct', 'A', 'shared', model, None, seed)
                     for model in ('node_generic', 'power_correct') for seed in seeds}
    for child in children:
        if child['role'] != 'E01':
            continue
        key = tuple(child['key'])
        if key not in expected_keys or key in measured or child['training_manifest'] not in records:
            raise ValueError('Unexpected, duplicated or unpaired rejection model/seed')
        value = audit_summary(child['path'])
        trained_path = run_directory(root, records[child['training_manifest']])
        trained = read_json(trained_path / 'summary.json')
        definition = cell('direct', 'A', 'shared', key[3], key[5])
        if (trained['cell_id'] != definition['cell_id'] or value['cell_id'] != definition['cell_id']
                or value['checkpoint_hash'] != sha256_file(trained_path / 'weights.pt')):
            raise ValueError('Rejection audit does not use its frozen paired checkpoint')
        value['training_qualified'] = (trained['convergence']['stabilized'] and
            trained['validation']['one_step'] < .5 * trained['constant_velocity_error'])
        if key[3] == 'node_generic':
            if child.get('solver_manifest') not in records:
                raise ValueError('Strong NODE comparator needs its same-weights solver audit')
            refined = audit_summary(child['solver_manifest'])
            if refined['checkpoint_hash'] != value['checkpoint_hash']:
                raise ValueError('Solver refinement changed the trained vector field')
            value['solver_metrics'] = _metrics(refined)
        measured[key] = value
    if set(measured) != expected_keys:
        raise ValueError('Incomplete rejection subset model/seed coverage')
    look = summary['look']
    recorded_look = read_json(root / 'decisions/statistical_looks' / f'look-{look["look_index"]:05d}.json')
    if (canonical_json(look) != canonical_json(recorded_look) or look['family_size'] != 280
            or look['evidence_identity']['run_hashes'] != [c['sha256'] for c in children]):
        raise ValueError('Rejection statistical look is stale or uses a reduced family')
    summaries = {c['criterion_id']: sensitivity_contrasts(c, measured, seeds) for c in scope_criteria()}
    if canonical_json(summaries) != canonical_json(summary['contrast_summaries']):
        raise ValueError('Paired subset contrasts disagree with raw evidence')
    rows = evaluate_subset(summaries, seeds, look['look_index'])
    if canonical_json(rows) != canonical_json(summary['criterion_evaluations']):
        raise ValueError('Subset gate disagrees with independent threshold evaluation')
    if any(row['status'] == 'invalid' for row in rows):
        raise ValueError('Invalid subset criterion measurement')
    outcome = ('NO-GO' if any(row['status'] == 'fail' for row in rows) else
               'GO' if all(row['status'] == 'pass' for row in rows) else 'INCONCLUSIVE')
    return outcome, 'valid', rows
