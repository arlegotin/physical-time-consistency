"""Independent rejection-only mechanism confirmation; uses the full K=280 family."""
from pathlib import Path

CHECKS = ('MR.freeze','MR.coverage','MR.statistics')


def describe(root):
    if (root / 'config/core_negative_development_freeze.json').exists():
        from poc.io import read_json
        projection = root / 'config/core_negative_resources.json'
        return {'expected_cells': list(CHECKS), 'negative_only': True,
            'expected_jobs': [{'job_id': 'registered-direct-NODE-rejection-subset', 'actuator': 'direct',
                'models': ['node_generic', 'power_correct'], 'training_seeds': [0, 1, 2, 3, 4],
                'model_fits': 10, 'E01_jobs': 15, 'family_size': 280, 'tested_atomic_criteria': 8}],
            'resource_projection': read_json(projection) if projection.exists() else {'status': 'needs_measured_readiness'},
            'prohibiting_conditions': ['Registered GX readiness and original full-family freeze required; no PASS from subset']}
    return {'expected_cells':list(CHECKS),'expected_jobs':[{'job_id':'rejection-only-core-confirmation',
            'training_seeds':[0,1,2,3,4],'family_size':280}],
            'resource_projection':{'wall_seconds':None,'owner_imposed_caps':None,
                                   'note':'Project from current M30 measured fits; may stop after a supported mandatory rejection'}}


def run(root: Path,resume=False):
    if (root / 'config/core_negative_development_freeze.json').exists():
        from poc.gates import load_inputs
        from poc.io import read_json, run_directory
        selected = load_inputs(root, 'G30')
        full_development = (len(selected) == 1 and
            set(selected[0]['expected_cells']) == {'M30.data', 'M30.training', 'M30.nominal_freeze', 'M30.audit'})
        if not full_development:
            from poc.rejection_subset import run_rejection
            return run_rejection(root, resume)
    from poc.mechanism_run import run_rejection
    return run_rejection(root,resume)


def evaluate(root,evidence,seen):
    from poc.io import read_json,run_directory
    if len(evidence) != 1 or evidence[0]['phase'] != 'confirmatory' or set(seen) != set(CHECKS):
        raise ValueError('GR requires a complete rejection-confirmation receipt')
    summary = read_json(run_directory(root,evidence[0])/'summary.json')
    if summary.get('negative_only'):
        from poc.rejection_subset import evaluate_rejection
        return evaluate_rejection(root, evidence, seen)
    evaluations = summary['criterion_evaluations']
    from poc.contracts import verify_run
    from poc.io import sha256_file,canonical_json
    from poc.probes import summarize_probes
    from poc.mechanism import sensitivity_contrasts,_metrics
    import numpy as np
    children = read_json(run_directory(root,evidence[0])/'children.json')['records']
    by_path = {}
    for child in children:
        if child['path'] in by_path or sha256_file(root/child['path']) != child['sha256']:
            raise ValueError('Duplicate or changed rejection child evidence')
        record = read_json(root/child['path'])
        verify_run(root,record)
        by_path[child['path']] = record
    measured = {}
    for child in children:
        if child['role'] != 'E01':
            continue
        folder = run_directory(root,by_path[child['path']])
        audit = read_json(folder/'summary.json')
        with np.load(folder/'jacobians.npz',allow_pickle=False) as archive:
            rows = {name:archive[name] for name in archive.files}
        metrics = summarize_probes(rows,[str(rows['cell_id'][0])])
        if canonical_json(metrics) != canonical_json(audit['metrics']):
            raise ValueError('Rejection metrics disagree with complete raw Jacobians')
        if child['training_manifest'] not in by_path:
            raise ValueError('Rejection audit lacks its qualified training checkpoint')
        trained = read_json(run_directory(root,by_path[child['training_manifest']])/'summary.json')
        audit['training_qualified'] = trained['convergence']['stabilized'] and trained['validation']['one_step'] < .5*trained['constant_velocity_error']
        if child.get('solver_manifest'):
            if child['solver_manifest'] not in by_path:
                raise ValueError('Rejection comparator lacks its same-field solver audit')
            folder = run_directory(root,by_path[child['solver_manifest']])
            refined = read_json(folder/'summary.json')
            with np.load(folder/'jacobians.npz',allow_pickle=False) as archive:
                rows = {name:archive[name] for name in archive.files}
            if canonical_json(summarize_probes(rows,[str(rows['cell_id'][0])])) != canonical_json(refined['metrics']):
                raise ValueError('Rejection solver summary disagrees with its raw Jacobians')
            audit['solver_metrics'] = _metrics(refined)
        key = tuple(child['key'])
        if key in measured:
            raise ValueError('Duplicate confirmatory model/seed cell')
        measured[key] = audit
    # Recompute from saved paired seed summaries; cached green/red decisions are not evidence.
    from poc.criterion_gates import evaluate_criterion
    from poc.criteria import expand_criteria
    registry = {c['criterion_id']:c for c in expand_criteria()}
    expected = {c['criterion_id'] for c in registry.values() if c['statistical'] and c['criterion_id'].startswith(
        ('K02.relative/','K02.absolute/','K02.order/','K02.gain/','K02.actuator/','K03.jacobian/'))
        and c['dimensions'].get('competitor','mlp_dt') in ('mlp_dt','gru_dt','node_generic')}
    if {row['criterion_id'] for row in evaluations} != expected or len(evaluations) != len(expected):
        raise ValueError('Rejection core criterion coverage changed')
    for identifier in expected:
        derived_summary = sensitivity_contrasts(registry[identifier],measured,list(range(5)))
        if canonical_json(derived_summary) != canonical_json(summary['contrast_summaries'][identifier]):
            raise ValueError('Rejection contrasts disagree with paired raw evidence')
    derived = [evaluate_criterion(registry[row['criterion_id']],summary['contrast_summaries'][row['criterion_id']],
                                  summary['look']['look_index']) for row in evaluations]
    if derived != evaluations:
        raise ValueError('Rejection gate disagrees with independent contrast re-evaluation')
    if any(row['status']=='invalid' for row in derived):
        raise ValueError('Invalid confirmatory contrast')
    outcome = 'NO-GO' if any(row['status']=='fail' for row in derived) else (
        'GO' if summary['complete_core_coverage'] and all(row['status']=='pass' for row in derived) else 'INCONCLUSIVE')
    return outcome,'valid',derived


def transition(outcome,evaluations):
    from poc.rejection_subset import scope_criteria, subset_transition
    if {row['criterion_id'] for row in evaluations} == {c['criterion_id'] for c in scope_criteria()}:
        transition = subset_transition(outcome)
        transition['look_index'] = next((row['look_index'] for row in evaluations if 'look_index' in row), None)
        transition['alpha_spent'] = next((row['alpha_spent'] for row in evaluations if 'alpha_spent' in row), None)
        return transition
    if outcome == 'NO-GO':
        actions,verdict = [{'milestone_id':'MF','branch':None,'question':'Report the valid scoped mandatory mechanism rejection'}],'POC FAIL'
    elif outcome == 'GO':
        actions,verdict = [{'milestone_id':'M40','branch':None,'question':'Core mechanism supports control evaluation; remaining suite still required'}],None
    else:
        branch = next((row['needs_branch'] for row in evaluations if row.get('needs_branch')),None)
        actions,verdict = [{'milestone_id':'MX','branch':(branch or 'precision') if outcome=='INCONCLUSIVE' else 'repair',
                          'question':'Resolve the named frozen sensitivity contrast'}],None
    return {'permitted_next_actions':actions,'terminal_verdict':verdict,
            'look_index':next((row['look_index'] for row in evaluations if 'look_index' in row),None),
            'alpha_spent':next((row['alpha_spent'] for row in evaluations if 'alpha_spent' in row),None)}
