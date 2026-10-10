"""Minimal exploratory mechanism audit; never a scientific rejection from one seed."""
from pathlib import Path

CHECKS = ('M30.data','M30.training','M30.nominal_freeze','M30.audit')


def describe(root):
    # Keep CLI dry-run standard-library-only and do not open model outcomes.
    models = ('mlp_dt','gru_dt','node_generic','power_correct','node_structured')
    jobs = []
    for actor in ('direct','lagged'):
        for condition in ('F','A'):
            for role,count in (('train',1024),('validation',256),('development',128)):
                jobs.append({'job_id':f'{actor}/{condition}/{role}','kind':'generate','trajectories':count})
            for model in models:
                jobs.append({'job_id':f'{actor}__{condition}__one_step__{model}__seed9000','kind':'train'})
        for model in models:
            jobs.append({'job_id':f'{actor}__A__shared__{model}__seed9000','kind':'train'})
    return {'expected_cells':list(CHECKS),'expected_jobs':jobs,
            'resource_projection':{'initial_training_fits':30,'label_transitions':180224,
              'expected_data_physics_steps':28_835_840,'owner_imposed_caps':None,
              'wall_seconds':None,'note':'Measure job throughput; nominal-only optimizer extensions are conditional'},
            'prohibiting_conditions':['G20 must be current GO','Confirmation remains sealed/absent until an actual freeze']}


def run(root: Path,resume=False):
    from poc.mechanism_run import run_development
    return run_development(root,resume)


def evaluate(root,evidence,seen):
    from poc.io import read_json,run_directory
    from poc.mechanism import development_screen
    if len(evidence) != 1 or evidence[0]['phase'] != 'exploratory' or set(seen) != set(CHECKS):
        raise ValueError('G30 requires the complete exploratory producer, not an individual dataset or fit')
    summary = read_json(run_directory(root,evidence[0])/'summary.json')
    if summary.get('qualification_unresolved'):
        return 'INCONCLUSIVE','valid',[{'criterion_id':'G30.nominal_adequacy','status':'unresolved',
            'estimate':None,'bounds':None,'threshold':None,'operator':'qualification',
            'reason':'A valid nominal fit remains unqualified after the prespecified optimizer grid; no fine-time outcomes opened'}]
    from poc.contracts import verify_run
    from poc.io import sha256_file,canonical_json
    children = read_json(run_directory(root,evidence[0])/'children.json')['records']
    paths = set()
    for child in children:
        if sha256_file(root/child['path']) != child['sha256']:
            raise ValueError('Development child manifest hash mismatch')
        verify_run(root,read_json(root/child['path']))
        paths.add(child['path'])
    from poc.probes import summarize_probes
    import numpy as np
    for observation in summary['observations']:
        if observation['audit_manifest'] not in paths:
            raise ValueError('Development summary names an unrecorded child audit')
        child = read_json(root/observation['audit_manifest'])
        with np.load(run_directory(root,child)/'jacobians.npz',allow_pickle=False) as archive:
            rows = {name:archive[name] for name in archive.files}
        derived = summarize_probes(rows,[str(rows['cell_id'][0])])
        if canonical_json(next(iter(derived.values()))) != canonical_json(observation['metrics']):
            raise ValueError('Development metrics disagree with complete raw probe rows')
        if observation.get('solver_audit_manifest'):
            if observation['solver_audit_manifest'] not in paths:
                raise ValueError('Development solver audit is not in the complete child closure')
            child = read_json(root/observation['solver_audit_manifest'])
            with np.load(run_directory(root,child)/'jacobians.npz',allow_pickle=False) as archive:
                rows = {name:archive[name] for name in archive.files}
            derived = summarize_probes(rows,[str(rows['cell_id'][0])])
            if canonical_json(next(iter(derived.values()))) != canonical_json(observation['solver_metrics']):
                raise ValueError('Development solver metrics disagree with raw probe rows')
    observations = {tuple(row['key']):row['metrics'] for row in summary['observations']}
    solvers = {tuple(row['key']):row['solver_metrics'] for row in summary['observations'] if row.get('solver_metrics')}
    result = development_screen(observations,solvers)
    return result['outcome'],'valid',result['criterion_evaluations']


def transition(outcome,evaluations):
    if outcome == 'GO':
        actions = [{'milestone_id':'M40','branch':None,'question':'Core exploratory mechanism supports testing controller relevance'}]
    elif outcome == 'INCONCLUSIVE':
        nominal = any(row['criterion_id']=='G30.nominal_adequacy' for row in evaluations)
        actions = [{'milestone_id':'MX' if nominal else 'MR','branch':'optimizer' if nominal else None,
                    'question':'Resolve nominal training adequacy' if nominal else 'Confirm a potential mechanism rejection with independent seeds'}]
    else:
        actions = [{'milestone_id':'MX','branch':'repair','question':'Repair the specifically invalid development producer'}]
    return {'permitted_next_actions':actions,'terminal_verdict':None}
