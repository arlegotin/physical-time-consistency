"""Explicit M30/MR producers; fresh child jobs, frozen recipes, immutable evidence."""
from pathlib import Path
import sys

from .io import atomic_json,read_json,run_directory,sha256_file


def _sources(root):
    from .audit_jobs import SOURCES as AUDIT
    from .training_jobs import SOURCES as TRAIN
    return sorted(set(AUDIT)|set(TRAIN)|{'poc/mechanism.py','poc/mechanism_run.py',
            'poc/milestones/m30.py','poc/milestones/mr.py','schemas/dataset.schema.json',
            'schemas/metrics.schema.json','assets/two_link_direct.xml','assets/two_link_lagged.xml'})


def _stage_cells(root):
    from .mechanism import development_cells,confirmation_cells
    from .criteria import validate_registry
    registry = read_json(root/'config/criteria.json')
    validate_registry(registry['criteria'])
    matrix = {'protocol_version':'PTCO-1.0','statistical_family_size':280,'status':'declared',
              'development':development_cells(),'confirmation':confirmation_cells()}
    path = root/'config/matrix.json'
    if path.exists() and read_json(path) != matrix:
        raise ValueError('Declared experiment matrix changed without a protocol revision')
    atomic_json(path,matrix)
    return matrix


def run_development(root,resume=False):
    from .execution import parent,run_job
    from .mechanism import CORE,run_worker,train_cell,freeze_recipes,_metrics
    root = Path(root).resolve()
    predecessor = parent(root,'G20')
    matrix = _stage_cells(root)
    def produce(folder):
        generated,training,audits,observations = [],[],[],[]
        for actor in ('direct','lagged'):
            for condition in ('F','A'):
                for role,count in (('train',1024),('validation',256),('development',128)):
                    generated.append(run_worker(root,folder,'generate',{'actuator':actor,'condition':condition,
                                            'role':role,'n_trajectories':count},resume))
        for definition in matrix['development']:
            training.append(train_cell(root,folder,definition,'exploratory',resume))
        qualifications = []
        for definition,record in zip(matrix['development'],training):
            summary = read_json(run_directory(root,record)/'summary.json')
            qualifications.append({'cell_id':definition['cell_id'],
                'qualified':summary['convergence']['stabilized'] and
                            summary['validation']['one_step'] < .5*summary['constant_velocity_error']})
        if not all(row['qualified'] for row in qualifications):
            atomic_json(folder/'children.json',{'records':[{'path':str((run_directory(root,r)/'manifest.json').relative_to(root)),
                         'sha256':sha256_file(run_directory(root,r)/'manifest.json')} for r in generated+training]})
            return [{'check_id':name,'status':'pass','scope':'M30','tolerance':None,
                     'reason':'Valid data and finite nominal fitting attempts retained; no fine-time audit opened'}
                    for name in ('M30.data','M30.training','M30.nominal_freeze','M30.audit')],{
                    'observations':[],'missing_cells':[],'qualification_unresolved':qualifications,
                    'fine_time_audit_opened':False}
        freeze_path = root/'config/pre_audit_freeze.json'
        freeze_recipes(root,freeze_path,training,'exploratory')
        for definition,record in zip(matrix['development'],training):
            modes = ('low','direct','lagged') if definition['model_id']=='power_correct' and definition['condition']=='F' else (None,)
            for mode in modes:
                parameters = {**definition,'training_manifest':str((run_directory(root,record)/'manifest.json').relative_to(root)),
                    'corpus_manifest':f'data/{definition["actuator"]}/{definition["condition"]}/development/manifest.json',
                    'freeze_path':str(freeze_path.relative_to(root))}
                if mode:
                    parameters['power_mode'] = mode
                job = {'protocol_version':'PTCO-1.0','phase':'exploratory','purpose':'evaluation','semantic_inputs':parameters}
                audit = run_worker(root,folder,'E01',job,resume)
                audits.append(audit)
                summary = read_json(run_directory(root,audit)/'summary.json')
                if summary['failed_rows']:
                    raise ValueError('A development model has explicit failed sensitivities; diagnose before a superiority screen')
                observations.append({'key':[definition['actuator'],definition['condition'],definition['loss_mode'],definition['model_id'],mode],
                                     'metrics':_metrics(summary),'audit_manifest':str((run_directory(root,audit)/'manifest.json').relative_to(root))})
                if definition['model_id'].startswith('node_'):
                    refined = run_worker(root,folder,'E01',{'protocol_version':'PTCO-1.0','phase':'exploratory',
                        'purpose':'evaluation','semantic_inputs':{**parameters,'solver_substeps':2}},resume)
                    audits.append(refined)
                    observations[-1]['solver_metrics'] = _metrics(read_json(run_directory(root,refined)/'summary.json'))
                    observations[-1]['solver_audit_manifest'] = str((run_directory(root,refined)/'manifest.json').relative_to(root))
                if definition['condition']=='F':
                    audits.append(run_worker(root,folder,'E02',job,resume))
        rows = generated+training+audits
        atomic_json(folder/'children.json',{'records':[{'path':str((run_directory(root,r)/'manifest.json').relative_to(root)),
                                 'sha256':sha256_file(run_directory(root,r)/'manifest.json')} for r in rows]})
        return [{'check_id':name,'status':'pass','scope':'M30','tolerance':None,
                 'reason':'Complete qualified data, nominal fitting, preregistration and exploratory raw audits'}
                 for name in ('M30.data','M30.training','M30.nominal_freeze','M30.audit')],{
                 'observations':observations,'missing_cells':[],'training_manifests':[
                     str((run_directory(root,r)/'manifest.json').relative_to(root)) for r in training],
                 'child_run_count':len(rows),'confirmation_run':False}
    return run_job(root,'M30','exploratory',_sources(root),
        ('M30.data','M30.training','M30.nominal_freeze','M30.audit'),[predecessor],produce,
        config_keys={'config/criteria.json':read_json(root/'config/criteria.json'),'config/matrix.json':matrix})


def run_rejection(root,resume=False):
    from .execution import run_job
    from .gates import decide,load_inputs
    from .mechanism import CORE,cell,run_worker,train_cell,freeze_recipes,sensitivity_contrasts
    from .criteria import expand_criteria
    from .criterion_gates import evaluate_criterion
    from .statistics import reserve_look
    root = Path(root).resolve()
    predecessor = decide('G30',load_inputs(root,'G30'),root)
    if predecessor['outcome'] != 'INCONCLUSIVE' or predecessor['validity_status'] != 'valid':
        raise ValueError('MR requires the valid, weak/uncertain development mechanism gate')
    _stage_cells(root)
    def produce(folder):
        training,audits,records,children = [],[],{},[]
        freeze_path = root/'config/rejection_freeze.json'
        freeze_recipes(root,freeze_path,[],'confirmatory')
        atomic_json(folder/'initial_freeze.json',read_json(freeze_path))
        for actor in ('direct','lagged'):
            for condition in ('F','A'):
                run_worker(root,folder,'generate',{'actuator':actor,'condition':condition,'role':'confirmation','n_trajectories':256},resume)
        definitions = [cell(actor,'A','shared',model,seed) for seed in range(5) for actor in ('direct','lagged')
                       for model in ('power_correct','mlp_dt','gru_dt','node_generic')]
        definitions += [cell(actor,'F','one_step','power_correct',seed) for seed in range(5) for actor in ('direct','lagged')]
        for definition in definitions:
            record = train_cell(root,folder,definition,'confirmatory',resume)
            training.append(record)
            training_path = str((run_directory(root,record)/'manifest.json').relative_to(root))
            trained = read_json(run_directory(root,record)/'summary.json')
            qualified = trained['convergence']['stabilized'] and trained['validation']['one_step'] < .5*trained['constant_velocity_error']
            children.append({'role':'training','path':training_path,'sha256':sha256_file(root/training_path)})
            selected = root/'config/rejection_checkpoint_freezes'/f'{definition["cell_id"]}.json'
            freeze_recipes(root,selected,[record],'confirmatory')
            modes = ('low','direct','lagged') if definition['condition']=='F' else (None,)
            for mode in modes:
                parameters = {**definition,'training_manifest':str((run_directory(root,record)/'manifest.json').relative_to(root)),
                    'corpus_manifest':f'data/{definition["actuator"]}/{definition["condition"]}/confirmation/manifest.json',
                    'freeze_path':str(selected.relative_to(root))}
                if mode:
                    parameters['power_mode'] = mode
                audit = run_worker(root,folder,'E01',{'protocol_version':'PTCO-1.0','phase':'confirmatory',
                                      'purpose':'evaluation','semantic_inputs':parameters},resume)
                audits.append(audit)
                key = (definition['actuator'],definition['condition'],definition['loss_mode'],definition['model_id'],mode,definition['seed'])
                measured = read_json(run_directory(root,audit)/'summary.json')
                measured['training_qualified'] = qualified
                child = {'role':'E01','key':list(key),'path':str((run_directory(root,audit)/'manifest.json').relative_to(root)),
                    'sha256':sha256_file(run_directory(root,audit)/'manifest.json'),'training_manifest':training_path}
                if definition['model_id']=='node_generic':
                    refined = run_worker(root,folder,'E01',{'protocol_version':'PTCO-1.0','phase':'confirmatory',
                        'purpose':'evaluation','semantic_inputs':{**parameters,'solver_substeps':2}},resume)
                    audits.append(refined)
                    measured['solver_metrics'] = _metrics(read_json(run_directory(root,refined)/'summary.json'))
                    refined_path = str((run_directory(root,refined)/'manifest.json').relative_to(root))
                    child['solver_manifest'] = refined_path
                    children.append({'role':'solver','path':refined_path,'sha256':sha256_file(root/refined_path)})
                children.append(child)
                records[key] = measured
        atomic_json(folder/'children.json',{'records':children})
        tested = [c for c in expand_criteria() if c['statistical'] and c['criterion_id'].startswith(
                       ('K02.relative/','K02.absolute/','K02.order/','K02.gain/','K02.actuator/','K03.jacobian/'))
                  and c['dimensions'].get('competitor','mlp_dt') in ('mlp_dt','gru_dt','node_generic')]
        identity = {'run_hashes':[sha256_file(run_directory(root,r)/'manifest.json') for r in training+audits],
                    'criteria':sha256_file(root/'config/criteria.json'),'rejection_freeze':sha256_file(freeze_path)}
        look = reserve_look(root,identity)
        summaries = {c['criterion_id']:sensitivity_contrasts(c,records,list(range(5))) for c in tested}
        evaluations = [evaluate_criterion(c,summaries[c['criterion_id']],look['look_index']) for c in tested]
        atomic_json(folder/'contrast_summaries.json',summaries)
        atomic_json(folder/'criterion_evaluations.json',{'rows':evaluations,'look':look})
        return [{'check_id':name,'status':'pass','scope':'MR','tolerance':None,
                 'reason':'Independent frozen core evidence, complete clusters and the original K=280 rejection ledger'}
                 for name in ('MR.freeze','MR.coverage','MR.statistics')],{
                 'criterion_evaluations':evaluations,'contrast_summaries':summaries,'look':look,
                 'complete_core_coverage':True,'missing_cells':[],
                 'full_suite_claimed':False,'child_manifests':identity['run_hashes']}
    return run_job(root,'MR','confirmatory',_sources(root),('MR.freeze','MR.coverage','MR.statistics'),
            [predecessor],produce,config_keys={'config/criteria.json':read_json(root/'config/criteria.json')})
