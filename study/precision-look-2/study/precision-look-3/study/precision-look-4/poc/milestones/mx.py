"""Named storage/input-cache repair receipts; unrelated MX work remains prohibited."""
from pathlib import Path

CHECKS = ('repair.counterexample','repair.resume','repair.regressions')


def _request_path(root):
    from poc.io import read_json,safe_path
    pointer = read_json(Path(root)/'decisions/next_action.json')
    if not pointer.get('request_path'):
        raise ValueError('MX needs the exact registered request path in the decision pointer')
    return safe_path(Path(root),pointer['request_path'])


def validate_request(request):
    if request.get('branch') == 'mechanism-negative':
        from poc.rejection_subset import validate_readiness_request
        return validate_readiness_request(request)
    if set(request.get('contract',{})) != {str(i) for i in range(1,13)}:
        raise ValueError('Repair request requires all twelve execution contract elements')
    if request.get('branch') != 'repair':
        raise ValueError('This registered MX implementation supports only its named checkpoint repair branch')
    scopes = {'checkpoint':{'poc/train.py','tests/test_training.py'},
              'history_cache':{'poc/history.py','poc/models/sequence.py','tests/test_history.py',
                               'tests/test_models.py','tests/test_training.py'}}
    kind = request.get('repair_kind','checkpoint')
    if kind not in scopes or set(request.get('changed_files',[])) != scopes[kind]:
        raise ValueError('Repair may change only the files in its named counterexample scope')
    for name in ('request_id','unresolved_question','counterexample','invalidated_run_ids','regression_tests','resource_projection'):
        if not request.get(name):
            raise ValueError('Missing repair contract field: '+name)
    if request.get('return_milestone') != 'M30' or request.get('protocol_version') != 'PTCO-1.0':
        raise ValueError('Checkpoint repair must return to the original M30 protocol')
    return request


def describe(root):
    from poc.io import read_json
    request = validate_request(read_json(_request_path(root)))
    if request['branch'] == 'mechanism-negative':
        from poc.rejection_subset import READINESS_CHECKS
        return {'expected_cells': list(READINESS_CHECKS),
            'expected_jobs': [{'job_id': 'qualify-prespecified-core-negative-scope', 'training_seed': 9000,
                'models': ['node_generic', 'power_correct'], 'condition': 'A', 'loss_mode': 'shared'}],
            'request_id': request['request_id'], 'resource_projection': request['resource_projection'],
            'prohibiting_conditions': ['No confirmation before qualified recipes and complete freeze']}
    return {'expected_cells':list(CHECKS),'expected_jobs':[{'job_id':'registered-checkpoint-repair'}],
            'request_id':request['request_id'],'resource_projection':request['resource_projection'],
            'prohibiting_conditions':['An explicit twelve-part checkpoint request must be registered']}


def run(root,resume=False):
    import hashlib
    import shutil
    import platform
    import resource
    import subprocess
    import sys
    import time
    import uuid
    from datetime import datetime,timezone
    from poc.contracts import verify_run
    from poc.execution import parent
    from poc.io import read_json,atomic_json,canonical_json,complete_run,job_key,job_lock,sha256_file,utc_now
    root = Path(root).resolve()
    request_path = _request_path(root)
    request = validate_request(read_json(request_path))
    if request['branch'] == 'mechanism-negative':
        from poc.rejection_subset import run_readiness
        return run_readiness(root, request, request_path, resume)
    registry = read_json(root/'config/milestones.json')['milestones']['MX']['requests']
    if not any(row['request_id']==request['request_id'] for row in registry):
        raise ValueError('Checkpoint repair request is not registered')
    predecessor = parent(root,'G20')
    files = ['poc/milestones/mx.py',*request['changed_files'],
             'poc/io.py','poc/contracts.py','poc/gates.py',str(request_path.relative_to(root))]
    dependencies = {name:sha256_file(root/name) for name in files}
    for entry in request['input_evidence']:
        path = root/entry['path']
        verify_run(root,read_json(path),check_dependencies=False)
        if sha256_file(path) != entry['sha256']:
            raise ValueError('Repair input-evidence hash mismatch')
        dependencies[str(path.relative_to(root))] = sha256_file(path)
    inputs = {'protocol_version':'PTCO-1.0','milestone_id':'MX','phase':'unit',
              'dependencies':dependencies,'request':request}
    key = job_key(inputs)
    with job_lock(key,root):
        for path in (root/'evidence/MX').glob('*/manifest.json'):
            record = read_json(path)
            if record['job_key']==key:
                verify_run(root,record)
                return record
        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8]
        folder = root/'evidence/MX'/run_id; folder.mkdir(parents=True)
        started,wall,cpu = utc_now(),time.perf_counter(),time.process_time()
        children_before = resource.getrusage(resource.RUSAGE_CHILDREN)
        argv = [sys.executable,'-m','pytest','-q',*request['regression_tests']]
        with (folder/'stdout.log').open('w') as output,(folder/'stderr.log').open('w') as errors:
            result = subprocess.run(argv,cwd=root,stdout=output,stderr=errors)
        if result.returncode:
            raise RuntimeError('Checkpoint repair regression still fails; preserve this attempt')
        (folder/'events.jsonl').write_text(canonical_json({'event':'checkpoint_repair_verified','utc':utc_now()}).decode()+'\n')
        atomic_json(folder/'request.json',request)
        shutil.copyfile(root/request['counterexample_log'],folder/'counterexample.log')
        checks = [{'check_id':name,'status':'pass',
                   'reason':request.get('validity_reason','Packed and legacy checkpoints preserve identical updates, targets and ragged permutations'),
                   'tolerance':0,'scope':request.get('repair_kind','checkpoint')} for name in CHECKS]
        atomic_json(folder/'validity.json',{'validity_status':'valid','checks':checks,'blocker':None})
        atomic_json(folder/'summary.json',{'request_id':request['request_id'],'return_milestone':'M30',
                    'changed_training_recipe':False,'scientific_result':None,'missing_cells':[],
                    'original_attempts_preserved':request['invalidated_run_ids']})
        usage = resource.getrusage(resource.RUSAGE_SELF)
        children = resource.getrusage(resource.RUSAGE_CHILDREN)
        atomic_json(folder/'resources.json',{'wall_seconds':time.perf_counter()-wall,
           'cpu_seconds':time.process_time()-cpu+children.ru_utime+children.ru_stime-children_before.ru_utime-children_before.ru_stime,
           'peak_rss_bytes':int(max(usage.ru_maxrss,children.ru_maxrss)*(1 if platform.system()=='Darwin' else 1024)),
           'rss_convention':'macOS bytes' if platform.system()=='Darwin' else 'Linux KiB normalized to bytes',
           'disk_bytes':sum(path.stat().st_size for path in folder.rglob('*') if path.is_file()),
           'original_estimate':request['resource_projection']})
        revision = subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,text=True)
        record = {'schema_version':1,'protocol_version':'PTCO-1.0','milestone_id':'MX','run_id':run_id,
          'job_key':key,'job_inputs':inputs,'phase':'unit','argv':argv,'code_revision':revision.stdout.strip(),'dirty':True,
          'code_tree_sha256':hashlib.sha256(canonical_json(dependencies)).hexdigest(),
          'configuration_sha256':sha256_file(request_path),'environment_sha256':sha256_file(root/'locks/environment.json'),
          'data_hashes':{},'model_hashes':{},'parent_decisions':[predecessor],'started_utc':started,'ended_utc':utc_now(),
          'completion_status':'complete','expected_cells':list(CHECKS),
          'produced_files':[{'path':str(path.relative_to(root)),'sha256':sha256_file(path)} for path in sorted(folder.rglob('*')) if path.is_file()],
          'invalidates':request['invalidated_run_ids'],'reason':request.get('invalidation_reason',
              'Incomplete attempt interrupted to repair measured checkpoint serialization overhead; no learned fine-time results inspected')}
        older = []
        for path in (root/'evidence/MX').glob('*/manifest.json'):
            prior = read_json(path)
            if prior['job_inputs'].get('request',{}).get('request_id') == request['request_id']:
                verify_run(root,prior,check_dependencies=False)
                older.append(prior['run_id'])
        if older:
            record['supersedes'] = older
        complete_run(root,record)
        return record


def evaluate(root,evidence,seen):
    from poc.rejection_subset import READINESS_CHECKS, evaluate_readiness
    if set(seen) == set(READINESS_CHECKS):
        return evaluate_readiness(root, evidence, seen)
    if len(evidence) != 1 or evidence[0]['phase'] != 'unit' or set(seen) != set(CHECKS) or any(c['status']!='pass' for c in seen.values()):
        raise ValueError('GX requires the exact registered repair and its completed regression evidence')
    return 'GO','valid',[{'criterion_id':name,'status':'pass','estimate':1.,'bounds':None,
                        'threshold':0.,'operator':'deterministic','reason':seen[name]['reason']} for name in CHECKS]


def transition(outcome,evaluations):
    if any(row['criterion_id'].startswith('negative.') for row in evaluations):
        return {'permitted_next_actions': [{'milestone_id': 'MR' if outcome == 'GO' else 'MX',
            'branch': None if outcome == 'GO' else 'optimizer',
            'question': 'Obtain independent rejection-only evidence; no full-PoC support from this shortcut'}],
            'terminal_verdict': None}
    return {'permitted_next_actions':[{'milestone_id':'M30' if outcome=='GO' else 'MX',
           'branch':None if outcome=='GO' else 'repair','question':'Resume the same fixed development design after checkpoint repair'}],
           'terminal_verdict':None}
