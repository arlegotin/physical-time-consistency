"""Prospectively registered control precision using unchanged scientific engines."""
from __future__ import annotations

from pathlib import Path

from .io import atomic_json, job_key, job_lock, read_json, safe_path, sha256_file, utc_now

SPEC = 'docs/superpowers/specs/2026-10-06-control-precision.md'
PLAN = 'docs/superpowers/plans/2026-10-06-control-precision.md'
REQUEST_PATH = 'config/revisions/MX-control-precision-0001.json'
CONTEXT = 'study/control-precision-look2'
ACTUATORS = ('direct', 'lagged')
MODELS = ('mlp_dt', 'power_correct')
CRITERIA = [f'utility_control/{a}/mlp_dt/h5000000' for a in ACTUATORS]
COORDINATORS = ('poc/control_precision.py', 'poc/control_precision_result.py',
                'poc/control_retraining.py', 'poc/control_confirmation.py')
# These two archived versions differ only in atomic selection packets and
# interrupted-job ancestry. Their optimizer, losses, minibatches, update session,
# validation and nominal selection calculations are identical (checked by AST
# fixtures and independently validated selected tensors). Unknown variants fail.
NOMINAL_RECOVERY_VARIANTS = {
    'poc/train.py': frozenset(('d56ccfaca398e08edc934b24c8b3ba7c7424628ffd5af4c753e36ef2bc5d7368',
                              '4f3dca540505a8cb4489944f00f4486b2adbc7713e24733ee6e7b7e94396c8c3')),
    'poc/training_jobs.py': frozenset(('97ec7fc02811efa99a370fa4dda667efbe2d58bdee8acd1e5a6db08b74bb3a4c',
                                      '6936a047ca67736c16c3bb17e5297ffbf1c57380f2c49df45c150d6649214ba4'))}
PARAMETERS = dict(retained_seed_ids=list(range(5)), new_seed_ids=list(range(5, 10)),
                  episode_ids=list(range(20)), criterion_ids=CRITERIA,
                  actuators=list(ACTUATORS), models=list(MODELS), condition='A', loss_mode='shared',
                  model_step_ns=5_000_000, physics_step_ns=250_000, finer_physics_step_ns=125_000,
                  hardware_clocks_ns=dict(observation=40_000_000, replanning=40_000_000, actuator=40_000_000),
                  family_size=8, look_index=2, look_alpha=.0125, directional_tail_alpha=.00078125,
                  bootstrap_seed_entropy=[9102, 2], bootstrap_resamples=20_000,
                  training_context=CONTEXT)


def _inputs(root):
    names = (SPEC, PLAN, 'config/authorization.json', 'config/planner.json',
             'config/control_confirmation_design.json', 'config/control_confirmation_cells.json',
             'reports/control_confirmation_result.json', 'reports/control_confirmation_numerical.json',
             'reports/control_confirmation_batch.json', 'reports/control_precision_planning_2026-10-06.json')
    hashes = {name: sha256_file(safe_path(root, name)) for name in names}
    for name in ('reports/control_confirmation_result.json', 'reports/control_confirmation_numerical.json',
                 'config/control_confirmation_cells.json'):
        value = read_json(root/name)
        for filename, digest in {**value.get('input_hashes', {}), **value.get('evaluator_hashes', {})}.items():
            if sha256_file(safe_path(root, filename)) != digest:
                raise ValueError('Changed original control dependency: '+filename)
            if filename in hashes and hashes[filename] != digest:
                raise ValueError('Conflicting precision parent dependency')
            hashes[filename] = digest
    return hashes


def build_request(root: Path) -> dict:
    """Bind the existing look and a sample size chosen before any new outcomes."""
    import copy
    root = Path(root).resolve()
    planning = read_json(root/'reports/control_precision_planning_2026-10-06.json')
    request = dict(schema_version=1, protocol_version='PTCO-1.0',
        protocol_id='UTILITY-2026-10-02-control-v1', request_id='MX-control-precision-0001',
        branch='precision', parent_gate='limited_utility_control_look_1', return_milestone='M70',
        claim_ids=['C04'], spec=SPEC, plan=PLAN, status='ready', recorded_utc=utc_now(),
        authorization_reference='config/authorization.json',
        authority='Owner authorized continued local current-branch execution and useful measured improvements',
        unresolved_question='Is mean realized control cost lower than the MLP at 5 ms on each actuator chain?',
        live_explanations=['Positive improvement survives additional independent fits',
                           'The population improvement is unresolved or absent'],
        new_information='Five additional ascending training-seed clusters with twenty paired new tasks each',
        progress_condition='Narrower signed uncertainty or a specifically identified influence/numerical limitation',
        action_parameters=copy.deepcopy(PARAMETERS), input_hashes=_inputs(root),
        contract={str(i): f'Exact element {i} in {SPEC}, Twelve-part execution contract' for i in range(1, 13)},
        expected_artifacts=['config/control_precision_nominal_jobs.json',
                            'reports/control_precision_nominal.json', 'config/control_precision_control_jobs.json',
                            'config/control_precision_finer_jobs.json', 'reports/control_precision_primary.json',
                            'reports/control_precision_numerical.json', 'reports/control_precision_result.json'],
        resource_projection=dict(n_new_control_episodes=400, minimum_new_finer_episodes=100,
            primary_cpu_seconds=planning['projection']['primary_cpu_seconds'],
            primary_artifact_bytes=planning['projection']['primary_artifact_bytes'], owner_imposed_caps=None,
            additional_resources='Exact context bytes and measured nominal/finer/audit projection required before dispatch'))
    return verify_request(root, request)


def verify_request(root: Path, request: dict) -> dict:
    root = Path(root).resolve()
    fixed = dict(schema_version=1, protocol_version='PTCO-1.0', protocol_id='UTILITY-2026-10-02-control-v1',
                 request_id='MX-control-precision-0001', branch='precision',
                 parent_gate='limited_utility_control_look_1', return_milestone='M70', claim_ids=['C04'],
                 spec=SPEC, plan=PLAN, status='ready', authorization_reference='config/authorization.json')
    if any(request.get(k) != v for k, v in fixed.items()) or request.get('action_parameters') != PARAMETERS:
        raise ValueError('Changed prospective precision sample, dimensions or inference')
    # JSON equality must not allow booleans to masquerade as seed integers.
    parameters = request['action_parameters']
    if any(type(s) is not int for name in ('retained_seed_ids', 'new_seed_ids', 'episode_ids')
           for s in parameters[name]):
        raise ValueError('Precision identities require integers')
    if request.get('input_hashes') != _inputs(root):
        raise ValueError('Changed or incomplete precision evidence closure')
    if request.get('contract') != {str(i): f'Exact element {i} in {SPEC}, Twelve-part execution contract'
                                  for i in range(1, 13)}:
        raise ValueError('Complete twelve-part precision contract required')
    authorization = read_json(root/'config/authorization.json')
    if authorization.get('local_compute_authorized') is not True or authorization.get('paid_services_authorized') is not False:
        raise ValueError('Existing local allocation required; no paid service authorized')
    from .control_confirmation import validate_design
    validate_design(read_json(root/'config/control_confirmation_design.json'))
    result = read_json(root/'reports/control_confirmation_result.json')
    numeric = read_json(root/'reports/control_confirmation_numerical.json')
    planning = read_json(root/'reports/control_precision_planning_2026-10-06.json')
    rows = {r['criterion_id']: r for r in result.get('criteria', [])}
    if (result.get('look_index') != 1 or result.get('family_size') != 8 or len(rows) != 8
            or any(not rows.get(c, {}).get('status', '').startswith('unresolved') for c in CRITERIA)
            or numeric.get('state') != 'qualified' or numeric.get('scope') != 'full'
            or numeric.get('required_actions') != []):
        raise ValueError('Qualified original look 1 and two live unresolved signs required')
    expected = dict(new_outcomes_opened=False, proposed_new_seed_ids=list(range(5, 10)),
                    proposed_total_seed_ids=list(range(10)), proposed_next_look=2,
                    proposed_primary_new_episodes=400, statistical_family_size=8,
                    look_alpha=.0125, directional_tail_alpha=.00078125,
                    preserve_all_prior_seeds_and_looks=True, status='planning_only_not_executable_request')
    if any(planning.get(k) != v for k, v in expected.items()):
        raise ValueError('Changed prospective sample-size analysis')
    for item in planning['evidence']:
        if sha256_file(safe_path(root, item['path'])) != item['sha256']:
            raise ValueError('Changed planning evidence')
    projection = request.get('resource_projection', {})
    if (projection.get('n_new_control_episodes') != 400 or projection.get('minimum_new_finer_episodes') != 100
            or any(projection.get(k) != planning['projection'][k]
                   for k in ('primary_cpu_seconds', 'primary_artifact_bytes'))
            or projection.get('owner_imposed_caps') is not None):
        raise ValueError('Missing or changed measured precision resource projection')
    return request


def union_nominal_inventory(source, inputs):
    """One verified two-plant nominal context, never inherited fitted checkpoints."""
    from .control_retraining import retraining_context_inventory
    if len(inputs) != 2 or {p['actuator'] for p, _ in inputs} != set(ACTUATORS):
        raise ValueError('Both actuator nominal corpora required')
    files, gates = {}, {}
    for parameters, parents in inputs:
        inventory = retraining_context_inventory(source, parameters, parents)
        for name, digest in inventory['files'].items():
            if name in files and files[name] != digest:
                raise ValueError('Conflicting nominal context inputs')
            files[name] = digest
        for name, refs in inventory['gate_inputs']['gates'].items():
            if name in gates and gates[name] != refs:
                raise ValueError('Conflicting early nominal validity')
            gates[name] = refs
    if set(gates) != {'G00', 'G10', 'G20'}:
        raise ValueError('Complete early nominal validity closure required')
    return dict(files=files, gate_inputs=dict(protocol_version='PTCO-1.0', gates=gates), cached_training_copied=False)


def reuse_coordinates(audit):
    """The complete predefined cohort, independent of prediction outcomes."""
    from .control_retraining import assert_validation_scores
    rows = audit.get('cells', [])
    expected = {f'direct__A__shared__{model}__seed{s}'
                for model in ('power_correct', 'node_generic') for s in range(40)}
    by_id = {r['cell_id']: r for r in rows}
    if (audit.get('state') != 'complete' or audit.get('passed') is not True
            or audit.get('expected_cells') != 80 or audit.get('completed_cells') != 80
            or audit.get('labels') != 'Nominal validation only' or len(rows) != 80 or set(by_id) != expected):
        raise ValueError('Complete independent forty-seed nominal selection audit required')
    for row in rows:
        if row.get('status') != 'pass':
            raise ValueError('Unqualified nominal reuse proof')
        assert_validation_scores(row['observed'], row['selected'])
    return [by_id[f'direct__A__shared__power_correct__seed{s}'] for s in range(5, 10)]


def verify_reuse_semantics(record, normalization, definition, inventory, expected_normalization):
    """Reuse is conditional on nominal semantics, not the prediction score."""
    parameters = record['job_inputs']['parameters']
    if (record.get('phase') != 'confirmatory' or record.get('milestone_id') != 'M30'
            or any(parameters.get(k) != definition[k]
                   for k in ('actuator', 'condition', 'loss_mode', 'model_id', 'seed'))
            or parameters.get('learning_rate') not in (.001, .0003)
            or normalization != expected_normalization):
        raise ValueError('Reused fit changed nominal identity, recipe or train-only scaler')
    data = parameters.get('data_hashes', {})
    prefix = f'data/{definition["actuator"]}/A/'
    for role in ('train', 'validation'):
        for filename in ('manifest.json', 'trajectories.npz'):
            name = prefix+role+'/'+filename
            if data.get(name) != inventory['files'].get(name) or name not in inventory['files']:
                raise ValueError('Reused fit has different nominal labels: '+name)
    # Infrastructure provenance can differ across archived contexts; all training
    # calculations, learned functions and physical-history semantics must agree.
    calculations = ('poc/train.py', 'poc/training_jobs.py', 'poc/mechanism.py',
                    'poc/history.py', 'poc/analytic.py', 'poc/clocks.py')
    for name, digest in record['job_inputs']['dependencies'].items():
        if name in calculations or name.startswith('poc/models/'):
            expected = inventory['files'].get(name)
            known_recovery = name in NOMINAL_RECOVERY_VARIANTS and {digest, expected} == NOMINAL_RECOVERY_VARIANTS[name]
            if expected != digest and not known_recovery:
                raise ValueError('Reused fit changed nominal training or model calculation: '+name)


def _nominal_inputs(root, request):
    from .control_confirmation import _check_selected_fit
    from .control_retraining import assert_validation_scores
    root = Path(root).resolve(); source = root/'study/control-development'
    frozen = read_json(root/'config/control_confirmation_cells.json')
    inputs = []
    hashes = {}
    normalizations = {}
    for actuator in ACTUATORS:
        candidates = [r for r in frozen['selected'] if (r['actuator'], r['model_id'], r['seed']) == (actuator, 'mlp_dt', 0)]
        if len(candidates) != 1:
            raise ValueError('Both qualified original nominal corpus references required')
        item = candidates[0]; _check_selected_fit(root, item)
        path = safe_path(root, item['manifest']); record = read_json(path)
        if path.parents[3] != source:
            raise ValueError('Nominal context must come from the original two-plant control source')
        inputs.append((record['job_inputs']['parameters'], record['parent_decisions']))
        name = f'data/{actuator}/A/normalization.json'
        normalizations[actuator] = read_json(source/name)
        for filename in (str(path.relative_to(root)), str((source/name).relative_to(root))):
            hashes[filename] = sha256_file(root/filename)
    inventory = union_nominal_inventory(source, inputs)
    for name, digest in inventory['files'].items():
        hashes[str((source/name).relative_to(root))] = digest
    audit_path = root/'reports/selection_integrity_forty.json'
    audit = read_json(audit_path); proofs = reuse_coordinates(audit)
    hashes[str(audit_path.relative_to(root))] = sha256_file(audit_path)
    external = Path(audit['source_context']).resolve()
    if not external.is_relative_to(root):
        raise ValueError('Reuse source must remain in the authorized repository')
    reuse = {}
    for proof in proofs:
        path = safe_path(external, proof['manifest'])
        record = read_json(path); seed = record['job_inputs']['parameters']['seed']
        definition = dict(actuator='direct', condition='A', loss_mode='shared', model_id='power_correct', seed=seed)
        item = dict(**definition, cell_id=proof['cell_id'], manifest=str(path.relative_to(root)),
                    manifest_sha256=proof['manifest_sha256'], checkpoint_sha256=sha256_file(path.parent/'weights.pt'),
                    canonical_tensor_sha256=proof['canonical_tensor_sha256'])
        _check_selected_fit(root, item)
        verify_reuse_semantics(record, read_json(path.parent/'normalization.json'), definition, inventory,
                               normalizations['direct'])
        if proof['canonical_tensor_sha256'] != record['model_hashes']['canonical_tensor_sha256']:
            raise ValueError('Reused canonical tensors differ from the independent selection audit')
        assert_validation_scores(proof['observed'], read_json(path.parent/'selection.json'))
        assert_validation_scores(proof['observed'], read_json(path.parent/'summary.json')['validation'])
        for filename in ('manifest.json', 'weights.pt', 'normalization.json', 'summary.json', 'selection.json'):
            name = str((path.parent/filename).relative_to(root)); hashes[name] = sha256_file(root/name)
        reuse[proof['cell_id']] = item
    return dict(source=str(source.relative_to(root)), inventory=inventory,
                normalization_by_actuator=normalizations, reuse=reuse, input_hashes=hashes)


def _nominal_jobs(inputs):
    jobs = []
    for actuator in ACTUATORS:
        for model in MODELS:
            for seed in range(5, 10):
                cell = f'{actuator}__A__shared__{model}__seed{seed}'
                definition = dict(actuator=actuator, condition='A', loss_mode='shared', model_id=model, seed=seed, cell_id=cell)
                external = inputs['reuse'].get(cell)
                jobs.append(dict(cell_id=cell, definition=definition, mode='reuse' if external else 'fit', external=external))
    return jobs


def prepare_nominal(root: Path, request: dict) -> dict:
    """Copy only nominal inputs and freeze all twenty coordinates, without fitting."""
    from .control_retraining import copy_retraining_context
    root = Path(root).resolve(); verify_request(root, request)
    if read_json(root/REQUEST_PATH) != request:
        raise ValueError('Register the prospective precision request before preparation')
    target = root/'config/control_precision_nominal_jobs.json'
    if target.exists(): return verify_nominal_registry(root, read_json(target))
    inputs = _nominal_inputs(root, request)
    jobs = _nominal_jobs(inputs)
    copy_retraining_context(root/inputs['source'], root/CONTEXT, inputs['inventory'])
    result = dict(state='complete', request_sha256=sha256_file(root/REQUEST_PATH), context=CONTEXT,
                  expected_fits=20, new_fits=sum(j['mode'] == 'fit' for j in jobs),
                  reused_fits=sum(j['mode'] == 'reuse' for j in jobs), jobs=jobs, **inputs,
                  context_input_bytes=sum((root/inputs['source']/n).stat().st_size for n in inputs['inventory']['files']),
                  selection='Unchanged nominal-only engine; entire eligible ascending direct-head cohort',
                  prior_prediction_outcomes_observed=True, new_control_outcomes_opened=False)
    # The coordinator is additive. Its source and helper hashes are frozen only
    # when the implementation is complete, before any real scientific dispatch.
    result['coordinator_hashes'] = {name: sha256_file(Path(__file__).resolve().parents[1]/name)
        for name in COORDINATORS}
    atomic_json(target, result)
    return verify_nominal_registry(root, result)


def verify_nominal_registry(root, registry):
    root = Path(root).resolve()
    request = verify_request(root, read_json(root/REQUEST_PATH))
    if (registry.get('state') != 'complete' or registry.get('expected_fits') != 20
            or registry.get('context') != CONTEXT or registry.get('request_sha256') != sha256_file(root/REQUEST_PATH)):
        raise ValueError('Complete prospectively registered nominal context required')
    inputs = _nominal_inputs(root, request)
    for key, value in inputs.items():
        if registry.get(key) != value: raise ValueError('Changed nominal inventory or reuse cohort: '+key)
    expected = _nominal_jobs(inputs)
    if (registry.get('jobs') != expected or len(expected) != 20
            or registry.get('new_fits') != sum(j['mode'] == 'fit' for j in expected)
            or registry.get('reused_fits') != sum(j['mode'] == 'reuse' for j in expected)):
        raise ValueError('Missing, duplicate or substituted nominal precision cell')
    if set(registry.get('coordinator_hashes', {})) != set(COORDINATORS):
        raise ValueError('Complete coordinator and consumer hash freeze required')
    for name, digest in registry['coordinator_hashes'].items():
        if sha256_file(Path(__file__).resolve().parents[1]/name) != digest:
            raise ValueError('Changed precision coordinator after nominal preparation')
    for name, digest in registry['input_hashes'].items():
        if sha256_file(safe_path(root, name)) != digest: raise ValueError('Changed nominal input: '+name)
    marker = read_json(root/CONTEXT/'FRESH_TRAINING_CONTEXT.json')
    if marker['inventory'] != inputs['inventory'] or marker.get('cached_training_copied') is not False:
        raise ValueError('Changed or partial independent nominal context')
    for name, digest in inputs['inventory']['files'].items():
        if sha256_file(safe_path(root/CONTEXT, name)) != digest:
            raise ValueError('Changed copied nominal input: '+name)
    return registry


def _observe_nominal(root, job, registry, path):
    """Recompute selected float32 nominal objectives without changing weights."""
    import torch
    from .contracts import verify_run
    from .control_confirmation import _check_selected_fit
    from .control_retraining import assert_validation_scores
    from .execution import cpu_defaults
    from .dataset import load_corpus
    from .models.common import Scales
    from .models.factory import build_model
    from .train import canonical_tensor_hash, qualify_nominal, validate_nominal
    root = Path(root).resolve(); path = Path(path).resolve(); source = path.parents[3]
    if job['mode'] == 'fit' and source != root/registry['context']:
        raise ValueError('New nominal fit is outside the cache-free precision context')
    if job['mode'] == 'reuse' and str(path.relative_to(root)) != job['external']['manifest']:
        raise ValueError('Substituted reused nominal checkpoint')
    record = read_json(path); verify_run(source, record)
    definition = job['definition']; parameters = record['job_inputs']['parameters']
    norm = read_json(path.parent/'normalization.json')
    verify_reuse_semantics(record, norm, definition, registry['inventory'],
                           registry['normalization_by_actuator'][definition['actuator']])
    if not qualify_nominal(record, source)['qualified']:
        raise ValueError('Precision checkpoint has not qualified nominal learning')
    cpu_defaults(definition['seed'])
    scales = Scales(*(torch.tensor(norm[k], dtype=torch.float32) for k in ('scale_q', 'scale_v', 'velocity_input_rms')))
    model = build_model(definition['model_id'], definition['actuator'], scales, definition['seed'])
    weights = torch.load(path.parent/'weights.pt', weights_only=True, map_location='cpu')
    if canonical_tensor_hash(weights) != record['model_hashes']['canonical_tensor_sha256']:
        raise ValueError('Precision canonical tensors differ from the fit manifest')
    model.load_state_dict(weights, strict=True)
    # Audit logs are new evidence, never appended into an archived fit context.
    arrays = load_corpus(safe_path(source, parameters['validation_manifest']), phase='confirmatory', role='validation',
                         purpose='selection', audit_path=root/'reports/control_precision_nominal_access.jsonl')
    observed = validate_nominal(model, arrays, scales, definition['loss_mode'])
    assert_validation_scores(observed, read_json(path.parent/'selection.json'))
    assert_validation_scores(observed, read_json(path.parent/'summary.json')['validation'])
    if canonical_tensor_hash(model.state_dict()) != record['model_hashes']['canonical_tensor_sha256']:
        raise ValueError('Independent nominal validation changed selected tensors')
    item = dict(actuator=definition['actuator'], condition='A', loss_mode='shared', model_id=definition['model_id'],
                seed=definition['seed'], cell_id=job['cell_id'], manifest=str(path.relative_to(root)),
                manifest_sha256=sha256_file(path), checkpoint_sha256=sha256_file(path.parent/'weights.pt'),
                canonical_tensor_sha256=record['model_hashes']['canonical_tensor_sha256'],
                qualified=True, mode=job['mode'], independently_recomputed_validation=observed,
                normalization_sha256=sha256_file(path.parent/'normalization.json'),
                resources=read_json(path.parent/'resources.json'))
    _check_selected_fit(root, item)
    return item


def _fit_in_child(root, job, registry, attempt):
    import json
    import os
    import subprocess
    import sys
    script = '''import json,sys
from pathlib import Path
from poc.mechanism import run_worker,train_cell
from poc.io import run_directory
from poc.train import qualify_nominal
root=Path.cwd();definition=json.loads(sys.argv[1]);folder=root/'precision_fit';folder.mkdir(exist_ok=True)
run_worker(root,folder,'controls',dict(actuator=definition['actuator'],model_id=definition['model_id'],seed=8100),True)
record=train_cell(root,folder,definition,'confirmatory',resume=True)
if not qualify_nominal(record,root)['qualified']:raise RuntimeError('Precision nominal fitting remains unqualified')
print(json.dumps(dict(manifest_path=str((run_directory(root,record)/'manifest.json').relative_to(root)))))
'''
    environment = dict(os.environ); environment.pop('PYTHONPATH', None)
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        environment[name] = '1'
    attempt.mkdir(parents=True)
    command = [sys.executable, '-c', script, json.dumps(job['definition'])]
    atomic_json(attempt/'attempt.json', dict(started_utc=utc_now(), cell_id=job['cell_id'], argv=command,
                                           context=registry['context'], returncode=None))
    with (attempt/'stdout.log').open('w') as out, (attempt/'stderr.log').open('w') as err:
        child = subprocess.run(command, cwd=root/registry['context'], env=environment, stdout=out, stderr=err)
    detail = read_json(attempt/'attempt.json'); detail.update(returncode=child.returncode, ended_utc=utc_now())
    atomic_json(attempt/'attempt.json', detail)
    if child.returncode:
        raise RuntimeError('Precision nominal child incomplete; preserved diagnostics: '+str(attempt.relative_to(root)))
    result = json.loads((attempt/'stdout.log').read_text().strip().splitlines()[-1])
    return safe_path(root/registry['context'], result['manifest_path'])


def run_nominal(root: Path, registry: dict, *, resume: bool) -> dict:
    root = Path(root).resolve(); assert_reproduction_complete(root)
    target = root/'reports/control_precision_nominal.json'; selected = []
    registry_hash = sha256_file(root/'config/control_precision_nominal_jobs.json')
    def save(state, **extra):
        value = dict(state=state, registry_sha256=registry_hash, expected_fits=len(registry['jobs']),
                     completed_fits=len(selected), selected=selected, new_control_outcomes_opened=False, **extra)
        atomic_json(target, value); return value
    with job_lock(job_key(dict(control_precision_nominal=registry['request_sha256'])), root):
        verify_nominal_registry(root, registry)
        reused = {}; jobs = {j['cell_id']: j for j in registry['jobs']}
        if len(jobs) != len(registry['jobs']): raise ValueError('Duplicate nominal precision coordinate')
        if target.exists():
            previous = read_json(target)
            if not resume or previous['registry_sha256'] != registry_hash:
                raise ValueError('Existing precision nominal receipt requires compatible --resume')
            for item in previous['selected']:
                cell = item['cell_id']; path = safe_path(root, item['manifest'])
                if cell not in jobs or cell in reused or sha256_file(path) != item['manifest_sha256']:
                    raise ValueError('Changed or duplicate completed nominal fit')
                if _observe_nominal(root, jobs[cell], registry, path) != item:
                    raise ValueError('Saved nominal freeze differs from actual selected weights')
                reused[cell] = item
            if previous['state'] == 'complete':
                if set(reused) != set(jobs) or previous['completed_fits'] != len(jobs):
                    raise ValueError('Incomplete claimed nominal precision freeze')
                return previous
        try:
            for index, job in enumerate(registry['jobs']):
                verify_nominal_registry(root, registry)
                if job['cell_id'] in reused:
                    selected.append(reused[job['cell_id']]); continue
                if job['mode'] == 'reuse': path = safe_path(root, job['external']['manifest'])
                else:
                    stamp = utc_now().replace(':', '').replace('-', '').replace('.', '')
                    attempt = root/'reports/control_precision_nominal_attempts'/f'{index:04d}-{stamp}'
                    print(f'Precision nominal {index+1}/{len(jobs)}: {job["cell_id"]}', flush=True)
                    path = _fit_in_child(root, job, registry, attempt)
                selected.append(_observe_nominal(root, job, registry, path)); save('running')
            return save('complete', diagnosis_required=False)
        except BaseException as error:
            save('interrupted', reason=f'{type(error).__name__}: {error}'); raise


def assert_reproduction_complete(root):
    """Read-only preparation is allowed; scientific dispatch waits for replication."""
    root = Path(root)
    path = root/'reports/reproduction_followup.json'
    if not path.exists() or read_json(path).get('state') != 'reproduction_ready_for_whole_branch_review':
        raise ValueError('Complete original reproduction required before precision scientific dispatch')
    required = [('control_replay', 60, 'episodes'), ('control_retraining', 12, 'fits'),
                ('control_retrained_policy_batch', 480, 'episodes')]
    for name, count, collection in required:
        path = root/'reports'/f'{name}.json'
        value = read_json(path) if path.exists() else {}
        if (value.get('state') != 'complete' or len(value.get(collection, [])) != count
                or value.get('completed_'+collection) != count or value.get('expected_'+collection) != count
                or value.get('diagnosis_required') is not False):
            raise ValueError('Complete qualified original reproduction required: '+name)
    result = read_json(root/'reports/control_retrained_policy_result.json')
    if (result.get('status') != 'pass' or result.get('diagnosis_required') is not False
            or result.get('criteria_unchanged') is not True or result.get('alpha_spent') != 0
            or result.get('new_independent_training_clusters') != 0):
        raise ValueError('Original reproduction criterion stability required')


def _verify_selected_nominal(root, selected):
    from .control_confirmation import _check_selected_fit
    path = root/'reports/control_precision_nominal.json'
    if read_json(path) != selected:
        raise ValueError('Use the exact complete independently verified nominal freeze')
    registry_path = root/'config/control_precision_nominal_jobs.json'
    verify_nominal_registry(root, read_json(registry_path))
    rows = selected.get('selected', [])
    expected = {(a, m, s) for a in ACTUATORS for m in MODELS for s in range(5, 10)}
    coordinates = [(r['actuator'], r['model_id'], r['seed']) for r in rows]
    if (selected.get('state') != 'complete' or selected.get('diagnosis_required') is not False
            or selected.get('expected_fits') != 20 or selected.get('completed_fits') != 20
            or selected.get('registry_sha256') != sha256_file(registry_path)
            or len(coordinates) != 20 or set(coordinates) != expected):
        raise ValueError('All twenty qualified checkpoints must be frozen before control')
    for row in rows:
        if row.get('qualified') is not True or type(row['seed']) is not int:
            raise ValueError('Unqualified nominal precision checkpoint')
        _check_selected_fit(root, row)
        manifest = safe_path(root, row['manifest'])
        if (sha256_file(manifest) != row['manifest_sha256']
                or sha256_file(manifest.parent/'weights.pt') != row['checkpoint_sha256']):
            raise ValueError('Changed selected precision manifest or checkpoint')
    return sorted(rows, key=lambda r: (r['actuator'], r['model_id'], r['seed']))


def _primary_requests(root, selected, request):
    from .control_jobs import validate_request
    settings = read_json(root/'config/planner.json')['settings']
    if (settings['candidate_count'], settings['iterations'], settings['elite_count']) != (128, 4, 16):
        raise ValueError('Preserve the qualified original common planner')
    rows = []
    for item in selected:
        for episode in range(20):
            cell = f'{item["actuator"]}__A__shared__{item["model_id"]}__seed{item["seed"]}__h5000000__episode{episode}'
            value = validate_request(dict(schema_version=1, phase='confirmatory', model_id=item['model_id'],
                actuator=item['actuator'], seed=item['seed'], episode_id=episode, condition='clean',
                model_partition_ns=[5_000_000]*8, actuator_interval_ns=40_000_000, physics_step_ns=250_000,
                power_mode=None, model_manifest=item['manifest'], manifest_sha256=item['manifest_sha256'],
                checkpoint_sha256=item['checkpoint_sha256'], planner_settings=settings,
                resource_projection=dict(cpu_seconds=request['resource_projection']['primary_cpu_seconds']/400,
                    artifact_bytes=request['resource_projection']['primary_artifact_bytes']/400,
                    basis='Measured same-size original paired fine-time cohort; actual jobs are recorded')))
            rows.append(dict(cell_id=cell, request=value))
    return rows


def _episode_dependencies(root, rows):
    from .control_jobs import SOURCES
    names = set(SOURCES) | {REQUEST_PATH, 'poc/control_precision.py', 'poc/control_precision_result.py',
        'config/planner.json', 'config/numerical.json',
        'config/revisions/owner-objective-2026-10-01.json', 'config/control_precision_nominal_jobs.json',
        'reports/control_precision_nominal.json', 'assets/two_link_direct.xml', 'assets/two_link_lagged.xml'}
    for row in rows:
        path = Path(row['request']['model_manifest'])
        names.update((str(path), str(path.parent/'weights.pt')))
    return {name: sha256_file(safe_path(root, name)) for name in sorted(names)}


def _write_episode_registry(root, request, rows, *, kind, path, **extra):
    if path.exists(): return verify_episode_registry(root, read_json(path))
    hashes = _episode_dependencies(root, rows)
    jobs = []
    for row in rows:
        filename = root/'config/control_precision_jobs'/f'{row["cell_id"]}.json'
        if filename.exists() and read_json(filename) != row['request']:
            raise ValueError('Conflicting prospective precision episode request')
        atomic_json(filename, row['request'])
        jobs.append({k: v for k, v in row.items() if k != 'request'} |
                    dict(path=str(filename.relative_to(root)), sha256=sha256_file(filename)))
    result = dict(schema_version=1, state='complete', kind=kind, request_sha256=sha256_file(root/REQUEST_PATH),
        registry_path=str(path.relative_to(root)), expected_episodes=len(jobs), jobs=jobs,
        created_utc=utc_now(), episode_seed_root=6101, input_hashes=hashes,
        resource_projection=dict(expected_episodes=len(jobs),
            cpu_seconds=sum(r['request']['resource_projection']['cpu_seconds'] for r in rows),
            artifact_bytes=sum(r['request']['resource_projection']['artifact_bytes'] for r in rows),
            basis='Original measured primary proxy; finer physics and actual representative jobs remain separate'), **extra)
    atomic_json(path, result)
    return verify_episode_registry(root, result)


def prepare_control(root, request, selected):
    root = Path(root).resolve(); verify_request(root, request)
    rows = _primary_requests(root, _verify_selected_nominal(root, selected), request)
    return _write_episode_registry(root, request, rows, kind='primary',
                                   path=root/'config/control_precision_control_jobs.json')


def prepare_finer(root, request, primary, *, full=False, actuators=ACTUATORS):
    root = Path(root).resolve(); verify_request(root, request); verify_episode_registry(root, primary)
    if (type(full) is not bool or not actuators or tuple(a for a in ACTUATORS if a in actuators) != tuple(actuators)
            or primary['kind'] != 'primary'):
        raise ValueError('Explicit ordered affected plants and a primary precision registry required')
    rows = []
    for job in primary['jobs']:
        value = read_json(safe_path(root, job['path']))
        if value['actuator'] not in actuators or (not full and value['episode_id'] >= 5): continue
        rows.append(dict(cell_id='physics125000__'+job['cell_id'], parent_cell_id=job['cell_id'],
            parent_request_sha256=job['sha256'], request=dict(value, physics_step_ns=125_000)))
    suffix = '' if not full and tuple(actuators) == ACTUATORS else '__'+('full' if full else 'subset')+'__'+'-'.join(actuators)
    path = root/f'config/control_precision_finer_jobs{suffix}.json'
    return _write_episode_registry(root, request, rows, kind='finer', path=path, scope='full' if full else 'subset',
        actuators=list(actuators), parent_registry_sha256=sha256_file(safe_path(root, primary['registry_path'])))


def verify_episode_registry(root, registry):
    root = Path(root).resolve(); request = verify_request(root, read_json(root/REQUEST_PATH))
    if (registry.get('state') != 'complete' or registry.get('schema_version') != 1
            or registry.get('request_sha256') != sha256_file(root/REQUEST_PATH)
            or registry.get('episode_seed_root') != 6101 or registry.get('kind') not in ('primary', 'finer')):
        raise ValueError('Changed complete precision registry identity or task namespace')
    selected = _verify_selected_nominal(root, read_json(root/'reports/control_precision_nominal.json'))
    rows = _primary_requests(root, selected, request)
    if registry['kind'] == 'primary':
        if registry['registry_path'] != 'config/control_precision_control_jobs.json':
            raise ValueError('Changed primary precision registry path')
    else:
        primary = read_json(root/'config/control_precision_control_jobs.json')
        verify_episode_registry(root, primary)
        if registry.get('parent_registry_sha256') != sha256_file(root/'config/control_precision_control_jobs.json'):
            raise ValueError('Changed finer precision parent registry')
        scope, actuators = registry.get('scope'), registry.get('actuators')
        if scope not in ('subset', 'full') or not actuators or [a for a in ACTUATORS if a in actuators] != actuators:
            raise ValueError('Unknown finer precision scope or affected plants')
        by_id = {j['cell_id']: j for j in primary['jobs']}
        rows = [dict(cell_id='physics125000__'+r['cell_id'], parent_cell_id=r['cell_id'],
                     parent_request_sha256=by_id[r['cell_id']]['sha256'], request=dict(r['request'], physics_step_ns=125_000))
                for r in rows if r['request']['actuator'] in actuators and (scope == 'full' or r['request']['episode_id'] < 5)]
    if (registry.get('expected_episodes') != len(rows) or len(registry.get('jobs', [])) != len(rows)
            or registry.get('input_hashes') != _episode_dependencies(root, rows)):
        raise ValueError('Incomplete precision registry or changed dependency closure')
    for job, expected in zip(registry['jobs'], rows):
        path = safe_path(root, job['path'])
        if (job['path'] != f'config/control_precision_jobs/{expected["cell_id"]}.json'
                or job['cell_id'] != expected['cell_id'] or sha256_file(path) != job['sha256']
                or read_json(path) != expected['request']
                or any(job.get(k) != v for k, v in expected.items() if k != 'request')):
            raise ValueError('Changed, duplicate or reordered precision episode cell')
    return registry


def _episode_in_child(root, job, attempt):
    import subprocess
    import sys
    attempt.mkdir(parents=True)
    command = [sys.executable, '-m', 'poc.control_jobs', job['path'], '--root', str(root)]
    atomic_json(attempt/'attempt.json', dict(cell_id=job['cell_id'], request_sha256=job['sha256'],
                                           argv=command, started_utc=utc_now(), returncode=None))
    with (attempt/'stdout.log').open('w') as out, (attempt/'stderr.log').open('w') as err:
        child = subprocess.run(command, cwd=root, stdout=out, stderr=err)
    detail = read_json(attempt/'attempt.json'); detail.update(returncode=child.returncode, ended_utc=utc_now())
    atomic_json(attempt/'attempt.json', detail)
    if child.returncode:
        raise RuntimeError('Precision control child incomplete; preserved diagnostics: '+str(attempt.relative_to(root)))
    text = (attempt/'stdout.log').read_text().strip().splitlines()[-1]
    # The existing producer emits one plain absolute manifest path.
    path = Path(text).resolve()
    if not path.is_relative_to(root/'evidence/M70') or path.name != 'manifest.json':
        raise ValueError('Control child returned an unexpected manifest path')
    return path


def _observe_episode(root, job, path):
    from .control_confirmation_batch import episode_receipt
    from .control_utility import audit_raw_episode
    value = episode_receipt(root, job, path)
    raw = audit_raw_episode(Path(path).parent, value['summary'],
                            expected_tick_ns=value['summary']['physics_step_ns'])
    return {**value, 'raw_audit': raw}


def run_episode_registry(root, registry, *, resume):
    import shutil
    root = Path(root).resolve(); assert_reproduction_complete(root)
    path = safe_path(root, registry['registry_path']); registry_hash = sha256_file(path)
    if read_json(path) != registry: raise ValueError('Dispatch requires the exact immutable episode registry')
    suffix = '' if registry['kind'] == 'primary' else Path(registry['registry_path']).stem.removeprefix('control_precision_finer_jobs')
    filename = 'control_precision_primary.json' if registry['kind'] == 'primary' else 'control_precision_finer'+suffix+'.json'
    target = root/'reports'/filename; episodes = []; jobs = {j['cell_id']: j for j in registry['jobs']}
    if len(jobs) != len(registry['jobs']): raise ValueError('Duplicate precision control coordinate')
    def save(state, **extra):
        value = dict(schema_version=1, state=state, kind=registry['kind'], registry_sha256=registry_hash,
                     expected_episodes=len(jobs), completed_episodes=len(episodes), episodes=episodes,
                     updated_utc=utc_now(), scientific_decision='not_evaluated', **extra)
        atomic_json(target, value); return value
    with job_lock(job_key({'control_precision_episode_registry': registry_hash}), root):
        verify_episode_registry(root, registry); reused = {}
        if target.exists():
            previous = read_json(target)
            if not resume or previous['registry_sha256'] != registry_hash:
                raise ValueError('Existing precision batch requires compatible --resume')
            for item in previous['episodes']:
                cell = item['cell_id']; manifest = safe_path(root, item['manifest'])
                if cell not in jobs or cell in reused or sha256_file(manifest) != item['manifest_sha256']:
                    raise ValueError('Conflicting, stale or duplicated precision episode attempt')
                if _observe_episode(root, jobs[cell], manifest) != item:
                    raise ValueError('Saved precision episode differs from complete raw evidence')
                reused[cell] = item
            if previous['state'] == 'complete':
                if (set(reused) != set(jobs) or previous['completed_episodes'] != len(jobs)
                        or previous['expected_episodes'] != len(jobs)):
                    raise ValueError('Partial coverage cannot claim a completed precision batch')
                return previous
        estimate = registry['resource_projection']['artifact_bytes']*(len(jobs)-len(reused))/len(jobs)
        if shutil.disk_usage(root).free < estimate:
            raise RuntimeError('Actual disk prerequisite prevents the projected precision batch; preserve evidence and provision disk')
        try:
            for index, job in enumerate(registry['jobs']):
                verify_episode_registry(root, registry)
                if job['cell_id'] in reused: episodes.append(reused[job['cell_id']]); continue
                if sha256_file(safe_path(root, job['path'])) != job['sha256']:
                    raise ValueError('Changed precision request before child dispatch')
                stamp = utc_now().replace(':', '').replace('-', '').replace('.', '')
                attempt = root/'reports/control_precision_episode_attempts'/f'{registry["kind"]}-{index:04d}-{stamp}'
                print(f'Precision {registry["kind"]} {index+1}/{len(jobs)}: {job["cell_id"]}', flush=True)
                manifest = _episode_in_child(root, job, attempt)
                episodes.append(_observe_episode(root, job, manifest)); save('running')
            return save('complete', diagnosis_required=False)
        except BaseException as error:
            save('interrupted', reason=f'{type(error).__name__}: {error}'); raise


def main(argv=None):
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request', type=Path)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv); root = args.root.resolve()
    path = args.request.resolve() if args.request.is_absolute() else safe_path(root, str(args.request))
    if path != root/REQUEST_PATH:
        raise ValueError('Only the exact registered precision contract can execute')
    request = verify_request(root, read_json(path))
    if args.dry_run:
        print(json.dumps(dict(execute=False, new_control_outcomes_opened=False,
            expected_nominal_coordinates=20, expected_new_primary_episodes=400,
            minimum_new_finer_episodes=100, retained_seed_ids=list(range(5)), new_seed_ids=list(range(5, 10)),
            inference_look=2, family_size=8,
            prerequisite='Complete original reproduction; freeze nominal checkpoints and measured resources')))
        return 0
    registry_path = root/'config/control_precision_nominal_jobs.json'
    registry = prepare_nominal(root, request) if args.prepare else verify_nominal_registry(root, read_json(registry_path))
    if args.prepare: return 0
    selected = run_nominal(root, registry, resume=True)
    primary = prepare_control(root, request, selected)
    run_episode_registry(root, primary, resume=True)
    finer = prepare_finer(root, request, primary)
    run_episode_registry(root, finer, resume=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
