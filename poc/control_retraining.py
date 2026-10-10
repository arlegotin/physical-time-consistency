"""Fixed-seed fresh-training comparisons under the original physical tolerances."""
from __future__ import annotations

import math
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from .control_confirmation import ACTUATORS, MODELS
from .io import atomic_json,job_key,job_lock,read_json,safe_path,sha256_file,utc_now


def training_replication_cells(frozen):
    """Choose all six implementations at seeds 0 and 4 before reading outcomes."""
    rows=frozen.get('selected',[])
    expected={(a,m,s) for a in ACTUATORS for m in MODELS for s in range(5)}
    coordinates=[(r['actuator'],r['model_id'],r['seed']) for r in rows]
    if (frozen.get('state')!='complete' or frozen.get('expected_episodes')!=1200
            or len(rows)!=30 or set(coordinates)!=expected
            or any(type(r['seed']) is not int for r in rows)):
        raise ValueError('Complete frozen thirty-checkpoint cohort required for fresh training')
    return [dict(row) for row in sorted(rows,key=lambda r:(r['actuator'],r['model_id'],r['seed']))
            if row['seed'] in (0,4)]


def compare_training(original,fresh,original_validation,fresh_validation):
    """Compare actual tensors and nominal scores; control replication stays separate."""
    if not isinstance(original,dict) or not isinstance(fresh,dict) or not original or set(original)!=set(fresh):
        raise ValueError('Missing or different selected tensor names')
    scores=[]
    for score in (original_validation,fresh_validation):
        value=score.get('objective')
        if type(value) not in (int,float) or not math.isfinite(value) or value<0:
            raise ValueError('Nonfinite or negative nominal validation objective')
        scores.append(float(value))
    parameters={};identical=True
    for name in sorted(original):
        first,second=original[name],fresh[name]
        if (not isinstance(first,torch.Tensor) or not isinstance(second,torch.Tensor)
                or first.dtype!=second.dtype or first.shape!=second.shape
                or not torch.is_floating_point(first) or not first.numel()
                or not torch.isfinite(first).all() or not torch.isfinite(second).all()):
            raise ValueError('Changed or nonfinite parameter shape/dtype: '+name)
        first,second=first.detach().cpu(),second.detach().cpu()
        difference=float((first.double()-second.double()).abs().max())
        limit=1e-6*(1+float(first.double().abs().max()))
        identical=identical and torch.equal(first,second)
        parameters[name]=dict(max_difference=difference,tolerance=limit,within_tolerance=difference<=limit)
    absolute=abs(scores[0]-scores[1])
    relative=absolute/scores[0] if scores[0]>0 else (0. if absolute==0 else None)
    passed=all(r['within_tolerance'] for r in parameters.values()) and relative is not None and relative<=1e-4
    return dict(status='pass' if passed else 'outside_tolerance',diagnosis_required=not passed,
        parameters=parameters,max_parameter_difference=max(r['max_difference'] for r in parameters.values()),
        canonical_tensors_identical=bool(identical),validation_objective_absolute_difference=absolute,
        validation_objective_relative_difference=relative,fresh_training=True,control_checked=False,
        scope='Selected tensors and nominal validation only; replicated control and criterion stability required separately')


def retraining_context_inventory(source,parameters,parents):
    """Carry required early validity forward while excluding every fitted-model cache."""
    source=Path(source).resolve();files={};visited=set();gates={}
    def add(name,digest=None):
        if name.startswith('evidence/M30/') or '/confirmation/' in name or '/development/' in name:
            raise ValueError('Fresh context cannot inherit fitted-model caches or held-out labels: '+name)
        path=safe_path(source,name);actual=sha256_file(path)
        if digest is not None and actual!=digest:
            raise ValueError('Changed fresh-training input: '+name)
        if name in files and files[name]!=actual:raise ValueError('Conflicting fresh-training dependency')
        files[name]=actual
    def manifest(name,digest):
        add(name,digest)
        if name in visited:return
        visited.add(name);path=safe_path(source,name);record=read_json(path)
        marker=str((path.parent/'COMPLETED').relative_to(source));add(marker)
        if read_json(source/marker)['manifest_sha256']!=digest:
            raise ValueError('Partial early reference before fresh training')
        for item in record['produced_files']:add(item['path'],item['sha256'])
        inputs=record['job_inputs']
        for filename,expected in inputs.get('dependencies',{}).items():add(filename,expected)
        for filename,expected in inputs.get('configuration_keys',{}).items():
            add(filename)
            if any(read_json(source/filename).get(k)!=v for k,v in expected.items()):
                raise ValueError('Changed early validity configuration')
        for filename,expected in inputs.get('wheel_hashes',{}).items():add('wheelhouse/'+filename,expected)
        for parent in record.get('parent_decisions',[]):decision(parent)
    def decision(record):
        if record['gate_id'] not in ('G00','G10','G20'):
            raise ValueError('Fresh-training context requires early validity only')
        filename=f"decisions/{record['gate_id']}/{record['decision_id']}.json";add(filename)
        if read_json(source/filename)!=record:raise ValueError('Changed early parent decision')
        refs=[]
        for ref in record['evidence']:
            if ref['role']=='manifest':
                manifest(ref['path'],ref['sha256']);refs.append(ref['path'])
            else:add(ref['path'],ref['sha256'])
        old=gates.setdefault(record['gate_id'],refs)
        if old!=refs:raise ValueError('Conflicting early gate evidence')
    for directory in ('poc','schemas'):
        for path in sorted((source/directory).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                add(str(path.relative_to(source)))
    for name in ('locks/environment.json','locks/wheels.json','locks/requirements.lock'):add(name)
    for role in ('train','validation'):
        name=parameters[role+'_manifest'];add(name);record=read_json(source/name)
        if (record['role']!=role or record['actuator']!=parameters['actuator']
                or record['condition']!='A' or parameters['condition']!='A'
                or set(record['files'])!={'trajectories.npz'}):
            raise ValueError('Changed fresh-training corpus role or actuator')
        for filename,digest in record['files'].items():
            add(str((Path(name).parent/filename)),digest)
    for parent in parents:decision(parent)
    if 'G20' not in gates:raise ValueError('Qualified plant evidence required for fresh training')
    return dict(files=files,gate_inputs=dict(protocol_version='PTCO-1.0',gates=gates),cached_training_copied=False)


def copy_retraining_context(source,target,inventory):
    """Independent copies keep even regenerated normalization away from archives."""
    source,target=Path(source).resolve(),Path(target).resolve()
    if target==source or target.is_relative_to(source):
        raise ValueError('Fresh training requires a separate context outside its source')
    marker=target/'FRESH_TRAINING_CONTEXT.json'
    if target.exists():
        if not marker.exists() or read_json(marker)['inventory']!=inventory:
            raise ValueError('Partial or different existing fresh-training context')
        for name,digest in inventory['files'].items():
            if sha256_file(safe_path(target,name))!=digest:
                raise ValueError('Changed frozen fresh-training input: '+name)
        if read_json(target/'decisions/gate_inputs.json')!=inventory['gate_inputs']:
            raise ValueError('Changed early gate input in fresh-training context')
        return target
    for name,digest in inventory['files'].items():
        if sha256_file(safe_path(source,name))!=digest:
            raise ValueError('Changed source before copying fresh-training context')
    temporary=target.with_name(target.name+'.preparing')
    if temporary.exists():raise ValueError('Interrupted context copy requires explicit diagnosis')
    temporary.mkdir(parents=True)
    for name,digest in inventory['files'].items():
        destination=safe_path(temporary,name);destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(safe_path(source,name),destination)
        if sha256_file(destination)!=digest:raise ValueError('Copied training input differs from source')
    atomic_json(temporary/'decisions/gate_inputs.json',inventory['gate_inputs'])
    atomic_json(temporary/'FRESH_TRAINING_CONTEXT.json',dict(inventory=inventory,
        source=str(source),created_utc=utc_now(),cached_training_copied=False))
    temporary.rename(target)
    return target


def prepare(root):
    """Freeze twelve independent fits only after the selected-weight replay completes."""
    root=Path(root).resolve()
    replay=read_json(root/'reports/control_replay.json')
    if (replay.get('state')!='complete' or replay.get('expected_episodes')!=60
            or replay.get('completed_episodes')!=60 or len(replay.get('episodes',[]))!=60
            or replay.get('diagnosis_required') is not False):
        raise ValueError('Complete qualified selected-weight replay required before fresh training')
    from .control_replay import _verify_registry,run_replays
    registry=read_json(root/'config/control_replay_jobs.json');_verify_registry(root,registry)
    run_replays(root,registry,sha256_file(root/'config/control_replay_jobs.json'),resume=True)
    frozen=read_json(root/'config/control_confirmation_cells.json')
    from .control_confirmation import _check_selected_fit,verify_freeze
    verify_freeze(root,read_json(root/'config/control_confirmation_design.json'),frozen)
    target=root/'config/control_retraining_jobs.json'
    if target.exists():return verify_registry(root,read_json(target))
    jobs=[];inputs={}
    for item in training_replication_cells(frozen):
        _check_selected_fit(root,item)
        path=safe_path(root,item['manifest']);record=read_json(path);source=path.parents[3]
        parameters=record['job_inputs']['parameters']
        inventory=retraining_context_inventory(source,parameters,record['parent_decisions'])
        cell=f"{item['actuator']}__A__shared__{item['model_id']}__seed{item['seed']}"
        context='study/control-retraining/'+cell
        definition=dict(actuator=item['actuator'],condition='A',loss_mode='shared',model_id=item['model_id'],
            seed=item['seed'],cell_id=cell)
        jobs.append(dict(cell_id=cell,definition=definition,original=dict(item),
            source=str(source.relative_to(root)),context=context,inventory=inventory))
        for name,digest in inventory['files'].items():inputs[str((source/name).relative_to(root))]=digest
        for name in ('manifest.json','weights.pt','summary.json','resources.json'):
            filename=str((path.parent/name).relative_to(root));inputs[filename]=sha256_file(root/filename)
    from .training_jobs import SOURCES
    for name in (*SOURCES,'poc/control_retraining.py','poc/control_confirmation.py',
                 'poc/control_replay.py','poc/control_models.py','config/control_confirmation_cells.json',
                 'config/control_replay_jobs.json','reports/control_replay.json'):
        inputs[name]=sha256_file(root/name)
    result=dict(state='complete',jobs=jobs,input_hashes=inputs,expected_fits=12,
        selection='Same source-context nominal-only engine, initialization and recipe; no copied neural fits',
        resource_projection=dict(context_copy_bytes=sum(sum((root/j['source']/n).stat().st_size
            for n in j['inventory']['files']) for j in jobs),
            interpretation='Exact context input bytes; fresh nominal convergence and control costs measured during execution'),
        control_reproduction='All twenty tasks for seeds 0 and 4 at both clocks, 480 episodes, required separately',
        criterion_stability_checked=False)
    atomic_json(target,result);return result


def verify_registry(root,registry):
    root=Path(root).resolve()
    for name,digest in registry['input_hashes'].items():
        if sha256_file(safe_path(root,name))!=digest:raise ValueError('Changed retraining dependency: '+name)
    frozen=read_json(root/'config/control_confirmation_cells.json')
    selected=training_replication_cells(frozen)
    if (registry.get('state')!='complete' or registry.get('expected_fits')!=12
            or len(registry.get('jobs',[]))!=12):
        raise ValueError('Complete fixed-seed retraining registry required')
    for item,job in zip(selected,registry['jobs']):
        if item!=job['original']:raise ValueError('Changed selected fit in retraining registry')
        cell=f"{item['actuator']}__A__shared__{item['model_id']}__seed{item['seed']}"
        definition=dict(actuator=item['actuator'],condition='A',loss_mode='shared',model_id=item['model_id'],
            seed=item['seed'],cell_id=cell)
        if (job['cell_id']!=cell or job['definition']!=definition
                or job['context']!='study/control-retraining/'+cell):
            raise ValueError('Changed fresh-training cell or independent context')
        path=safe_path(root,item['manifest']);record=read_json(path)
        if (safe_path(root,job['source'])!=path.parents[3] or job['inventory']!=
                retraining_context_inventory(path.parents[3],record['job_inputs']['parameters'],record['parent_decisions'])):
            raise ValueError('Changed fresh-training context inventory')
    return registry


def assert_validation_scores(observed,saved):
    """A selected-score receipt must describe the actual selected tensors."""
    for name in ('objective','one_step','four_step'):
        first,second=observed.get(name),saved.get(name)
        if (type(first) not in (int,float) or type(second) not in (int,float)
                or not math.isfinite(first) or not math.isfinite(second)
                or not math.isclose(first,second,rel_tol=1e-10,abs_tol=1e-12)):
            raise ValueError('Selected tensors disagree with nominal validation: '+name)
    for name in ('one_count','four_count'):
        if type(observed.get(name)) is not int or observed[name]!=saved.get(name):
            raise ValueError('Selected validation coverage differs: '+name)


def _observe_fit(root,job,path):
    from .contracts import verify_run
    from .train import qualify_nominal
    root=Path(root).resolve();context=safe_path(root,job['context']);path=Path(path).resolve()
    if path.parent.parent!=context/'evidence/M30':
        raise ValueError('Fresh fit manifest is outside its independent training context')
    record=read_json(path);verify_run(context,record)
    parameters=record['job_inputs']['parameters']
    if (record['milestone_id']!='M30' or record['phase']!='replication'
            or any(parameters.get(k)!=job['definition'][k]
                   for k in ('actuator','condition','loss_mode','model_id','seed'))
            or not qualify_nominal(record,context)['qualified']):
        raise ValueError('Fresh fit changed identity or has not qualified nominal training')
    original=safe_path(root,job['original']['manifest'])
    weights=torch.load(original.parent/'weights.pt',weights_only=True,map_location='cpu')
    repeated=torch.load(path.parent/'weights.pt',weights_only=True,map_location='cpu')
    from .train import canonical_tensor_hash
    first_record=read_json(original)
    for tensors,manifest in ((weights,first_record),(repeated,record)):
        if canonical_tensor_hash(tensors)!=manifest['model_hashes']['canonical_tensor_sha256']:
            raise ValueError('Fresh or original selected tensor hash differs from its manifest')
    from .execution import cpu_defaults
    from .dataset import load_corpus
    from .models.common import Scales
    from .models.factory import build_model
    from .train import validate_nominal
    cpu_defaults(parameters['seed'])
    norm=read_json(path.parent/'normalization.json')
    if norm!=read_json(original.parent/'normalization.json'):
        raise ValueError('Fresh training changed the original train-only normalization')
    scales=Scales(*(torch.tensor(norm[k],dtype=torch.float32)
                   for k in ('scale_q','scale_v','velocity_input_rms')))
    model=build_model(parameters['model_id'],parameters['actuator'],scales,parameters['seed'])
    model.load_state_dict(repeated,strict=True)
    arrays=load_corpus(safe_path(context,parameters['validation_manifest']),
        phase='replication',role='validation',purpose='selection')
    observed=validate_nominal(model,arrays,scales,parameters['loss_mode'])
    assert_validation_scores(observed,read_json(path.parent/'selection.json'))
    assert_validation_scores(observed,read_json(path.parent/'summary.json')['validation'])
    if canonical_tensor_hash(model.state_dict())!=record['model_hashes']['canonical_tensor_sha256']:
        raise ValueError('Independent nominal validation changed selected tensors')
    comparison=compare_training(weights,repeated,read_json(original.parent/'summary.json')['validation'],
                                read_json(path.parent/'summary.json')['validation'])
    return dict(cell_id=job['cell_id'],manifest=str(path.relative_to(root)),manifest_sha256=sha256_file(path),
        checkpoint_sha256=sha256_file(path.parent/'weights.pt'),comparison=comparison,
        independently_recomputed_validation=observed,
        resources=read_json(path.parent/'resources.json'))


def run_fits(root,registry,registry_hash,*,resume=False):
    """New child interpreters import the exact archived training engine, without fit caches."""
    root=Path(root).resolve();target=root/'reports/control_retraining.json';fits=[]
    jobs={j['cell_id']:j for j in registry['jobs']}
    if len(jobs)!=len(registry['jobs']):raise ValueError('Duplicate fresh-training cells')
    def save(state,**extra):
        result=dict(state=state,registry_sha256=registry_hash,expected_fits=len(jobs),completed_fits=len(fits),
            fits=fits,control_checked=False,criterion_stability_checked=False,**extra)
        atomic_json(target,result);return result
    with job_lock(job_key({'fixed_control_retraining':'seeds0and4'}),root):
        verify_registry(root,registry);reused={}
        if target.exists():
            previous=read_json(target)
            if not resume or previous['registry_sha256']!=registry_hash:
                raise ValueError('Existing fresh-training receipt requires compatible --resume')
            for item in previous['fits']:
                cell=item['cell_id'];path=safe_path(root,item['manifest'])
                if cell not in jobs or cell in reused or sha256_file(path)!=item['manifest_sha256']:
                    raise ValueError('Changed or duplicate completed fresh fit')
                actual=_observe_fit(root,jobs[cell],path)
                if actual!=item:raise ValueError('Fresh fit receipt differs from actual tensors or scores')
                reused[cell]=item
            if previous['state']=='complete':
                if set(reused)!=set(jobs) or previous['completed_fits']!=len(jobs):
                    raise ValueError('Incomplete completed fresh-training cohort')
                return previous
        script='''import json,sys
from pathlib import Path
from poc.mechanism import run_worker,train_cell
from poc.io import run_directory
from poc.train import qualify_nominal
root=Path.cwd();definition=json.loads(sys.argv[1]);folder=root/'fresh_fit';folder.mkdir(exist_ok=True)
run_worker(root,folder,'controls',dict(actuator=definition['actuator'],model_id=definition['model_id'],seed=8100),True)
record=train_cell(root,folder,definition,'replication',resume=True)
if not qualify_nominal(record,root)['qualified']:raise RuntimeError('Fresh nominal fitting remains unqualified')
print(json.dumps(dict(manifest_path=str((run_directory(root,record)/'manifest.json').relative_to(root)))))
'''
        environment=dict(os.environ)
        for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
            environment[name]='1'
        environment.pop('PYTHONPATH',None)
        try:
            for i,job in enumerate(registry['jobs']):
                verify_registry(root,registry)
                if job['cell_id'] in reused:fits.append(reused[job['cell_id']]);continue
                context=copy_retraining_context(safe_path(root,job['source']),safe_path(root,job['context']),job['inventory'])
                print(f"Fresh nominal training {i+1}/{len(jobs)}: {job['cell_id']}",flush=True)
                child=subprocess.run([sys.executable,'-c',script,json.dumps(job['definition'])],
                    cwd=context,env=environment,capture_output=True,text=True)
                attempt=root/'reports/control_retraining_attempts'/f'{i:04d}-{utc_now().replace(":","")}'
                attempt.mkdir(parents=True);(attempt/'stdout.log').write_text(child.stdout);(attempt/'stderr.log').write_text(child.stderr)
                if child.returncode:raise RuntimeError('Fresh nominal child incomplete: '+child.stderr[-2000:])
                result=json.loads(child.stdout.strip().splitlines()[-1])
                path=safe_path(context,result['manifest_path'])
                fits.append(_observe_fit(root,job,path));save('running')
            return save('complete',diagnosis_required=any(f['comparison']['diagnosis_required'] for f in fits))
        except BaseException as error:
            save('interrupted',reason=f'{type(error).__name__}: {error}');raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv);root=args.root.resolve()
    if args.dry_run:
        print(dict(execute=False,expected_fits=12,seeds=[0,4],models=list(MODELS),actuators=list(ACTUATORS),
            prerequisite='Complete qualified selected-weight replay',cached_training_copied=False,
            control_reproduction_required_episodes=480))
        return 0
    if sys.version_info[:2]!=(3,11):raise ValueError('Fresh training requires pinned CPython 3.11')
    from .study_reproduce import validate_runtime_versions
    import importlib.metadata
    versions=read_json(root/'locks/environment.json')['versions']
    validate_runtime_versions(versions,{n:importlib.metadata.version(n) for n in versions})
    authorization=read_json(root/'config/authorization.json')
    if authorization.get('local_compute_authorized') is not True or authorization.get('paid_services_authorized') is not False:
        raise ValueError('Fresh training requires existing local compute authorization')
    path=root/'config/control_retraining_jobs.json'
    registry=prepare(root) if args.prepare else verify_registry(root,read_json(path))
    if args.prepare and not args.resume:return 0
    result=run_fits(root,registry,sha256_file(path),resume=args.resume)
    return 1 if result.get('diagnosis_required') else 0


if __name__=='__main__':
    raise SystemExit(main())
