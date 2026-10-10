"""Fresh, version-specific data and nominal paired fits for PTCO-AFF-0.2."""
from __future__ import annotations

import argparse
import io
import os
from pathlib import Path

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'

import numpy as np
import torch

from .affine_experiment import train_block, verify_bindings
from .io import atomic_bytes, atomic_json, job_key, read_json, sha256_file, utc_now
from .train import (TrainingSession, analytic_control_arrays, array_scales, batch_objective,
                    canonical_tensor_hash)

PROTOCOL = 'PTCO-AFF-0.2'
FREEZE = 'config/affine_fresh_freeze.json'
SPEC = 'docs/superpowers/specs/2026-10-08-affine-independent-confirmation.md'
MODELS = ('affine_flow', 'node_structured')
SEEDS = tuple(range(2026100800, 2026100805))
ROLES = {'train':0, 'validation':1, 'confirmation':2}


def fresh_entropy(role, trajectory_id, channel):
    if role not in ROLES or type(trajectory_id) is not int or trajectory_id < 0 or channel not in (0,1,2,3):
        raise ValueError('Explicit fresh role, trajectory identity and channel required')
    return [20261008, 2, ROLES[role], 1, trajectory_id, channel]


def fresh_rng(role, trajectory_id, channel):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(fresh_entropy(role, trajectory_id, channel))))


def build_fresh_model(name, scales, seed):
    if name == 'affine_flow':
        from .affine_experiment import build_candidate
        return build_candidate('lagged', scales, seed)
    if name == 'node_structured':
        from .models.factory import build_model
        return build_model(name, 'lagged', scales, seed)
    raise ValueError('Unregistered fresh comparator')


def fit_configuration(freeze, name, seed, data, replica_id=None):
    if name not in MODELS or type(seed) is not int or seed < SEEDS[0]:
        raise ValueError('Invalid paired fresh fit identity')
    if replica_id not in (None,'fresh_process_1'):
        raise ValueError('Unknown fresh training replication')
    return dict(freeze=freeze, model_id=name, actuator='lagged', seed=seed,
                loss_mode='shared', data=data, replica_id=replica_id)


def generate_fresh_arrays(role, count, physics_step_ns, assets, resume_dir=None):
    from .envs import Plant
    fresh_entropy(role, 0, 0)
    if type(count) is not int or count <= 0:
        raise ValueError('Positive complete trajectory count required')
    plant = Plant('lagged', physics_step_ns, Path(assets))
    identity = dict(protocol_id=PROTOCOL, role=role, count=count, physics_step_ns=physics_step_ns,
                    xml_sha256=plant.xml_sha256, source_sha256=sha256_file(Path(__file__)))
    resume_dir = Path(resume_dir) if resume_dir is not None else None
    if resume_dir:
        resume_dir.mkdir(parents=True, exist_ok=True)
        request = resume_dir/'request.json'
        if request.exists() and read_json(request) != identity:
            raise ValueError('Changed fresh corpus is not a resumption')
        atomic_json(request, identity)
    arrays = dict(state=np.empty((count,33,6),dtype=np.float64),
                  command=np.empty((count,32,2),dtype=np.float64),
                  duration_ns=np.empty((count,32),dtype=np.int64),
                  time_ns=np.zeros((count,33),dtype=np.int64), trajectory_id=np.arange(count,dtype=np.int64))
    for i in range(count):
        checkpoint = resume_dir/f'trajectory-{i}.npz' if resume_dir else None
        marker = checkpoint.with_suffix('.json') if checkpoint else None
        if checkpoint and checkpoint.exists() and marker.exists():
            if read_json(marker)['sha256'] != sha256_file(checkpoint):
                raise ValueError('Fresh resumed trajectory hash mismatch')
            with np.load(checkpoint, allow_pickle=False) as stored:
                for key in ('state', 'command', 'duration_ns', 'time_ns'):
                    arrays[key][i] = stored[key]
            continue
        rng = fresh_rng(role,i,0)
        initial = np.r_[rng.uniform(-np.pi,np.pi,2),rng.uniform(-2,2,2),rng.uniform(-2,2,2)]
        commands = fresh_rng(role,i,1).uniform(-2,2,(32,2))
        durations = fresh_rng(role,i,2).choice(np.array([20,30,40,50,60],dtype=np.int64)*1_000_000,32)
        plant.reset(initial)
        arrays['state'][i,0] = initial
        arrays['command'][i], arrays['duration_ns'][i] = commands, durations
        arrays['time_ns'][i,1:] = np.cumsum(durations)
        for j,(command,duration) in enumerate(zip(commands,durations)):
            arrays['state'][i,j+1] = plant.advance(command,int(duration))['state']
        if checkpoint:
            buffer=io.BytesIO(); np.savez_compressed(buffer,**{k:arrays[k][i] for k in
                                                   ('state','command','duration_ns','time_ns')})
            atomic_bytes(checkpoint,buffer.getvalue()); atomic_json(marker,dict(sha256=sha256_file(checkpoint)))
    return arrays


def validate_fresh_manifest(manifest):
    if manifest.get('protocol_id') != PROTOCOL:
        raise ValueError('Wrong fresh-data protocol')
    if (manifest.get('role') not in ROLES or manifest.get('actuator') != 'lagged'
            or manifest.get('condition') != 'A' or set(manifest.get('files',{})) != {'trajectories.npz'}):
        raise ValueError('Wrong fresh-data role/plant/files')


def load_fresh_arrays(path, purpose):
    from .dataset import validate_arrays
    path = Path(path); manifest = read_json(path)
    validate_fresh_manifest(manifest)
    allowed = {'training':{'train'}, 'selection':{'validation'}, 'evaluation':{'confirmation'}}
    if purpose not in allowed or manifest['role'] not in allowed[purpose]:
        raise ValueError('Forbidden fresh-data role for purpose')
    if sha256_file(path.parent/'trajectories.npz') != manifest['files']['trajectories.npz']:
        raise ValueError('Fresh data hash mismatch')
    with np.load(path.parent/'trajectories.npz',allow_pickle=False) as stored:
        arrays = {key:stored[key].copy() for key in stored.files}
    validate_arrays(arrays,manifest)
    return arrays


def source_files():
    from .affine_measurement import _sources
    return sorted(set(_sources())|{'poc/affine_fresh.py', 'poc/dataset.py'})


def learning_bindings(root):
    from .contracts import verify_run
    from .train import qualified_learning_control
    development=read_json(root/'config/affine_measurement_freeze.json')
    candidate=development['selections']['lagged']; candidate_path=root/candidate['model_manifest']
    verify_bindings(root,{candidate['model_manifest']:candidate['manifest_sha256']})
    verify_run(root,read_json(candidate_path))
    first=read_json(candidate_path.parent/'learning_controls.json')
    if first['tiny_loss'] >= .001 or first['analytic_validation'] >= .05:
        raise ValueError('Original affine learning controls did not qualify')
    baseline_path=root/development['baseline_selections']['lagged:node_structured']['model_manifest']
    context=baseline_path.parents[3]
    control_path=qualified_learning_control(context,'node_structured','lagged')
    control=read_json(control_path); second=read_json(control_path.parent/'summary.json')
    if second['tiny_loss'] >= .001 or second['validation']['one_step'] >= .05:
        raise ValueError('Original NODE learning controls did not qualify')
    for filename,expected in control['job_inputs']['dependencies'].items():
        if filename.startswith('poc/models/') or filename in {'poc/train.py','poc/analytic.py','poc/history.py'}:
            if sha256_file(root/filename) != expected:
                raise ValueError('Current model/training differs from bound analytic control')
    paths=[candidate_path,candidate_path.parent/'learning_controls.json',control_path,control_path.parent/'summary.json']
    return {str(p.relative_to(root)):sha256_file(p) for p in paths}


def prepare(root):
    root = Path(root).resolve()
    if (root/FREEZE).exists():
        return load_freeze(root)
    authority = read_json(root/'config/authorization.json')
    if not authority['local_compute_authorized'] or authority['paid_services_authorized']:
        raise ValueError('Existing local CPU authorization required')
    files = source_files()
    inputs = {name:sha256_file(root/name) for name in
              (SPEC,'config/authorization.json','config/numerical.json','config/planner.json',
               'assets/two_link_lagged.xml','locks/environment.json','locks/wheels.json',
               'reports/affine_prediction_measurement.json','reports/affine_prediction_result.json')}
    controls=learning_bindings(root); inputs.update(controls)
    freeze = dict(protocol_id=PROTOCOL,created_utc=utc_now(),initial_seeds=list(SEEDS),models=list(MODELS),
                  namespace=[20261008,2],actuator='lagged',loss_mode='shared',epoch_block=100,
                  learning_rates=[.001,.0003],counts={'train':1024,'validation':256,'confirmation':256},
                  physics_step_ns=125000,control_physics_step_ns=250000,initial_episodes_per_seed=10,
                  control_configurations=[['affine_flow',5000000],['node_structured',5000000],['node_structured',40000000]],
                  planner_settings=read_json(root/'config/planner.json')['settings'],
                  source_hashes={name:sha256_file(root/name) for name in files},input_hashes=inputs,
                  learning_controls=controls,
                  work_context='study/affine-fresh',initial_alpha=.025,
                  primary_statistics=['q_error_5ms','v_error_5ms','batch128_latency_5ms'],
                  inference_scope='Fresh model seeds and tasks; fixed version-specific train/probe corpus; one known-lag robot')
    atomic_json(root/FREEZE,freeze)
    return freeze


def load_freeze(root):
    freeze = read_json(root/FREEZE)
    if (freeze['protocol_id'] != PROTOCOL or freeze['models'] != list(MODELS)
            or freeze['initial_seeds'] != list(SEEDS) or freeze['namespace'] != [20261008,2]
            or freeze['counts'] != {'train':1024,'validation':256,'confirmation':256}):
        raise ValueError('Changed fresh confirmation design')
    verify_bindings(root,{**freeze['source_hashes'],**freeze['input_hashes']})
    return freeze


def assert_development_controls_complete(root):
    batch = read_json(root/'reports/affine_control_batch.json')
    if batch['state'] != 'complete' or len(batch['completed']) != 80:
        raise ValueError('Finish the independent development control batch before fresh computation')


def _receipt(root, record):
    if record['completion_status'] != 'complete':
        raise ValueError('Fresh producer failed; preserve and diagnose '+record['run_id'])
    name = f'evidence/{record["milestone_id"]}/{record["run_id"]}/manifest.json'
    return dict(path=name,sha256=sha256_file(root/name))


def job_configuration(root, stage, values):
    """Bind semantic values through real immutable files consumed by verify_run."""
    path = 'config/affine_fresh_requests/'+stage+'-'+job_key(values)+'.json'
    target = Path(root)/path
    if target.exists():
        if read_json(target) != values:
            raise ValueError('Changed immutable fresh job configuration')
    else:
        atomic_json(target, values)
    return {path:values}


def dataset(root, role):
    from .execution import cpu_defaults,run_job
    from .dataset import validate_arrays
    root = Path(root).resolve(); freeze=load_freeze(root); assert_development_controls_complete(root)
    cpu_defaults(SEEDS[0])
    if role not in ROLES:
        raise ValueError('Unknown fresh corpus role')
    if role == 'confirmation':
        selected_freeze(root)  # Required before held-out inputs/labels exist.
    request = dict(protocol_id=PROTOCOL,role=role,count=freeze['counts'][role],freeze=freeze)
    files=sorted(set(freeze['source_hashes'])|set(freeze['input_hashes'])|{FREEZE})
    work=root/freeze['work_context']/('data-'+role)
    def produce(folder):
        arrays=generate_fresh_arrays(role,request['count'],freeze['physics_step_ns'],root/'assets',work)
        buffer=io.BytesIO(); np.savez_compressed(buffer,**arrays); atomic_bytes(folder/'trajectories.npz',buffer.getvalue())
        manifest=dict(protocol_id=PROTOCOL,role=role,actuator='lagged',condition='A',
                      trajectory_count=request['count'],transition_count=request['count']*32,
                      physics_step_ns=freeze['physics_step_ns'],namespace=freeze['namespace'],
                      entropy_tuples=[[fresh_entropy(role,i,c) for c in range(4)] for i in range(request['count'])],
                      arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in arrays.items()},
                      files={'trajectories.npz':sha256_file(folder/'trajectories.npz')},sealed=role=='confirmation',
                      source_sha256=freeze['source_hashes']['poc/affine_fresh.py'])
        validate_fresh_manifest(manifest); validate_arrays(arrays,manifest); atomic_json(folder/'corpus.json',manifest)
        check=dict(check_id='fresh.corpus',status='pass',scope=role,tolerance=None,
                   reason='New namespace, complete trajectories and integer physical clocks')
        return [check],dict(protocol_id=PROTOCOL,role=role,missing_cells=[],trajectory_count=request['count'])
    configuration=job_configuration(root,'corpus',request)
    record=run_job(root,'M30','confirmatory',sorted(set(files)|set(configuration)),('fresh.corpus',),[],produce,
                   config_keys=configuration)
    receipt=_receipt(root,record); atomic_json(root/f'reports/affine_fresh_data_{role}.json',receipt)
    return receipt


def data_inputs(root):
    from .contracts import verify_run
    entries={role:read_json(root/f'reports/affine_fresh_data_{role}.json') for role in ('train','validation')}
    arrays={}
    for role,entry in entries.items():
        verify_bindings(root,{entry['path']:entry['sha256']}); path=root/entry['path']
        verify_run(root,read_json(path)); arrays[role]=load_fresh_arrays(path.parent/'corpus.json',
                                                                 'training' if role=='train' else 'selection')
    return entries,arrays


def fit(root,name,seed,replica_id=None):
    from .execution import cpu_defaults,run_job
    root=Path(root).resolve(); freeze=load_freeze(root); assert_development_controls_complete(root)
    entries,arrays=data_inputs(root); cpu_defaults(seed)
    configuration=fit_configuration(freeze,name,seed,entries,replica_id)
    files=set(freeze['source_hashes'])|set(freeze['input_hashes'])|{FREEZE}
    for entry in entries.values():
        path=root/entry['path']; files|={entry['path'],str((path.parent/'corpus.json').relative_to(root)),
                                      str((path.parent/'trajectories.npz').relative_to(root))}
    work=root/freeze['work_context']/job_key(configuration)
    def produce(folder):
        train,validation=arrays['train'],arrays['validation']; scales=array_scales(train)
        chosen,result=None,None
        for rate in freeze['learning_rates']:
            output=work/('lr'+str(rate).replace('.','_')); output.mkdir(parents=True,exist_ok=True)
            session=TrainingSession(build_fresh_model(name,scales,seed),train,scales,seed,'shared',learning_rate=rate)
            if (output/'resume.complete.json').exists() or (output/'resume.prev.complete.json').exists():
                session.restore_latest(output/'resume.pt')
            end=max(100,((session.epoch+99)//100)*100)
            while True:
                result=train_block(session,validation,output,end)
                if result['qualified']:
                    chosen=output;break
                if result['convergence']['stabilized']:
                    break
                end+=100
            if chosen is not None:
                break
        if chosen is None:
            raise ValueError('Both prespecified nominal rates plateaued unqualified; optimizer diagnosis required')
        if result['valid_target_count'] != result['epochs']*train['command'].shape[0]*32:
            raise ValueError('Fresh dataset-equivalent exposure mismatch')
        for filename in ('weights.pt','selection.json','validation.csv','training.csv','resume.pt',
                         'resume.rng.json','resume.complete.json','selection.pt','selection.complete.json'):
            atomic_bytes(folder/filename,(chosen/filename).read_bytes())
        normalization=dict(protocol_id=PROTOCOL,scale_q=scales.q.tolist(),scale_v=scales.v.tolist(),
                           velocity_input_rms=scales.v_input.tolist(),training_receipt=entries['train'],
                           transition_count=train['command'].shape[0]*32)
        atomic_json(folder/'normalization.json',normalization); atomic_json(folder/'request.json',configuration)
        summary=dict(protocol_id=PROTOCOL,actuator='lagged',model_id=name,seed=seed,phase='confirmatory',
                     qualification=result,parameter_count=sum(p.numel() for p in session.model.parameters()),
                     validation=result['selected'],training_updates=result['updates'],missing_cells=[],
                     data_hashes=entries,model_hashes={'canonical_tensor_sha256':result['canonical_tensor_sha256']},
                     work_context=str(work.relative_to(root)))
        return [dict(check_id='fresh.nominal_fit',status='pass',scope=f'{name}:{seed}',tolerance=None,
                     reason='Independent initialization, complete exposure and nominal-only qualified selection')],summary
    print(f'Fresh nominal fit {name} seed{seed}',flush=True)
    keys=job_configuration(root,'fit',configuration)
    record=run_job(root,'M30','confirmatory',sorted(files|set(keys)),('fresh.nominal_fit',),[],produce,
                   config_keys=keys)
    receipt=_receipt(root,record)
    summary=read_json(root/receipt['path']).get('model_hashes',{})
    path=(root/receipt['path']).parent
    receipt.update(model_id=name,seed=seed,checkpoint_sha256=sha256_file(path/'weights.pt'),
                   canonical_tensor_sha256=summary['canonical_tensor_sha256'])
    directory='affine_fresh_fits' if replica_id is None else 'affine_fresh_replicas'
    atomic_json(root/f'reports/{directory}/{name}__seed{seed}.json',receipt)
    return receipt


def seal(root):
    from .contracts import verify_run
    root=Path(root).resolve(); freeze=load_freeze(root)
    selections={}
    for seed in SEEDS:
        for name in MODELS:
            receipt=read_json(root/f'reports/affine_fresh_fits/{name}__seed{seed}.json')
            path=root/receipt['path']; verify_bindings(root,{receipt['path']:receipt['sha256']})
            verify_run(root,read_json(path))
            if receipt['model_id'] != name or receipt['seed'] != seed or sha256_file(path.parent/'weights.pt') != receipt['checkpoint_sha256']:
                raise ValueError('Selected fresh fit identity mismatch')
            selections[f'{name}:{seed}']=receipt
    value=dict(protocol_id=PROTOCOL,freeze_sha256=sha256_file(root/FREEZE),selections=selections,
               data=data_inputs(root)[0],created_utc=utc_now())
    path=root/'config/affine_fresh_selected.json'
    if path.exists():
        existing=read_json(path)
        if {k:v for k,v in existing.items() if k!='created_utc'} != {k:v for k,v in value.items() if k!='created_utc'}:
            raise ValueError('Fresh selections already sealed under different inputs')
        return existing
    atomic_json(path,value); return value


def selected_freeze(root):
    value=read_json(root/'config/affine_fresh_selected.json')
    if value['protocol_id'] != PROTOCOL or value['freeze_sha256'] != sha256_file(root/FREEZE):
        raise ValueError('Fresh selected freeze mismatch')
    return value


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','data','train','retrain','seal','status'))
    parser.add_argument('--root',type=Path,default=Path.cwd()); parser.add_argument('--role',choices=tuple(ROLES))
    parser.add_argument('--model',choices=MODELS); parser.add_argument('--seed',type=int)
    args=parser.parse_args(argv); root=args.root.resolve()
    if args.command=='prepare':
        value=prepare(root); print(dict(protocol_id=PROTOCOL,state='frozen',initial_fits=len(SEEDS)*len(MODELS)))
    elif args.command=='data':
        if args.role is None: parser.error('--role required')
        print(dataset(root,args.role))
    elif args.command in {'train','retrain'}:
        initial=SEEDS if args.command=='train' else (SEEDS[0],SEEDS[-1])
        for seed in (initial if args.seed is None else (args.seed,)):
            for name in (MODELS if args.model is None else (args.model,)):
                print(fit(root,name,seed,None if args.command=='train' else 'fresh_process_1'),flush=True)
    elif args.command=='seal':
        print(dict(state='selected_sealed',count=len(seal(root)['selections'])))
    else:
        value=load_freeze(root); print(dict(protocol_id=PROTOCOL,initial_seeds=value['initial_seeds'],
                           completed_fits=len(list((root/'reports/affine_fresh_fits').glob('*.json')))))


if __name__=='__main__':
    main()
