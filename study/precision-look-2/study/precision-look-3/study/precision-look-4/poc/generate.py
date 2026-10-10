"""PCG64 trajectory generation with complete-trajectory resume boundaries."""
from __future__ import annotations

import io
from pathlib import Path
import numpy as np

from .dataset import ROLES, validate_arrays
from .envs import Plant
from .io import atomic_bytes, atomic_json, canonical_json, job_key, read_json, sha256_file

ROOT_SEEDS = {'train':(1201,1202),'validation':(2201,2202),
              'confirmation':(3201,3202),'development':(4201,4202)}


def entropy(actuator: str, condition: str, role: str, trajectory_id: int, channel: int) -> list[int]:
    if actuator not in {'direct','lagged'} or condition not in {'F','A'} or role not in ROLES:
        raise ValueError('Unknown actuator, condition or corpus role')
    return [ROOT_SEEDS[role][condition == 'A'],int(actuator == 'lagged'),int(trajectory_id),int(channel)]


def rng_for(actuator,condition,role,trajectory_id,channel):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy(actuator,condition,role,trajectory_id,channel))))


def generate_arrays(actuator: str, condition: str, role: str, n_trajectories: int,
                    physics_step_ns: int, asset_root: Path, resume_dir: Path | None = None) -> dict:
    if type(n_trajectories) is not int or n_trajectories <= 0:
        raise ValueError('A positive whole-trajectory count is required')
    entropy(actuator,condition,role,0,0)
    robot = Plant(actuator,physics_step_ns,asset_root)
    dimension = 4 if actuator == 'direct' else 6
    arrays = {'state':np.empty((n_trajectories,33,dimension),dtype=np.float64),
              'command':np.empty((n_trajectories,32,2),dtype=np.float64),
              'duration_ns':np.empty((n_trajectories,32),dtype=np.int64),
              'time_ns':np.zeros((n_trajectories,33),dtype=np.int64),
              'trajectory_id':np.arange(n_trajectories,dtype=np.int64)}
    identity = {'actuator':actuator,'condition':condition,'role':role,'count':n_trajectories,
                'step_ns':physics_step_ns,'xml_sha256':robot.xml_sha256,
                'source_sha256':sha256_file(Path(__file__))}
    if resume_dir:
        resume_dir.mkdir(parents=True,exist_ok=True)
        request = resume_dir / 'request.json'
        if request.exists() and read_json(request) != identity:
            raise ValueError('Changed corpus generation is not a resumption')
        atomic_json(request,identity)
    for trajectory_id in range(n_trajectories):
        checkpoint = resume_dir / f'trajectory-{trajectory_id}.npz' if resume_dir else None
        if checkpoint and checkpoint.exists() and checkpoint.with_suffix('.json').exists():
            marker = read_json(checkpoint.with_suffix('.json'))
            if marker['sha256'] != sha256_file(checkpoint):
                raise ValueError('Resumed trajectory hash mismatch')
            with np.load(checkpoint,allow_pickle=False) as saved:
                for key in ('state','command','duration_ns','time_ns'):
                    arrays[key][trajectory_id] = saved[key]
            continue
        initial = rng_for(actuator,condition,role,trajectory_id,0)
        state = np.r_[initial.uniform(-np.pi,np.pi,2),initial.uniform(-2,2,2)]
        if actuator == 'lagged':
            state = np.r_[state,initial.uniform(-2,2,2)]
        commands = rng_for(actuator,condition,role,trajectory_id,1).uniform(-2,2,(32,2))
        durations = np.full(32,40_000_000,dtype=np.int64) if condition == 'F' else rng_for(actuator,condition,role,trajectory_id,2).choice(np.array([20,30,40,50,60],dtype=np.int64)*1_000_000,size=32)
        robot.reset(state)
        arrays['state'][trajectory_id,0] = state
        arrays['command'][trajectory_id],arrays['duration_ns'][trajectory_id] = commands,durations
        arrays['time_ns'][trajectory_id,1:] = np.cumsum(durations)
        for step,(command,duration) in enumerate(zip(commands,durations)):
            arrays['state'][trajectory_id,step+1] = robot.advance(command,int(duration))['state']
        if checkpoint:
            buffer = io.BytesIO()
            np.savez_compressed(buffer,**{key:arrays[key][trajectory_id] for key in ('state','command','duration_ns','time_ns')})
            atomic_bytes(checkpoint,buffer.getvalue())
            atomic_json(checkpoint.with_suffix('.json'),{'sha256':sha256_file(checkpoint),'trajectory_id':trajectory_id})
    return arrays


def corpus_manifest(arrays: dict, actuator: str, condition: str, role: str, file: Path,
                    numerical: dict, parent_decision: dict | None, root: Path, phase: str) -> dict:
    if parent_decision is None and phase != 'unit':
        raise ValueError('Measured corpora require a qualified numerical decision')
    robot = Plant(actuator,numerical['physics_step_ns'],root / 'assets' if (root / 'assets').exists() else Path('assets'))
    compiled = robot.compiled_manifest()
    import hashlib
    result = {'schema_version':1,'protocol_version':'PTCO-1.0','role':role,'phase':phase,'actuator':actuator,
              'condition':condition,'trajectory_count':int(arrays['state'].shape[0]),'transition_count':int(arrays['command'].shape[0]*32),
              'arrays':{key:{'shape':list(value.shape),'dtype':str(value.dtype),
                        'unit':{'state':'rad, rad/s[, N m]','command':'N m','duration_ns':'ns','time_ns':'ns','trajectory_id':'identifier'}[key]}
                        for key,value in arrays.items()},
              'files':{'trajectories.npz':sha256_file(file)},'generation_code_sha256':sha256_file(Path(__file__)),
              'xml_sha256':robot.xml_sha256,'compiled_physics_sha256':hashlib.sha256(canonical_json(compiled)).hexdigest(),
              'physics_step_ns':numerical['physics_step_ns'],'numerical':numerical,'parent_numerical_decision':parent_decision,
              'seeds':{'root_seed':ROOT_SEEDS[role][condition == 'A'],'actuator_code':int(actuator == 'lagged'),
                       'trajectory_ids':arrays['trajectory_id'].tolist(),'channel_codes':[0,1,2],
                       'entropy_tuples':[[entropy(actuator,condition,role,int(i),channel) for channel in (0,1,2)] for i in arrays['trajectory_id']]},
              'warnings':[],'command_range':[float(arrays['command'].min()),float(arrays['command'].max())],
              'state_range':[float(arrays['state'].min()),float(arrays['state'].max())],
              'sealed':role == 'confirmation','allowed_readers':['evaluation','replication'] if role == 'confirmation' else []}
    validate_arrays(arrays,result)
    return result


def generate_corpus(root: Path, actuator: str, condition: str, role: str, n_trajectories: int) -> dict:
    """Produce a qualified corpus and an immutable standard run receipt."""
    from .execution import parent, run_job
    from .dataset import load_corpus
    root = Path(root).resolve()
    entropy(actuator,condition,role,0,0)
    freeze = None
    if role == 'confirmation':
        for filename in ('frozen.json','rejection_freeze.json'):
            candidate = root / 'config' / filename
            if candidate.exists():
                freeze = read_json(candidate)
                if freeze.get('protocol_version') != 'PTCO-1.0' or freeze.get('statistical_family_size') != 280:
                    raise ValueError('Confirmation requires a complete preregistered freeze')
                declared = freeze.get('generation_hashes',{})
                if declared.get('poc/generate.py') != sha256_file(Path(__file__)):
                    raise ValueError('Confirmation generation is absent/stale in the freeze')
                break
        if freeze is None:
            raise ValueError('Confirmation generation requires an actual freeze')
    predecessor = parent(root,'G20')
    numerical = read_json(root / 'config/numerical.json')
    from .io import run_directory
    numerical_record = read_json(root / predecessor['evidence'][0]['path'])
    if numerical != read_json(run_directory(root,numerical_record) / 'numerical.json'):
        raise ValueError('Numerical configuration differs from qualified reference')
    folder = root / 'data' / actuator / condition / role
    phase = 'confirmatory' if role == 'confirmation' else 'exploratory'
    request = {'actuator':actuator,'condition':condition,'role':role,'trajectory_count':n_trajectories,
               'freeze':freeze,'numerical':numerical}
    request_path = root / 'config/corpus_requests' / f'{actuator}__{condition}__{role}.json'
    if request_path.exists() and read_json(request_path) != request:
        raise ValueError('Changed corpus inputs require versioned output, not overwrite')
    atomic_json(request_path,request)
    files = ['poc/__init__.py','poc/__main__.py','poc/io.py','poc/contracts.py','poc/gates.py',
             'poc/execution.py','poc/generate.py','poc/dataset.py','poc/envs.py','poc/clocks.py',
             'poc/research_jobs.py',
             'poc/sensitivity.py','poc/oracle_rigidbody.py','poc/milestones/__init__.py',
             'poc/milestones/m10.py','poc/milestones/m20.py','schemas/dataset.schema.json',
             f'assets/two_link_{actuator}.xml']
    def produce(run_folder):
        if (folder / 'manifest.json').exists():
            load_corpus(folder / 'manifest.json',phase=phase,role=role,
                        purpose={'train':'training','validation':'selection','development':'calibration','confirmation':'evaluation'}[role])
            raise ValueError('Existing corpus lacks matching completed job; preserve it for diagnosis')
        arrays = generate_arrays(actuator,condition,role,n_trajectories,numerical['physics_step_ns'],
                                 root / 'assets',folder / '.generation')
        buffer = io.BytesIO()
        np.savez_compressed(buffer,**arrays)
        atomic_bytes(run_folder / 'trajectories.npz',buffer.getvalue())
        atomic_bytes(folder / 'trajectories.npz',buffer.getvalue())
        manifest = corpus_manifest(arrays,actuator,condition,role,folder / 'trajectories.npz',
                                   numerical,predecessor,root,phase)
        atomic_json(run_folder / 'corpus.json',manifest)
        atomic_json(folder / 'manifest.json',manifest)
        return [{'check_id':'dataset','status':'pass','reason':'Complete deterministic trajectories, qualified plant and hashes',
                 'tolerance':None,'scope':f'{actuator}/{condition}/{role}'}], {
                 'manifest_path':str((folder / 'manifest.json').relative_to(root)),
                 'manifest_sha256':sha256_file(folder / 'manifest.json'),'missing_cells':[],
                 'trajectory_count':n_trajectories,'transition_count':32*n_trajectories}
    return run_job(root,'M30',phase,files,('dataset',),[predecessor],produce,
                   config_keys={str(request_path.relative_to(root)):request})
