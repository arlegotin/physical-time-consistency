"""Explicit label-access policy and integrity checks for physical trajectories."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .contracts import PHASES, _validate_schema, verify_decision
from .io import append_event, atomic_json, read_json, sha256_file, utc_now

ROLES = ('train','validation','confirmation','development')
PURPOSE_ROLES = {'training':{'train'}, 'selection':{'validation'},
                 'evaluation':{'train','validation','development','confirmation'},
                 'calibration':{'development'}}
ARRAYS = ('state','command','duration_ns','time_ns','trajectory_id')


def project_root(manifest_path: Path) -> Path:
    path = Path(manifest_path).resolve()
    if len(path.parents) < 5 or path.parents[3].name != 'data':
        raise ValueError('Corpus manifest must live under data/actuator/condition/role')
    return path.parents[4]


def validate_arrays(arrays: dict, manifest: dict) -> None:
    if set(arrays) != set(ARRAYS):
        raise ValueError('Missing or unknown corpus arrays')
    count,dimension = manifest['trajectory_count'], 4 if manifest['actuator'] == 'direct' else 6
    shapes = {'state':(count,33,dimension),'command':(count,32,2),'duration_ns':(count,32),
              'time_ns':(count,33),'trajectory_id':(count,)}
    for key,value in arrays.items():
        expected_dtype = np.float64 if key in {'state','command'} else np.int64
        if value.shape != shapes[key] or value.dtype != expected_dtype or not np.isfinite(value).all():
            raise ValueError('Invalid shape, dtype or nonfinite values: '+key)
        if manifest['arrays'][key]['shape'] != list(value.shape) or manifest['arrays'][key]['dtype'] != str(value.dtype):
            raise ValueError('Manifest dimension/dtype disagrees with arrays: '+key)
    if len(np.unique(arrays['trajectory_id'])) != count or np.any(arrays['trajectory_id'] < 0):
        raise ValueError('Duplicate or negative trajectory identities')
    if np.any(np.abs(arrays['command']) > 2):
        raise ValueError('Physical command exceeds torque bounds')
    allowed = [40_000_000] if manifest['condition'] == 'F' else [20_000_000,30_000_000,40_000_000,50_000_000,60_000_000]
    durations = arrays['duration_ns']
    if not np.isin(durations,allowed).all() or np.any(durations % manifest['physics_step_ns']):
        raise ValueError('Duration law or numerical tick is invalid')
    if np.any(arrays['time_ns'][:,0]) or not np.array_equal(arrays['time_ns'][:,1:],np.cumsum(durations,axis=1)):
        raise ValueError('Physical time must equal cumulative integer durations')


def load_corpus(manifest_path: Path, phase: str, role: str, purpose: str,
                *, audit_path: Path | None = None) -> dict[str,np.ndarray]:
    path = Path(manifest_path).resolve()
    if phase not in PHASES or purpose not in PURPOSE_ROLES or role not in ROLES:
        raise ValueError('Undeclared phase, role or purpose for label access')
    if role not in PURPOSE_ROLES[purpose] or (role == 'confirmation' and phase not in {'confirmatory','replication','unit'}):
        raise ValueError('Forbidden confirmation/role access for this purpose or phase')
    root = project_root(path)
    manifest = read_json(path)
    _validate_schema(manifest,read_json(Path(__file__).resolve().parent.parent / 'schemas/dataset.schema.json'))
    if manifest.get('role') != role or path.parent.name != role:
        raise ValueError('Declared corpus role disagrees with manifest/path')
    if manifest.get('protocol_version') != 'PTCO-1.0' or manifest.get('schema_version') != 1:
        raise ValueError('Unsupported dataset schema/protocol')
    if manifest.get('actuator') not in {'direct','lagged'} or manifest.get('condition') not in {'F','A'}:
        raise ValueError('Unknown actuator or training condition')
    if path.parents[2].name != manifest['actuator'] or path.parents[1].name != manifest['condition']:
        raise ValueError('Dataset path actuator/condition conflict')
    if type(manifest['physics_step_ns']) is not int or manifest['physics_step_ns'] <= 0 or manifest['transition_count'] != manifest['trajectory_count']*32:
        raise ValueError('Invalid numerical tick or corpus count')
    if manifest['phase'] != 'unit':
        if manifest['parent_numerical_decision'] is None:
            raise ValueError('Missing qualified numerical provenance')
        verify_decision(root,manifest['parent_numerical_decision'])
    if role == 'confirmation' and not manifest.get('sealed'):
        raise ValueError('Confirmation inputs must be sealed before evaluation')
    for filename,expected in manifest['files'].items():
        if filename != 'trajectories.npz' or sha256_file(path.parent / filename) != expected:
            raise ValueError('Dataset file/hash mismatch')
    if set(manifest['files']) != {'trajectories.npz'}:
        raise ValueError('Missing corpus file hash')
    with np.load(path.parent / 'trajectories.npz',allow_pickle=False) as archive:
        arrays = {key:archive[key].copy() for key in archive.files}
    validate_arrays(arrays,manifest)
    append_event(Path(audit_path) if audit_path is not None else root / 'evidence/data_access.jsonl',
          {'event':'dataset_open','manifest':str(path.relative_to(root)), 'source_root':str(root),
           'manifest_sha256':sha256_file(path),'phase':phase,'role':role,'purpose':purpose,'utc':utc_now()})
    return arrays


def fit_normalization(train_manifest: Path) -> dict:
    if read_json(train_manifest).get('role') != 'train':
        raise ValueError('Normalization requires train labels only')
    arrays = load_corpus(train_manifest,phase='exploratory',role='train',purpose='training')
    states,h = arrays['state'],arrays['duration_ns'][:,:,None]/1e9
    q_residual = states[:,1:,:2]-states[:,:-1,:2]-h*states[:,:-1,2:4]
    v_residual = states[:,1:,2:4]-states[:,:-1,2:4]
    result = {'schema_version':1,'protocol_version':'PTCO-1.0',
              'scale_q':np.maximum(np.sqrt(np.mean(q_residual**2,axis=(0,1))),1e-4).tolist(),
              'scale_v':np.maximum(np.sqrt(np.mean(v_residual**2,axis=(0,1))),1e-3).tolist(),
              'velocity_input_rms':np.maximum(np.sqrt(np.mean(states[:,:-1,2:4]**2,axis=(0,1))),1.).tolist(),
              'training_manifest_sha256':sha256_file(train_manifest),'transition_count':int(h.size),
              'training_manifest':str(Path(train_manifest).resolve().relative_to(project_root(train_manifest)))}
    atomic_json(Path(train_manifest).parent.parent / 'normalization.json',result)
    return result
