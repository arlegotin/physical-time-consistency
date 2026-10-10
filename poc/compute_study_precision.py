"""Additive independent-seed continuation; the original freeze stays immutable."""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import numpy as np
import torch
from .compute_study import (CONFIGS,MODELS,ALLOCATION,StudyStore,context as initial_context,
                           build as build_quadrature)
from .io import atomic_bytes,atomic_json,read_json,sha256_file,utc_now

REQUEST='config/compute_study_precision_002.json'
FREEZE='config/compute_study_precision_002_freeze.json'
PREFIX='reports/compute_study_precision_002'


def validate_request(r):
    old,new=r['prior_seeds'],r['new_seeds']
    if (r['look']!=2 or not old or len(new)<5 or any(type(s) is not int for s in old+new)
            or old!=sorted(set(old)) or new!=list(range(max(old)+1,max(old)+1+len(new)))
            or r['episodes_per_seed']!=10 or not 0<=r['within_episode_fraction']<=.7
            or not math.isfinite(r['variance']) or r['variance']<=0
            or not math.isfinite(r['mean_contrast']) or r['planning_sd_margin']!=1.25
            or not r['reason'] or any(len(r[k])!=64 for k in ('prior_result_sha256','allocation_sha256'))
            or any(not math.isfinite(v) or v<=0 for v in r['resource_projection'].values())):
        raise ValueError('Prospective next independent seed block and measured precision justification required')
    return old+new


def build(name,scales,seed):
    from .affine_fresh import build_fresh_model
    if name=='node_quadrature':return build_quadrature(scales,seed)
    return build_fresh_model(name,scales,seed)


def joined_cells(old,new):
    rows=list(old)+list(new)
    keys=[(r['model_id'],r['step_ns'],r['seed'],r['episode_id']) for r in rows]
    if len(keys)!=len(set(keys)):raise ValueError('Duplicate precision observations')
    return rows


def prepare(root):
    root=Path(root);f,old_store=initial_context(root);r=read_json(root/REQUEST);validate_request(r)
    if (r['prior_seeds']!=f['seeds'] or sha256_file(root/'reports/provenance/compute_study_look_001_result.json')!=r['prior_result_sha256']
            or sha256_file(root/ALLOCATION)!=r['allocation_sha256']):raise ValueError('Changed precision parent evidence')
    if (root/FREEZE).exists():return context(root)[0]
    contract=r['contract_path']
    hashes={**old_store.hashes,REQUEST:sha256_file(root/REQUEST),'poc/compute_study_precision.py':sha256_file(Path(__file__)),
            contract:sha256_file(root/contract)}
    value=dict(request=r,source_hashes=hashes,store_root=str(old_store.base/'precision-002'),created_utc=utc_now())
    atomic_json(root/FREEZE,value);return value


def context(root):
    root=Path(root);initial_context(root);f=read_json(root/FREEZE);r=read_json(root/REQUEST)
    validate_request(r)
    if f['request']!=r:raise ValueError('Changed precision request')
    hashes={**f['source_hashes'],FREEZE:sha256_file(root/FREEZE)}
    store=StudyStore(f['store_root'],{k:root/k for k in hashes},hashes);store.check_inputs();return f,store


def fit_path(root,name,seed,replica=False):
    return Path(root)/f'{PREFIX}_fits/{name}__seed{seed}{"__replica" if replica else ""}.json'


def fit(root,name,seed,replica=False):
    from .affine_fresh import data_inputs
    from .affine_experiment import train_block
    from .train import TrainingSession,array_scales,canonical_tensor_hash
    f,store=context(root)
    if name not in MODELS or seed not in f['request']['new_seeds']:raise ValueError('Unregistered precision fit')
    entries,arrays=data_inputs(Path(root));scales=array_scales(arrays['train'])
    def produce(folder):
        chosen=None
        for rate in (.001,.0003):
            out=folder/('lr'+str(rate));out.mkdir(exist_ok=True)
            session=TrainingSession(build(name,scales,seed),arrays['train'],scales,seed,'shared',learning_rate=rate)
            if (out/'resume.complete.json').exists() or (out/'resume.prev.complete.json').exists():session.restore_latest(out/'resume.pt')
            end=max(100,((session.epoch+99)//100)*100)
            while True:
                result=train_block(session,arrays['validation'],out,end)
                if result['qualified']:chosen=out;break
                if result['convergence']['stabilized']:break
                end+=100
            if chosen is not None:break
        if chosen is None:raise ValueError('Nominal optimizer qualification unresolved')
        if result['valid_target_count']!=result['epochs']*32768:raise ValueError('Unmatched training exposure')
        for filename in ('weights.pt','selection.json','validation.csv','training.csv'):atomic_bytes(folder/filename,(chosen/filename).read_bytes())
        atomic_json(folder/'normalization.json',dict(scale_q=scales.q.tolist(),scale_v=scales.v.tolist(),velocity_input_rms=scales.v_input.tolist(),train=entries['train']))
        return dict(model_id=name,seed=seed,qualified=True,qualification=result,
            parameter_count=sum(p.numel() for p in session.model.parameters()),
            canonical_tensor_sha256=canonical_tensor_hash(torch.load(folder/'weights.pt',weights_only=True)),
            checkpoint_sha256=sha256_file(folder/'weights.pt'))
    print(f'Precision fit {name} seed{seed} replica={replica}',flush=True)
    receipt=store.produce('training',dict(seed=seed,model_id=name,replica=replica,data=entries),produce)
    atomic_json(fit_path(root,name,seed,replica),receipt);return receipt


def load_model(root,name,seed):
    from .models.common import Scales
    from .train import canonical_tensor_hash
    _,store=context(root);receipt=read_json(fit_path(root,name,seed));folder,summary=store.read(receipt)
    if summary['model_id']!=name or summary['seed']!=seed or not summary['qualified']:raise ValueError('Precision checkpoint mismatch')
    norm=read_json(folder/'normalization.json');scales=Scales(*(torch.tensor(norm[k],dtype=torch.float32) for k in ('scale_q','scale_v','velocity_input_rms')))
    model=build(name,scales,seed);weights=torch.load(folder/'weights.pt',weights_only=True,map_location='cpu')
    if canonical_tensor_hash(weights)!=summary['canonical_tensor_sha256']:raise ValueError('Precision tensor mismatch')
    model.load_state_dict(weights);return model.double().eval()


def controls(root,kind='primary'):
    from .compute_study_runs import execute_episode,npz
    f,store=context(root);_,old_store=initial_context(root);allocation=read_json(Path(root)/ALLOCATION);old_store.read(allocation['receipt'])
    path=Path(root)/f'{PREFIX}_{kind}.json';ledger=read_json(path) if path.exists() else dict(kind=kind,completed=[],state='running')
    for seed in f['request']['new_seeds']:
        models={name:load_model(root,name,seed) for name in MODELS}
        for episode in range(10 if kind=='primary' else 1):
            for name,step in CONFIGS:
                settings=allocation['settings'][f'{name}:{step}'];receipt=read_json(fit_path(root,name,seed))
                def produce(folder):
                    raw,summary=execute_episode(models[name],settings,seed,episode,step,125000 if kind=='finer' else 250000,
                        allocation['maximum_candidates'],Path(root)/'assets')
                    npz(folder/'episode.npz',**raw);summary.update(model_id=name,kind=kind);return summary
                record=store.produce('control',dict(model_id=name,seed=seed,episode_id=episode,step_ns=step,kind=kind,
                    settings=settings,maximum=allocation['maximum_candidates'],allocation_sha256=sha256_file(Path(root)/ALLOCATION),selected_model=receipt),produce)
                if record not in ledger['completed']:ledger['completed'].append(record)
                ledger.update(completed_count=len(ledger['completed']),updated_utc=utc_now());atomic_json(path,ledger)
                print(f'Precision {kind} {name}/{step//1000000}ms seed{seed} task{episode} [{len(ledger["completed"])}]',flush=True)
    ledger['state']='complete';atomic_json(path,ledger)


def measure(root):
    from .affine_confirmation import _profile
    from .compute_study_runs import npz
    f,store=context(root);_,old_store=initial_context(root);receipt=read_json(Path(root)/'reports/compute_study_references.json');reference_folder,_=old_store.read(receipt)
    with np.load(reference_folder/'references.npz',allow_pickle=False) as data:ref={k:data[k].copy() for k in data.files}
    fits={f'{name}:{seed}':read_json(fit_path(root,name,seed)) for seed in f['request']['new_seeds'] for name in MODELS}
    def produce(folder):
        profiles=[]
        for seed in f['request']['new_seeds']:
            for name in MODELS if seed%2==0 else MODELS[::-1]:
                model=load_model(root,name,seed);state=torch.tensor(ref['states']);command=torch.tensor(ref['commands'])
                raw={key:np.empty((512,3,6)) for key in ('model_zero','model_command','model_composed')}
                with torch.no_grad():
                    for hi,h_ns in enumerate(ref['horizons_ns']):
                        h=float(h_ns)/1e9;raw['model_zero'][:,hi]=model.predict(state,torch.zeros_like(command),h).next_state.numpy()
                        raw['model_command'][:,hi]=model.predict(state,command,h).next_state.numpy()
                        mid=model.predict(state,command,h/2).next_state;raw['model_composed'][:,hi]=model.predict(mid,command,h/2).next_state.numpy()
                if any(not np.isfinite(v).all() for v in raw.values()):raise ValueError('Nonfinite precision prediction')
                npz(folder/f'{name}__seed{seed}.npz',**raw)
                rows=_profile(model,state[:1],command[:1],name,seed)
                for row in rows:row['neural_rows']=row['batch_size']*(4 if name=='node_structured' else 2)
                profiles.extend(rows)
        atomic_json(folder/'profiles.json',dict(rows=profiles));return dict(reference=receipt,profiles=profiles,seeds=f['request']['new_seeds'],models=list(MODELS))
    record=store.produce('predictions',dict(reference=receipt,fits=fits),produce);atomic_json(Path(root)/f'{PREFIX}_measurement.json',record)


def records(root,kind):
    _,store=context(root);ledger=read_json(Path(root)/f'{PREFIX}_{kind}.json')
    if ledger['state']!='complete':raise ValueError('Incomplete precision cohort')
    rows=[];paths=[]
    for receipt in ledger['completed']:
        folder,summary=store.read(receipt);rows.append(summary);paths.append(folder)
    return rows,paths,ledger


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','train','measure','controls','reproduce'))
    parser.add_argument('--root',type=Path,default=Path.cwd());args=parser.parse_args(argv)
    from .execution import cpu_defaults
    cpu_defaults(2026100811)
    if args.command=='prepare':prepare(args.root)
    elif args.command=='train':
        f,_=context(args.root)
        for seed in f['request']['new_seeds']:
            for name in MODELS:fit(args.root,name,seed)
    elif args.command=='measure':measure(args.root)
    elif args.command=='controls':controls(args.root)
    else:
        controls(args.root,'replay');controls(args.root,'finer');f,_=context(args.root)
        fit(args.root,'node_quadrature',f['request']['new_seeds'][-1],True)


if __name__=='__main__':main()
