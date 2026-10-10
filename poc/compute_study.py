"""Targeted, additive compute-matched NODE study; heavy work on local Data volume."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import resource
import time

for _var in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_var]='1'
import numpy as np
import torch
from .io import atomic_bytes, atomic_json, append_event, read_json, sha256_file, job_key, job_lock, safe_path, utc_now

PROTOCOL='PTCO-COMPUTE-0.1'
FREEZE='config/compute_study_freeze.json'
ALLOCATION='config/compute_study_allocation.json'
PREFIX='reports/compute_study'
SPEC='docs/superpowers/specs/2026-10-09-compute-matched-node.md'
MODELS=('affine_flow','node_quadrature','node_structured')
CONFIGS=tuple((name,step) for name in MODELS for step in (5_000_000,40_000_000))
STORE=Path.home() / 'physical-time-consistency-artifacts/study/compute-matched-node'


class StudyStore:
    """Explicit second-volume store. Original project path validation is untouched."""
    def __init__(self, base, sources, expected):
        self.base=Path(base).resolve()
        self.sources={k:Path(v).resolve() for k,v in sources.items()}
        self.hashes={k:sha256_file(v) if expected[k] is None else expected[k] for k,v in self.sources.items()}

    def check_inputs(self):
        if any(sha256_file(path)!=self.hashes[k] for k,path in self.sources.items()):
            raise ValueError('Changed study source/input binding')

    def read(self, receipt):
        self.check_inputs()
        folder=safe_path(self.base,receipt['path'])
        manifest=read_json(folder/'manifest.json')
        if (sha256_file(folder/'manifest.json')!=receipt['manifest_sha256']
                or read_json(folder/'COMPLETED')['manifest_sha256']!=receipt['manifest_sha256']
                or manifest['inputs']!=self.hashes or manifest['job_key']!=receipt['job_key']):
            raise ValueError('Incomplete/stale study receipt')
        for filename,digest in manifest['files'].items():
            if sha256_file(safe_path(folder,filename))!=digest:
                raise ValueError('Study artifact hash mismatch: '+filename)
        return folder,manifest['summary']

    def produce(self, stage, request, producer):
        self.check_inputs()
        identity=dict(protocol=PROTOCOL,stage=stage,request=request,inputs=self.hashes)
        key=job_key(identity); final=self.base/'jobs'/key
        with job_lock(key,self.base):
            if (final/'COMPLETED').exists():
                receipt=dict(job_key=key,path=str(final.relative_to(self.base)),manifest_sha256=sha256_file(final/'manifest.json'))
                self.read(receipt);return receipt
            work=self.base/'partial'/key; work.mkdir(parents=True,exist_ok=True)
            append_event(work/'attempts.jsonl',dict(event='start',utc=utc_now(),pid=os.getpid(),request=request))
            wall,cpu=time.perf_counter(),time.process_time()
            try:
                summary=producer(work)
            except BaseException as error:
                append_event(work/'attempts.jsonl',dict(event='interrupted',utc=utc_now(),error=repr(error),
                    wall_seconds=time.perf_counter()-wall,cpu_seconds=time.process_time()-cpu))
                raise
            consumed=dict(wall_seconds=time.perf_counter()-wall,cpu_seconds=time.process_time()-cpu,
                          peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if os.uname().sysname=='Darwin' else 1024))
            append_event(work/'attempts.jsonl',dict(event='complete',utc=utc_now(),**consumed))
            manifest=dict(protocol_id=PROTOCOL,job_key=key,identity=identity,inputs=self.hashes,summary=summary,
                resources=consumed,completed_utc=utc_now(),files={str(p.relative_to(work)):sha256_file(p)
                    for p in sorted(work.rglob('*')) if p.is_file() and p.name not in {'manifest.json','COMPLETED'}})
            atomic_json(work/'manifest.json',manifest)
            atomic_json(work/'COMPLETED',dict(manifest_sha256=sha256_file(work/'manifest.json')))
            final.parent.mkdir(parents=True,exist_ok=True);work.rename(final)
            receipt=dict(job_key=key,path=str(final.relative_to(self.base)),manifest_sha256=sha256_file(final/'manifest.json'))
            self.read(receipt);return receipt


def closest_budget(timings, target):
    if (not timings or not np.isfinite(target) or target<=0 or any(type(k) is not int or k<16 or k%16
            or not np.isfinite(v) or v<=0 for k,v in timings.items())):
        raise ValueError('Positive measured timings and sixteen-candidate increments required')
    count=min(timings,key=lambda k:(abs(timings[k]-target),k))
    return dict(candidate_count=count,iterations=4,elite_count=count//8,initial_std=1.,std_floor=.1)


def selected_receipt(root, name, seed, freeze):
    if name=='node_quadrature':return read_json(Path(root)/f'{PREFIX}_fits/seed{seed}.json')
    return freeze['original_selections'][f'{name}:{seed}']


def prepare(root):
    from .affine_precision import frozen
    root=Path(root).resolve()
    if (root/FREEZE).exists():return context(root)[0]
    prior=frozen(root)
    names=set(prior['source_hashes'])|set(prior['input_hashes'])|{SPEC,'poc/quadrature_node.py',
        'poc/stateless_mpc.py','poc/compute_study.py','poc/compute_study_runs.py','config/authorization.json'}
    hashes={name:sha256_file(root/name) for name in sorted(names)}
    value=dict(protocol_id=PROTOCOL,created_utc=utc_now(),seeds=prior['seeds'],configurations=[list(c) for c in CONFIGS],
        original_selections=prior['selections'],store_root=str(STORE),source_hashes=hashes,episodes_per_seed=10,
        namespace=[20261009,3],epoch_block=100,learning_rates=[.001,.0003],loss_mode='shared',
        old_evaluation_freeze='config/affine_precision_freeze_001.json',probe_count=512,
        physics_step_ns=250000,finer_physics_step_ns=125000,comparison_family_size=8,
        scope='Conditional fixed training corpus; new comparator and independent probe/task namespace')
    atomic_json(root/FREEZE,value);return value


def context(root):
    root=Path(root).resolve();f=read_json(root/FREEZE)
    if f['protocol_id']!=PROTOCOL or f['configurations']!=[list(c) for c in CONFIGS] or f['namespace']!=[20261009,3]:
        raise ValueError('Invalid compute study design')
    hashes={**f['source_hashes'],FREEZE:sha256_file(root/FREEZE)}
    store=StudyStore(f['store_root'],{k:str(root/k) for k in hashes},hashes)
    store.check_inputs();return f,store


def build(scales,seed):
    from .quadrature_node import QuadratureNODE
    torch.manual_seed(seed);return QuadratureNODE('lagged',scales)


def fit(root,seed,replica=False):
    from .affine_fresh import data_inputs
    from .affine_experiment import train_block
    from .train import TrainingSession,array_scales,canonical_tensor_hash
    f,store=context(root)
    if seed not in f['seeds']:raise ValueError('Unknown training cluster')
    entries,arrays=data_inputs(root);scales=array_scales(arrays['train'])
    def produce(folder):
        chosen=None;result=None
        for rate in f['learning_rates']:
            out=folder/('lr'+str(rate));out.mkdir(exist_ok=True)
            session=TrainingSession(build(scales,seed),arrays['train'],scales,seed,'shared',learning_rate=rate)
            if (out/'resume.complete.json').exists() or (out/'resume.prev.complete.json').exists():
                session.restore_latest(out/'resume.pt')
            end=max(100,((session.epoch+99)//100)*100)
            while True:
                result=train_block(session,arrays['validation'],out,end)
                if result['qualified']:chosen=out;break
                if result['convergence']['stabilized']:break
                end+=100
            if chosen is not None:break
        if chosen is None:raise ValueError('Nominal optimizer qualification unresolved; diagnostic required')
        if result['valid_target_count']!=result['epochs']*32768:raise ValueError('Dataset exposure mismatch')
        for name in ('weights.pt','selection.json','validation.csv','training.csv'):
            atomic_bytes(folder/name,(chosen/name).read_bytes())
        atomic_json(folder/'normalization.json',dict(scale_q=scales.q.tolist(),scale_v=scales.v.tolist(),velocity_input_rms=scales.v_input.tolist(),train=entries['train']))
        return dict(model_id='node_quadrature',seed=seed,qualified=True,qualification=result,
                    parameter_count=sum(p.numel() for p in session.model.parameters()),
                    canonical_tensor_sha256=canonical_tensor_hash(torch.load(folder/'weights.pt',weights_only=True)),
                    checkpoint_sha256=sha256_file(folder/'weights.pt'))
    print(f'Quadrature nominal fit seed {seed} replica={replica}',flush=True)
    receipt=store.produce('training',dict(seed=seed,replica=replica,data=entries),produce)
    sub='replicas' if replica else 'fits'
    atomic_json(Path(root)/f'{PREFIX}_{sub}/seed{seed}.json',receipt);return receipt


def load_model(root,name,seed):
    from .affine_confirmation import load_model as load_original
    from .models.common import Scales
    from .train import canonical_tensor_hash
    f,store=context(root)
    if name!='node_quadrature':return load_original(Path(root),name,seed,dict(selections=f['original_selections']))
    folder,summary=store.read(read_json(Path(root)/f'{PREFIX}_fits/seed{seed}.json'))
    if summary['model_id']!=name or summary['seed']!=seed or not summary['qualified']:
        raise ValueError('Selected comparator mismatch')
    norm=read_json(folder/'normalization.json')
    scales=Scales(*(torch.tensor(norm[k],dtype=torch.float32) for k in ('scale_q','scale_v','velocity_input_rms')))
    model=build(scales,seed);weights=torch.load(folder/'weights.pt',weights_only=True,map_location='cpu')
    if canonical_tensor_hash(weights)!=summary['canonical_tensor_sha256']:raise ValueError('Comparator tensor mismatch')
    model.load_state_dict(weights);return model.double().eval()


def calibrate(root):
    from .stateless_mpc import search
    from .compute_study_runs import task_conditions,normal_bank
    f,store=context(root); destination=Path(root)/ALLOCATION
    if destination.exists():
        result=read_json(destination)
        if result['freeze_sha256']!=sha256_file(Path(root)/FREEZE):raise ValueError('Stale allocation')
        store.read(result['receipt']);return result
    seed=f['seeds'][0]; inputs=task_conditions(9000,0);models={n:load_model(root,n,seed) for n in MODELS}
    def produce(folder):
        readings={};target=None
        def timing(name,step,count):
            key=f'{name}:{step}:{count}'
            if key in readings:return float(np.median(readings[key]))
            samples=[];settings=closest_budget({count:1.},1.)
            for repeat in range(4):
                start=time.perf_counter_ns()
                search(models[name],torch.tensor(inputs['initial_state']),torch.tensor(inputs['target']),None,None,
                       normal_bank(9000,repeat,count,count)[0],settings,(step,)*(40_000_000//step),40_000_000)
                elapsed=(time.perf_counter_ns()-start)/1e9
                if repeat:samples.append(elapsed)
            readings[key]=samples;print(f'Calibration {key}: {np.median(samples):.4f}s',flush=True)
            return float(np.median(samples))
        target=timing('node_structured',5_000_000,128)
        allocations={};details={}
        for name,step in CONFIGS:
            counts={};count=16
            while True:
                counts[count]=timing(name,step,count)
                if counts[count]>=target:break
                count*=2
            low=max((k for k,v in counts.items() if v<target),default=0);high=count
            while high-low>16:
                mid=((low+high)//32)*16
                if mid<=low:mid=low+16
                counts[mid]=timing(name,step,mid)
                if counts[mid]<target:low=mid
                else:high=mid
            cfg=closest_budget(counts,target);key=f'{name}:{step}'
            allocations[key]=cfg;details[key]=dict(seconds=counts[cfg['candidate_count']],ratio=counts[cfg['candidate_count']]/target)
            if abs(details[key]['ratio']-1)>.1:
                raise ValueError('Timing allocation exceeds ten-percent calibration mismatch; repeat measurement required')
        validation={};keys=list(allocations)
        for repeat in range(3):
            order=keys[repeat:]+keys[:repeat]
            for key in order:
                name,step_text=key.split(':');step=int(step_text);cfg=allocations[key]
                start=time.perf_counter_ns()
                search(models[name],torch.tensor(inputs['initial_state']),torch.tensor(inputs['target']),None,None,
                       normal_bank(9000,100+repeat,cfg['candidate_count'],cfg['candidate_count'])[0],cfg,
                       (step,)*(40_000_000//step),40_000_000)
                validation.setdefault(key,[]).append((time.perf_counter_ns()-start)/1e9)
        actual_target=float(np.median(validation['node_structured:5000000']))
        for key,values in validation.items():
            ratio=float(np.median(values))/actual_target
            details[key].update(rotated_seconds=float(np.median(values)),rotated_ratio=ratio)
            if abs(ratio-1)>.1:raise ValueError('Rotated validation outside ten-percent allocation matching tolerance')
        atomic_json(folder/'timings.json',dict(samples=readings,rotated_samples=validation,target_seconds=target,selected=details))
        return dict(target_seconds=target,settings=allocations,details=details,
                    maximum_candidates=max(s['candidate_count'] for s in allocations.values()))
    selections={name:selected_receipt(root,name,seed,f) for name in MODELS}
    receipt=store.produce('timing-calibration',dict(selections=selections),produce);_,summary=store.read(receipt)
    result=dict(protocol_id=PROTOCOL,freeze_sha256=sha256_file(Path(root)/FREEZE),receipt=receipt,**summary)
    atomic_json(destination,result);return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','train','calibrate','measure','controls','reproduce','status'))
    parser.add_argument('--root',type=Path,default=Path.cwd());parser.add_argument('--seed',type=int)
    args=parser.parse_args(argv);root=args.root.resolve()
    from .execution import cpu_defaults
    cpu_defaults(2026100800)
    if args.command=='prepare':print(dict(protocol=prepare(root)['protocol_id'],state='specified'))
    elif args.command=='train':
        f,_=context(root)
        for seed in f['seeds'] if args.seed is None else (args.seed,):fit(root,seed)
    elif args.command=='calibrate':print(calibrate(root))
    elif args.command in {'measure','controls','reproduce'}:
        from . import compute_study_runs
        getattr(compute_study_runs,args.command)(root)
    else:
        f,s=context(root);print(dict(protocol=f['protocol_id'],fits=len(list((root/(PREFIX+'_fits')).glob('*.json'))),
                                     free_store_bytes=os.statvfs(s.base).f_bavail*os.statvfs(s.base).f_frsize))


if __name__=='__main__':main()
