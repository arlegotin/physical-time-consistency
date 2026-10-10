"""Matched endpoint training, physical-time rollouts and exact optimizer resume."""
from __future__ import annotations

import hashlib
import io
import os
import random
import time
from pathlib import Path
for _thread_variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_thread_variable] = '1'
import numpy as np
import torch

from .history import HistoryBatch,HistoryToken
from .io import atomic_bytes,atomic_json,canonical_json,read_json,sha256_file
from .models.common import Scales
from .models.sequence import Recurrent


def endpoint_loss(predicted,target,scales,*,reduction='mean'):
    q = ((predicted[:,:2]-target[:,:2])/scales.q.to(predicted))**2
    v = ((predicted[:,2:4]-target[:,2:4])/scales.v.to(predicted))**2
    values = (q.sum(dim=1)+v.sum(dim=1))/4
    if reduction == 'none':
        return values
    if reduction != 'mean':
        raise ValueError('Unsupported endpoint loss reduction')
    return values.mean()


def optimizer_for(parameters,learning_rate=.001):
    if learning_rate not in {.001,.0003}:
        raise ValueError('Only the preregistered nominal learning rates are permitted')
    return torch.optim.AdamW(parameters,lr=learning_rate,weight_decay=1e-6,betas=(.9,.999),eps=1e-8,
                            amsgrad=False,foreach=False,fused=False)


def epoch_starts(arrays,rng):
    count,steps = arrays['command'].shape[:2]
    windows = [np.array([(trajectory,index) for index in range(start,min(start+8,steps))],dtype=np.int64)
               for trajectory in range(count) for start in range(0,steps,8)]
    return [windows[i] for i in rng.permutation(len(windows))]


def histories_for(arrays,starts):
    sequences,times = [],[]
    for trajectory,start in starts:
        query_time = int(arrays['time_ns'][trajectory,start])
        sequences.append(tuple(HistoryToken(int(arrays['time_ns'][trajectory,index]),arrays['state'][trajectory,index],
                           arrays['command'][trajectory,index],int(arrays['duration_ns'][trajectory,index]),'observed',
                           f't{int(trajectory)}-s{index}') for index in range(start)
                           if query_time-320_000_000 <= arrays['time_ns'][trajectory,index] < query_time))
        times.append(query_time)
    return HistoryBatch(tuple(sequences),np.array(times,dtype=np.int64))


def rollout_loss(model,window,history=None):
    state = window['start_state']
    losses,anchors = [],[]
    tokens = 0
    scales = window['scales']
    context = history.fork() if history is not None else None
    for step in range(4):
        command,h = window['commands'][:,step],window['durations_s'][:,step]
        prior = state
        if context is not None:
            context = context.at_time(window['times_ns'][:,step])
        result = model.predict(state,command,h,history=context)
        if any(status != 'ok' for status in result.status):
            raise ValueError('Failed imagined rollout during supervised training')
        state = result.next_state
        losses.append(endpoint_loss(state,window['targets'][:,step],scales,reduction='none'))
        if context is not None:
            context = context.at_time(window['times_ns'][:,step+1])
            additions = tuple(HistoryToken(int(window['times_ns'][index,step]),prior[index],
                 command[index].detach().cpu().numpy(),int(window['times_ns'][index,step+1]-window['times_ns'][index,step]),'imagined',
                 f'imagined-{int(window["times_ns"][index,step])}-{step}') for index in range(len(state)))
            context = context.append_completed(additions)
            tokens += len(state)
        anchors.extend(window['times_ns'][:,step+1].tolist())
    return torch.stack(losses).mean(),tokens,anchors


def batch_objective(model,arrays,starts,scales,loss_mode='one_step'):
    if loss_mode not in {'one_step','shared'}:
        raise ValueError('Unknown supervised loss mode')
    starts = np.asarray(starts,dtype=np.int64)
    if starts.ndim != 2 or starts.shape[1] != 2 or len(starts) == 0:
        raise ValueError('A nonempty explicit trajectory/start batch is required')
    parameter = next(model.parameters())
    def tensor(values):
        return torch.as_tensor(values,dtype=parameter.dtype,device=parameter.device)
    trajectories,indices = starts.T
    state = tensor(arrays['state'][trajectories,indices])
    command = tensor(arrays['command'][trajectories,indices])
    h = tensor(arrays['duration_ns'][trajectories,indices,None]/1e9)
    uses_history = isinstance(model,Recurrent) or getattr(model,'uses_history',False)
    history = histories_for(arrays,starts) if uses_history else None
    output = model.predict(state,command,h,history=history)
    if any(status != 'ok' for status in output.status):
        raise ValueError('Failed one-step model output during training')
    one = endpoint_loss(output.next_state,tensor(arrays['state'][trajectories,indices+1]),scales)
    valid = indices+4 <= arrays['command'].shape[1]
    four = one.new_zeros(())
    token_count,anchors = 0,[]
    if loss_mode == 'shared' and np.any(valid):
        ti,si = trajectories[valid],indices[valid]
        offsets = np.arange(4)
        selected = starts[valid]
        window = {'start_state':tensor(arrays['state'][ti,si]),
                  'commands':tensor(arrays['command'][ti[:,None],si[:,None]+offsets]),
                  'durations_s':tensor(arrays['duration_ns'][ti[:,None],si[:,None]+offsets,None]/1e9),
                  'targets':tensor(arrays['state'][ti[:,None],si[:,None]+offsets+1]),
                  'times_ns':arrays['time_ns'][ti[:,None],si[:,None]+np.arange(5)],'scales':scales}
        four,token_count,anchors = rollout_loss(model,window,histories_for(arrays,selected) if uses_history else None)
    return {'loss':one+.25*four if loss_mode == 'shared' else one,'one_loss':one,'four_loss':four,
            'one_count':len(starts),'four_count':int(valid.sum()) if loss_mode == 'shared' else 0,
            'imagined_history_anchors':anchors,'imagined_tokens':token_count}


def select_checkpoint(rows):
    if not rows:
        raise ValueError('No nominal validation checkpoints')
    selected = rows[0]
    for row in rows[1:]:
        if not np.isfinite(row['objective']):
            raise ValueError('Nonfinite nominal validation objective')
        if row['objective'] < selected['objective']-1e-8*abs(selected['objective']):
            selected = row
    return dict(selected)


def nominal_convergence(rows,epoch):
    by_epoch = {row['epoch']:row['objective'] for row in rows}
    recent,older = [epoch-i for i in (20,15,10,5,0)],[epoch-i for i in (45,40,35,30,25)]
    if epoch < 100 or any(i not in by_epoch for i in recent+older):
        return {'stabilized':False,'reason':'Incomplete moving convergence window'}
    previous,current = np.median([by_epoch[i] for i in older]),np.median([by_epoch[i] for i in recent])
    if previous <= 0:
        return {'stabilized':bool(previous == current),'relative_improvement':None}
    improvement = float((previous-current)/previous)
    return {'stabilized':improvement < .01,'relative_improvement':improvement}


def canonical_tensor_hash(state_dict):
    digest = hashlib.sha256()
    for name,value in sorted(state_dict.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(canonical_json({'name':name,'dtype':str(tensor.dtype),'shape':list(tensor.shape)}))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _tuples(value):
    return tuple(_tuples(item) for item in value) if isinstance(value,list) else value


class TrainingSession:
    """An optimizer-update boundary is the only valid resumable training state."""
    def __init__(self,model,arrays,scales,seed,loss_mode='one_step',effective_batch=256,learning_rate=.001):
        if type(effective_batch) is not int or effective_batch <= 0 or (effective_batch%8 and arrays['command'].shape[1]>1):
            raise ValueError('Effective batches must contain complete eight-start windows')
        self.model,self.arrays,self.scales = model,arrays,scales
        self.loss_mode,self.effective_batch = loss_mode,effective_batch
        self.optimizer = optimizer_for(model.parameters(),learning_rate)
        self.recipe = {'learning_rate':learning_rate,'loss_mode':loss_mode,'effective_batch':effective_batch,
                       'seed':seed,'model_type':type(model).__name__,
                       'model_options':{key:getattr(model,key) for key in ('actuator','structured','time_scaled','time_blind') if hasattr(model,key)}}
        self.rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([5501,int(seed)])))
        self.epoch,self.window_position,self.valid_target_count,self.updates = 1,0,0,0
        self.windows = epoch_starts(arrays,self.rng)
        self.training_rows,self.update_timing = [],[]
        digest = hashlib.sha256()
        for key,value in sorted(arrays.items()):
            digest.update(canonical_json({'name':key,'dtype':str(value.dtype),'shape':list(value.shape)}))
            digest.update(np.ascontiguousarray(value).tobytes())
        self.data_hash = digest.hexdigest()
        self.scale_hash = canonical_tensor_hash({'q':scales.q,'v':scales.v,'input':scales.v_input})

    def advance_updates(self,count):
        for _ in range(count):
            if self.window_position == len(self.windows):
                self.epoch += 1
                self.windows,self.window_position = epoch_starts(self.arrays,self.rng),0
            indices,valid = [],0
            while self.window_position < len(self.windows) and valid < self.effective_batch:
                window = self.windows[self.window_position]
                indices.append(window)
                valid += len(window)
                self.window_position += 1
            starts = np.concatenate(indices)
            if valid > self.effective_batch:
                raise ValueError('Window batching exceeds effective target count')
            began = time.perf_counter_ns()
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            result = batch_objective(self.model,self.arrays,starts,self.scales,self.loss_mode)
            if not torch.isfinite(result['loss']):
                raise ValueError('Nonfinite supervised training loss')
            result['loss'].backward()
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.,error_if_nonfinite=True,foreach=False)
            self.optimizer.step()
            self.valid_target_count += valid
            self.updates += 1
            self.training_rows.append({'epoch':self.epoch,'update':self.updates,'loss':float(result['loss'].detach()),
                  'one_step':float(result['one_loss'].detach()),'four_step':float(result['four_loss'].detach()),
                  'one_count':valid,'four_count':result['four_count'],'valid_target_count':self.valid_target_count,
                  'gradient_norm':float(norm),'clipped':bool(norm>1)})
            self.update_timing.append({'update':self.updates,'elapsed_seconds':(time.perf_counter_ns()-began)/1e9})

    def advance_epoch(self,path):
        """Persist at a completed optimizer boundary; uncommitted work may be replayed."""
        while self.window_position < len(self.windows):
            self.advance_updates(1)
        self.save(path)

    def save(self,path):
        path = Path(path)
        marker_path = path.with_suffix('.complete.json')
        if marker_path.exists():
            marker = read_json(marker_path)
            if marker['checkpoint_sha256'] == sha256_file(path) and marker['rng_sha256'] == sha256_file(path.with_suffix('.rng.json')):
                previous = path.with_name(path.stem+'.prev.pt')
                atomic_bytes(previous,path.read_bytes())
                atomic_bytes(previous.with_suffix('.rng.json'),path.with_suffix('.rng.json').read_bytes())
                atomic_json(previous.with_suffix('.complete.json'),marker)
        lengths = np.asarray([len(window) for window in self.windows],dtype=np.int64)
        flat_windows = np.concatenate(self.windows)
        offsets = np.r_[0,np.cumsum(lengths)].astype(np.int64)
        checkpoint = {'checkpoint_format_version':2,'weights':self.model.state_dict(),'optimizer':self.optimizer.state_dict(),
                  'epoch':self.epoch,'window_position':self.window_position,'valid_target_count':self.valid_target_count,
                  'updates':self.updates,'windows':torch.from_numpy(flat_windows),
                  'window_offsets':torch.from_numpy(offsets),
                  'training_rows':self.training_rows,'update_timing':self.update_timing,
                  'data_hash':self.data_hash,'scale_hash':self.scale_hash,'loss_mode':self.loss_mode,
                  'effective_batch':self.effective_batch,'recipe':self.recipe,'torch_rng':torch.get_rng_state()}
        buffer = io.BytesIO()
        torch.save(checkpoint,buffer)
        atomic_bytes(path,buffer.getvalue())
        atomic_json(Path(path).with_suffix('.rng.json'),{'numpy':self.rng.bit_generator.state,
                     'python':random.getstate(),'torch':torch.get_rng_state().tolist()})
        atomic_json(Path(path).with_suffix('.complete.json'),{'checkpoint_sha256':sha256_file(path),
                     'rng_sha256':sha256_file(Path(path).with_suffix('.rng.json')),'update':self.updates})

    def restore(self,path):
        marker = read_json(Path(path).with_suffix('.complete.json'))
        if marker['checkpoint_sha256'] != sha256_file(path) or marker['rng_sha256'] != sha256_file(Path(path).with_suffix('.rng.json')):
            raise ValueError('Incomplete or inconsistent checkpoint hash closure')
        saved = torch.load(path,weights_only=True,map_location='cpu')
        if saved['recipe'] != self.recipe:
            raise ValueError('Changed training recipe is not an exact resumption')
        for key in ('data_hash','scale_hash','loss_mode','effective_batch'):
            if saved[key] != getattr(self,key):
                raise ValueError('Changed training inputs are not an exact resumption: '+key)
        self.model.load_state_dict(saved['weights'])
        self.optimizer.load_state_dict(saved['optimizer'])
        for key in ('epoch','window_position','valid_target_count','updates','training_rows','update_timing'):
            setattr(self,key,saved[key])
        if isinstance(saved['windows'],torch.Tensor):
            flat = saved['windows'].numpy()
            offsets = saved['window_offsets'].numpy()
            if offsets.ndim != 1 or offsets[0] != 0 or offsets[-1] != len(flat) or np.any(np.diff(offsets)<=0):
                raise ValueError('Packed checkpoint window offsets are invalid')
            self.windows = [flat[start:end].copy() for start,end in zip(offsets[:-1],offsets[1:])]
        else:
            self.windows = [window.numpy() for window in saved['windows']]
        rng = read_json(Path(path).with_suffix('.rng.json'))
        self.rng.bit_generator.state = rng['numpy']
        random.setstate(_tuples(rng['python']))
        torch.set_rng_state(saved['torch_rng'])

    def tensor_hash(self):
        return canonical_tensor_hash(self.model.state_dict())

    def restore_latest(self,path):
        path = Path(path)
        try:
            self.restore(path)
            return path
        except (ValueError,OSError):
            previous = path.with_name(path.stem+'.prev.pt')
            self.restore(previous)
            return previous


def analytic_control_arrays(actuator,role):
    from .analytic import transition
    code = 0 if actuator == 'direct' else 1
    if role == 'tiny':
        count,entropy = 16,[8101,code,0]
    elif role in {'train','validation'}:
        count,entropy = (1024 if role == 'train' else 256),[8102,code,int(role == 'validation')]
    else:
        raise ValueError('Unknown analytic plumbing-control role')
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))
    initial = rng.uniform(-1,1,(count,4 if actuator == 'direct' else 6))
    command = rng.uniform(-1,1,(count,2))
    endpoint = np.stack([transition(state,u,.04,actuator) for state,u in zip(initial,command)])
    return {'state':np.stack((initial,endpoint),axis=1),'command':command[:,None,:],
            'duration_ns':np.full((count,1),40_000_000,dtype=np.int64),
            'time_ns':np.tile(np.array([0,40_000_000],dtype=np.int64),(count,1)),
            'trajectory_id':np.arange(count,dtype=np.int64)}


def array_scales(arrays):
    states,h = arrays['state'],arrays['duration_ns'][:,:,None]/1e9
    q = states[:,1:,:2]-states[:,:-1,:2]-h*states[:,:-1,2:4]
    v = states[:,1:,2:4]-states[:,:-1,2:4]
    return Scales(torch.tensor(np.maximum(np.sqrt(np.mean(q*q,axis=(0,1))),1e-4),dtype=torch.float32),
                  torch.tensor(np.maximum(np.sqrt(np.mean(v*v,axis=(0,1))),1e-3),dtype=torch.float32),
                  torch.tensor(np.maximum(np.sqrt(np.mean(states[:,:-1,2:4]**2,axis=(0,1))),1.),dtype=torch.float32))


def validate_nominal(model,arrays,scales,loss_mode,effective_batch=256):
    windows = epoch_starts(arrays,np.random.Generator(np.random.PCG64(0)))
    one_sum,four_sum,one_count,four_count = 0.,0.,0,0
    model.eval()
    with torch.no_grad():
        pending,count = [],0
        for index,window in enumerate(windows):
            pending.append(window)
            count += len(window)
            if count < effective_batch and index+1 < len(windows):
                continue
            result = batch_objective(model,arrays,np.concatenate(pending),scales,loss_mode)
            one_sum += float(result['one_loss'])*result['one_count']
            four_sum += float(result['four_loss'])*result['four_count']
            one_count += result['one_count']
            four_count += result['four_count']
            pending,count = [],0
    one,four = one_sum/one_count,four_sum/four_count if four_count else 0.
    return {'objective':one+(.25*four if loss_mode == 'shared' else 0.),'one_step':one,'four_step':four,
            'one_count':one_count,'four_count':four_count}


def _write_training_csv(path,rows,timing=None):
    import csv
    if not rows:
        return
    elapsed = {row['update']:row['elapsed_seconds'] for row in timing or []}
    records = [{**row,**({'elapsed_seconds':elapsed[row['update']]} if timing is not None else {})} for row in rows]
    with Path(path).open('w',newline='') as output:
        writer = csv.DictWriter(output,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def run_learning_controls(model_id,actuator,root,resume=False):
    from .execution import cpu_defaults,parent
    from .models.factory import build_model
    from .training_jobs import training_job
    cpu_defaults(8100)
    predecessor = parent(root,'G20')
    parameters = {'model_id':model_id,'actuator':actuator,'seed':8100,'tiny_updates':5000,
                  'analytic_epochs':100,'tiny_threshold':.001,'validation_threshold':.05}
    def produce(folder,resume_from=None):
        tiny = analytic_control_arrays(actuator,'tiny')
        tiny_scales = array_scales(tiny)
        tiny_model = build_model(model_id,actuator,tiny_scales,8100)
        session = TrainingSession(tiny_model,tiny,tiny_scales,8100,effective_batch=16)
        starts = np.array([[i,0] for i in range(16)],dtype=np.int64)
        loss = float(batch_objective(tiny_model,tiny,starts,tiny_scales)['loss'].detach())
        while loss >= .001 and session.updates < 5000:
            session.advance_updates(1)
            loss = float(batch_objective(tiny_model,tiny,starts,tiny_scales)['loss'].detach())
        _write_training_csv(folder / 'tiny_training.csv',session.training_rows,session.update_timing)
        np.savez_compressed(folder / 'tiny_arrays.npz',**tiny)
        session.save(folder / 'tiny_resume.pt')
        if loss >= .001:
            raise ValueError(f'Analytic tiny-fit control needs optimizer/implementation diagnosis: {loss}')
        arrays,validation = analytic_control_arrays(actuator,'train'),analytic_control_arrays(actuator,'validation')
        norm = array_scales(arrays)
        model = build_model(model_id,actuator,norm,8100)
        full = TrainingSession(model,arrays,norm,8100)
        validation_rows,best,best_weights = [],None,None
        for epoch in range(1,101):
            while full.epoch == epoch and full.window_position < len(full.windows):
                full.advance_updates(1)
            if epoch%5 == 0:
                scores = {'epoch':epoch,**validate_nominal(model,validation,norm,'one_step')}
                validation_rows.append(scores)
                selected = select_checkpoint(validation_rows)
                if best is None or selected['epoch'] != best['epoch']:
                    best,best_weights = selected,{name:value.detach().clone() for name,value in model.state_dict().items()}
            if epoch < 100:
                full.epoch += 1
                full.windows,full.window_position = epoch_starts(arrays,full.rng),0
        _write_training_csv(folder / 'training.csv',full.training_rows,full.update_timing)
        _write_training_csv(folder / 'validation.csv',validation_rows)
        np.savez_compressed(folder / 'analytic_train.npz',**arrays)
        np.savez_compressed(folder / 'analytic_validation.npz',**validation)
        full.save(folder / 'resume.pt')
        buffer = io.BytesIO()
        torch.save(best_weights,buffer)
        atomic_bytes(folder / 'weights.pt',buffer.getvalue())
        atomic_json(folder / 'selection.json',best)
        if best['one_step'] >= .05:
            raise ValueError(f'Analytic nominal validation control requires diagnosis: {best["one_step"]}')
        checks = [{'check_id':criterion,'status':'pass','reason':'Exact analytic data learned with declared optimizer and nominal selection',
                   'scope':f'{model_id}/{actuator}','tolerance':.001 if criterion == 'tiny_fit' else .05}
                   for criterion in ('tiny_fit','analytic_validation')]
        return checks,{'model_id':model_id,'actuator':actuator,'tiny_loss':loss,'tiny_updates':session.updates,
                       'validation':best,'training_updates':session.updates+full.updates,'missing_cells':[],
                       'robot_result':False,'model_hashes':{'canonical_tensor_sha256':canonical_tensor_hash(best_weights)}}
    return training_job(root,f'learning_control__{actuator}__{model_id}','unit',parameters,
                        ('tiny_fit','analytic_validation'),[predecessor],produce,resume=resume)


def constant_velocity_error(arrays,scales):
    state = torch.tensor(arrays['state'][:,:-1].reshape(-1,arrays['state'].shape[2]),dtype=torch.float32)
    target = torch.tensor(arrays['state'][:,1:].reshape(-1,arrays['state'].shape[2]),dtype=torch.float32)
    h = torch.tensor(arrays['duration_ns'].reshape(-1,1)/1e9,dtype=torch.float32)
    predicted = torch.cat((state[:,:2]+h*state[:,2:4],state[:,2:4]),dim=1)
    return float(endpoint_loss(predicted,target,scales))


def qualified_learning_control(root,model_id,actuator):
    from .contracts import verify_run
    matches = []
    for path in sorted((Path(root) / 'evidence/M30').glob('*/manifest.json')):
        record = read_json(path)
        if record['job_inputs'].get('cell_id') != f'learning_control__{actuator}__{model_id}' or record['completion_status'] != 'complete':
            continue
        try:
            verify_run(Path(root),record)
        except (ValueError,OSError):
            continue  # Historical control cannot qualify changed current code.
        matches.append(path)
    if len(matches) != 1:
        raise ValueError('A unique current qualified analytic learning control is required')
    return matches[0]


def fit(job,root,resume=False):
    from .dataset import fit_normalization,load_corpus
    from .execution import cpu_defaults,parent
    from .models.factory import build_model
    from .training_jobs import training_job
    root = Path(root).resolve()
    parameters = dict(job['semantic_inputs'])
    train_manifest = root / parameters['train_manifest']
    validation_manifest = root / parameters['validation_manifest']
    if read_json(train_manifest).get('role') != 'train':
        raise ValueError('Training labels must be train; confirmation is forbidden')
    if read_json(validation_manifest).get('role') != 'validation':
        raise ValueError('Checkpoint selection requires validation labels only')
    if job['purpose'] != 'training' or job['protocol_version'] != 'PTCO-1.0':
        raise ValueError('Training job must declare its exact purpose and protocol')
    epochs = parameters['epochs']
    if type(epochs) is not int or epochs < 100 or epochs%100:
        raise ValueError('Nominal training uses complete prespecified 100-epoch blocks')
    cpu_defaults(parameters['seed'])
    predecessor = parent(root,'G20')
    control_manifest = qualified_learning_control(root,parameters['model_id'],parameters['actuator'])
    arrays = load_corpus(train_manifest,phase=job['phase'],role='train',purpose='training')
    validation = load_corpus(validation_manifest,phase=job['phase'],role='validation',purpose='selection')
    normalization = fit_normalization(train_manifest)
    for manifest in (train_manifest,validation_manifest):
        identity = read_json(manifest)
        if identity['actuator'] != parameters['actuator'] or identity['condition'] != parameters['condition']:
            raise ValueError('Training cell and dataset actuator/condition disagree')
    norm = Scales(torch.tensor(normalization['scale_q']),torch.tensor(normalization['scale_v']),
                  torch.tensor(normalization['velocity_input_rms']))
    parameters['data_hashes'] = {str(path.relative_to(root)):sha256_file(path) for path in
                       (train_manifest,validation_manifest,train_manifest.parent / 'trajectories.npz',
                        validation_manifest.parent / 'trajectories.npz',control_manifest,
                        train_manifest.parent.parent / 'normalization.json')}
    if parameters.get('continue_from'):
        from .contracts import verify_run
        previous_folder = root/parameters['continue_from']
        verify_run(root,read_json(previous_folder/'manifest.json'))
        for path in (previous_folder/'manifest.json',previous_folder/'resume.pt',
                     previous_folder/'resume.rng.json',previous_folder/'resume.complete.json',
                     previous_folder/'weights.pt',previous_folder/'validation_history.json'):
            parameters['data_hashes'][str(path.relative_to(root))] = sha256_file(path)
    parameters['normalization_sha256'] = hashlib.sha256(canonical_json(normalization)).hexdigest()
    cell = f'{parameters["actuator"]}__{parameters["condition"]}__{parameters["loss_mode"]}__{parameters["model_id"]}__seed{parameters["seed"]}'
    def produce(folder,resume_from=None):
        model = build_model(parameters['model_id'],parameters['actuator'],norm,parameters['seed'])
        session = TrainingSession(model,arrays,norm,parameters['seed'],parameters['loss_mode'],
                                  learning_rate=parameters['learning_rate'])
        rows,best,best_weights = [],None,None
        continuation = str(resume_from.relative_to(root)) if resume_from else parameters.get('continue_from')
        if continuation:
            previous = root / continuation
            session.restore_latest(previous / 'resume.pt')
            if (previous / 'validation_history.json').exists():
                rows = read_json(previous / 'validation_history.json')['rows']
                rows = [row for row in rows if row['epoch'] <= session.epoch]
                if rows:
                    best = select_checkpoint(rows)
                    best_weights = torch.load(previous / 'weights.pt',weights_only=True,map_location='cpu')
            if session.window_position == len(session.windows) and rows and rows[-1]['epoch'] == session.epoch:
                session.epoch += 1
                session.windows,session.window_position = epoch_starts(arrays,session.rng),0
        while session.epoch <= epochs:
            epoch = session.epoch
            session.advance_epoch(folder / 'resume.pt')
            if epoch%5 == 0:
                row = {'epoch':epoch,**validate_nominal(model,validation,norm,parameters['loss_mode'])}
                if parameters['condition'] == 'A':
                    durations = validation['duration_ns']
                    nominal_starts = np.argwhere(durations == 40_000_000)
                    with torch.no_grad():
                        nominal = batch_objective(model,validation,nominal_starts,norm,'one_step')
                    row['nominal_40ms_one_step'] = float(nominal['one_loss'])
                rows.append(row)
                selected = select_checkpoint(rows)
                if best is None or best['epoch'] != selected['epoch']:
                    best,best_weights = selected,{name:value.detach().clone() for name,value in model.state_dict().items()}
                atomic_json(folder / 'validation_history.json',{'rows':rows})
                buffer = io.BytesIO()
                torch.save(best_weights,buffer)
                atomic_bytes(folder / 'weights.pt',buffer.getvalue())
            if epoch == epochs:
                break
            session.epoch += 1
            session.windows,session.window_position = epoch_starts(arrays,session.rng),0
        _write_training_csv(folder / 'training.csv',session.training_rows,session.update_timing)
        _write_training_csv(folder / 'validation.csv',rows)
        atomic_json(folder / 'selection.json',best)
        atomic_json(folder / 'normalization.json',normalization)
        checks = [{'check_id':criterion,'status':'pass','reason':'Finite supervised fit, complete exposure and nominal checkpoint selection',
                   'scope':cell,'tolerance':None} for criterion in ('accounting','checkpoint','selection')]
        if session.valid_target_count != epochs*arrays['command'].shape[0]*arrays['command'].shape[1]:
            raise ValueError('Dataset-equivalent exposure accounting mismatch')
        return checks,{'cell_id':cell,'parameters':parameters,'validation':best,
               'validation_history':rows,'convergence':nominal_convergence(rows,epochs),
               'constant_velocity_error':constant_velocity_error(validation,norm),
               'valid_target_count':session.valid_target_count,'training_updates':session.updates,
               'parameter_count':sum(parameter.numel() for parameter in model.parameters()),'missing_cells':[],
               'model_hashes':{'canonical_tensor_sha256':canonical_tensor_hash(best_weights)}}
    return training_job(root,cell,job['phase'],parameters,('accounting','checkpoint','selection'),[predecessor],produce,resume=resume)


def qualify_nominal(training_run,root=Path('.')):
    from .contracts import verify_run
    from .io import run_directory
    verify_run(Path(root).resolve(),training_run)
    summary = read_json(run_directory(Path(root).resolve(),training_run) / 'summary.json')
    score,reference = summary['validation']['one_step'],summary['constant_velocity_error']
    learned = np.isfinite(score) and np.isfinite(reference) and score < .5*reference
    return {'qualified':bool(learned and summary['convergence']['stabilized']),
            'nominal_learning':bool(learned),'convergence':summary['convergence'],
            'reason':'Nominal learning and convergence checked independently of fine-time outcomes'}


def main(argv=None):
    import argparse
    import json
    from .models.factory import CORE_MODELS
    from .io import run_directory
    parser = argparse.ArgumentParser(description='Explicit PTCO supervised-training utilities')
    commands = parser.add_subparsers(dest='command',required=True)
    control = commands.add_parser('controls',help='Learn exact analytic plants before robot fitting')
    control.add_argument('--model',choices=CORE_MODELS)
    control.add_argument('--actuator',choices=('direct','lagged'))
    control.add_argument('--all-core',action='store_true')
    control.add_argument('--resume',action='store_true')
    control.add_argument('--root',type=Path,default=Path.cwd())
    args = parser.parse_args(argv)
    if not args.all_core and (args.model is None or args.actuator is None):
        parser.error('Specify --model and --actuator, or --all-core')
    models = CORE_MODELS if args.all_core else (args.model,)
    actuators = ('direct','lagged') if args.all_core else (args.actuator,)
    for actuator in actuators:
        for name in models:
            record = run_learning_controls(name,actuator,args.root,resume=args.resume)
            summary = read_json(run_directory(args.root.resolve(),record) / 'summary.json')
            print(json.dumps({'model':name,'actuator':actuator,'run_id':record['run_id'],
                              'completion_status':record['completion_status'],**summary},allow_nan=False),flush=True)
            if record['completion_status'] != 'complete':
                return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
