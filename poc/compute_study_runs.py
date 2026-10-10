"""Independent task streams and compact raw producers for the matched study."""
from __future__ import annotations
import io
import time
from pathlib import Path
import numpy as np
import torch
from .io import atomic_bytes,atomic_json,read_json,sha256_file,utc_now
from .compute_study import CONFIGS,MODELS,PREFIX,ALLOCATION,context,load_model,selected_receipt


def rng(seed,episode,channel):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261009,3,seed,episode,channel])))


def task_conditions(seed,episode):
    if type(seed) is not int or seed<0 or type(episode) is not int or episode<0:raise ValueError('Explicit task identity required')
    return dict(initial_state=np.r_[rng(seed,episode,0).uniform(-.5,.5,2),np.zeros(4)],
                target=rng(seed,episode,1).uniform(-1,1,2))


def normal_bank(seed,episode,count,maximum):
    if type(count) is not int or count<2 or type(maximum) is not int or count>maximum:
        raise ValueError('Valid candidate prefix and maximum required')
    return rng(seed,episode,2).normal(size=(40,4,maximum,6,2))[:,:,:count]


def npz(path,**arrays):
    buffer=io.BytesIO();np.savez_compressed(buffer,**arrays);atomic_bytes(path,buffer.getvalue())


def execute_episode(model,settings,seed,episode_id,step_ns,physics_step_ns,maximum=None,assets=Path('assets')):
    from .envs import Plant
    from .stateless_mpc import search
    from .action_path import ActionSegment
    from .control import anchor_components
    if step_ns not in (5_000_000,40_000_000):raise ValueError('Frozen two query intervals required')
    inputs=task_conditions(seed,episode_id);plant=Plant('lagged',physics_step_ns,assets)
    plant.reset(inputs['initial_state']);maximum=maximum or settings['candidate_count']
    normals=normal_bank(seed,episode_id,settings['candidate_count'],maximum)
    states=[plant.state()];times=[0];anchors=[];actions=[];selected=[];latencies=[];banks=[];proposals=[]
    predicted=[];selected_ids=[];fallbacks=0;previous=None;work={k:0 for k in ('network_calls','vector_field_calls','scalar_candidate_evaluations','history_tokens_processed')}
    for index in range(40):
        if plant.time_ns!=index*40_000_000:raise ValueError('Fresh observation clock mismatch')
        start=time.perf_counter_ns()
        result=search(model,torch.tensor(plant.state(),dtype=torch.float64),torch.tensor(inputs['target']),None,
                      previous,normals[index],settings,(step_ns,)*(40_000_000//step_ns),40_000_000)
        latencies.append((time.perf_counter_ns()-start)/1e9)
        previous=result['knots'].copy();selected.append(previous);banks.append(result['candidates']);proposals.append(result['proposals'])
        selected_ids.append(-1 if result['selected_global_id'] is None else result['selected_global_id'])
        predicted.append(np.full((6,6),np.nan) if result['predicted_anchors'] is None else result['predicted_anchors'])
        fallbacks+=int(result['status']=='all_candidates_failed')
        if result['status']=='ok' and not np.array_equal(previous,result['candidates']['knots'][result['selected_global_id']]):
            raise ValueError('Executed unscored candidate')
        response=plant.advance(previous[0],40_000_000,log_every_step=True)
        if response['status']!='ok':raise ValueError('True plant failure invalidates episode')
        states.extend(response['trajectory']['state']);times.extend(response['trajectory']['time_ns']);anchors.append(plant.state())
        actions.append(ActionSegment(index*40_000_000,40_000_000,previous[0]))
        for key in work:work[key]+=result['work'][key]
    states=np.asarray(states);times=np.asarray(times,dtype=np.int64);anchors=np.asarray(anchors)
    if not np.array_equal(times,np.arange(0,1_600_000_000+1,physics_step_ns)):raise ValueError('Partial/dense trajectory invalid')
    parts=anchor_components(anchors,inputs['target'],tuple(actions));error=states[:,:2]-inputs['target'];final=times>=1_400_000_000
    raw=dict(time_ns=times,state=states,anchors=anchors,target=inputs['target'],initial_state=inputs['initial_state'],
             commands=np.stack([a.command for a in actions]),selected_knots=np.stack(selected),selected_ids=np.array(selected_ids),
             predicted_anchors=np.stack(predicted),latency_seconds=np.array(latencies),
             candidate_knots=np.stack([b['knots'] for b in banks]),candidate_costs=np.stack([b['costs'] for b in banks]),
             candidate_statuses=np.array([b['statuses'] for b in banks]),candidate_reasons=np.array([b['reasons'] for b in banks]),
             proposal_means=np.array([[p['mean'] for p in ps] for ps in proposals]),
             proposal_stds=np.array([[p['std'] for p in ps] for ps in proposals]))
    summary=dict(complete=True,seed=seed,episode_id=episode_id,step_ns=step_ns,physics_step_ns=physics_step_ns,
            realized_cost=parts['total'],q_rmse_rad=float(np.sqrt(np.mean(error**2))),terminal_q_error_rad=float(np.linalg.norm(error[-1])),
            velocity_rms_rad_s=float(np.sqrt(np.mean(states[:,2:4]**2))),effort_integral=parts['effort']/.001,
            success=bool(np.all(np.linalg.norm(error[final],axis=1)<.1) and np.all(np.linalg.norm(states[final,2:4],axis=1)<.2)),
            fallback_count=fallbacks,planning_median_seconds=float(np.median(latencies)),planning_p95_seconds=float(np.quantile(latencies,.95)),
            settings=settings,maximum_candidates=maximum,work=work,target=inputs['target'].tolist(),initial_state=inputs['initial_state'].tolist())
    return raw,summary


def controls(root,kind='primary'):
    f,store=context(root);allocation=read_json(Path(root)/ALLOCATION);store.read(allocation['receipt'])
    path=Path(root)/f'{PREFIX}_{kind}.json'
    ledger=read_json(path) if path.exists() else dict(kind=kind,completed=[],state='running')
    for seed in f['seeds']:
        models={n:load_model(root,n,seed) for n in MODELS}
        for episode in range(f['episodes_per_seed'] if kind=='primary' else 1):
            for name,step in CONFIGS:
                settings=allocation['settings'][f'{name}:{step}']
                request=dict(model_id=name,seed=seed,episode_id=episode,step_ns=step,kind=kind,
                             settings=settings,maximum=allocation['maximum_candidates'],allocation_sha256=sha256_file(Path(root)/ALLOCATION),
                             selected_model=selected_receipt(root,name,seed,f))
                def produce(folder):
                    tick=f['finer_physics_step_ns'] if kind=='finer' else f['physics_step_ns']
                    raw,summary=execute_episode(models[name],settings,seed,episode,step,tick,
                                               allocation['maximum_candidates'],Path(root)/'assets')
                    npz(folder/'episode.npz',**raw);summary.update(model_id=name,kind=kind)
                    return summary
                receipt=store.produce('control',request,produce)
                if receipt not in ledger['completed']:ledger['completed'].append(receipt)
                ledger.update(updated_utc=utc_now(),completed_count=len(ledger['completed']))
                atomic_json(path,ledger)
                _,summary=store.read(receipt)
                print(f'{kind} {name}/{step//1000000}ms seed{seed} task{episode}: cost={summary["realized_cost"]:.6f} planner={summary["planning_median_seconds"]:.4f}s [{len(ledger["completed"])}]',flush=True)
    ledger['state']='complete';atomic_json(path,ledger)


def references(root):
    from .envs import Plant
    f,store=context(root)
    def produce(folder):
        source=Plant('lagged',125000,Path(root)/'assets');states=[];commands=[];strata=[]
        for trajectory in range(256):
            generator=rng(trajectory,0,10);s=np.r_[generator.uniform(-np.pi,np.pi,2),generator.uniform(-2,2,4)]
            source.reset(s);states.append(s.copy());commands.append(rng(trajectory,0,13).uniform(-1.5,1.5,2));strata.append('cold')
            action_rng=rng(trajectory,0,11);duration_rng=rng(trajectory,0,12)
            while source.time_ns<320_000_000:
                source.advance(action_rng.uniform(-2,2,2),int(duration_rng.choice([20,30,40,50,60]))*1_000_000)
            states.append(source.state());commands.append(rng(trajectory,1,13).uniform(-1.5,1.5,2));strata.append('warm')
        states=np.array(states);commands=np.array(commands);horizons=np.array([5_000_000,10_000_000,40_000_000]);truth={};envelopes={'q':0.,'v':0.}
        for tick in (250000,125000):
            plant=Plant('lagged',tick,Path(root)/'assets');outputs={k:np.empty((512,3,6)) for k in ('zero','command')}
            for i,s in enumerate(states):
                for j,h in enumerate(horizons):
                    for key,u in (('zero',np.zeros(2)),('command',commands[i])):
                        plant.reset(s);outputs[key][i,j]=plant.advance(u,int(h))['state']
            truth[tick]=outputs
        for c,index in (('q',slice(0,2)),('v',slice(2,4))):
            envelopes[c]=max(float(np.max(np.linalg.norm(truth[250000][k][:,:,index]-truth[125000][k][:,:,index],axis=-1))) for k in ('zero','command'))
        if envelopes['q']>1e-7 or envelopes['v']>1e-6:raise ValueError('Probe truth unconverged')
        npz(folder/'references.npz',states=states,commands=commands,strata=np.array(strata),horizons_ns=horizons,
            truth_zero=truth[125000]['zero'],truth_command=truth[125000]['command'],coarse_zero=truth[250000]['zero'],coarse_command=truth[250000]['command'])
        return dict(n_probes=512,numerical_envelope=envelopes,namespace=f['namespace'])
    receipt=store.produce('references',{},produce);atomic_json(Path(root)/f'{PREFIX}_references.json',receipt);return receipt


def measure(root):
    from .affine_confirmation import _profile
    f,store=context(root);reference=references(root);folder,_=store.read(reference)
    with np.load(folder/'references.npz',allow_pickle=False) as data:ref={k:data[k].copy() for k in data.files}
    fits={str(seed):read_json(Path(root)/f'{PREFIX}_fits/seed{seed}.json') for seed in f['seeds']}
    def produce(folder):
        profiles=[]
        for si,seed in enumerate(f['seeds']):
            for name in MODELS if si%2==0 else MODELS[::-1]:
                model=load_model(root,name,seed);raw={}
                for key in ('zero','command','composed'):raw['model_'+key]=np.empty((512,3,6))
                state=torch.tensor(ref['states']);u=torch.tensor(ref['commands'])
                with torch.no_grad():
                    for hi,h_ns in enumerate(ref['horizons_ns']):
                        h=float(h_ns)/1e9
                        raw['model_zero'][:,hi]=model.predict(state,torch.zeros_like(u),h).next_state.numpy()
                        raw['model_command'][:,hi]=model.predict(state,u,h).next_state.numpy()
                        mid=model.predict(state,u,h/2).next_state
                        raw['model_composed'][:,hi]=model.predict(mid,u,h/2).next_state.numpy()
                if any(not np.isfinite(v).all() for v in raw.values()):raise ValueError('Nonfinite probe prediction')
                npz(folder/f'{name}__seed{seed}.npz',**raw)
                rows=_profile(model,state[:1],u[:1],name,seed)
                for row in rows:row['neural_rows']=row['batch_size']*(4 if name=='node_structured' else 2)
                profiles.extend(rows);print(f'New probes/profile {name} seed{seed}',flush=True)
        atomic_json(folder/'profiles.json',dict(rows=profiles))
        return dict(reference=reference,profiles=profiles,seeds=f['seeds'],models=list(MODELS))
    receipt=store.produce('predictions',dict(reference=reference,fits=fits),produce)
    atomic_json(Path(root)/f'{PREFIX}_measurement.json',receipt)


def reproduce(root):
    from .compute_study import fit
    controls(root,'replay');controls(root,'finer')
    f,store=context(root)
    for seed in (f['seeds'][0],f['seeds'][-1]):fit(root,seed,replica=True)
