"""Open-loop partition tests using only common physical anchors and imagined history."""
from __future__ import annotations

from pathlib import Path
import copy
import numpy as np
import torch

from .clocks import jitter_partition, partition
from .history import HistoryBatch, HistoryToken


def rollout_path(model, initial_state, commands, durations_ns, model_step_ns,
                 seed_entropy=(1201,0,0,4), power_mode=None):
    commands = np.asarray(commands,dtype=np.float64)
    durations = np.asarray(durations_ns)
    state = np.asarray(initial_state,dtype=np.float64)
    if commands.ndim != 2 or commands.shape[1] != 2 or durations.dtype != np.int64 or durations.shape != (len(commands),):
        raise ValueError('An explicit physical command path and integer hold durations are required')
    if not np.isfinite(commands).all() or np.any(np.abs(commands)>2) or np.any(durations != 40_000_000):
        raise ValueError('E02 requires original 40 ms held-command paths')
    if state.shape not in ((4,),(6,)) or not np.isfinite(state).all():
        raise ValueError('A finite fully observed initial state is required')
    audit = copy.deepcopy(model).double().eval()
    output = np.full((len(commands)+1,len(state)),np.nan,dtype=np.float64)
    output[0] = state
    statuses = ['ok']
    times = np.r_[0,np.cumsum(durations)].astype(np.int64)
    context = HistoryBatch(((),),np.array([0],dtype=np.int64))
    x = torch.tensor(state[None,:],dtype=torch.float64)
    traces, commits = [], []
    work = {'network_calls':0,'vector_field_calls':0,'history_tokens_processed':0,
            'scalar_candidate_evaluations':0}
    failed = False
    with torch.no_grad():
        for hold,(command,duration) in enumerate(zip(commands,durations)):
            if failed:
                statuses.append('model_failure')
                continue
            start = x[0].detach().numpy().copy()
            time = int(times[hold])
            steps = jitter_partition((*seed_entropy,hold)) if model_step_ns == 'jitter' else partition(int(duration),model_step_ns)
            for step in steps:
                # Read-only eligible history; never commit a model substep.
                query = context.at_time(np.array([time],dtype=np.int64))
                kwargs = {'history':query}
                if power_mode is not None:
                    kwargs['power_mode'] = power_mode
                try:
                    packet = audit.predict(x,torch.tensor(command[None,:],dtype=torch.float64),step/1e9,**kwargs)
                    for key in work:
                        work[key] += getattr(packet,key)
                    failed = packet.status != ('ok',)
                except (RuntimeError,ValueError) as error:
                    failed = True
                    traces.append({'hold':hold,'query_ns':time,'h_ns':step,'status':'model_failure','reason':str(error)})
                if failed:
                    break
                traces.append({'hold':hold,'query_ns':time,'h_ns':step,'status':'ok',
                               'observations_received':1,'history_packet_ids':[t.packet_id for t in query.eligible().tokens[0]]})
                x, time = packet.next_state, time+step
            if failed:
                statuses.append('model_failure')
                continue
            if time != int(times[hold+1]):
                raise ValueError('Model partition changed the physical command/anchor clock')
            output[hold+1] = x[0].numpy()
            statuses.append('ok')
            token = HistoryToken(int(times[hold]),start,command,int(duration),'imagined',f'hold-{hold}')
            context = context.at_time(np.array([time],dtype=np.int64)).append_completed(token).eligible()
            commits.append({'anchor_ns':time,'start_ns':token.start_ns,'packet_id':token.packet_id,'provenance':'imagined'})
    return {'states':output,'time_ns':times,'statuses':np.array(statuses), 'traces':traces,
            'history_commits':commits,'observations_received':1,**work}


def prediction_metrics(result, truth, nominal, normalization):
    truth = np.asarray(truth,dtype=np.float64)
    if result['states'].shape != truth.shape or nominal['states'].shape != truth.shape:
        raise ValueError('Truth and predictions must share every common physical anchor')
    if np.any(result['statuses'] != 'ok') or np.any(nominal['statuses'] != 'ok'):
        return {'status':'model_failure','q_rmse_rad':None,'v_rmse_rad_s':None,
                'normalized_endpoint_error':None,'max_state_error':None,'partition_discrepancy':None}
    error = result['states'][1:,:4]-truth[1:,:4]
    scale = np.r_[normalization['scale_q'],normalization['scale_v']]
    return {'status':'ok','q_rmse_rad':float(np.sqrt(np.mean(error[:,:2]**2))),
            'v_rmse_rad_s':float(np.sqrt(np.mean(error[:,2:4]**2))),
            'normalized_endpoint_error':float(np.sqrt(np.mean((error[-1]/scale)**2))),
            'max_state_error':float(np.max(np.abs(error))),
            'partition_discrepancy':float(np.sqrt(np.mean(((result['states'][1:,:4]-nominal['states'][1:,:4])/scale)**2))),
            'q_rmse_by_anchor':np.sqrt(np.mean(error[:,:2]**2,axis=1)).tolist(),
            'v_rmse_by_anchor':np.sqrt(np.mean(error[:,2:4]**2,axis=1)).tolist()}


def run_e02(job,root: Path,resume=False):
    from .probes import load_audit_inputs,write_metric_csv
    from .audit_jobs import audit_job
    from .io import atomic_bytes,append_event
    import io
    root = Path(root).resolve()
    parameters = dict(job['semantic_inputs'])
    model,arrays,normalization,metadata = load_audit_inputs(parameters,job['phase'],root)
    if metadata['corpus']['condition'] != 'F':
        raise ValueError('E02 uses fixed-data original command paths only')
    modes = parameters.get('partitions_ns',[40_000_000,20_000_000,10_000_000,5_000_000,2_500_000,'jitter'])
    def produce(folder):
        states,statuses,metrics = [],[],[]
        for trajectory in range(len(arrays['state'])):
            common = (model,arrays['state'][trajectory,0],arrays['command'][trajectory],arrays['duration_ns'][trajectory])
            nominal = rollout_path(*common,40_000_000,power_mode=parameters.get('power_mode'))
            for mode in modes:
                result = nominal if mode == 40_000_000 else rollout_path(*common,mode,
                        seed_entropy=(metadata['corpus']['seeds']['root_seed'],int(parameters['actuator']=='lagged'),trajectory,4),
                        power_mode=parameters.get('power_mode'))
                states.append(result['states']); statuses.append(result['statuses'])
                metrics.append({'trajectory_id':trajectory,'partition_ns':mode,
                                **prediction_metrics(result,arrays['state'][trajectory],nominal,normalization)})
                append_event(folder/'partition_traces.jsonl',{'trajectory_id':trajectory,'partition_ns':mode,
                            'traces':result['traces'],'history_commits':result['history_commits']})
        buffer = io.BytesIO()
        np.savez_compressed(buffer,states=np.stack(states),statuses=np.stack(statuses),
                            time_ns=arrays['time_ns'],truth=arrays['state'])
        atomic_bytes(folder/'anchors.npz',buffer.getvalue())
        common = {'run_id':folder.name,'experiment_id':'E02','phase':job['phase'],
                  'model_id':parameters['model_id'],'actuator':parameters['actuator'],
                  'condition':parameters['condition'],'loss_mode':parameters['loss_mode'],
                  'seed':parameters['seed'],'workload_id':parameters['corpus_manifest']}
        write_metric_csv(folder/'prediction_metrics.csv',[{**common,'trajectory_id':row['trajectory_id'],
               'partition_ns':row['partition_ns'],'metric':name,'statistic':'trajectory',
               'value':row.get(name),'unit':'rad' if name == 'q_rmse_rad' else 'rad/s' if name == 'v_rmse_rad_s' else 'normalized',
               'sample_count':32,'status':row['status'],'evidence':'anchors.npz'} for row in metrics for name in
               ('q_rmse_rad','v_rmse_rad_s','normalized_endpoint_error','partition_discrepancy')])
        return [{'check_id':'E02.coverage','status':'pass','scope':'E02','tolerance':None,
                 'reason':'Every held path, partition and common anchor retained; failures explicit'}],{
                 'metrics':metrics,'missing_cells':[], 'model_metadata':metadata['model'],
                 'cell_count':len(metrics),'failed_cells':sum(row['status'] != 'ok' for row in metrics)}
    return audit_job(root,'E02',job,metadata,('E02.coverage',),produce)
