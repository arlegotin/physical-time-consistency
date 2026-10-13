"""Fresh-root numerical replication; invoked after the complete primary study."""
from pathlib import Path
import gzip,json,sys,hashlib,time
sys.path.insert(0,str(Path.cwd()))
from poc.calibration_study import open_store, _execution_settings
_execution_settings()
import numpy as np
import torch
from poc.calibration_predictions import (prepare_evaluation,load_evaluation,load_prediction_truth,truth_cases,case_path,_predict_paths,_replay,refinement_required)
from poc.calibration_decisions import (decision_state_ids,load_decision_truth,score_candidates,decision_outcome,_bank,_replay as decision_replay)
from poc.calibration_training import load_selected
from poc.calibration_resources import ResourceTracker
from poc.envs import Plant
from poc.io import canonical_json,atomic_json

primary_root=Path(sys.argv[1]); replica_root=Path(sys.argv[2]); root=Path.cwd()
primary=open_store(root,primary_root,'hybrid'); replica=open_store(root,replica_root,'hybrid')
assert {k:v for k,v in primary.bindings.items() if k != 'output_root'} == {k:v for k,v in replica.bindings.items() if k != 'output_root'}
assert primary.recipe['evaluation']['seed_session_id'] == 1
assert replica.recipe['evaluation']['seed_session_id'] == 1
models=('hybrid_affine_flow','hybrid_affine_rk4','hybrid_affine_aware')

def products(store,stage):
    results=[]
    for marker in sorted((store.output_root/'jobs'/stage).glob('*/COMPLETED')):
        receipt={'job_key':marker.parent.name,'path':str(marker.parent.relative_to(store.output_root)),
                 'manifest_sha256':json.loads(marker.read_text())['manifest_sha256']}
        folder,manifest=store.read(receipt)
        results.append((receipt,folder,manifest))
    return results

def coordinate(manifest):
    return {k:manifest['request'][k] for k in ('session_id','budget_s','model_id','initialization_id')}

def selected(store,model):
    matches=[p for p in products(store,'fit') if coordinate(p[2])=={'session_id':1,'budget_s':30,'model_id':model,'initialization_id':0}]
    assert len(matches)==1
    return matches[0]

def one(store,stage):
    matches=products(store,stage)
    if stage=='collect':matches=[p for p in matches if p[2]['request']['session_id']==1]
    assert len(matches)==1,(stage,len(matches));return matches[0]

def arrays(path):
    with np.load(path,allow_pickle=False) as packet:return {k:packet[k] for k in packet.files}

def identical(a,b):
    return bool(np.array_equal(np.asarray(a),np.asarray(b),equal_nan=True))

def records(path):
    with gzip.open(path,'rt') as stream:
        for line in stream:yield json.loads(line)

result={'scope':'independent fresh-root fitting and numerical replication; resources excluded from primary commissioning projection',
        'primary_output_root':str(primary_root),'replica_output_root':str(replica_root),'fits':[],'prediction_checks':[],
        'decision_checks':[],'truth_checks':{},'all_equal':True}
started=time.perf_counter()
with ResourceTracker(replica_root/'execution-logs/numerical-replication-resources.json') as tracker:
    pc=one(primary,'collect');rc=one(replica,'collect')
    pa=arrays(pc[1]/'session.npz');ra=arrays(rc[1]/'session.npz')
    session_equal=set(pa)==set(ra) and all(identical(pa[k],ra[k]) for k in pa)
    result['session_arrays_equal']=session_equal;result['all_equal'] &= session_equal
    pe=one(primary,'evaluation');re_receipt=prepare_evaluation(replica,root=root)
    eva=load_evaluation(replica,re_receipt);original=load_evaluation(primary,pe[0])
    eval_equal=set(eva)==set(original) and all(identical(eva[k],original[k]) for k in eva)
    result['evaluation_arrays_equal']=eval_equal;result['all_equal'] &= eval_equal
    truth_product=one(primary,'prediction_truth');prediction_truth=load_prediction_truth(primary,pe[0],truth_product[0])
    cases=truth_cases(replica.recipe);audit_ids=np.flatnonzero(np.isin(eva['trajectory_id'],[0,63]))
    resolutions=[250_000,125_000,62_500];plants={h:Plant('lagged',h,root/'assets') for h in resolutions}
    truth_exact=True;resolution_discrepancy={};fresh_truth={}
    for s in audit_ids.tolist():
        for c,case in enumerate(cases):
            values=[_replay(plants[h],eva['state'][s],case_path(eva,s,case)) for h in resolutions]
            fresh_truth.update({(s,c,h):value for h,value in zip(resolutions,values)})
            expected=[prediction_truth['endpoint'][s,c],*prediction_truth['refined_endpoint'][s,c]]
            truth_exact &= all(identical(a,b) for a,b in zip(values,expected))
            for i in (0,1):
                key=str(resolutions[i])+'/'+str(resolutions[i+1]);resolution_discrepancy[key]=max(resolution_discrepancy.get(key,0.),float(np.max(np.abs(values[i]-values[i+1]))))
    result['truth_checks']['predictions']={'audit_trajectory_ids':[0,63],'state_count':len(audit_ids),
        'case_count':len(cases),'resolution_ns':resolutions,'all_values_equal':truth_exact,
        'maximum_resolution_discrepancy':resolution_discrepancy}
    result['all_equal'] &= truth_exact
    dt_product=one(primary,'decision_truth');dt=load_decision_truth(primary,pe[0],dt_product[0]);decision_ids=decision_state_ids(eva,replica.recipe)
    chosen=[0,len(decision_ids)-1];truth_exact=True;resolution_discrepancy={};dtruth={}
    banks={i:_bank(eva,decision_ids[i],replica.recipe) for i in chosen}
    bank_equal=all(identical(banks[i],dt['bank'][i]) for i in chosen)
    result['independently_regenerated_candidate_banks_equal']=bank_equal;result['all_equal'] &= bank_equal
    for i in chosen:
        s=decision_ids[i];target=eva['target'][int(eva['trajectory_id'][s])]
        regenerated={k:np.empty_like(dt[k][i]) for k in ['cost','sampled_feasible','envelope','feasibility_status']}
        for c,bank in enumerate(banks[i]):
            runs=[decision_replay(plants[h],eva['state'][s],target,bank) for h in resolutions]
            truth_exact &= identical([r['cost'] for r in runs],dt['cost_by_resolution'][i,c,:3])
            truth_exact &= identical([r['sampled_feasible'] for r in runs],dt['feasibility_by_resolution'][i,c,:3])
            for slot in (3,4):
                h=replica.recipe['evaluation']['adaptive_refinement_steps_ns'][slot-3]
                if np.isfinite(dt['cost_by_resolution'][i,c,slot]):
                    if h not in plants:plants[h]=Plant('lagged',h,root/'assets')
                    run=decision_replay(plants[h],eva['state'][s],target,bank);runs.append(run)
                    truth_exact &= identical(run['cost'],dt['cost_by_resolution'][i,c,slot])
                    truth_exact &= identical(run['sampled_feasible'],dt['feasibility_by_resolution'][i,c,slot])
            regenerated['cost'][c]=runs[-1]['cost'];regenerated['sampled_feasible'][c]=runs[-1]['sampled_feasible']
            regenerated['feasibility_status'][c]='resolved' if runs[-1]['sampled_feasible']==runs[-2]['sampled_feasible'] else 'unresolved'
            truth_exact &= identical(runs[-1]['samples'],dt['samples'][i,c]) and identical(runs[0]['samples'][8::8],dt['base_anchors'][i,c])
            truth_exact &= all(identical(runs[-1][key],dt[key][i,c]) for key in ('dense_max_abs','dense_violation_count','dense_first_violation_ns'))
            regenerated['envelope'][c]=max(abs(runs[-1]['cost']-runs[-2]['cost']),abs(runs[-2]['cost']-runs[-3]['cost']))
            for level in (0,1):
                key=str(resolutions[level])+'/'+str(resolutions[level+1]);resolution_discrepancy[key]=max(resolution_discrepancy.get(key,0.),abs(runs[level]['cost']-runs[level+1]['cost']))
        truth_exact &= all(bool(np.array_equal(v,dt[k][i])) if k=='feasibility_status' else identical(v,dt[k][i]) for k,v in regenerated.items());dtruth[i]=regenerated
    result['truth_checks']['decisions']={'trajectory_ids':[0,63],'state_ids':[decision_ids[i] for i in chosen],
        'candidate_count':256,'resolution_ns':resolutions,'replayed_primary_adaptive_resolution_choices':True,
        'all_cost_feasibility_and_envelope_values_equal':truth_exact,'maximum_resolution_cost_discrepancy':resolution_discrepancy}
    result['all_equal'] &= truth_exact
    primary_prediction_ids=sorted(set([0,len(eva['state'])-1,*np.flatnonzero(eva['primary'])[[0,-1]].tolist()]))
    for model_id in models:
        pf=selected(primary,model_id);rf=selected(replica,model_id)
        pm=load_selected(primary,pf[0]);rm=load_selected(replica,rf[0])
        tensor_equal=pf[2]['summary']['canonical_tensor_sha256']==rf[2]['summary']['canonical_tensor_sha256']
        weights_equal=set(pm.state_dict())==set(rm.state_dict()) and all(torch.equal(pm.state_dict()[k],rm.state_dict()[k]) for k in pm.state_dict())
        validation_equal=pf[2]['summary']['selected']==rf[2]['summary']['selected']
        result['fits'].append({'coordinate':coordinate(rf[2]),'primary_receipt':pf[0],'replica_receipt':rf[0],
            'canonical_tensor_sha256':rf[2]['summary']['canonical_tensor_sha256'],'canonical_tensor_hash_equal':tensor_equal,
            'all_selected_tensors_equal':weights_equal,'selected_validation_equal':validation_equal,'selected':rf[2]['summary']['selected']})
        result['all_equal'] &= tensor_equal and weights_equal and validation_equal
        pp=[p for p in products(primary,'predictions') if p[2]['request']['fit_receipt']==pf[0]];assert len(pp)==1
        expected={(r['state_id'],r['scenario'],r['horizon_ns'],r['query_step_ns'],r['component']):r
                  for r in records(pp[0][1]/'rows.jsonl.gz') if r['kind']=='endpoint' and r['state_id'] in primary_prediction_ids}
        count=0;exact=True;max_error=0.
        for case_id,case in enumerate(cases):
            if case['branch']=='zero':continue
            paths=[case_path(eva,s,case) for s in range(len(eva['state']))]
            steps=replica.recipe['evaluation']['query_steps_ns'] if case['branch']=='recorded' else [case['horizon_ns']]
            for step in steps:
                outputs,statuses=_predict_paths(rm,eva['state'],paths,step)
                for s in primary_prediction_ids:
                    for component,cut in [('q',slice(0,2)),('v',slice(2,4))]:
                        row=expected[s,case['scenario'],case['horizon_ns'],step,component]
                        eq=statuses[s]==row['status'] and (identical(outputs[s,cut],row['prediction']) if statuses[s]=='ok' else row['prediction'] is None)
                        exact &= eq
                        h=row['truth_resolution_ns'][0]
                        grids=[250_000,125_000,62_500] if h==250_000 else [125_000,62_500,31_250] if h==31_250 else [62_500,31_250,15_625]
                        assert h in (250_000,31_250,15_625),h
                        for grid in grids:
                            if (s,case_id,grid) not in fresh_truth:
                                if grid not in plants:plants[grid]=Plant('lagged',grid,root/'assets')
                                fresh_truth[s,case_id,grid]=_replay(plants[grid],eva['state'][s],case_path(eva,s,case))
                        truth_value=fresh_truth[s,case_id,h]
                        bound=np.maximum(np.abs(fresh_truth[s,case_id,grids[0]]-fresh_truth[s,case_id,grids[1]]),np.abs(fresh_truth[s,case_id,grids[1]]-fresh_truth[s,case_id,grids[2]]))
                        exact &= identical(truth_value[cut],row['truth']) and identical(bound[cut],row['envelope'])
                        if statuses[s]=='ok':
                            error=outputs[s,cut]-truth_value[cut]
                            exact &= identical(error,row['error'])
                            bound_norm=float(np.linalg.norm(bound[cut]));error_norm=float(np.linalg.norm(error))
                            truth_status='numerically_equal' if error_norm==0 and bound_norm==0 else 'requires_refinement' if refinement_required(bound_norm,error_norm) else 'audited'
                            exact &= truth_status==row['truth_status']
                            max_error=max(max_error,float(np.max(np.abs(outputs[s,cut]-row['prediction']))))
                        count+=1
        result['prediction_checks'].append({'model_id':model_id,'state_ids':primary_prediction_ids,'checked_component_coordinates':count,
            'all_predictions_truth_errors_envelopes_and_statuses_equal':exact,'max_abs_difference':max_error});result['all_equal'] &= exact
        term_rows={(r['state_id'],r['scenario'],r['horizon_ns'],r['query_step_ns'],r['kind'],r['component']):r
                   for r in records(pp[0][1]/'rows.jsonl.gz') if r['kind']!='endpoint' and r['state_id'] in primary_prediction_ids}
        term_count=0;term_exact=True
        def truth_value_and_envelope(s,c,h):
            grids=[250_000,125_000,62_500] if h==250_000 else [125_000,62_500,31_250] if h==31_250 else [62_500,31_250,15_625]
            assert h in (250_000,31_250,15_625),h
            for grid in grids:
                if (s,c,grid) not in fresh_truth:
                    if grid not in plants:plants[grid]=Plant('lagged',grid,root/'assets')
                    fresh_truth[s,c,grid]=_replay(plants[grid],eva['state'][s],case_path(eva,s,cases[c]))
            bound=np.maximum(np.abs(fresh_truth[s,c,grids[0]]-fresh_truth[s,c,grids[1]]),np.abs(fresh_truth[s,c,grids[1]]-fresh_truth[s,c,grids[2]]))
            return fresh_truth[s,c,h],bound
        for c,case in enumerate(cases):
            if case['branch']!='new':continue
            step=case['horizon_ns'];paths=[case_path(eva,s,case) for s in range(len(eva['state']))]
            new,ns=_predict_paths(rm,eva['state'],paths,step)
            zero,zs=_predict_paths(rm,eva['state'],[case_path(eva,s,cases[c+1]) for s in range(len(eva['state']))],step)
            half,hs=_predict_paths(rm,eva['state'],paths,step//2)
            terms=(('drift',None if zero is None else zero-eva['state'],zs),
                   ('command_response',None if new is None or zero is None else new-zero,[a if a!='ok' else b for a,b in zip(ns,zs)]),
                   ('composition',None if new is None or half is None else new-half,[a if a!='ok' else b for a,b in zip(ns,hs)]))
            for kind,values,statuses in terms:
                for s in primary_prediction_ids:
                    for component,cut in [('q',slice(0,2)),('v',slice(2,4))]:
                        row=term_rows[s,case['scenario'],case['horizon_ns'],step,kind,component]
                        if kind=='composition':target=np.zeros(6);bound=np.zeros(6)
                        elif kind=='drift':
                            zero_truth,bound=truth_value_and_envelope(s,c+1,row['truth_resolution_ns'][0]);target=zero_truth-eva['state'][s]
                        else:
                            new_truth,new_bound=truth_value_and_envelope(s,c,row['truth_resolution_ns'][0])
                            zero_truth,zero_bound=truth_value_and_envelope(s,c+1,row['truth_resolution_ns'][1])
                            target=new_truth-zero_truth;bound=new_bound+zero_bound
                        usable=values is not None and statuses[s]=='ok'
                        error=values[s,cut]-target[cut] if usable else None
                        envelope_norm=float(np.linalg.norm(bound[cut]))
                        truth_status='not_applicable' if kind=='composition' else ('numerically_equal' if usable and float(np.linalg.norm(error))==0 and envelope_norm==0 else 'requires_refinement' if usable and refinement_required(envelope_norm,float(np.linalg.norm(error))) else 'audited')
                        term_exact &= statuses[s]==row['status'] and identical(target[cut],row['truth']) and identical(bound[cut],row['envelope'])
                        term_exact &= envelope_norm==row['envelope_norm'] and truth_status==row['truth_status']
                        term_exact &= (identical(values[s,cut],row['prediction']) and identical(error,row['error'])) if usable else row['prediction'] is None and row['error'] is None
                        term_count+=1
        result['prediction_checks'][-1].update(checked_additional_component_coordinates=term_count,
            all_drift_command_response_composition_values_errors_envelopes_and_statuses_equal=term_exact)
        result['all_equal'] &= term_exact
        pd=[p for p in products(primary,'decisions') if p[2]['request']['fit_receipt']==pf[0]];assert len(pd)==1
        expected={(r['state_id'],r['query_step_ns'],r['candidate_id']):r for r in records(pd[0][1]/'rows.jsonl.gz') if r['state_id'] in [decision_ids[i] for i in chosen]}
        outcomes={(r['state_id'],r['query_step_ns']):r for r in pd[0][2]['summary']['outcomes']};exact=True;count=0;max_error=0.
        for i in chosen:
            s=decision_ids[i]
            for step in replica.recipe['evaluation']['query_steps_ns']:
                scored=score_candidates(rm,torch.from_numpy(eva['state'][s]),eva['target'][int(eva['trajectory_id'][s])],banks[i],query_step_ns=step)
                for c,cost in enumerate(scored['costs']):
                    row=expected[s,step,c];cost_equal=identical(cost,row['predicted_cost']) if np.isfinite(cost) else row['predicted_cost'] is None
                    exact &= cost_equal and bool(scored['sampled_feasible'][c])==row['predicted_sampled_feasible'] and scored['statuses'][c]==row['status']
                    if np.isfinite(cost) and row['predicted_cost'] is not None:max_error=max(max_error,abs(float(cost)-row['predicted_cost']))
                    count+=1
                own=dtruth[i];outcome=decision_outcome(scored,own['cost'],own['sampled_feasible'],own['envelope'],own['feasibility_status']);base=outcomes[s,step]
                exact &= all(outcome[k]==base[k] for k in outcome) and scored['work']==base['work'] and scored['query_log']==base['query_log']
        result['decision_checks'].append({'model_id':model_id,'state_ids':[decision_ids[i] for i in chosen],'checked_candidate_coordinates':count,
            'all_costs_feasibility_selection_and_regret_equal':exact,'max_abs_predicted_cost_difference':max_error});result['all_equal'] &= exact
result['resources']=tracker.summary();result['elapsed_wall_s']=time.perf_counter()-started
atomic_json(replica_root/'numerical-replication.json',result)
print(canonical_json(result).decode())
if not result['all_equal']:raise SystemExit('Fresh-process replication disagreement; retained for diagnosis')
