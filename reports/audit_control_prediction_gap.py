"""Reconcile selected-plan next-interval errors on the completed development cells."""
from pathlib import Path
import json
import math
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np

from poc.contracts import verify_run
from poc.control_gap import first_interval_metrics
from poc.io import atomic_json,canonical_json,read_json,safe_path,sha256_file


def audit(root=ROOT):
    root=Path(root).resolve();rows=[];groups={};inputs={};dependencies={}
    definitions=[('augmented_shared','reports/control_development_direct.json',60),
                 ('fixed_one_step','reports/control_attribution_development.json',160)]
    for label,name,count in definitions:
        batch=read_json(root/name);inputs[name]=sha256_file(root/name)
        if (batch['state']!='complete' or batch['completed_episodes']!=count
                or batch['expected_episodes']!=count or len(batch['episodes'])!=count):
            raise ValueError('Incomplete fixed development workload')
        if label=='augmented_shared':
            expected={('direct',model,None,h,e) for model in ('mlp_dt','node_generic','power_correct')
                      for h in (40_000_000,5_000_000) for e in range(10)}
        else:
            expected={(a,model,mode,h,e) for a in ('direct','lagged')
                      for model,mode in [('power_correct','low'),('power_correct','direct'),
                                         ('power_correct','lagged'),('node_structured',None)]
                      for h in (40_000_000,5_000_000) for e in range(10)}
        seen=set()
        for entry in batch['episodes']:
            path=safe_path(root,entry['manifest'])
            if sha256_file(path)!=entry['manifest_sha256']:raise ValueError('Changed saved development manifest')
            record=read_json(path);verify_run(root,record,check_dependencies=False)
            summary=read_json(path.parent/'summary.json')
            if canonical_json(summary)!=canonical_json(entry['summary']):raise ValueError('Changed development summary')
            h=summary['model_partition_ns'][0]
            coordinate=(summary['actuator'],summary['model_id'],summary['power_mode'],h,summary['episode_id'])
            if (coordinate not in expected or coordinate in seen or summary['seed']!=9000
                    or summary['phase']!='exploratory' or summary['condition']!='clean'
                    or summary['model_partition_ns']!=[h]*(40_000_000//h)
                    or summary['actuator_interval_ns']!=40_000_000 or summary['physics_step_ns']!=250_000):
                raise ValueError('Unexpected development phase, coordinate or clock')
            seen.add(coordinate)
            with np.load(path.parent/'true_trajectories.npz',allow_pickle=False) as data:
                anchors=data['anchors'].copy();target=data['target'].copy()
            decisions=[json.loads(line) for line in (path.parent/'decisions.jsonl').read_text().splitlines()]
            segments=[json.loads(line) for line in (path.parent/'action_segments.jsonl').read_text().splitlines()]
            gap=first_interval_metrics(decisions,anchors,segments,target)
            rows.append(dict(cohort=label,manifest=entry['manifest'],manifest_sha256=entry['manifest_sha256'],
                model_id=summary['model_id'],actuator=summary['actuator'],power_mode=summary['power_mode'],
                model_step_ns=h,episode_id=summary['episode_id'],**gap))
            key=(label,*coordinate[:-1]);groups.setdefault(key,[]).append(gap)
            dependencies[entry['manifest']]=dict(artifact_sha256=entry['manifest_sha256'],
                semantic_dependency_hashes=record['job_inputs']['dependencies'])
        if seen!=expected:raise ValueError('Missing original development tasks')
    aggregates=[]
    for key,values in sorted(groups.items(),key=lambda item:tuple(str(v) for v in item[0])):
        label,a,model,mode,h=key
        def mean(name):
            observed=[v[name] for v in values if v[name] is not None]
            return math.fsum(observed)/len(observed) if observed else None
        aggregates.append(dict(cohort=label,actuator=a,model_id=model,power_mode=mode,model_step_ns=h,
            n_training_seeds=1,n_episodes=len(values),n_decisions=sum(v['n_decisions'] for v in values),
            n_predictions=sum(v['n_predictions'] for v in values),fallback_count=sum(v['fallback_count'] for v in values),
            mean_episode_q_prediction_rmse_rad=mean('q_prediction_rmse_rad'),
            mean_episode_v_prediction_rmse_rad_s=mean('v_prediction_rmse_rad_s'),
            mean_absolute_first_interval_cost_gap=mean('mean_absolute_cost_gap'),
            mean_real_minus_predicted_first_interval_cost=mean('mean_real_minus_predicted_cost')))
    result=dict(schema_version=1,scope='Descriptive development audit at training seed9000; no confirmation or causal ranking',
        input_receipts=inputs,evaluator_hashes={name:sha256_file(root/name) for name in
            ('poc/control_gap.py','reports/audit_control_prediction_gap.py')},
        n_episodes=220,n_decisions=sum(row['n_decisions'] for row in rows),groups=aggregates,episodes=rows,
        provenance=dependencies,statistical_look=None,scientific_decision='not_evaluated',
        limitation='Each policy has its own selected commands and observed states; these errors do not isolate a cause of the between-policy control-cost difference')
    atomic_json(root/'reports/control_prediction_gap_development.json',result)
    print(json.dumps(dict(n_episodes=220,n_decisions=result['n_decisions'],groups=aggregates),allow_nan=False))
    return result


if __name__=='__main__':audit()
