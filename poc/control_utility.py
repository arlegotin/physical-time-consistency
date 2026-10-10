"""Registered signed control differences with training-seed cluster uncertainty."""
from __future__ import annotations

import math
import json
import hashlib
import re
import argparse
from fractions import Fraction
from pathlib import Path
import numpy as np

from .control_confirmation import ACTUATORS, MODELS, STEPS, validate_design
from .statistics import contrast_bounds, look_alpha
from .io import atomic_json, canonical_json, job_lock, read_json, safe_path, sha256_file, utc_now


def audit_raw_episode(folder: Path, summary: dict, *, expected_tick_ns=250_000) -> dict:
    """Independent scalar cost sums; the caller verifies the complete artifact manifest."""
    folder=Path(folder)
    tick=summary['physics_step_ns']
    if expected_tick_ns not in (250_000,125_000) or tick != expected_tick_ns:
        raise ValueError('Raw audit requires its explicitly registered physics tick')
    dimension=4 if summary['actuator']=='direct' else 6
    with np.load(folder/'true_trajectories.npz',allow_pickle=False) as data:
        times,states,anchors,target=(data[k] for k in ('time_ns','state','anchors','target'))
        expected=np.arange(0,1_600_000_001,tick,dtype=np.int64)
        stride=40_000_000//tick
        if (not np.array_equal(times,expected) or states.shape!=(len(expected),dimension)
                or not np.isfinite(states).all() or anchors.shape!=(40,dimension)
                or not np.array_equal(anchors,states[stride::stride])
                or not np.array_equal(target,summary['target'])
                or not np.array_equal(states[0],summary['initial_state'])
                or not np.array_equal(data['anchors_with_initial'],np.vstack([states[0],anchors]))):
            raise ValueError('Incomplete physical truth clocks, initial state or anchors')
        position=.04*math.fsum(float((x-y)**2) for row in anchors[:,:2] for x,y in zip(row,target))
        velocity=.04*.05*math.fsum(float(x*x) for row in anchors[:,2:4] for x in row)
        terminal=5*math.fsum(float((x-y)**2) for x,y in zip(anchors[-1,:2],target))
        final=times>=1_400_000_000
        success=bool(np.all(np.linalg.norm(states[final,:2]-target,axis=1)<.1)
                     and np.all(np.linalg.norm(states[final,2:4],axis=1)<.2))
        observation_times=np.arange(0,1_600_000_000,40_000_000,dtype=np.int64)
        with np.load(folder/'observations.npz',allow_pickle=False) as observations:
            if (not np.array_equal(observations['time_ns'],observation_times)
                    or not np.array_equal(observations['state'],states[::stride][:-1])
                    or not np.array_equal(observations['packet_id'],[f'obs-{t}' for t in observation_times])
                    or observations['noise_time_ns'].size):
                raise ValueError('Changed clean observations or extra fresh measurements')
    segments=[json.loads(line) for line in (folder/'action_segments.jsonl').read_text().splitlines()]
    cursor=0;terms=[]
    for segment in segments:
        command=np.asarray(segment['command'])
        if (segment['start_ns']!=cursor or segment['duration_ns']!=40_000_000
                or command.shape!=(2,) or not np.isfinite(command).all() or np.any(np.abs(command)>2)):
            raise ValueError('Changed actual held command path')
        cursor+=segment['duration_ns']
        terms.append(.04*math.fsum(float(x*x) for x in command))
    if cursor!=1_600_000_000:
        raise ValueError('Incomplete executed command path')
    decisions=[json.loads(line) for line in (folder/'decisions.jsonl').read_text().splitlines()]
    if (len(decisions)!=40 or any(d['time_ns']!=int(t) or d['observation_packet_id']!=f'obs-{t}'
            or d['status'] not in ('ok','all_candidates_failed') for d,t in zip(decisions,observation_times))):
        raise ValueError('Incomplete real decisions or observation packet identities')
    failures=sum(d['status']=='all_candidates_failed' for d in decisions)
    cost=math.fsum([position,velocity,.001*math.fsum(terms),terminal])
    if (not math.isclose(cost,summary['realized_cost'],rel_tol=1e-12,abs_tol=1e-12)
            or success != summary['success'] or failures != summary['fallback_count']):
        raise ValueError('Raw cost, success or fallback does not reconcile with its measured summary')
    return dict(raw_cost=cost,success=success,fallback_count=failures,n_real_anchors=40,n_observations=40)


def seed_costs(episodes: list[dict], design: dict) -> dict:
    validate_design(design)
    seeds, ids = design['initial_training_seed_ids'], design['initial_episode_ids']
    expected = {(a,m,h,s,e) for a in ACTUATORS for m in MODELS for h in STEPS for s in seeds for e in ids}
    indexed = {}
    for row in episodes:
        h = row['model_partition_ns'][0] if row['model_partition_ns'] else None
        key = row['actuator'], row['model_id'], h, row['seed'], row['episode_id']
        if key not in expected or key in indexed:
            raise ValueError('Unexpected or duplicated registered control episode')
        if (row['phase'] != 'confirmatory' or row['complete'] is not True or row['condition'] != 'clean'
                or row['power_mode'] is not None or row['model_partition_ns'] != [h]*(40_000_000//h)
                or row['actuator_interval_ns'] != 40_000_000 or row['physics_step_ns'] != 250_000
                or not math.isfinite(row['realized_cost']) or row['realized_cost'] < 0
                or type(row['fallback_count']) is not int or not 0 <= row['fallback_count'] <= 40
                or type(row['success']) is not bool):
            raise ValueError('Incomplete, nonfinite or changed registered episode')
        # Independently regenerate only known inputs, never future plant states.
        s,e = row['seed'],row['episode_id']
        def rng(channel):
            return np.random.Generator(np.random.PCG64(np.random.SeedSequence([6101,s,e,channel])))
        initial=np.r_[rng(0).uniform(-.5,.5,2),np.zeros(2 if row['actuator']=='direct' else 4)]
        if (not np.array_equal(row['initial_state'],initial) or not np.array_equal(row['target'],rng(1).uniform(-1,1,2))
                or row['seed_namespaces'] != {'cem':[6101,s,e,2],'disturbance':[6101,s,e//20,4]}):
            raise ValueError('Changed paired task or random-number namespace')
        indexed[key]=row
    if set(indexed) != expected:
        raise ValueError('Missing registered paired control episodes')
    rows=[]
    for a in ACTUATORS:
        for m in MODELS:
            for h in STEPS:
                clusters=[[indexed[(a,m,h,s,e)] for e in ids] for s in seeds]
                rows.append(dict(actuator=a,model_id=m,model_step_ns=h,
                    cost_by_seed=[math.fsum(r['realized_cost'] for r in cluster)/len(ids) for cluster in clusters],
                    fallback_by_seed=[sum(r['fallback_count'] for r in cluster) for cluster in clusters],
                    success_by_seed=[sum(r['success'] for r in cluster) for cluster in clusters]))
    return dict(seed_ids=seeds,episodes_per_seed=len(ids),n_episodes=len(episodes),rows=rows)


def utility_bounds(costs: dict, design: dict, look_index: int, numerical_envelopes: dict) -> dict:
    validate_design(design)
    if (costs['seed_ids'] != design['initial_training_seed_ids']
            or costs['episodes_per_seed'] != len(design['initial_episode_ids'])
            or costs['n_episodes'] != 1200):
        raise ValueError('Changed independent control cluster design')
    expected={(a,m,h) for a in ACTUATORS for m in MODELS for h in STEPS}
    indexed={(r['actuator'],r['model_id'],r['model_step_ns']):r for r in costs['rows']}
    if len(costs['rows']) != 12 or set(indexed) != expected:
        raise ValueError('Incomplete seed-level control costs')
    n=len(costs['seed_ids'])
    for row in costs['rows']:
        values=np.asarray(row['cost_by_seed'],dtype=np.float64)
        if values.shape!=(n,) or not np.isfinite(values).all() or np.any(values<0):
            raise ValueError('Invalid independent seed-level costs')
    ids={r['criterion_id'] for r in design['criteria']}
    if set(numerical_envelopes) != ids or any(set(e)!= {'lower','upper'} for e in numerical_envelopes.values()):
        raise ValueError('Complete measured numerical envelopes required; no zero default')
    alpha=look_alpha(look_index)
    tail=alpha/16
    indices=np.random.Generator(np.random.PCG64(np.random.SeedSequence([9102,look_index]))).integers(
        0,n,size=(20_000,n),dtype=np.int64)
    results=[]
    for criterion in design['criteria']:
        a,h,m=criterion['actuator'],criterion['model_step_ns'],criterion['competitor']
        baseline=np.asarray(indexed[(a,m,h)]['cost_by_seed'])
        method=np.asarray(indexed[(a,'power_correct',h)]['cost_by_seed'])
        difference=baseline-method
        envelope=numerical_envelopes[criterion['criterion_id']]
        bounds=contrast_bounds(difference,tail,indices,envelope)
        status='unresolved'
        if bounds['status']=='unresolved_variance':
            status='unresolved_variance'
        elif bounds['influence']['unstable']:
            status='unresolved_influence'
        else:
            central=bounds['central_bootstrap']
            if bounds['lower']>0 and central[0]-envelope['lower']>0:
                status='lower_cost'
            elif bounds['upper']<0 and central[1]+envelope['upper']<0:
                status='higher_cost'
        bmean,mmean=float(baseline.mean()),float(method.mean())
        results.append(dict(criterion_id=criterion['criterion_id'],actuator=a,competitor=m,model_step_ns=h,
            n_training_seeds=n,episodes_per_seed=costs['episodes_per_seed'],status=status,
            baseline_mean_cost=bmean,method_mean_cost=mmean,point_difference=float(difference.mean()),
            relative_reduction=1-mmean/bmean if bmean>0 else None,
            difference_by_seed=difference.tolist(),bounds=bounds,
            fallback_by_seed=indexed[(a,'power_correct',h)]['fallback_by_seed']))
    return dict(protocol_id=design['protocol_id'],look_index=look_index,look_alpha=alpha,
                tail_alpha=tail,family_size=8,bootstrap_seed_entropy=[9102,look_index],criteria=results,
                scope=design['scope'],population_assumption='Approximately regular independent training-seed contrasts')


def reserve_control_look(root: Path, design: dict, evidence_identity: dict) -> dict:
    """Reserve the new eight-comparison look; the historical PTCO ledger stays intact."""
    validate_design(design)
    if (set(evidence_identity)!= {'registry_sha256','batch_sha256','numerical_sha256'}
            or any(not isinstance(v,str) or not re.fullmatch('[0-9a-f]{64}',v) for v in evidence_identity.values())):
        raise ValueError('Complete frozen control evidence identity required')
    identity_hash=hashlib.sha256(canonical_json(evidence_identity)).hexdigest()
    design_hash=hashlib.sha256(canonical_json(design)).hexdigest()
    folder=Path(root)/'decisions/control_utility_looks'
    key=hashlib.sha256(b'UTILITY-2026-10-02-control-v1 look ledger').hexdigest()
    with job_lock(key,Path(root)):
        previous=[read_json(path) for path in sorted(folder.glob('look-*.json'))]
        for index,record in enumerate(previous,1):
            expected=dict(protocol_id=design['protocol_id'],family_size=8,look_index=index,
                alpha=look_alpha(index),tail_alpha=look_alpha(index)/16,design_sha256=design_hash,
                cumulative_alpha=float(Fraction(1,20)*(1-Fraction(1,2)**index)),
                bootstrap_seed_entropy=[9102,index])
            if (any(record.get(k)!=v for k,v in expected.items())
                    or record['evidence_sha256']!=hashlib.sha256(canonical_json(record['evidence_identity'])).hexdigest()):
                raise ValueError('Control statistical-look ledger is inconsistent')
        matching=[r for r in previous if r['evidence_sha256']==identity_hash]
        if matching:
            if len(matching)!=1:
                raise ValueError('Control statistical-look ledger contains duplicate evidence')
            return matching[0]
        index=len(previous)+1
        record=dict(schema_version=1,protocol_id=design['protocol_id'],family_size=8,
            look_index=index,alpha=look_alpha(index),tail_alpha=look_alpha(index)/16,
            cumulative_alpha=float(Fraction(1,20)*(1-Fraction(1,2)**index)),design_sha256=design_hash,
            evidence_identity=evidence_identity,evidence_sha256=identity_hash,
            bootstrap_seed_entropy=[9102,index],created_utc=utc_now())
        atomic_json(folder/f'look-{index:05d}.json',record)
        return record


def audit_batch(root: Path) -> dict:
    """Close the complete frozen primary batch before any control sign is interpreted."""
    root=Path(root).resolve()
    receipt_path=root/'reports/control_confirmation_batch.json';receipt=read_json(receipt_path)
    if (receipt.get('state')!='complete' or receipt.get('expected_episodes')!=1200
            or receipt.get('completed_episodes')!=1200 or len(receipt.get('episodes',[]))!=1200):
        raise ValueError('Complete paired primary control batch required before scientific audit')
    design=read_json(root/'config/control_confirmation_design.json')
    registry_path=root/'config/control_confirmation_cells.json'
    if receipt['registry_sha256']!=sha256_file(registry_path):
        raise ValueError('Primary batch differs from its frozen registry')
    from .control_confirmation import verify_freeze
    from .control_confirmation_batch import episode_receipt
    freeze=read_json(registry_path);verify_freeze(root,design,freeze)
    jobs={job['cell_id']:job for job in freeze['jobs']}
    seen=set();rows=[];audited=[]
    for entry in receipt['episodes']:
        cell=entry['cell_id']
        if cell not in jobs or cell in seen:
            raise ValueError('Missing or duplicated primary audit cell')
        seen.add(cell)
        manifest=safe_path(root,entry['manifest'])
        if sha256_file(manifest)!=entry['manifest_sha256']:
            raise ValueError('Changed primary episode manifest')
        observed=episode_receipt(root,jobs[cell],manifest)
        if any(entry.get(k)!=v for k,v in observed.items()):
            raise ValueError('Primary receipt differs from its completed observation')
        raw=audit_raw_episode(manifest.parent,observed['summary'])
        audited.append(dict(cell_id=cell,manifest=entry['manifest'],manifest_sha256=entry['manifest_sha256'],**raw))
        rows.append(observed['summary'])
    if seen!=set(jobs):
        raise ValueError('Incomplete primary raw audit coverage')
    costs=seed_costs(rows,design)
    result=dict(schema_version=1,state='complete',protocol_id=design['protocol_id'],
        recorded_utc=utc_now(),registry_sha256=sha256_file(registry_path),
        batch_sha256=sha256_file(receipt_path),design_sha256=sha256_file(root/'config/control_confirmation_design.json'),
        evaluator_sha256=sha256_file(root/'poc/control_utility.py'),audited_episodes=audited,seed_costs=costs,
        scientific_decision='pending_numerical_reference_confirmation_and_registered_statistical_look')
    atomic_json(root/'reports/control_confirmation_raw_audit.json',result)
    return result


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--audit-only',action='store_true',required=True,
                        help='Recompute complete raw primary costs; do not assign scientific signs')
    args=parser.parse_args(argv)
    result=audit_batch(args.root)
    print(json.dumps(dict(raw_episode_count=len(result['audited_episodes']),
                          scientific_decision=result['scientific_decision'])))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
