"""Frozen prediction, inference and paired-control comparison of affine flow."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

for _variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_variable] = '1'

import numpy as np
import torch

from .affine_experiment import (ACTUATORS, FREEZE as NOMINAL_FREEZE, PROTOCOL, SEED,
                                build_candidate, load_freeze, verify_bindings)
from .io import atomic_bytes, atomic_json, job_key, job_lock, read_json, safe_path, sha256_file, utc_now
from .models.common import Scales

FREEZE = 'config/affine_measurement_freeze.json'
MODELS = ('affine_flow', 'power_correct', 'mlp_h', 'node_generic', 'node_structured', 'physics_fit')


def summarize_prediction(raw, coordinates, actor):
    from .shared_state_diagnostic import decompose
    for key in ('probe_ids', 'strata', 'horizons_ns'):
        if not np.array_equal(raw[key], coordinates[key]):
            raise ValueError('Prediction coordinate coverage mismatch')
    shape = (len(raw['probe_ids']), len(raw['horizons_ns']), 4 if actor == 'direct' else 6)
    fields = ('truth_zero', 'truth_command', 'model_zero', 'model_command', 'model_composed')
    if any(raw[k].shape != shape or not np.isfinite(raw[k]).all() for k in fields):
        raise ValueError('Prediction coordinate shape/nonfinite mismatch')
    errors = decompose(raw['model_zero'], raw['model_command'], raw['truth_zero'],
                       raw['truth_command'], raw['model_composed'])
    rows = []
    for stratum in ('cold', 'warm'):
        mask = raw['strata'] == stratum
        if not mask.any():
            raise ValueError('Prediction coordinate missing stratum')
        for hi, horizon in enumerate(raw['horizons_ns']):
            for component, indices in (('q', slice(0,2)), ('v', slice(2,4))):
                row = dict(actuator=actor, stratum=stratum, h_ns=int(horizon), component=component,
                           model_id='affine_flow', n_probes=int(mask.sum()))
                for metric, values in errors.items():
                    norms = np.linalg.norm(values[mask, hi, indices], axis=-1)
                    row.update({metric+'_mean':float(norms.mean()), metric+'_median':float(np.median(norms)),
                                metric+'_p90':float(np.quantile(norms,.9,method='linear'))})
                rows.append(row)
    return rows


def validate_prediction_result(raw_by_actor, summary, freeze):
    if set(raw_by_actor) != set(ACTUATORS):
        raise ValueError('Prediction reaggregation missing actuator')
    rows, count = [], 0
    for actor in ACTUATORS:
        raw = raw_by_actor[actor]
        rows.extend(summarize_prediction(raw, freeze['coordinates'][actor], actor))
        count += len(raw['probe_ids'])*len(raw['horizons_ns'])
    if (count != freeze['expected_new_prediction_rows'] or summary['new_model_rows'] != count
            or summary['rows'] != rows):
        raise ValueError('Prediction raw-to-summary reaggregation mismatch')
    for row in summary.get('profiles', []):
        samples = np.asarray(row['samples_ns'], dtype=np.float64)
        if (len(samples) != (500 if row['batch_size'] == 1 else 100) or not np.isfinite(samples).all()
                or np.any(samples <= 0) or row['median_ns'] != float(np.median(samples))
                or row['p95_ns'] != float(np.quantile(samples,.95,method='linear'))):
            raise ValueError('Inference timing raw-to-summary reaggregation mismatch')


def _coordinate(row):
    return (row['actuator'], row['model_id'], row['actuator_interval_ns'], row['model_partition_ns'][0], row['episode_id'])


def summarize_control(rows, expected):
    ids = [_coordinate(row) for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(map(tuple,expected)):
        raise ValueError('Control coordinate coverage mismatch')
    groups = {}
    for row in rows:
        if (row['complete'] is not True or row['phase'] != 'exploratory' or row['seed'] != SEED
                or not all(math.isfinite(row[k]) and row[k] >= 0 for k in
                           ('realized_cost','planner_latency_median_seconds','planner_latency_p95_seconds',
                            'q_rmse_rad','terminal_q_error_rad','neural_rows','fallback_count'))):
            raise ValueError('Invalid/incomplete control episode')
        groups.setdefault(_coordinate(row)[:4], []).append(row)
    conditions, comparisons = [], []
    for key, episodes in groups.items():
        conditions.append(dict(actuator=key[0], model_id=key[1], actuator_interval_ns=key[2],
              nominal_model_partition_ns=key[3], actual_max_query_ns=min(key[2],key[3]), n_episodes=len(episodes),
              mean_cost=float(np.mean([e['realized_cost'] for e in episodes])),
              mean_episode_median_planner_seconds=float(np.mean([e['planner_latency_median_seconds'] for e in episodes])),
              mean_q_rmse_rad=float(np.mean([e['q_rmse_rad'] for e in episodes])),
              successes=sum(e['success'] for e in episodes), fallback_count=sum(e['fallback_count'] for e in episodes)))
        if key[1] != 'affine_flow':
            continue
        method = {e['episode_id']:e for e in episodes}
        for other, baseline_rows in groups.items():
            if (other[0],other[2],other[3]) != (key[0],key[2],key[3]) or other[1] == 'affine_flow':
                continue
            baseline = {e['episode_id']:e for e in baseline_rows}
            if method.keys() != baseline.keys() or any(
                method[t]['initial_state'] != baseline[t]['initial_state'] or method[t]['target'] != baseline[t]['target']
                for t in method):
                raise ValueError('Different paired task inputs')
            tasks = sorted(method)
            m, b = (np.array([group[t]['realized_cost'] for t in tasks]) for group in (method,baseline))
            mt, bt = (np.array([group[t]['planner_latency_median_seconds'] for t in tasks]) for group in (method,baseline))
            comparisons.append(dict(actuator=key[0], competitor=other[1], actuator_interval_ns=key[2],
                nominal_model_partition_ns=key[3], n_paired_tasks=len(tasks),
                cost_reduction=None if b.mean() <= 0 else float(1-m.mean()/b.mean()),
                planning_time_reduction=None if bt.mean() <= 0 else float(1-mt.mean()/bt.mean()),
                paired_cost_differences=(b-m).tolist(), method_lower_cost_tasks=int((m < b).sum()),
                method_fallback_count=sum(e['fallback_count'] for e in episodes)))
    return dict(conditions=conditions, comparisons=comparisons)


def recompute_episode_cost(folder):
    from .action_path import ActionSegment
    from .control import anchor_components
    folder = Path(folder)
    segments = [ActionSegment(r['start_ns'],r['duration_ns'],np.array(r['command'])) for r in
                map(json.loads,(folder/'action_segments.jsonl').read_text().splitlines())]
    with np.load(folder/'true_trajectories.npz',allow_pickle=False) as arrays:
        cost = anchor_components(arrays['anchors'], arrays['target'], segments)['total']
    if not math.isclose(cost, read_json(folder/'summary.json')['realized_cost'],rel_tol=1e-12,abs_tol=1e-12):
        raise ValueError('Episode cost differs from raw truth/commands')
    return cost


def _sources():
    from .control_jobs import SOURCES
    from .training_jobs import SOURCES as TRAIN_SOURCES
    return sorted(set(SOURCES)|set(TRAIN_SOURCES)|{'poc/affine_flow.py','poc/affine_experiment.py',
            'poc/affine_measurement.py','poc/shared_state_diagnostic.py','poc/efficient_power.py',
            'poc/efficiency_result.py','poc/efficiency_pilot.py','poc/efficiency_episode.py','poc/control_profile.py'})


def load_candidate(root, actor, selected):
    from .contracts import verify_run
    from .train import canonical_tensor_hash
    path = safe_path(root,selected['model_manifest'])
    verify_bindings(root,{selected['model_manifest']:selected['manifest_sha256'],
                         str((path.parent/'weights.pt').relative_to(root)):selected['checkpoint_sha256']})
    record = read_json(path)
    verify_run(root,record)
    summary = read_json(path.parent/'summary.json')
    if (summary['protocol_id'] != PROTOCOL or summary['actuator'] != actor or summary['model_id'] != 'affine_flow'
            or summary['seed'] != SEED or summary['qualification']['qualified'] is not True):
        raise ValueError('Candidate fit identity/qualification mismatch')
    normalization = read_json(path.parent/'normalization.json')
    scales = Scales(*(torch.tensor(normalization[k],dtype=torch.float32) for k in
                     ('scale_q','scale_v','velocity_input_rms')))
    model = build_candidate(actor,scales,SEED)
    weights = torch.load(path.parent/'weights.pt',weights_only=True,map_location='cpu')
    if canonical_tensor_hash(weights) != record['model_hashes']['canonical_tensor_sha256']:
        raise ValueError('Candidate canonical tensor hash changed')
    model.load_state_dict(weights,strict=True)
    return model.double().eval()


def prepare(root):
    from .contracts import verify_run
    root = Path(root).resolve()
    if (root/FREEZE).exists():
        return measurement_freeze(root)
    nominal = load_freeze(root)
    inputs = {NOMINAL_FREEZE:sha256_file(root/NOMINAL_FREEZE)}
    selections = {}
    for actor in ACTUATORS:
        pointer_path = f'reports/affine_refinement_{actor}_selection.json'
        selected = read_json(root/pointer_path)
        load_candidate(root,actor,selected)  # Restoration/identity only; no prediction query.
        selections[actor] = selected
        path = root/selected['model_manifest']
        for name in (pointer_path,selected['model_manifest'],str((path.parent/'weights.pt').relative_to(root)),
                     str((path.parent/'normalization.json').relative_to(root))):
            inputs[name] = sha256_file(root/name)
    diagnostic_pointer = read_json(root/'reports/shared_state_diagnostic_run.json')
    diagnostic_path = root/diagnostic_pointer['manifest_path']
    verify_bindings(root,{diagnostic_pointer['manifest_path']:diagnostic_pointer['manifest_sha256']})
    verify_run(root,read_json(diagnostic_path))
    for name in ('reports/shared_state_diagnostic_run.json',diagnostic_pointer['manifest_path'],
                 'config/shared_state_diagnostic_freeze.json','reports/shared_state_diagnostic_result.json',
                 'config/efficiency_pilot_cells.json','reports/efficiency_pilot_batch.json',
                 'reports/efficiency_pilot_result.json','config/efficiency_pilot_freeze.json',
                 'config/planner.json','config/numerical.json'):
        inputs[name] = sha256_file(root/name)
    for actor in ACTUATORS:
        for stem in ('measurements','references'):
            name = str((diagnostic_path.parent/f'{actor}_{stem}.npz').relative_to(root))
            inputs[name] = sha256_file(root/name)
    pilot = read_json(root/'config/efficiency_pilot_cells.json')
    request = read_json(root/pilot['jobs'][0]['path'])
    settings = request['planner_settings']
    if settings != read_json(root/'config/planner.json')['settings']:
        raise ValueError('Original qualified pilot planner differs')
    jobs = []
    for actor in ACTUATORS:
        for rate in (40_000_000,10_000_000):
            for step in (40_000_000,5_000_000):
                for task in range(10):
                    cell = f'{actor}__affine_flow__rate{rate}__step{step}__task{task}'
                    jobs.append(dict(cell_id=cell,actuator=actor,actuator_interval_ns=rate,
                       model_partition_ns=[step]*(40_000_000//step),episode_id=task,
                       path='config/affine_control_jobs/'+cell+'.json'))
    prior_coordinates = read_json(root/'config/shared_state_diagnostic_freeze.json')['coordinates']
    coordinates={actor:{k:prior_coordinates[actor][k] for k in ('probe_ids','strata')} for actor in ACTUATORS}
    for actor in ACTUATORS:
        coordinates[actor]['horizons_ns']=[5_000_000,10_000_000,40_000_000]
    freeze = dict(schema_version=1, protocol_id=PROTOCOL, phase='exploratory', seed=SEED,
                  created_utc=utc_now(), selections=selections, baseline_selections=nominal['comparators'],
                  diagnostic_manifest=diagnostic_pointer['manifest_path'], models=list(MODELS),
                  planner_settings=settings, physics_step_ns=250_000, jobs=jobs,
                  expected_candidate_episodes=80, expected_new_prediction_rows=768,
                  source_hashes={name:sha256_file(root/name) for name in _sources()}, input_hashes=inputs,
                  nominal_freeze_sha256=sha256_file(root/NOMINAL_FREEZE), coordinates=coordinates, alpha_spent=0)
    atomic_json(root/FREEZE,freeze)
    for job in jobs:
        atomic_json(root/job['path'],{**job,'protocol_id':PROTOCOL,'freeze_sha256':sha256_file(root/FREEZE),
                                     'seed':SEED,'model_id':'affine_flow','selected':selections[job['actuator']]})
    return freeze


def measurement_freeze(root):
    freeze = read_json(root/FREEZE)
    if (freeze['protocol_id'] != PROTOCOL or freeze['seed'] != SEED or freeze['phase'] != 'exploratory'
            or freeze['models'] != list(MODELS) or len(freeze['jobs']) != 80
            or freeze['expected_new_prediction_rows'] != 768 or freeze['expected_candidate_episodes'] != 80):
        raise ValueError('Changed affine measurement design')
    verify_bindings(root,{**freeze['source_hashes'],**freeze['input_hashes']})
    return freeze


def _baseline(root, actor, model_id, freeze):
    from .control_models import load_selected
    from .efficient_power import EfficientPowerHead
    selections = freeze['baseline_selections']
    if model_id == 'physics_fit':
        selections = read_json(root/'config/efficiency_pilot_freeze.json')['selections']
    item = selections[actor+':'+model_id]
    model = load_selected(safe_path(root,item['model_manifest']),item['manifest_sha256'],item['checkpoint_sha256'],
                          model_id=model_id,actuator=actor,seed=SEED)
    return EfficientPowerHead.from_head(model) if model_id == 'power_correct' else model


def measure(root):
    from .execution import cpu_defaults, run_job
    from .shared_state_diagnostic import model_endpoints
    root = Path(root).resolve()
    freeze = measurement_freeze(root)
    cpu_defaults(SEED)
    files = sorted(set(freeze['source_hashes'])|set(freeze['input_hashes'])|{FREEZE})
    expected = ('affine.prediction_coverage','affine.reference_reuse','affine.latency')
    def produce(folder):
        rows, timings = [], []
        for actor in ACTUATORS:
            source = root/freeze['diagnostic_manifest']
            with np.load(source.parent/f'{actor}_measurements.npz',allow_pickle=False) as archive:
                old = {k:archive[k].copy() for k in archive.files}
            model = load_candidate(root,actor,freeze['selections'][actor])
            raw = {k:old[k] for k in ('probe_ids','strata','horizons_ns','truth_zero','truth_command')}
            raw.update({k:np.empty_like(old['truth_command']) for k in ('model_zero','model_command','model_composed')})
            for hi,horizon in enumerate(raw['horizons_ns']):
                endpoints = model_endpoints(model,old['states'],old['commands'],int(horizon))
                for key,value in endpoints.items():
                    raw['model_'+key][:,hi] = value
            coordinates = freeze['coordinates'][actor]
            if len(raw['probe_ids']) != 128:
                raise ValueError('Expected all 128 frozen probes')
            rows.extend(summarize_prediction(raw,coordinates,actor))
            np.savez_compressed(folder/f'{actor}_predictions.npz',**raw)
            from .control_episode import episode_conditions
            start_state = torch.tensor(episode_conditions(SEED,0,actor,'exploratory')['initial_state'],dtype=torch.float64)[None]
            command = torch.tensor([[.7,-1.1]],dtype=torch.float64)
            for name in MODELS:
                predictor = model if name == 'affine_flow' else _baseline(root,actor,name,freeze)
                with torch.no_grad():
                    for batch in (1,128):
                        x,u = start_state.repeat(batch,1),command.repeat(batch,1)
                        for h in (.005,.01,.04):
                            for _ in range(50): predictor.predict(x,u,h)
                            samples=[]
                            for _ in range(500 if batch == 1 else 100):
                                started=time.perf_counter_ns(); output=predictor.predict(x,u,h)
                                samples.append(time.perf_counter_ns()-started)
                                if any(s != 'ok' for s in output.status):
                                    raise ValueError('Failed inference cannot be dropped from profile')
                            timings.append(dict(actuator=actor,model_id=name,batch_size=batch,h_s=h,
                                samples_ns=samples,median_ns=float(np.median(samples)),
                                p95_ns=float(np.quantile(samples,.95,method='linear')),
                                network_dispatches=output.network_calls,vector_field_calls=output.vector_field_calls,
                                neural_rows=batch*(2 if name in {'affine_flow'} or name=='power_correct' and actor=='lagged'
                                                  else 1 if name in {'power_correct','mlp_h'} else 4 if name.startswith('node_') else 0)))
                print(actor+' '+name+' prediction/profile complete',flush=True)
        atomic_json(folder/'profiles.json',dict(rows=timings,warmup=50,repeats_by_batch={'1':500,'128':100}))
        summary=dict(protocol_id=PROTOCOL,phase='exploratory',new_model_rows=768,rows=rows,profiles=timings,
                     missing_cells=[],alpha_spent=0,scope='One development seed; no population or control conclusion')
        checks=[dict(check_id=k,status='pass',scope=PROTOCOL,tolerance=None,
                     reason='Complete frozen probe rows, hash-verified qualified references, all timing samples retained') for k in expected]
        verify_bindings(root,{**freeze['source_hashes'],**freeze['input_hashes']})
        return checks,summary
    record=run_job(root,'M40','exploratory',files,expected,[],produce,config_keys={FREEZE:freeze})
    if record['completion_status'] != 'complete':
        raise ValueError('Affine measurement failed: '+read_json(root/f'evidence/M40/{record["run_id"]}/summary.json').get('failure','unknown'))
    pointer=dict(path=f'evidence/M40/{record["run_id"]}/manifest.json',sha256=sha256_file(root/f'evidence/M40/{record["run_id"]}/manifest.json'))
    atomic_json(root/'reports/affine_prediction_measurement.json',pointer)
    return pointer


def verify_prediction(root):
    from .contracts import verify_run
    root=Path(root).resolve(); freeze=measurement_freeze(root)
    pointer=read_json(root/'reports/affine_prediction_measurement.json')
    verify_bindings(root,{pointer['path']:pointer['sha256']})
    path=root/pointer['path']; verify_run(root,read_json(path))
    raw={}
    for actor in ACTUATORS:
        with np.load(path.parent/f'{actor}_predictions.npz',allow_pickle=False) as archive:
            raw[actor]={k:archive[k].copy() for k in archive.files}
    summary=read_json(path.parent/'summary.json')
    validate_prediction_result(raw,summary,freeze)
    result={**summary,'evidence':pointer,'measurement_freeze_sha256':sha256_file(root/FREEZE)}
    atomic_json(root/'reports/affine_prediction_result.json',result)
    return result


def run_episode(root, job, freeze):
    from .control_episode import execute_episode
    from .control_artifacts import save_episode
    from .envs import Plant
    from .execution import cpu_defaults, run_job
    request=read_json(root/job['path'])
    required={**job,'protocol_id':PROTOCOL,'freeze_sha256':sha256_file(root/FREEZE),'seed':SEED,
              'model_id':'affine_flow','selected':freeze['selections'][job['actuator']]}
    if request != required:
        raise ValueError('Changed affine episode request')
    cpu_defaults(SEED)
    files=sorted(set(freeze['source_hashes'])|set(freeze['input_hashes'])|{FREEZE,job['path']})
    expected=('affine.control_episode','affine.control_clocks','affine.control_provenance')
    def produce(folder):
        actor=job['actuator']; model=load_candidate(root,actor,freeze['selections'][actor])
        plant=Plant(actor,freeze['physics_step_ns'],root/'assets')
        atomic_json(folder/'request.json',request)
        result=execute_episode(plant,model,freeze['planner_settings'],seed=SEED,episode_id=job['episode_id'],
                               phase='exploratory',condition='clean',model_partition=tuple(job['model_partition_ns']),
                               actuator_interval_ns=job['actuator_interval_ns'])
        summary=save_episode(folder,result,model_id='affine_flow',
                    checkpoint_hash=freeze['selections'][actor]['checkpoint_sha256'],run_id=folder.name)
        summary.update(protocol_id=PROTOCOL,neural_rows=2*summary['work']['scalar_candidate_evaluations'],missing_cells=[])
        checks=[dict(check_id=k,status='pass',scope=job['cell_id'],tolerance=None,
                     reason='Full physical episode, common clocks, frozen candidate/controller') for k in expected]
        return checks,summary
    return run_job(root,'M40','exploratory',files,expected,[],produce,config_keys={job['path']:request,FREEZE:freeze})


def controls(root):
    from .contracts import verify_run
    root=Path(root).resolve(); freeze=measurement_freeze(root)
    path=root/'reports/affine_control_batch.json'
    batch=read_json(path) if path.exists() else dict(state='partial',freeze_sha256=sha256_file(root/FREEZE),completed={})
    if batch['freeze_sha256'] != sha256_file(root/FREEZE):
        raise ValueError('Control batch belongs to another freeze')
    with job_lock(job_key(dict(protocol=PROTOCOL,stage='control_batch',freeze_sha256=sha256_file(root/FREEZE))),root):
        for job in freeze['jobs']:
            if job['cell_id'] in batch['completed']:
                entry=batch['completed'][job['cell_id']]
                verify_bindings(root,{entry['path']:entry['sha256']}); verify_run(root,read_json(root/entry['path']))
                continue
            print(f'{len(batch["completed"])}/80 {job["cell_id"]}',flush=True)
            record=run_episode(root,job,freeze)
            if record['completion_status'] != 'complete':
                raise ValueError('Invalid candidate episode; retain attempt and repair before comparison')
            name=f'evidence/M40/{record["run_id"]}/manifest.json'
            batch['completed'][job['cell_id']]=dict(path=name,sha256=sha256_file(root/name))
            atomic_json(path,batch)
        batch['state']='complete'; atomic_json(path,batch)
    return batch


def report(root):
    from .contracts import verify_run
    from .efficiency_result import summarize as pilot_summary
    root=Path(root).resolve(); freeze=measurement_freeze(root)
    original=pilot_summary(root,read_json(root/'config/efficiency_pilot_cells.json'),
                    read_json(root/'reports/efficiency_pilot_batch.json'),read_json(root/'reports/efficiency_pilot_profiles.json'))
    rows=[row for row in original['episodes'] if row['model_id'] != 'sim_oracle']
    batch=read_json(root/'reports/affine_control_batch.json')
    if batch['state'] != 'complete' or set(batch['completed']) != {j['cell_id'] for j in freeze['jobs']}:
        raise ValueError('Control coverage incomplete')
    for job in freeze['jobs']:
        entry=batch['completed'][job['cell_id']]; path=root/entry['path']
        verify_bindings(root,{entry['path']:entry['sha256']}); verify_run(root,read_json(path))
        recompute_episode_cost(path.parent); rows.append(read_json(path.parent/'summary.json'))
    expected=[(a,m,r,s,e) for a in ACTUATORS for m in MODELS for r in (40_000_000,10_000_000)
              for s in (40_000_000,5_000_000) for e in range(10)]
    result=dict(protocol_id=PROTOCOL,status='complete_exploratory',alpha_spent=0,
                **summarize_control(rows,expected),episodes=rows,evidence=batch['completed'],
                limitation='One development training seed; new architecture; fresh independent confirmation needed for outcome-selected improvement.')
    atomic_json(root/'reports/affine_control_result.json',result)
    lines=['# Affine-flow refinement: paired practical comparison','',result['limitation'],'',
           '| Plant | Hold ms | Query partition ms | Baseline | Cost reduction | Planning time reduction | Paired wins | Fallbacks |',
           '|---|---:|---:|---|---:|---:|---:|---:|']
    for row in result['comparisons']:
        def percent(x):return 'unresolved' if x is None else f'{100*x:+.2f}%'
        lines.append(f'| {row["actuator"]} | {row["actuator_interval_ns"]/1e6:g} | {row["nominal_model_partition_ns"]/1e6:g} | {row["competitor"]} | {percent(row["cost_reduction"])} | {percent(row["planning_time_reduction"])} | {row["method_lower_cost_tasks"]}/10 | {row["method_fallback_count"]} |')
    atomic_bytes(root/'reports/affine_control_result.md',('\n'.join(lines)+'\n').encode())
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','measure','verify-prediction','controls','report'))
    parser.add_argument('--root',type=Path,default=Path.cwd())
    args=parser.parse_args(argv); root=args.root.resolve()
    value={'prepare':prepare,'measure':measure,'verify-prediction':verify_prediction,
           'controls':controls,'report':report}[args.command](root)
    print(dict(command=args.command,status=value.get('state',value.get('status','complete')),protocol_id=PROTOCOL),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
