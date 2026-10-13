"""Read-only raw evidence aggregation for the provisional PTCO-CAL-0.1 screen."""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import html
import io
import json
from pathlib import Path

import numpy as np

from .calibration_models import propagation_family
from .calibration_store import CalibrationStore, PROTOCOL_ID
from .calibration_predictions import (aggregate_prediction_rows, load_evaluation,
    load_prediction_truth, truth_cases, _selected)
from .calibration_decisions import decision_outcome, decision_state_ids, load_decision_truth
from .calibration_timing import load_timing_finalization, summarize_timing
from .io import atomic_bytes, atomic_json, canonical_json, read_json

COORDINATE = ('session_id', 'budget_s', 'model_id', 'initialization_id')


def _coordinates(matrix, recipe):
    from .calibration_study import _coordinates as membership
    return membership(recipe, matrix)


def expected_coordinates(matrix: str) -> list[dict]:
    """Prospective empirical fits, independent of available products (no nominal fits)."""
    from .calibration_study import prospective_recipe
    return _coordinates(matrix, prospective_recipe(matrix=matrix))


def _coordinate(row):
    return {k: row[k] for k in COORDINATE}


def _key(row):
    return canonical_json(_coordinate(row))


def _rows(folder, name):
    if name.endswith('.gz'):
        from .calibration_records import read_records
        return read_records(folder/name)
    return [json.loads(line) for line in (folder/name).read_text().splitlines()]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _equal(actual, expected, message):
    _require(canonical_json(actual) == canonical_json(expected), message)


def _reference(store, manifest, evaluation):
    """Reconstruct final reference/envelopes from immutable base and refinement arrays."""
    request = manifest['request']
    raw = load_prediction_truth(store, request['evaluation_receipt'], request['truth_receipt'])
    reference, envelope = raw['endpoint'].copy(), raw['envelope'].copy()
    previous = raw['refined_endpoint'][:, :, 1].copy()
    delta = np.abs(previous-raw['refined_endpoint'][:, :, 0])
    predecessors = {}
    resolution = np.full(reference.shape[:2], store.recipe['evaluation']['truth_step_ns'])
    for receipt in manifest['summary']['refinement_receipts']:
        folder, refined = store.read(receipt)
        r = refined['request']
        _require(refined['stage'] == 'prediction_refinement' and
                 r['evaluation_receipt'] == request['evaluation_receipt'] and
                 r['truth_receipt'] == request['truth_receipt'], 'Refinement identity mismatch')
        step = r['resolution_ns']
        _require(step in store.recipe['evaluation']['refinement_steps_ns']+
                 store.recipe['evaluation']['adaptive_refinement_steps_ns'], 'Refinement resolution mismatch')
        with np.load(folder/'refinement.npz', allow_pickle=False) as saved:
            coords, endpoints = saved['coordinates'], saved['endpoint']
        _equal(coords.tolist(), r['coordinates'], 'Refinement coordinates mismatch')
        _require(len({tuple(c) for c in coords}) == len(coords) and np.isfinite(endpoints).all(),
                 'Invalid refinement arrays')
        for (s, c), endpoint in zip(coords, endpoints):
            s,c = int(s),int(c)
            _require(0 <= s < len(reference) and 0 <= c < reference.shape[1], 'Refinement outside truth')
            resolution[s,c] = min(resolution[s,c], step)
            if step in store.recipe['evaluation']['refinement_steps_ns']:
                predecessors[s,c,step] = endpoint
                if step == store.recipe['evaluation']['refinement_steps_ns'][-1]:
                    prior = predecessors[s,c,store.recipe['evaluation']['refinement_steps_ns'][0]]
                    previous[s,c],delta[s,c] = endpoint,np.abs(endpoint-prior)
            else:
                _require(np.isfinite(previous[s,c]).all(), 'Missing refinement predecessor')
                new_delta = np.abs(endpoint-previous[s,c])
                envelope[s,c] = np.maximum(delta[s,c],new_delta)
                reference[s,c],previous[s,c],delta[s,c] = endpoint,endpoint,new_delta
    return reference,envelope,resolution


def _prediction_metrics(store, folder, manifest):
    request = manifest['request']
    evaluation = load_evaluation(store, request['evaluation_receipt'])
    _, fit = store.read(request['fit_receipt'])
    coordinate = _coordinate(fit['request'])
    comparison = ('same_weight_propagation_diagnostic' if manifest['stage']=='propagation' else
                  'unfitted_reference' if fit['stage']=='nominal' else 'own_solver_trained')
    _require(request['comparison']==comparison, 'Prediction comparison mismatch')
    _equal(manifest['summary']['canonical_tensor_sha256'], fit['summary'].get('canonical_tensor_sha256'),
           'Prediction selected tensor identity mismatch')
    propagators = propagation_family(coordinate['model_id']) if manifest['stage']=='propagation' else [coordinate['model_id']]
    _require(bool(propagators), 'Propagation diagnostic requires a coefficient model')
    reference,envelope,resolution = _reference(store, manifest, evaluation)
    templates = {}
    for case_id,case in enumerate(truth_cases(store.recipe)):
        if case['branch']=='zero':
            continue
        kinds = ['endpoint','drift','command_response','composition'] if case['branch']=='new' else ['endpoint']
        steps = store.recipe['evaluation']['query_steps_ns'] if case['branch']=='recorded' else [case['horizon_ns']]
        for p in propagators:
            for s in range(len(evaluation['state'])):
                for step in steps:
                    for kind in kinds:
                        for component,cut,unit in [('q',slice(0,2),'rad'),('v',slice(2,4),'rad/s')]:
                            templates[p,s,case['scenario'],case['horizon_ns'],step,kind,component] = (case_id,cut,unit)
    rows = _rows(folder,'rows.jsonl.gz')
    seen = set()
    for row in rows:
        key = tuple(row[k] for k in ('propagator','state_id','scenario','horizon_ns','query_step_ns','kind','component'))
        _require(key in templates and key not in seen, 'Duplicate or unexpected prediction row')
        seen.add(key)
        case_id,cut,unit = templates[key]
        s = row['state_id'];kind = row['kind']
        _equal(_coordinate(row), coordinate, 'Prediction fit coordinate mismatch')
        _require(row['own_solver_trained'] == (comparison == 'own_solver_trained'),
                 'Prediction own-solver label mismatch')
        _require(row['comparison']==comparison and row['unit']==unit and
                 row['trajectory_id']==int(evaluation['trajectory_id'][s]) and
                 row['anchor_ns']==int(evaluation['time_ns'][s]) and
                 row['stratum']==('primary' if evaluation['primary'][s] else 'initial_secondary'),
                 'Prediction observed coordinate mismatch')
        truth = reference[s,case_id,cut];bound = envelope[s,case_id,cut]
        ids = [[s,case_id]]
        if kind=='drift':
            truth = reference[s,case_id+1,cut]-evaluation['state'][s,cut]
            bound = envelope[s,case_id+1,cut];ids = [[s,case_id+1]]
        elif kind=='command_response':
            truth = truth-reference[s,case_id+1,cut]
            bound = bound+envelope[s,case_id+1,cut];ids.append([s,case_id+1])
        elif kind=='composition':
            truth,bound,ids = np.zeros(2),np.zeros(2),[]
        _equal(row['truth'],truth.tolist(),'Prediction truth mismatch')
        _equal(row['envelope'],bound.tolist(),'Prediction truth envelope mismatch')
        _equal(row['truth_coordinates'],ids,'Prediction truth provenance mismatch')
        _equal(row['truth_resolution_ns'],[int(resolution[a,b]) for a,b in ids], 'Truth resolution mismatch')
        _require(row['envelope_norm']==float(np.linalg.norm(bound)), 'Prediction envelope norm mismatch')
        if row['status']=='ok':
            value = np.asarray(row['prediction'],dtype=np.float64)
            _require(value.shape==(2,) and np.isfinite(value).all(), 'Invalid finite prediction')
            _equal(row['error'],(value-truth).tolist(),'Prediction raw residual mismatch')
        else:
            _require(row['prediction'] is None and row['error'] is None, 'Failed prediction carries a metric')
    _require(seen==set(templates),'Missing prediction rows')
    metrics = aggregate_prediction_rows(rows)
    _equal(manifest['summary']['metrics'],metrics,'Prediction raw summary mismatch')
    _require(manifest['summary']['row_count']==len(rows),'Prediction row count mismatch')
    return metrics


def _decision_records(store, folder, manifest):
    request = manifest['request']
    arrays = load_evaluation(store,request['evaluation_receipt'])
    truth = load_decision_truth(store,request['evaluation_receipt'],request['truth_receipt'])
    _,fit = store.read(request['fit_receipt']);coordinate = _coordinate(fit['request'])
    groups = defaultdict(list)
    for row in _rows(folder,'rows.jsonl.gz'):
        _equal(_coordinate(row),coordinate,'Decision coordinate mismatch')
        groups[row['state_id'],row['query_step_ns']].append(row)
    ids = decision_state_ids(arrays,store.recipe)
    expected = {(s,h) for s in ids for h in store.recipe['evaluation']['query_steps_ns']}
    _require(set(groups)==expected,'Missing or unexpected decision groups')
    raw_outcomes = _rows(folder,'outcomes.jsonl')
    indexed = {(o['state_id'],o['query_step_ns']):o for o in raw_outcomes}
    _require(set(indexed)==expected and len(indexed)==len(raw_outcomes),'Duplicate or missing decision outcomes')
    outcomes = []
    for (s,h),rows in sorted(groups.items()):
        rows.sort(key=lambda r:r['candidate_id'])
        i = ids.index(s)
        _require([r['candidate_id'] for r in rows]==list(range(store.recipe['decisions']['candidate_count'])),
                 'Duplicate or missing candidate rows')
        for c,row in enumerate(rows):
            _require(row['trajectory_id']==int(arrays['trajectory_id'][s]) and row['anchor_ns']==int(arrays['time_ns'][s]),
                     'Decision anchor mismatch')
            for key,rawkey in [('truth_cost','cost'),('truth_envelope','envelope'),
                ('truth_sampled_feasible','sampled_feasible'),('truth_resolution_ns','resolution_ns'),
                ('truth_feasibility_disagreement','feasibility_disagreement'),
                ('truth_feasibility_status','feasibility_status'),
                ('truth_refinement_status','refinement_status'),('truth_dense_max_abs','dense_max_abs'),
                ('truth_dense_violation_count','dense_violation_count'),('truth_dense_first_violation_ns','dense_first_violation_ns')]:
                _equal(row[key],np.asarray(truth[rawkey][i,c]).tolist(),'Decision truth binding mismatch: '+key)
            _require(row['truth_cost_lower']==row['truth_cost']-row['truth_envelope'] and
                     row['truth_cost_upper']==row['truth_cost']+row['truth_envelope'], 'Decision cost bounds mismatch')
        costs = np.array([r['predicted_cost'] if r['predicted_cost'] is not None else np.inf for r in rows])
        feasible = np.array([r['predicted_sampled_feasible'] for r in rows],bool)
        _require(all(not r['predicted_sampled_feasible'] or (r['status']=='ok' and np.isfinite(costs[c]))
                     for c,r in enumerate(rows)), 'Invalid predicted feasible candidate')
        valid = np.flatnonzero(feasible)
        selected = int(valid[np.argmin(costs[valid])]) if len(valid) else None
        _require([r['selected'] for r in rows]==[c==selected for c in range(len(rows))], 'Decision selection mismatch')
        predicted = {'selected_index':selected,'sampled_feasible':feasible,'statuses':[r['status'] for r in rows]}
        outcome = decision_outcome(predicted,truth['cost'][i],truth['sampled_feasible'][i],truth['envelope'][i],truth['feasibility_status'][i])
        actual = indexed[s,h]
        _equal(_coordinate(actual),coordinate,'Decision outcome coordinate mismatch')
        _equal({k:actual[k] for k in outcome},outcome,'Decision raw outcome mismatch')
        outcomes.append(actual)
    _equal(manifest['summary']['outcomes'],raw_outcomes,'Decision summary mismatch')
    _require(manifest['summary']['row_count']==sum(map(len,groups.values())), 'Decision row count mismatch')
    if 'hybrid_component' in store.recipe:
        # Enrich copies only AFTER validating immutable stored outcomes/summary.
        enriched = []
        for actual in outcomes:
            i = ids.index(actual['state_id'])
            selected = actual['selected_index'] if not actual['decision_failure'] else None
            enriched.append({**actual, 'truth_receipt':request['truth_receipt'],
                'selected_truth_cost':float(truth['cost'][i,selected]) if selected is not None else None,
                'selected_truth_envelope':float(truth['envelope'][i,selected]) if selected is not None else None,
                'selected_command_sha256':hashlib.sha256(truth['bank'][i,selected].tobytes()).hexdigest()
                    if selected is not None else None})
        return enriched
    return outcomes


def _timing_records(store, receipt, folder, manifest):
    request = manifest['request']
    arrays = load_evaluation(store,request['evaluation_receipt'])
    state_id = decision_state_ids(arrays,store.recipe)[0]
    _require(request['recipe_receipt']==store.recipe_receipt and request['timing']==store.recipe['timing'] and
             request['phase']==store.recipe['phase'] and request['representative_state_id']==state_id,
             'Timing representative input mismatch')
    from .calibration_decisions import _bank
    import hashlib
    bank = _bank(arrays,state_id,store.recipe)
    target = arrays['target'][arrays['trajectory_id'][state_id]]
    _require(request['input_sha256']==hashlib.sha256(arrays['state'][state_id].tobytes()+target.tobytes()+bank.tobytes()).hexdigest(),
             'Timing bank input mismatch')
    _equal([_coordinate(store.read(r)[1]['request']) for r in request['fit_receipts']],request['coordinates'],
           'Timing fit coordinates mismatch')
    expected = []
    for c in request['coordinates']:
        for step in store.recipe['evaluation']['query_steps_ns']:
            for category,batches in [('network_forward',[1,128,256]),('prediction',[1,128,256]),('complete_scorer',[len(bank)])]:
                for batch in batches:
                    expected.append({**c,'category':category,'batch_size':batch,'query_step_ns':step,'state_id':state_id,
                                     'trajectory_id':int(arrays['trajectory_id'][state_id]),'anchor_ns':int(arrays['time_ns'][state_id])})
    registry = json.loads((folder/'registry.json').read_text())
    _require(len(registry)==len(expected) and {canonical_json(c) for c in registry}=={canonical_json(c) for c in expected},
             'Timing registry coverage mismatch')
    rows = _rows(folder,'samples.jsonl')
    keys = [(canonical_json(r['coordinate']),r['repetition']) for r in rows]
    wanted = {(canonical_json(c),i) for c in expected for i in range(50)}
    _require(len(keys)==len(set(keys)) and set(keys)==wanted,'Duplicate or missing timing samples')
    summary = summarize_timing(rows)
    _equal(manifest['summary']['groups'],summary['groups'],'Timing raw summary mismatch')
    _require(manifest['summary']['sample_count']==len(rows),'Timing sample count mismatch')
    groups = summary['groups']
    if 'hybrid_component' in store.recipe:
        grouped = defaultdict(list)
        for row in rows:
            grouped[canonical_json(row['coordinate'])].append(row)
        groups = [{**g, 'samples':[{k:r[k] for k in ('coordinate','repetition','order_index','wall_s','cpu_s','status')}
                   for r in sorted(grouped[canonical_json(g['coordinate'])],key=lambda r:r['repetition'])]}
                  if g['coordinate']['category']=='complete_scorer' else g for g in groups]
    return groups,load_timing_finalization(store,receipt)


def select_timing_products(entries: list, wanted: list[dict]) -> tuple[list, list]:
    """Select requested-matrix measurements by membership, never by observed speed.

    An exact complete request supersedes subset runs. Without it, strictly larger
    membership supersedes contained subsets; disjoint subsets can close coverage.
    Equal or partially overlapping selected requests are ambiguous. Supersets
    belonging to a later matrix cannot replace the earlier matrix's measurement.
    """
    wanted_keys = {_key(c) for c in wanted}
    candidates,ignored = [],[]
    for entry in entries:
        keys = {_key(c) for c in entry[2]['request']['coordinates']}
        _require(len(keys)==len(entry[2]['request']['coordinates']), 'Duplicate timing request coordinates')
        if not keys <= wanted_keys:
            ignored.append(entry[0])
        else:
            candidates.append((entry,keys))
    exact = [e for e,k in candidates if k==wanted_keys]
    _require(len(exact)<=1, 'Ambiguous complete-matrix timing observations')
    if exact:
        return exact,ignored+[e[0] for e,k in candidates if e is not exact[0]]
    selected = []
    for entry,keys in candidates:
        if any(keys < others for _,others in candidates):
            ignored.append(entry[0])
            continue
        for _,others in selected:
            _require(keys!=others,'Ambiguous repeated timing coordinate sets')
            _require(not keys & others,'Overlapping timing subset measurements')
        selected.append((entry,keys))
    return [entry for entry,_ in selected],ignored


def _evidence(store, matrix):
    """Catch evidence-format failures only here, never turn them into method failures."""
    expected = _coordinates(matrix,store.recipe)
    _equal(store.coordinates,expected,'Prospective matrix mismatch')
    nominal = [{'session_id':s,'budget_s':b,'model_id':'physics_nominal','initialization_id':None}
               for s,b in sorted({(c['session_id'],c['budget_s']) for c in expected})]
    wanted = expected+nominal
    entries,invalid = [],[]
    for marker in sorted(store.output_root.glob('jobs/*/*/COMPLETED')):
        try:
            receipt = {'job_key':marker.parent.name,'path':str(marker.parent.relative_to(store.output_root)),
                       'manifest_sha256':read_json(marker)['manifest_sha256']}
            folder,manifest = store.read(receipt)
            entries.append((receipt,folder,manifest))
        except (ValueError,OSError,KeyError,TypeError) as error:
            invalid.append({'path':str(marker.parent.relative_to(store.output_root)),'error':str(error)})
    timing_entries = [e for e in entries if e[2]['stage']=='timing']
    try:
        selected_timing,ignored_timing = select_timing_products(timing_entries,wanted)
    except ValueError as error:
        selected_timing,ignored_timing = [],[]
        invalid.append({'stage':'timing','error':str(error)})
    selected_timing_keys = {e[0]['job_key'] for e in selected_timing}
    present = defaultdict(dict)
    metrics,outcomes,timing,finalization = [],[],[],[]
    resources,failed,limited = [],[],[]
    for receipt,folder,manifest in entries:
        stage,request = manifest['stage'],manifest['request']
        try:
            c = None
            if stage in {'fit','nominal'}:
                c = _coordinate(request)
            elif stage in {'predictions','decisions','propagation'}:
                c = _coordinate(store.read(request['fit_receipt'])[1]['request'])
            if c is not None and c not in wanted:
                continue  # expansion products outside the requested pilot are not pilot coverage
            if c is not None:
                _require(_key(c) not in present[stage],'Duplicate '+stage+' coordinate')
            if stage in {'fit','nominal'}:
                _selected(store,receipt)
                if stage=='fit' and manifest['summary'].get('fit_status')=='failed':
                    failed.append(c)
                if manifest['summary'].get('optimization_limited'):
                    limited.append(c)
            elif stage in {'predictions','propagation'}:
                metrics.extend(_prediction_metrics(store,folder,manifest))
            elif stage=='decisions':
                outcomes.extend(_decision_records(store,folder,manifest))
            elif stage=='timing':
                if receipt['job_key'] not in selected_timing_keys:
                    continue
                groups,event = _timing_records(store,receipt,folder,manifest)
                for g in groups:
                    tc = _coordinate(g['coordinate'])
                    if tc not in wanted:
                        continue
                    tk = canonical_json(g['coordinate'])
                    _require(tk not in present['timing'],'Duplicate timing coordinate')
                    present['timing'][tk] = g
                    timing.append(g)
                finalization.append(event)
            elif stage=='evaluation':
                load_evaluation(store,receipt)
            elif stage=='prediction_truth':
                load_prediction_truth(store,request['evaluation_receipt'],receipt)
            elif stage=='decision_truth':
                load_decision_truth(store,request['evaluation_receipt'],receipt)
            if c is not None:
                present[stage][_key(c)] = receipt
            resources.append({'stage':stage,'receipt':receipt,'producer_attempts':manifest['resources'],
                              **({'coordinate':c,'account':manifest['summary']['account']} if stage in {'fit','nominal'} else {}),
                              **({'noncritical':manifest['summary']['noncritical'],
                                  'machine':manifest['summary']['resources']} if stage=='timing' else {})})
        except (ValueError,OSError,KeyError,TypeError,IndexError) as error:
            invalid.append({'stage':stage,'receipt':receipt,'error':str(error)})
    missing = []
    # Hybrid comparisons require one joint rotated request, not union coverage
    # assembled from separate runs. Keep selected subsets as diagnostic evidence.
    if matrix=='hybrid' and not any(
            {_key(c) for c in entry[2]['request']['coordinates']}=={_key(c) for c in wanted}
            for entry in selected_timing):
        missing.append({'stage':'timing','reason':'complete_pooled_request_required',
                        'expected_coordinate_count':len(wanted)})
    for c in wanted:
        stages = ['nominal' if c['model_id']=='physics_nominal' else 'fit','predictions','decisions']
        if propagation_family(c['model_id']):
            stages.append('propagation')
        for stage in stages:
            if _key(c) not in present[stage]:
                missing.append({'stage':stage,'coordinate':c})
        covered = [g['coordinate'] for g in timing if _coordinate(g['coordinate'])==c]
        for step in store.recipe['evaluation']['query_steps_ns']:
            for category,batches in [('network_forward',[1,128,256]),('prediction',[1,128,256]),
                                     ('complete_scorer',[store.recipe['decisions']['candidate_count']])]:
                for batch in batches:
                    if not any(g['query_step_ns']==step and g['category']==category and g['batch_size']==batch for g in covered):
                        missing.append({'stage':'timing','coordinate':{**c,'query_step_ns':step,'category':category,'batch_size':batch}})
    event_path = store.output_root/'events.jsonl'
    events = []
    if event_path.exists():
        for line_number,line in enumerate(event_path.read_text().splitlines(),1):
            try:
                event = json.loads(line)
                _require(isinstance(event,dict),'Event must be an object')
                events.append(event)
            except ValueError as error:
                invalid.append({'path':'events.jsonl','line':line_number,'error':str(error)})
    coverage = {'complete':not missing and not invalid,'missing':missing,'invalid':invalid,'failed':failed,
                'optimization_limited':limited,'expected_fit_count':len(expected),'expected_nominal_count':len(nominal),
                'counts':{stage:len(items) for stage,items in present.items()},
                'failed_attempts':[e for e in events if e.get('event')=='failed'],
                'expected_coordinates':expected,'nominal_coordinates':nominal,
                'timing_selection_policy':('exact requested-matrix set required for coverage; maximal disjoint subsets retained as diagnostics; no speed selection'
                    if matrix=='hybrid' else 'exact requested-matrix set; otherwise maximal disjoint subsets; no speed selection'),
                'ignored_timing_receipts':ignored_timing}
    return coverage,metrics,outcomes,timing,resources,finalization


def verify(store: CalibrationStore, matrix: str) -> dict:
    """Read and recompute evidence; incomplete or invalid is distinct from failed performance."""
    return _evidence(store,matrix)[0]


def _screen_statistics(metrics: list[dict], statistics: dict) -> dict:
    reasons = []
    by_component = {r['component']:r for r in metrics}
    failures = statistics.get('prediction_failure_count',0)
    if failures:
        reasons.append('prediction_failure')
    missing = set(by_component)!={'q','v'}
    for component,rmse,p95 in [('q',.01,.03),('v',.10,.30)]:
        row = by_component.get(component,{})
        for field,limit in [('conservative_rmse_upper',rmse),('conservative_p95_upper',p95)]:
            value = row.get(field)
            if value is None:
                missing = missing or not failures
                reasons.append(component+'_'+field+'_undefined')
            elif value > limit:
                reasons.append(component+'_'+field+'_over_screen')
    zero = statistics.get('mean_zero_command_cost')
    failure_fraction = statistics.get('decision_failure_fraction')
    upper = statistics.get('mean_regret_upper')
    unresolved = missing or zero is None or failure_fraction is None or statistics.get('no_true_feasible_bank',False) or zero<=1e-8
    if statistics.get('truth_feasibility_unresolved',False):
        unresolved = True
        reasons.append('sampled_truth_feasibility_unresolved')
    if failure_fraction is not None and failure_fraction > .01:
        reasons.append('decision_failure_fraction_over_screen')
    if not unresolved and (upper is None or upper > .01*zero):
        reasons.append('conservative_valid_regret_over_screen')
    if unresolved:
        reasons.append('missing_records_or_no_true_feasible_bank_or_zero_cost_undefined')
    return {'status':'unresolved' if unresolved else 'fail' if reasons else 'pass','reasons':reasons,
            **statistics,'normalized_regret_upper':upper/zero if upper is not None and zero is not None and zero>1e-8 else None}


def configuration_screen(metrics: list[dict], outcomes: list[dict]) -> dict:
    good = [o for o in outcomes if not o['decision_failure'] and o['regret'] is not None]
    def mean(key):
        return float(np.mean([o[key] for o in good])) if good else None
    statistics = {'prediction_failure_count':sum(r['failure_count'] for r in metrics),
        'decision_failure_fraction':sum(o['decision_failure'] for o in outcomes)/len(outcomes) if outcomes else None,
        'mean_valid_regret':mean('regret'),'mean_regret_upper':mean('regret_upper'),
        'mean_regret_lower':float(np.mean([o.get('regret_lower',o['regret']) for o in good])) if good else None,
        'mean_zero_command_cost':float(np.mean([o['zero_command_truth_cost'] for o in outcomes])) if outcomes else None,
        'truth_feasibility_unresolved':any(o.get('truth_feasibility_status')=='unresolved' for o in outcomes),
        'no_true_feasible_bank':any(o['no_true_feasible_bank'] for o in outcomes)}
    return _screen_statistics(metrics,statistics)


def threshold_curve(rows: list[dict]) -> dict:
    rows = sorted(rows,key=lambda r:r['budget_s'])
    passing = [r['budget_s'] for r in rows if r['status']=='pass']
    first = min(passing) if passing else None
    lower = max((r['budget_s'] for r in rows if r['status']=='fail' and (first is None or r['budget_s']<first)),default=0)
    unresolved = any(r['status']=='unresolved' for r in rows)
    return {'rows':rows,'first_passing_budget_s':first,'bracket_s':[lower,first],
            'interval_censored':True,'never_passing':not passing,'unresolved':unresolved,
            'nonmonotone':first is not None and any(r['budget_s']>first and r['status']!='pass' for r in rows)}


def threshold_ratio(flow: dict, alternative: dict) -> dict:
    flow_lower,flow_upper = flow['bracket_s']
    other_lower,other_upper = alternative['bracket_s']
    return {'flow_bracket_s':flow['bracket_s'],'alternative_bracket_s':alternative['bracket_s'],
            'observed_budget_ratio':other_upper/flow_upper if other_upper is not None and flow_upper else None,
            'ratio_bracket':[other_lower/flow_upper if flow_upper else 0.,
                             other_upper/flow_lower if other_upper is not None and flow_lower else None],
            'interval_censored':True,'interpretation':'brackets, not an interpolated precise data requirement'}


def cluster_bootstrap(rows: list[dict], keys: list[str], fields: list[str], recipe: dict) -> dict:
    sessions = sorted({r['session_id'] for r in rows})
    namespace = recipe['bootstrap_namespace']
    draws = recipe['bootstrap_draws']
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(namespace)))
    indices = rng.integers(0,len(sessions),(draws,len(sessions))) if sessions else np.empty((draws,0),int)
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)
    results = []
    for key,group in sorted(grouped.items(),key=lambda item:str(item[0])):
        values,intervals,means = {},{},{}
        for field in fields:
            session_values = []
            for session in sessions:
                items = [r.get(field) for r in group if r['session_id']==session]
                # Undefined replicas remain undefined; no imputation or nanmean.
                session_values.append(float(np.mean(items)) if items and all(v is not None for v in items) else None)
            values[field] = session_values
            if session_values and all(v is not None for v in session_values):
                a = np.asarray(session_values)
                means[field] = float(a.mean())
                intervals[field] = np.percentile(a[indices].mean(axis=1),[2.5,97.5]).tolist()
            else:
                means[field],intervals[field] = None,[None,None]
        results.append({'coordinate':dict(zip(keys,key)),'session_ids':sessions,'session_values':values,
                        'means':means,'intervals':intervals})
    return {'independent_session_count':len(sessions),'session_ids':sessions,'draws':draws,
            'namespace':namespace,'paired_draws':True,'initializations':'average within session',
            'powered_claim':False,'groups':results}


def summarize_configurations(replicas: list[dict], recipe: dict) -> tuple[list[dict], dict]:
    fields = ['q_rmse','q_p95','v_rmse','v_p95','q_conservative_rmse_upper','q_conservative_p95_upper',
              'v_conservative_rmse_upper','v_conservative_p95_upper','decision_failure_fraction',
              'mean_valid_regret','mean_regret_upper','mean_regret_lower','mean_zero_command_cost',
              'scorer_wall_median_s','scorer_wall_p95_s']
    uncertainty = cluster_bootstrap(replicas,['model_id','budget_s','query_step_ns'],fields,recipe)
    configurations = []
    for group in uncertainty['groups']:
        coordinate = group['coordinate']
        cells = [row for row in replicas if all(row[k]==v for k,v in coordinate.items())]
        means = group['means']
        metrics = [{'component':component,
                    'conservative_rmse_upper':means[component+'_conservative_rmse_upper'],
                    'conservative_p95_upper':means[component+'_conservative_p95_upper']}
                   for component in ('q','v')]
        statistics = {k:means[k] for k in ['decision_failure_fraction','mean_valid_regret',
            'mean_regret_upper','mean_regret_lower','mean_zero_command_cost']}
        statistics.update(truth_feasibility_unresolved=any(row['screen'].get('truth_feasibility_unresolved',False) for row in cells),
                          prediction_failure_count=sum(row['screen']['prediction_failure_count'] for row in cells),
                          no_true_feasible_bank=any(row['screen'].get('no_true_feasible_bank',False) for row in cells))
        screen = _screen_statistics(metrics,statistics)
        if screen['status']=='pass' and means['scorer_wall_median_s'] is None:
            screen = {**screen,'status':'unresolved','reasons':screen['reasons']+['missing_or_failed_matching_scorer_timing']}
        configurations.append({**coordinate,**means,'screen':screen,
            'optimization_limited':any(row['optimization_limited'] for row in cells),
            'replica_count':len(cells),'initializations_per_session':max(sum(row['session_id']==sid for row in cells) for sid in group['session_ids']),
            'independent_session_count':len(group['session_ids']),
            'intervals':group['intervals'],'session_values':group['session_values']})
    return configurations,uncertainty


def gate_decision(result: dict) -> dict:
    def decision(name,*reasons):
        return {'decision':name,'reasons':list(reasons),'stage_c_launched':False,
                'scope':'provisional simulator mechanism screen; no deadline or hardware claim'}
    if not result['coverage']['complete']:
        return decision('unresolved','incomplete_or_invalid_evidence')
    if not result.get('empirical_claims_allowed',False):
        return decision('unresolved','unit_fixture_ineligible_for_empirical_claims')
    passing = [c for c in result['configurations'] if c['screen']['status']=='pass' and c['scorer_wall_median_s'] is not None]
    flow = [c for c in passing if c['model_id']=='affine_flow']
    if not flow:
        neural = [c for c in result['configurations'] if c['model_id'] not in {'physics_fit','physics_nominal'} and c['scorer_wall_median_s'] is not None]
        if neural and any(c['model_id'] in {'physics_fit','physics_nominal'} and c['scorer_wall_median_s']<=min(n['scorer_wall_median_s'] for n in neural) for c in passing):
            return decision('no_practical_headroom','simple_physics_passes_cheaply_while_flow_never_passes')
        return decision('unresolved' if any(c['screen']['status']=='unresolved' for c in result['configurations']) else 'no_advantage',
                        'affine_flow_has_no_resolved_passing_configuration')
    best_flow = min(flow,key=lambda c:(c['budget_s'],c['scorer_wall_median_s']))
    physics = [c for c in passing if c['model_id'] in {'physics_fit','physics_nominal'} and
               c['budget_s']<=best_flow['budget_s'] and c['scorer_wall_median_s']<=best_flow['scorer_wall_median_s']]
    if physics:
        return decision('no_practical_headroom','simple_physics_passes_at_no_greater_budget_or_scoring_cost')
    alternatives = [c for c in passing if c['model_id'] != 'affine_flow']
    def removes_advantage(model):
        return any(c['model_id']==model and c['budget_s']<=best_flow['budget_s'] and
                   c['scorer_wall_median_s']<=best_flow['scorer_wall_median_s'] and
                   c['decision_failure_fraction']<=best_flow['decision_failure_fraction'] and
                   c['mean_regret_upper']<=best_flow['mean_regret_upper'] for c in alternatives)
    if removes_advantage('node_actuator_aware'):
        return decision('cheap_solver_explains_gain','actuator_aware_node_matches_screen_budget_scoring_and_decisions')
    all_alternatives = [c for c in result['configurations'] if c['model_id'] != 'affine_flow']
    def unresolved(c):
        return c['screen']['status']=='unresolved' or (c['screen']['status']=='pass' and c['scorer_wall_median_s'] is None)
    if any(unresolved(c) for c in all_alternatives):
        return decision('unresolved','alternative_screen_or_matching_timing_unresolved')

    def opportunity(f, competitors, all_competitors):
        earlier = bool(competitors) and f['budget_s'] < min(c['budget_s'] for c in competitors)
        censored_earlier = not competitors and bool(all_competitors) and all(c['screen']['status']=='fail' for c in all_competitors) and f['budget_s'] < max(c['budget_s'] for c in all_competitors)
        earlier = earlier or censored_earlier
        # Every configuration retains its own prediction/decision/timing partition.
        # A previously available smaller-budget fit remains available at this budget.
        available = [c for c in competitors if c['budget_s']<=f['budget_s']]
        faster = bool(available) and f['scorer_wall_median_s'] < min(c['scorer_wall_median_s'] for c in available)
        compared = (competitors or [c for c in all_competitors if c['budget_s']==f['budget_s']]) if earlier else available
        no_loss = bool(compared) and all(f['decision_failure_fraction'] is not None and c['decision_failure_fraction'] is not None and
                   f['decision_failure_fraction']<=c['decision_failure_fraction'] and
                   f['mean_regret_upper'] is not None and c['mean_valid_regret'] is not None and
                   f['mean_regret_upper']<=c.get('mean_regret_lower',c['mean_valid_regret']) for c in compared)
        return ('earlier_screen_budget' if earlier else 'lower_complete_scoring_time') if (earlier or faster) and no_loss else None

    unrestricted = [c for c in alternatives if c['model_id']!='affine_rk4']
    all_unrestricted = [c for c in all_alternatives if c['model_id']!='affine_rk4']
    if removes_advantage('affine_rk4') and opportunity(best_flow,unrestricted,all_unrestricted):
        return decision('representation_only','both_f_B_solvers_match_screen_budget_scoring_and_decisions',
                        'representation_opportunity_over_unrestricted_alternatives')
    for f in flow:
        reason = opportunity(f,alternatives,all_alternatives)
        if reason:
            return decision('proceed_to_deadline_spec',reason,
                            'decision_regret_and_reliability_not_worse','new_stage_c_freeze_required')
    return decision('no_advantage','no_earlier_budget_or_faster_equal_screen_without_decision_loss')


def resource_accounting(categories: list[dict]) -> dict:
    """Sum measured store attempts once, excluding children covered by a parent.

    Inclusive raw spans remain available for stage diagnostics. A completed child
    whose parent is absent still contributes its observed work to a partial report.
    """
    indexed = {}
    for row in categories:
        resources = row['producer_attempts']
        for attempt in resources['attempts']:
            key = (resources['producer_key'],attempt['attempt_id'])
            _require(key not in indexed, 'Duplicate producer resource attempt')
            indexed[key] = attempt
    retained = []
    for attempt in indexed.values():
        parent = attempt['parent_producer_attempt']
        if parent is None or (parent['job_key'],parent['attempt_id']) not in indexed:
            retained.append(attempt)
    unknown = any(a['unknown_tail'] for a in retained)
    return {**{kind+'_s':sum(a['active_'+kind+'_s']+a['persistence_'+kind+'_s'] for a in retained)
               for kind in ('wall','cpu')},
            'accounting_status':'lower_bound_unknown_tail' if unknown else 'measured',
            'included_attempts':len(retained), 'excluded_nested_attempts':len(indexed)-len(retained),
            'scope':'nonoverlapping reported producer attempts; inclusive stage and rate spans must not be added; incomplete coverage can omit unfinished work'}


def aggregate(store: CalibrationStore, matrix: str) -> dict:
    coverage,metrics,outcomes,timing,resources,finalization = _evidence(store,matrix)
    if coverage['invalid']:
        raise ValueError('Invalid calibration evidence: '+canonical_json(coverage['invalid']).decode())
    wanted = coverage['expected_coordinates']+coverage['nominal_coordinates']
    replicas = []
    for c in wanted:
        for step in store.recipe['evaluation']['query_steps_ns']:
            selected = [m for m in metrics if _coordinate(m)==c and m['comparison']!='same_weight_propagation_diagnostic'
                        and m['scenario']=='recorded' and m['horizon_ns']==160_000_000 and
                        m['stratum']=='primary' and m['kind']=='endpoint' and m['query_step_ns']==step]
            decisions = [o for o in outcomes if _coordinate(o)==c and o['query_step_ns']==step]
            screen = configuration_screen(selected,decisions)
            complete = [t for t in timing if _coordinate(t['coordinate'])==c and
                        t['coordinate']['query_step_ns']==step and t['coordinate']['category']=='complete_scorer']
            usable = complete and complete[0]['failure_count']==0 and complete[0]['valid_timing_calls']==50
            row = {**c,'query_step_ns':step,'screen':screen,
                   'optimization_limited':c in coverage['optimization_limited'],
                   'scorer_wall_median_s':complete[0]['wall_s']['median'] if usable else None,
                   'scorer_wall_p95_s':complete[0]['wall_s']['p95'] if usable else None,
                   **{k:screen.get(k) for k in ['decision_failure_fraction','mean_valid_regret','mean_regret_upper','mean_regret_lower','mean_zero_command_cost']}}
            for component in ('q','v'):
                m = next((m for m in selected if m['component']==component),{})
                for field in ('rmse','p95','conservative_rmse_upper','conservative_p95_upper'):
                    row[component+'_'+field] = m.get(field)
            replicas.append(row)
    configurations,uncertainty = summarize_configurations(replicas,store.recipe['aggregation'])
    primary_model = store.recipe.get('hybrid_component',{}).get('primary_model_id','affine_flow')
    curves = []
    for model in (*store.recipe['models'],'physics_nominal'):
        for step in store.recipe['evaluation']['query_steps_ns']:
            curve = threshold_curve([{'budget_s':c['budget_s'],'status':c['screen']['status']} for c in configurations
                                     if c['model_id']==model and c['query_step_ns']==step])
            curves.append({'model_id':model,'query_step_ns':step,**curve})
    ratios = [{'model_id':c['model_id'],'query_step_ns':c['query_step_ns'],
               **threshold_ratio(next(f for f in curves if f['model_id']==primary_model and f['query_step_ns']==c['query_step_ns']),c)}
              for c in curves if c['model_id']!=primary_model]
    result = {'protocol_id':PROTOCOL_ID,'matrix':matrix,'phase':store.recipe['phase'],
              'empirical_claims_allowed':store.recipe['empirical_claims_allowed'],
              'coverage':coverage,'metrics':metrics,'decision_outcomes':outcomes,'timing':timing,
              'replica_configurations':replicas,'configurations':configurations,'learning_curves':curves,'threshold_ratios':ratios,
              'uncertainty':{**uncertainty,'interpretation':
                  'descriptive only; three sessions are exploratory, not powered confirmation' if matrix=='hybrid' else
                  'descriptive only; five sessions are not powered confirmation'},
              'recipe':store.recipe,'resources':{'categories':resources,'nonoverlapping_producers':resource_accounting(resources),
              'timing_finalization':finalization,
              'calibration_motion_seconds_proxy':[{'session_id':s,'available_observation_s':store.recipe['collector']['segment_count'],
                   'budgets_s':store.recipe['budgets_s']} for s in sorted({c['session_id'] for c in wanted})],
              'fitting_scope':'use nonoverlapping_producers for totals; inclusive producer_attempts and nested rate diagnostics overlap and must not be added; active and checkpoint persistence spans exclude inactive gaps, ledger observation writes and post-producer finalization',
              'projection_scope':'unknown-tail records are measured lower bounds, never complete throughput estimates; pilot projections must retain accounting_status and separately report actual compressed bytes and finalization overhead',
              'evaluator_scope':'evaluation, truth replay/refinement, predictions, decisions and timing resources reported separately',
              'finalization_provenance':'unique exact-receipt-bound original observational event; events.jsonl is outside immutable artifact hashes; resource-record write excluded'},
              'reproduction':{'output_root':str(store.output_root),'recipe_receipt':store.recipe_receipt,
                              'source_root':store.bindings['root'],'raw_boundary':'arrays and weights remain in the selected ignored output store'}}
    # JSON normalization makes writing/reading portable and exact (no tuple metadata).
    result = json.loads(canonical_json(result))
    if matrix=='hybrid':
        from .hybrid_report import hybrid_gate
        result['reproduction']['bindings'] = dict(store.bindings)
        result['gate'] = hybrid_gate(result)
    else:
        result['gate'] = gate_decision(result)
    return result


def _reproduce(result):
    recipe = result['recipe']
    matrix = result['matrix']
    commands = ['python3.11 -m venv .venv','.venv/bin/python -m pip install --only-binary=:all: -r requirements.in']
    initial_matrix = 'hybrid' if matrix=='hybrid' else 'pilot'
    prepare = f'.venv/bin/python -m poc.calibration_study prepare --root . --output-root "$CALIBRATION_OUTPUT" --matrix {initial_matrix}'
    if recipe['phase']=='unit':
        prepare += ' --unit-fixture-segments '+str(recipe['collector']['segment_count'])
        prepare += ' --unit-fixture-max-epochs '+str(recipe['training']['max_epochs_per_rate'])
        prepare += ' --unit-fixture-physical-epochs '+str(recipe['training']['physical_epochs'])
        prepare += ' --unit-fixture-evaluation-trajectories '+str(recipe['evaluation']['trajectory_count'])
        if 'unit_anchor_times_ns' in recipe['evaluation']:
            prepare += ' --unit-fixture-evaluation-anchors-ns '+' '.join(map(str,recipe['evaluation']['unit_anchor_times_ns']))
        prepare += ' --unit-fixture-decision-candidates '+str(recipe['decisions']['candidate_count'])
    commands += ['CALIBRATION_OUTPUT="study/hybrid-component"' if matrix=='hybrid' else 'CALIBRATION_OUTPUT="study/short-calibration"',prepare]
    commands += [f'.venv/bin/python -m poc.calibration_study {stage} --root . --output-root "$CALIBRATION_OUTPUT" --matrix {initial_matrix}'
                 for stage in ['collect','train','predict','decisions','time','report','verify','status']]
    if matrix=='development':
        commands.append(prepare.replace('--matrix pilot','--matrix development'))
        commands += [f'.venv/bin/python -m poc.calibration_study {stage} --root . --output-root "$CALIBRATION_OUTPUT" --matrix development'
                     for stage in ['collect','train','predict','decisions','time','report','verify','status']]
    return ('Run from the same public source revision with native CPython 3.11 and the pinned requirements. '+
            ('Use a fresh ignored output directory for the explicit hybrid cohort.\n\n```sh\n' if matrix=='hybrid' else
             'Use a fresh ignored output directory. Complete the pilot before explicit development expansion.\n\n```sh\n')+
            '\n'.join(commands)+'\n```\n\nTo reverify existing evidence without acquisition or fitting:\n\n```sh\n'+
            f'.venv/bin/python -m poc.calibration_study verify --root . --output-root "$CALIBRATION_OUTPUT" --matrix {matrix}\n'+
            f'.venv/bin/python -m poc.calibration_study report --root . --output-root "$CALIBRATION_OUTPUT" --matrix {matrix}\n```\n\n'+
            'Prediction and candidate rows use deterministic level-1 gzip JSONL (rows.jsonl.gz); decode with Python gzip.open. '
            'Compression is lossless and occurs outside scorer timing. Raw arrays, selected weights and optimizer products stay in the selected ignored output store. '
            'Immutable receipts bind source, runtime, assets, recipe and selected tensors. Copy summaries together with '
            'that store to preserve external artifacts. The output path is bound into receipts; a moved store requires '
            'a fresh reproducible run, not edited bindings. Finalization costs use unique original receipt-bound observations '
            'from events.jsonl, outside immutable artifact hashes. Observation-record persistence is excluded.\n')


def _svg(result):
    width,height = 1200,800
    title = ('Diagnostic session means; all-session component eligibility in component_gate.json'
             if result['matrix']=='hybrid' else 'Provisional simulator learning curves; all partitions / baselines')
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="25" y="25" font-size="16">{title}</text>']
    colors = ['#004488','#bb5566','#228833','#aa3377','#66ccee','#ccbb44','#882255','#777777']
    budgets = result['recipe']['budgets_s'];lo,hi=min(budgets),max(budgets)
    for panel,(field,label,limit) in enumerate([('q_rmse','q RMSE (rad)',.01),('v_rmse','v RMSE (rad/s)',.1),
                                              ('q_p95','q p95 endpoint norm (rad)',.03),('v_p95','v p95 endpoint norm (rad/s)',.3)]):
        ox,oy = 80+(panel%2)*590,65+(panel//2)*335
        rows = result['configurations']
        maximum = max([limit]+[c[field] for c in rows if c[field] is not None])*1.1
        def point(c):
            return ox+460*(c['budget_s']-lo)/max(1,hi-lo),oy+235-235*c[field]/maximum
        parts += [f'<text x="{ox}" y="{oy}">{html.escape(label)}</text>',
                  f'<path d="M{ox},{oy+10} V{oy+235} H{ox+460}" fill="none" stroke="black"/>',
                  f'<text x="{ox}" y="{oy+260}" font-size="12">Calibration motion seconds proxy (s)</text>',
                  f'<text x="{ox-60}" y="{oy+20}" font-size="10">{maximum:.4g}</text>',
                  f'<text x="{ox-15}" y="{oy+235}" font-size="10">0</text>',
                  f'<path d="M{ox},{oy+235-235*limit/maximum} H{ox+460}" stroke="#999" stroke-dasharray="3 4"/>']
        for b in budgets:
            x = ox+460*(b-lo)/max(1,hi-lo)
            parts.append(f'<text x="{x}" y="{oy+249}" font-size="10">{b}</text>')
        for index,model in enumerate((*result['recipe']['models'],'physics_nominal')):
            for step in result['recipe']['evaluation']['query_steps_ns']:
                group = sorted([c for c in rows if c['model_id']==model and c['query_step_ns']==step],key=lambda c:c['budget_s'])
                points = [point(c) for c in group if c[field] is not None]
                dash = {5_000_000:'',10_000_000:'6 3',20_000_000:'2 3',40_000_000:'8 3 2 3'}[step]
                if points:
                    parts.append(f'<polyline points="'+ ' '.join(f'{x},{y}' for x,y in points)+f'" fill="none" stroke="{colors[index]}" stroke-dasharray="{dash}"/>')
                for c in group:
                    if c[field] is None:
                        x=ox+460*(c['budget_s']-lo)/max(1,hi-lo);y=oy+225-index*5
                        parts.append(f'<text x="{x}" y="{y}" fill="{colors[index]}" font-size="10">×<title>{model} {step/1e6:g} ms: undefined / failed</title></text>')
                    else:
                        x,y=point(c)
                        parts.append(f'<circle cx="{x}" cy="{y}" r="2" fill="{colors[index]}"><title>{model}, {step/1e6:g} ms, {c["screen"]["status"]}, optimization limited={c["optimization_limited"]}</title></circle>')
    for index,model in enumerate((*result['recipe']['models'],'physics_nominal')):
        parts.append(f'<text x="{20+(index%4)*300}" y="{745+(index//4)*20}" fill="{colors[index]}" font-size="12">{model}</text>')
    parts.append('<text x="20" y="795" font-size="11">Partitions: solid 5 ms; long dash 10 ms; dots 20 ms; dash-dot 40 ms. × undefined. No hardware commissioning claim.</text></svg>')
    return ''.join(parts)


def write_report(result: dict, destination: Path) -> list[Path]:
    destination = Path(destination);destination.mkdir(parents=True,exist_ok=True)
    names = ['short_calibration_findings.md','short_calibration_result.json','short_calibration_metrics.csv',
             'short_calibration_learning_curves.svg','short_calibration_reproduce.md']
    paths = [destination/name for name in names]
    atomic_json(paths[1],result)
    buffer = io.StringIO();rows = result['configurations']
    fields = ['model_id','budget_s','query_step_ns','q_rmse','q_p95','v_rmse','v_p95',
              'decision_failure_fraction','mean_valid_regret','mean_regret_upper','scorer_wall_median_s','scorer_wall_p95_s',
              'optimization_limited','independent_session_count']
    component_rows = {}
    if result['matrix']=='hybrid':
        fields += ['component_eligibility','pooled_scorer_wall_median_s','pooled_scorer_wall_p95_s']
        component_rows = {(c['model_id'],c['budget_s'],c['query_step_ns']):c for c in result['gate']['configurations']}
    writer = csv.DictWriter(buffer,fieldnames=fields+['screen_status'],extrasaction='ignore',
                            **({'lineterminator':'\n'} if result['matrix']=='hybrid' else {}));writer.writeheader()
    for row in rows:
        component = component_rows.get((row['model_id'],row['budget_s'],row['query_step_ns']))
        extra = ({'component_eligibility':component['status'],
                  'pooled_scorer_wall_median_s':component['scorer_pool']['wall_s']['median'],
                  'pooled_scorer_wall_p95_s':component['scorer_pool']['wall_s']['p95']} if component else {})
        writer.writerow({**row,**extra,'screen_status':row['screen']['status']})
    atomic_bytes(paths[2],buffer.getvalue().encode())
    atomic_bytes(paths[3],_svg(result).encode())
    atomic_bytes(paths[4],_reproduce(result).encode())
    coverage = result['coverage']
    lines = [f'# Short calibration A/B ({result["matrix"]}, {result["phase"]})','',
             f'Gate: **{result["gate"]["decision"]}**. Reasons: '+', '.join(result['gate']['reasons'])+'.','',
             f'Coverage complete: {coverage["complete"]}. Expected fits: {coverage["expected_fit_count"]}; nominal construction references: {coverage["expected_nominal_count"]}. '
             f'Missing cells: {len(coverage["missing"])}; invalid: {len(coverage["invalid"])}; retained failed fits: {len(coverage["failed"])}; optimization limited: {len(coverage["optimization_limited"])}.','',
             'This is a provisional simulator screen. Motion seconds are a calibration interaction proxy. '
             'There is no feedback deadline, deployment or hardware commissioning result. Held-response improvements '
             'do not establish decision improvements. Reduced unit fixtures are ineligible for empirical claims.','',
             'Prediction at 160 ms recorded motion must satisfy q RMSE ≤ .01 rad, v RMSE ≤ .10 rad/s, '
             'q p95 ≤ .03 rad and v p95 ≤ .30 rad/s, with zero prediction failures. RMSE is '
             'sqrt(mean joint-component squared error); p95 is the percentile of two-joint endpoint norms. '
             'Screens use absolute errors plus numerical envelopes. Sub-envelope differences cannot establish superiority. '
             'Decision failures must be ≤ 1%; conservative valid regret must be ≤ 1% of mean truth zero-command cost. '
             'Zero-command cost ≤ 1e-8 or an empty feasible truth bank leaves the screen unresolved. '
             'Unresolved sampled truth feasibility makes conditional regret undefined and the failure fraction a conservative upper bound. '
             'Prediction, decision and timing partitions are matched in every row.','',
             f'Independent session clusters: {result["uncertainty"]["independent_session_count"]}. '
             'Initialization replicas are averaged within session; configuration screens use those session-averaged metrics. '
             'All paired model/budget/task rows use the same 10,000 fixed-namespace cluster draws. '
             f'Bootstrap namespace: {result["uncertainty"]["namespace"]}. Individual fitted-model screens and all session values '
             'remain in JSON. Passing an average does not mean every fitted model passes. '
             'Intervals are descriptive; five sessions are not powered confirmation or a customer reliability claim.','',
             '| Model | Motion proxy (s) | Main step (ms) | q RMSE (rad) | v RMSE (rad/s) | q p95 (rad) | v p95 (rad/s) | Failure fraction | Valid regret | Scorer median / p95 (s) | Screen | Limited |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|']
    def fmt(value):
        return 'undefined' if value is None else f'{value:.6g}'
    for c in rows:
        lines.append('| '+' | '.join([c['model_id'],str(c['budget_s']),str(c['query_step_ns']/1e6),
            *(fmt(c[k]) for k in ['q_rmse','v_rmse','q_p95','v_p95','decision_failure_fraction','mean_valid_regret']),
            fmt(c['scorer_wall_median_s'])+' / '+fmt(c['scorer_wall_p95_s']),c['screen']['status'],str(c['optimization_limited'])])+' |')
    lines += ['','Threshold brackets are interval-censored; no interpolated data requirement. All budgets, failures and nonmonotone curves remain visible.','',
              '| Model | Step (ms) | First pass (s) | Bracket (s) | Nonmonotone |','|---|---:|---|---|---|']
    for c in result['learning_curves']:
        lines.append(f'| {c["model_id"]} | {c["query_step_ns"]/1e6:g} | {c["first_passing_budget_s"]} | {c["bracket_s"]} | {c["nonmonotone"]} |')
    lines += ['','Resources are separated in JSON: motion seconds proxy; fit execution including validation/selection; '
              'shared evaluation/truth replay/refinement; predictions; decision scoring; timing critical samples; '
              'audit/logging/serialization; original numeric store finalization. Producer spans exclude '
              'post-producer artifact hashing and manifest/marker/receipt/completed-event persistence. '
              'Timing measures one frozen earliest-primary trajectory-0 bank; decision records cover the complete fixed '
              'trajectory library. Network-forward diagnostics never replace complete scorer cost.','',
              result['resources']['fitting_scope']+'. '+result['resources']['projection_scope']+'.','',
              result['resources']['finalization_provenance']+'. These observations validate exact receipt identity and '
              'schema but cannot prove the event log has never been edited. Reuse costs cannot replace original production costs.','',
              'Raw rows, decisions, numerical envelopes, all baselines, selected receipts, missing/failed/limited coordinates, '
              'session values, resource breakdown and bootstrap intervals are in short_calibration_result.json. '
              'Arrays and weights remain external in the selected ignored output store. See short_calibration_reproduce.md '
              'for exact portable acquisition and read-only verification commands.', '',
              'Observed threshold budget ratios and their interval-censored ratio brackets are in JSON. Undefined ratios are not imputed.']
    if result['matrix']=='hybrid':
        lines[0] = '# Hybrid component raw diagnostics'
        lines = [line.replace('five sessions','three sessions') for line in lines]
        lines.insert(2, 'The component gate uses the fixed 30 seconds / 40 ms configuration and all-session eligibility. '
                     'The metrics, plots and threshold curves below describe session-averaged provisional screens; '
                     'they do not replace the component rule. See component_findings.md and component_gate.json for '
                     'every session comparison and pooled actual scorer calls.')
        from .hybrid_report import component_findings
        # Preserve all five raw-report formats; append the component interpretation.
        paths += [destination/'component_findings.md',destination/'component_gate.json']
        atomic_bytes(paths[-2],component_findings(result).encode())
        atomic_json(paths[-1],result['gate'])
    atomic_bytes(paths[0],('\n'.join(lines)+'\n').encode())
    return paths
