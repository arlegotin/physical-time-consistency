"""Representative-input runtime measurements for PTCO-CAL-0.1.

Each timer encloses a complete synchronous operation. Audit, serialization,
receipt validation and hashes are measured separately. Fixed call counts are
measurement controls, never a feedback deadline or an elapsed-time stop rule.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager, ExitStack
import functools
import hashlib
import json
import platform
from pathlib import Path
import resource
import subprocess
import time
import types

import numpy as np
import torch

from .actuator_integrals import response_ratios
from .affine_flow import lag_coefficients
from .calibration_decisions import score_candidates, decision_state_ids, _bank
from .calibration_predictions import _recipe, _selected, load_evaluation
from .calibration_store import CalibrationStore, Receipt
from .io import atomic_bytes, atomic_json, canonical_json
from .models.common import features
from .mpc import WORK_KEYS
from .train import canonical_tensor_hash


def processor_identity() -> str:
    """Read local CPU hardware identification, falling back to observed ISA."""
    if platform.system()=='Darwin':
        try:
            result = subprocess.run(['/usr/sbin/sysctl','-n','machdep.cpu.brand_string'],
                                    capture_output=True,text=True,timeout=5,check=False)
            if result.returncode==0 and result.stdout.strip():
                return result.stdout.strip()
        except (OSError,subprocess.TimeoutExpired):
            pass
    elif platform.system()=='Linux':
        try:
            for line in Path('/proc/cpuinfo').read_text().splitlines():
                if line.startswith('model name'):
                    return line.partition(':')[2].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


class FixedCoefficientCache:
    """Detached interval-only coefficients; differentiable intervals bypass it.

    Keys include exact interval values, shape, tau, dtype and device. No learned
    coefficients or physical field evaluations are cached. Inference tensors
    are converted to ordinary detached tensors before storage for safe reuse.
    """
    def __init__(self):
        self.values = {}
        self.hits = 0
        self.misses = 0
        self.stats_by_helper = defaultdict(lambda:{'hits':0,'misses':0,'differentiable_bypasses':0})

    def key(self, h, tau_s):
        return (tuple(h.shape), h.detach().cpu().numpy().tobytes(), float(tau_s), str(h.dtype), str(h.device))

    def _get(self, name, h, tau_s, operation):
        if h.requires_grad:
            self.stats_by_helper[name]['differentiable_bypasses'] += 1
            return operation(h)
        key = (name, *self.key(h,tau_s))
        if key not in self.values:
            self.misses += 1
            self.stats_by_helper[name]['misses'] += 1
            with torch.inference_mode(False), torch.no_grad():
                value = operation(h.detach().clone())
                self.values[key] = tuple(v.detach() for v in value) if isinstance(value,tuple) else value.detach()
        else:
            self.hits += 1
            self.stats_by_helper[name]['hits'] += 1
        return self.values[key]

    def activation(self, state, command, h):
        beta = self._get('activation',h,.08,lambda x:-torch.expm1(-x/.08))
        w = state[:,4:6]
        return w+(command-w)*beta

    def lag_coefficients(self, h):
        return self._get('flow_lag',h,.08,lag_coefficients)

    def response_ratios(self, h, tau_s):
        return self._get('response',h,tau_s,lambda x:response_ratios(x,tau_s))


@contextmanager
def fixed_coefficient_cache(model):
    """Rebind only this instance's methods; preserve the selected numerical map.

    Copied function globals replace interval helpers without changing historical
    sources, process-wide modules or any field/network weights. Qualified
    PhysicsFit partial arguments survive rebinding. Restore on every exit.
    """
    cache = FixedCoefficientCache()
    saved = {}
    try:
        for name in ('predict','_step'):
            method = getattr(model,name,None)
            partial = method if isinstance(method,functools.partial) else None
            bound = partial.func if partial else method
            if not isinstance(bound,types.MethodType):
                continue
            function = bound.__func__
            scope = dict(function.__globals__)
            for helper in ('activation','lag_coefficients','response_ratios'):
                if helper in scope:
                    scope[helper] = getattr(cache,helper)
            replacement = types.FunctionType(function.__code__,scope,function.__name__,function.__defaults__,function.__closure__)
            replacement.__kwdefaults__ = function.__kwdefaults__
            replacement = types.MethodType(replacement,model)
            if partial:
                replacement = functools.partial(replacement,*partial.args,**partial.keywords)
            saved[name] = (name in model.__dict__,method)
            setattr(model,name,replacement)
        yield cache
    finally:
        for name,(was_local,method) in saved.items():
            if was_local:
                setattr(model,name,method)
            else:
                delattr(model,name)


def _result_record(result):
    if hasattr(result,'next_state'):
        return {'status':'ok' if all(s=='ok' for s in result.status) else 'prediction_failed',
                'work':{k:int(getattr(result,k)) for k in WORK_KEYS}}
    if isinstance(result,torch.Tensor):
        return {'status':'ok' if bool(torch.isfinite(result).all()) else 'nonfinite', 'work':{}}
    if isinstance(result,dict):
        out = {'status':result.get('status','ok'), 'work':result.get('work',{})}
        if 'selected_index' in result:
            out['status'] = 'ok' if all(s=='ok' for s in result['statuses']) else 'prediction_failed'
            out['selection'] = {'selected_index':result['selected_index'],
                'all_predicted_infeasible':result['all_predicted_infeasible'],
                'prediction_failure_count':sum(s!='ok' for s in result['statuses'])}
            out['query_log'] = result['query_log']
        return out
    raise ValueError('Unsupported timing operation result')


def _invoke(operation):
    try:
        return operation()
    except (RuntimeError,ValueError,FloatingPointError) as error:
        return {'status':'predictor_exception:'+type(error).__name__+':'+str(error),'work':{}}


def run_rotated_cells(cells, *, clock=time.perf_counter, cpu_clock=time.process_time):
    """20 excluded warmups then 50 retained rounds, rotating cell order each round.

    Each (coordinate, zero-argument operation) occurs once per round. Result
    inspection/logging and clock injection are outside the operation interval.
    """
    keys = [canonical_json(c) for c,_ in cells]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate timing coordinate')
    comparable = defaultdict(list)
    for cell in cells:
        identity = {k:v for k,v in cell[0].items() if k not in
                    {'model_id','session_id','budget_s','initialization_id'}}
        comparable[canonical_json(identity)].append(cell)
    samples = []
    with torch.inference_mode():
        for phase,count in (('warmup',20),('measured',50)):
            for repetition in range(count):
                rotated = []
                for group in comparable.values():
                    offset = repetition % len(group)
                    rotated.extend(group[offset:]+group[:offset])
                for order,(coordinate,operation) in enumerate(rotated):
                    wall,cpu = clock(),cpu_clock()
                    result = _invoke(operation)
                    cpu_s,wall_s = cpu_clock()-cpu,clock()-wall
                    if phase == 'measured':
                        samples.append({'coordinate':coordinate,'repetition':repetition,'order_index':order,
                            'wall_s':wall_s,'cpu_s':cpu_s,**_result_record(result)})
    return samples


def _fingerprint(result):
    """Byte parity audit excludes internal timers, retaining outputs and work."""
    digest = hashlib.sha256()
    if hasattr(result,'next_state'):
        digest.update(result.next_state.detach().cpu().numpy().tobytes())
        digest.update(canonical_json({'status':result.status,**_result_record(result)}))
    elif 'costs' in result:
        for name in ('costs','anchors','samples','sampled_feasible'):
            digest.update(np.ascontiguousarray(result[name]).tobytes())
        digest.update(canonical_json({k:result[k] for k in
            ('statuses','initial_feasible','selected_index','all_predicted_infeasible','work','query_log')}))
    else:
        digest.update(canonical_json(result))
    return digest.hexdigest()


def summarize_timing(samples: list[dict]) -> dict:
    """Recompute per-cell median/p95, failure coverage and actual work totals."""
    groups = defaultdict(list)
    for sample in samples:
        if any(not np.isfinite(sample[k]) or sample[k]<0 for k in ('wall_s','cpu_s')):
            raise ValueError('Timing durations must be finite and nonnegative')
        groups[canonical_json(sample['coordinate'])].append(sample)
    result = []
    for _,rows in sorted(groups.items()):
        coordinate = rows[0]['coordinate']
        work = defaultdict(int)
        for row in rows:
            for k,v in row['work'].items():
                work[k] += v
        timed = [r for r in rows if r['status'] not in {'fit_failed','not_applicable'}]
        result.append({'coordinate':coordinate,'measured_calls':len(rows),
            'valid_timing_calls':len(timed),
            'decision_critical':coordinate['category'] in {'prediction','complete_scorer'},
            'failure_count':sum(r['status'] not in {'ok','not_applicable'} for r in rows),
            'not_applicable_count':sum(r['status']=='not_applicable' for r in rows),
            'work_totals':dict(work),
            **{k:{'median':float(np.median([r[k] for r in timed])) if timed else None,
                    'p95':float(np.percentile([r[k] for r in timed],95)) if timed else None} for k in ('wall_s','cpu_s')}})
    return {'groups':result,'sample_count':len(samples),
            'critical_categories':['prediction','complete_scorer'],
            'noncritical_categories':['audit','logging_serialization_write','store_finalization'],
            'network_forward_role':'diagnostic; never a substitute for complete scorer'}


def _network_inputs(model,state,command,h):
    # Match each registered architecture's actual field input transforms.
    size = model.network[0].in_features
    if size == 6:
        q,v = state[:,:2],state[:,2:4]
        return torch.cat((torch.stack((q[:,0].sin(),q[:,0].cos(),q[:,1].sin(),q[:,1].cos()),1),v/model.v_input),1)
    effort = state[:,4:6]
    return features(state[:,:4],effort,torch.zeros_like(h),model.v_input)[:,:-1]


def load_timing_finalization(store: CalibrationStore, receipt: Receipt) -> dict:
    """Load exactly one original-production resource observation for a receipt.

    Verify the immutable product first. Events are observational append-only
    logs, not hash-covered evidence; binding and numeric checks cannot certify
    that logs were never edited. Reuse records are ignored. Missing or ambiguous
    original traces are invalid timing resource evidence.
    """
    _,manifest = store.read(receipt)
    if manifest['stage']!='timing':
        raise ValueError('Expected timing receipt for finalization resources')
    try:
        events = [json.loads(line) for line in (store.output_root/'events.jsonl').read_text().splitlines()]
    except (OSError,ValueError) as error:
        raise ValueError('Missing or malformed store finalization event log') from error
    matches = [e for e in events if e.get('event')=='store_finalization' and
               e.get('reused') is False and e.get('receipt')==receipt]
    if not matches:
        raise ValueError('Missing original timing store finalization trace')
    if len(matches)!=1:
        raise ValueError('Ambiguous original timing store finalization traces')
    event = matches[0]
    if any(event.get(k)!=v for k,v in {'protocol_id':manifest['protocol_id'],
            'stage':'timing','job_key':receipt['job_key']}.items()):
        raise ValueError('Store finalization resource identity mismatch')
    resources = event.get('resources',{})
    durations = [category+'_'+kind+'_s' for category in
        ('artifact_hash','metadata_write_hash','receipt_create_validate','completion_event_write','store_finalization')
        for kind in ('wall','cpu')]
    if resources.get('resource_record_write_measured') is not False or any(
            type(resources.get(k)) not in {int,float} or not np.isfinite(resources[k]) or resources[k]<0 for k in durations):
        raise ValueError('Malformed store finalization resource durations')
    return event


def measure_timing(store: CalibrationStore, fit_receipts: list[Receipt], evaluation_receipt: Receipt,
                   *, clock=time.perf_counter, cpu_clock=time.process_time) -> Receipt:
    """Rotate requested selected predictors on one frozen common observed state.

    The full-bank category encloses score_candidates, including every packed
    auxiliary query, construction, 5 ms feasibility, cost and candidate selection.
    Raw identity/audit/files never enter these durations. Failed fits stay visible.
    """
    audit_wall,audit_cpu = clock(),cpu_clock()
    recipe = _recipe(store)
    arrays = load_evaluation(store,evaluation_receipt)
    selected = [_selected(store,r) for r in fit_receipts]
    if not selected:
        raise ValueError('Timing requires at least one selected fit/reference')
    coordinates = [{k:m['request'][k] for k in ('session_id','budget_s','model_id','initialization_id')} for _,m in selected]
    scoring_hashes = [canonical_tensor_hash(model.state_dict()) if model is not None else None for model,_ in selected]
    if len({canonical_json(c) for c in coordinates})!=len(coordinates):
        raise ValueError('Duplicate timing model coordinate')
    state_id = decision_state_ids(arrays,recipe)[0]
    bank = _bank(arrays,state_id,recipe)
    state = torch.from_numpy(arrays['state'][state_id].copy())
    target = arrays['target'][arrays['trajectory_id'][state_id]].copy()
    input_hash = hashlib.sha256(state.numpy().tobytes()+target.tobytes()+bank.tobytes()).hexdigest()
    request = {'recipe_receipt':store.recipe_receipt,'evaluation_receipt':evaluation_receipt,
        'fit_receipts':fit_receipts,'coordinates':coordinates,'timing':recipe['timing'],
        'phase':recipe['phase'],'representative_state_id':state_id,'input_sha256':input_hash,
        'clock_kind':'native' if clock is time.perf_counter and cpu_clock is time.process_time else 'injected_unit'}
    if request['clock_kind']=='injected_unit' and recipe['phase']!='unit':
        raise ValueError('Injected clocks require explicitly nonempirical unit recipes')
    audit = {'audit_wall_s':clock()-audit_wall,'audit_cpu_s':cpu_clock()-audit_cpu}
    def produce(folder):
        from .calibration_study import _execution_settings
        execution = _execution_settings()
        resources = {**execution,'dtype':'float64','processor':processor_identity(),
            'platform':platform.platform(),'machine':platform.machine(),
            'synchronization':'synchronous CPU; no device synchronization',
            'perf_counter_resolution_s':time.get_clock_info('perf_counter').resolution,
            'process_time_resolution_s':time.get_clock_info('process_time').resolution,
            'clock_kind':request['clock_kind'],
            'peak_rss_native_unit':'bytes on macOS; KiB on Linux'}
        ticks = [time.perf_counter() for _ in range(100)]
        deltas = np.diff(ticks)
        resources['observed_positive_timer_delta_s'] = float(np.min(deltas[deltas>0])) if np.any(deltas>0) else None
        cells = []
        parity_wall,parity_cpu = clock(),cpu_clock()
        originals = {}
        # Audits are independent uncached outputs from the same selected object,
        # never alternative trained weights or a different numerical solver.
        with torch.inference_mode():
            for index,(model,_) in enumerate(selected):
                if model is None:
                    continue
                for step in recipe['evaluation']['query_steps_ns']:
                    originals[index,step,'complete_scorer',len(bank)] = _fingerprint(
                        score_candidates(model,state,target,bank,query_step_ns=step))
                    for batch in recipe['timing']['batches']:
                        x = state[None].repeat(batch,1)
                        command = torch.from_numpy(np.repeat(bank[0,0][None],batch,axis=0))
                        h = torch.full((batch,1),step/1e9,dtype=torch.float64)
                        originals[index,step,'prediction',batch] = _fingerprint(_invoke(functools.partial(model.predict,x,command,h)))
        with ExitStack() as stack:
            caches = [stack.enter_context(fixed_coefficient_cache(model)) if model is not None else None for model,_ in selected]
            # Coordinate-first, method-last traversal makes same-input methods
            # adjacent; round rotation shares first/last positions across methods.
            for step in recipe['evaluation']['query_steps_ns']:
                for category,batches in (('network_forward',recipe['timing']['batches']),
                    ('prediction',recipe['timing']['batches']),('complete_scorer',[len(bank)])):
                    for batch in batches:
                        for index,((model,manifest),coordinate) in enumerate(zip(selected,coordinates)):
                            identity = {**coordinate,'category':category,'batch_size':batch,'query_step_ns':step,
                                'state_id':state_id,'trajectory_id':int(arrays['trajectory_id'][state_id]),
                                'anchor_ns':int(arrays['time_ns'][state_id])}
                            x = state[None].repeat(batch,1)
                            command = torch.from_numpy(np.repeat(bank[0,0][None],batch,axis=0))
                            h = torch.full((batch,1),step/1e9,dtype=torch.float64)
                            if model is None:
                                operation = lambda:{'status':'fit_failed','work':{}}
                            elif category=='complete_scorer':
                                operation = functools.partial(score_candidates,model,state,target,bank,query_step_ns=step)
                            elif category=='prediction':
                                operation = functools.partial(model.predict,x,command,h)
                            elif getattr(model,'network',None) is None:
                                operation = lambda:{'status':'not_applicable','work':{}}
                            else:
                                inputs = _network_inputs(model,x,command,h).detach()
                                def operation(model=model,inputs=inputs,batch=batch):
                                    values = model.network(inputs)
                                    return {'status':'ok' if bool(torch.isfinite(values).all()) else 'nonfinite',
                                        'work':{'network_calls':1,'scalar_candidate_evaluations':batch}}
                            cells.append((identity,operation))
                            if model is not None and category!='network_forward':
                                with torch.inference_mode():
                                    if _fingerprint(_invoke(operation))!=originals[index,step,category,batch]:
                                        raise ValueError('Fixed coefficient cache changed selected numerical map')
            parity_resources = {'cache_parity_audit_wall_s':clock()-parity_wall,
                                'cache_parity_audit_cpu_s':cpu_clock()-parity_cpu}
            samples = run_rotated_cells(cells,clock=clock,cpu_clock=cpu_clock)
            cache_stats = [{'coordinate':c,'hits':cache.hits,'misses':cache.misses,
                            'entries':len(cache.values),'by_helper':dict(cache.stats_by_helper)} if cache else {'coordinate':c,'status':'fit_failed'}
                           for c,cache in zip(coordinates,caches)]
        resources['peak_rss_native'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        hash_wall,hash_cpu = clock(),cpu_clock()
        if [canonical_tensor_hash(model.state_dict()) if model is not None else None for model,_ in selected]!=scoring_hashes:
            raise ValueError('Timing operation changed selected model tensors')
        hash_resources = {'selected_tensor_audit_wall_s':clock()-hash_wall,
                          'selected_tensor_audit_cpu_s':cpu_clock()-hash_cpu}
        write_wall,write_cpu = clock(),cpu_clock()
        atomic_json(folder/'registry.json',[c for c,_ in cells])
        atomic_bytes(folder/'samples.jsonl',b''.join(canonical_json(row)+b'\n' for row in samples))
        logging = {'logging_serialization_write_wall_s':clock()-write_wall,
                   'logging_serialization_write_cpu_s':cpu_clock()-write_cpu}
        return {**summarize_timing(samples),'phase':recipe['phase'],
            'empirical_claims_allowed':recipe['empirical_claims_allowed'] and request['clock_kind']=='native',
            'measurement_scope':'representative-input measurements; no feedback deadline claim',
            'resources':resources,'noncritical':{**audit,**parity_resources,**hash_resources,**logging,
                'store_finalization_record':{'path':'events.jsonl','event':'store_finalization',
                    'selection':'unique original-production event bound to exact final receipt; ignore reused records',
                    'integrity':'observational event log, outside immutable artifact hashes'}},
            'cache_parity_audit':{'status':'exact','cell_count':len(originals),
                'scope':'all requested predict batches/partitions and complete scorers; uncached versus cached selected object',
                'output_sha256':{str(key):value for key,value in originals.items()}},
            'cache':cache_stats,'selected_tensor_hashes':[m['summary'].get('canonical_tensor_sha256') for _,m in selected],
            'scoring_tensor_hashes':scoring_hashes,
            'optimization_limited':[m['summary'].get('optimization_limited',False) for _,m in selected]}
    receipt = store.produce('timing',request,produce,measure_finalization=True,clock=clock,cpu_clock=cpu_clock)
    load_timing_finalization(store,receipt)
    return receipt
