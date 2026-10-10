"""The small fixed development matrix and prespecified rejection-only sensitivity cells."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import numpy as np

from .criteria import ACTUATORS,COMPETITORS,STRATA,HORIZONS,expand_criteria,validate_registry
from .io import append_event,atomic_json,canonical_json,job_key,read_json,run_directory,sha256_file,utc_now

CORE = ('mlp_dt','gru_dt','node_generic','power_correct','node_structured')


def cell(actuator,condition,loss,model,seed):
    return {'cell_id':f'{actuator}__{condition}__{loss}__{model}__seed{seed}',
            'actuator':actuator,'condition':condition,'loss_mode':loss,'model_id':model,'seed':seed}


def development_cells():
    return [cell(a,c,'one_step',m,9000) for a in ACTUATORS for c in ('F','A') for m in CORE]+[
            cell(a,'A','shared',m,9000) for a in ACTUATORS for m in CORE]


def confirmation_cells(seeds=range(5)):
    result = []
    for seed in seeds:
        for actor in ACTUATORS:
            result.extend(cell(actor,'A','shared',model,seed) for model in (*COMPETITORS,'power_correct','node_structured'))
            result.extend(cell(actor,'F','one_step',model,seed) for model in
                          ('power_correct','node_structured','mlp_dt','gru_dt','node_generic'))
            result.extend(cell(actor,'A','one_step',model,seed) for model in
                          ('power_correct','node_structured','mlp_dt','gru_dt'))
    return result


def _metrics(summary):
    value = summary['metrics']
    return value if 'cold' in value else next(iter(value.values()))


def development_screen(observations,solver_observations=None):
    checks = []
    for actor in ACTUATORS:
        for stratum in STRATA:
            for horizon in HORIZONS:
                try:
                    method = observations[(actor,'A','shared','power_correct',None)][stratum][str(horizon)]
                    baseline = min(observations[(actor,'A','shared',m,None)][stratum][str(horizon)]['primary_error']
                                   for m in ('mlp_dt','gru_dt','node_generic'))
                    correct_mode = actor
                    correct = observations[(actor,'F','one_step','power_correct',correct_mode)][stratum][str(horizon)]['primary_error']
                    low = observations[(actor,'F','one_step','power_correct','low')][stratum][str(horizon)]['primary_error']
                except (KeyError,TypeError):
                    raise ValueError('Development screen has a missing mandatory observation')
                reduction = 1-method['primary_error']/baseline if baseline > 0 else None
                passed = (reduction is not None and reduction >= .25 and correct < low and
                          all(.5 <= method['gain_'+r+'_median'] <= 2 for r in ('q','v')))
                node_key = (actor,'A','shared','node_generic',None)
                if solver_observations and node_key in solver_observations:
                    coarse = observations[node_key][stratum][str(horizon)]['primary_error']
                    refined = solver_observations[node_key][stratum][str(horizon)]['primary_error']
                    m = method['primary_error']
                    if abs(refined-coarse) > .1*abs(coarse-m) or (.75*coarse-m)*(.75*refined-m)<0:
                        passed = False
                checks.append({'criterion_id':f'G30.screen/{actor}/{stratum}/{horizon}',
                    'status':'pass' if passed else 'unresolved','estimate':reduction,'bounds':None,
                    'threshold':.25,'operator':'>=','reason':'Exploratory error, gain and fixed-weight direction screen; one seed cannot reject'})
    return {'outcome':'GO' if all(c['status']=='pass' for c in checks) else 'INCONCLUSIVE',
            'criterion_evaluations':checks,'terminal_verdict':None}


def sensitivity_contrasts(criterion,records,seeds):
    d = criterion['dimensions']
    family = criterion['criterion_id'].split('/')[0]
    values,lowers,uppers,baselines,methods = [],[],[],[],[]
    baseline_lower,baseline_upper,method_lower,method_upper = [],[],[],[]
    def retrieve(model,mode,seed,condition=None,loss=None):
        key = (d['actuator'],condition or d['training_condition'],loss or d['loss_mode'],model,mode,seed)
        if key not in records:
            raise ValueError('Sensitivity contrast is missing a frozen seed/cell: '+str(key))
        return records[key]
    def stat(summary,statistic):
        row = _metrics(summary)[d['stratum']][str(d['h_ns'])]
        if statistic == 'primary_error':
            return row[statistic],row[statistic+'_lower'],row[statistic+'_upper']
        component,quantile = d['component'],d['statistic']
        return row[component+'_'+quantile],row['error_'+component+'_lower_'+quantile],row['error_'+component+'_upper_'+quantile]
    mode = d['actuator'] if family in ('K02.actuator','K03.jacobian') else None
    required = [('power_correct',mode)]
    if family == 'K02.relative':
        required.append((d['competitor'],None))
    elif family in ('K02.actuator','K03.jacobian'):
        required.append(('power_correct','direct' if family=='K02.actuator' else 'low'))
    for seed in seeds:
        for model,power in required:
            record = retrieve(model,power,seed)
            if record.get('training_qualified') is False:
                return {'validity':'valid','values':[],'seed_ids':list(seeds),'qualification_status':'unresolved',
                        'qualification_reason':'Nominal fitting remains underqualified; retain every seed and diagnose optimization',
                        'needs_branch':'optimizer'}
        if family == 'K02.relative' and d['competitor']=='node_generic':
            base = retrieve('node_generic',None,seed)
            if 'solver_metrics' in base:
                b = stat(base,'primary_error')[0]
                m = stat(retrieve('power_correct',None,seed),'primary_error')[0]
                refined = base['solver_metrics'][d['stratum']][str(d['h_ns'])]['primary_error']
                r = criterion['threshold'] if d['contrast']=='relative' else 0.
                if abs(refined-b) > .1*abs(b-m) or (b-m)*(refined-m)<0 or ((1-r)*b-m)*((1-r)*refined-m)<0:
                    return {'validity':'valid','values':[],'seed_ids':list(seeds),'qualification_status':'unresolved',
                            'qualification_reason':'Same-field two-half-step audit changes the contrast or exceeds 10% of improvement',
                            'needs_branch':'solver'}
    for seed in seeds:
        if family in ('K02.relative','K02.actuator','K03.jacobian'):
            if family == 'K02.relative':
                correct = retrieve('power_correct',None,seed)
                baseline = retrieve(d['competitor'],None,seed)
                r = criterion['threshold'] if d['contrast']=='relative' else 0.
            else:
                correct = retrieve('power_correct',d['actuator'],seed)
                baseline = retrieve('power_correct','direct' if family=='K02.actuator' else 'low',seed)
                r = criterion['threshold']
            m,ml,mh = stat(correct,'primary_error'); b,bl,bh = stat(baseline,'primary_error')
            values.append((1-r)*b-m); lowers.append((1-r)*bl-mh); uppers.append((1-r)*bh-ml)
            baselines.append(b); methods.append(m)
            baseline_lower.append(bl); baseline_upper.append(bh)
            method_lower.append(ml); method_upper.append(mh)
        elif family == 'K02.absolute':
            m,ml,mh = stat(retrieve('power_correct',None,seed),'absolute')
            values.append(criterion['threshold']-m); lowers.append(criterion['threshold']-mh); uppers.append(criterion['threshold']-ml)
        elif family == 'K02.gain':
            row = _metrics(retrieve('power_correct',None,seed))[d['stratum']][str(d['h_ns'])]
            component = d['component']
            values.append(row['gain_'+component+'_fraction'])
            lowers.append(row['gain_'+component+'_certain_fraction']); uppers.append(row['gain_'+component+'_possible_fraction'])
        elif family == 'K02.order':
            rows = [row for row in retrieve('power_correct',None,seed)['slopes']
                    if row['stratum']==d['stratum'] and row['component']==d['component']]
            if not rows or any(row['status'] != 'ok' for row in rows):
                raise ValueError('Sensitivity slope contrast has missing/unresolved probe gains')
            errors,lo,hi = [],[],[]
            for row in rows:
                p = row['theoretical_order']; l,u = row['slope_bounds']
                errors.append(abs(row['slope']-p)); lo.append(0. if l <= p <= u else min(abs(l-p),abs(u-p)))
                hi.append(max(abs(l-p),abs(u-p)))
            probability = .5 if d['statistic']=='median' else .9
            q,ql,qh = [float(np.quantile(x,probability,method='linear')) for x in (errors,lo,hi)]
            values.append(criterion['threshold']-q); lowers.append(criterion['threshold']-qh); uppers.append(criterion['threshold']-ql)
        else:
            raise ValueError('Not a sensitivity contrast: '+family)
    values,lowers,uppers = map(np.asarray,(values,lowers,uppers))
    result = {'validity':'valid','values':values.tolist(),'seed_ids':list(seeds),
              'numerical_envelope':{'lower':float(np.mean(values-lowers)),'upper':float(np.mean(uppers-values))}}
    if baselines:
        result.update(baseline=baselines,method=methods)
        b,m = float(np.mean(baselines)),float(np.mean(methods))
        bl,bh,ml,mh = [float(np.mean(rows)) for rows in (baseline_lower,baseline_upper,method_lower,method_upper)]
        result['ratio_numerical_envelope'] = None if bl <= 0 or b <= 0 else {
            'lower':max(0.,mh/bl-m/b),'upper':max(0.,m/b-max(0.,ml)/bh)}
    return result


def run_worker(root,folder,command,request,resume=False):
    """Fresh imports keep declared semantic dependencies local to each real job."""
    key = job_key({'command':command,'request':request})
    request_path = root/'config/job_requests'/f'{command}-{key}.json'
    atomic_json(request_path,request)
    log = folder/f'{command}-{key}.stdout.log'
    errors = folder/f'{command}-{key}.stderr.log'
    argv = [sys.executable,'-m','poc.research_jobs',command,str(request_path),'--root',str(root)]
    if resume:
        argv.append('--resume')
    append_event(folder/'events.jsonl',{'event':'child_job_start','command':command,'request_hash':key,'utc':utc_now()})
    print(f'PTCO {command}: '+request.get('semantic_inputs',request).get('cell_id',
          '/'.join(str(request.get(k,'')) for k in ('actuator','condition','role'))),flush=True)
    with log.open('w') as out,errors.open('w') as err:
        process = subprocess.run(argv,cwd=root,stdout=out,stderr=err)
    if process.returncode:
        raise RuntimeError(f'Child {command} failed; preserved diagnostics: {errors.relative_to(root)}')
    result = json.loads(log.read_text().splitlines()[-1])
    record = read_json(root/result['manifest_path'])
    from .contracts import verify_run
    verify_run(root,record)
    append_event(folder/'events.jsonl',{'event':'child_job_complete','run_id':record['run_id'],'utc':utc_now()})
    return record


def train_cell(root,folder,definition,phase,resume=False):
    candidates,exposure,ledger = [],100,[]
    for learning_rate in (.001,.0003):
        epochs,previous = exposure,None
        while True:
            parameters = {**definition,'train_manifest':f'data/{definition["actuator"]}/{definition["condition"]}/train/manifest.json',
                  'validation_manifest':f'data/{definition["actuator"]}/{definition["condition"]}/validation/manifest.json',
                  'epochs':epochs,'learning_rate':learning_rate}
            if previous:
                parameters['continue_from'] = str(run_directory(root,previous).relative_to(root))
            job = {'protocol_version':'PTCO-1.0','phase':phase,'purpose':'training','semantic_inputs':parameters}
            record = run_worker(root,folder,'train',job,resume)
            summary = read_json(run_directory(root,record)/'summary.json')
            converged = summary['convergence']['stabilized']
            learned = summary['validation']['one_step'] < .5*summary['constant_velocity_error']
            ledger.append({'manifest':str((run_directory(root,record)/'manifest.json').relative_to(root)),
                 'learning_rate':learning_rate,'epochs':epochs,'nominal_learning':learned,'stabilized':converged,
                 'diagnostic':'Exact analytic learning, gradients and masks already qualified; inspect nominal curves only'})
            atomic_json(folder/(definition['cell_id']+'-nominal_grid.json'),{'attempts':ledger,'effective_batch':256,
                         'maximum_exposure_offered':epochs,'selection':'nominal validation only'})
            if converged:
                candidates.append((summary['validation']['objective'],record,learned))
                exposure = max(exposure,epochs)
                if learned:
                    return min((candidate for candidate in candidates if candidate[2]),key=lambda c:c[0])[1]
                break
            previous,epochs = record,epochs+100
    # A converged valid optimization limitation is retained; the stage routes a named
    # adequacy diagnostic instead of calling its data or measurements invalid.
    return min(candidates,key=lambda c:c[0])[1]


def freeze_recipes(root,path,training,phase):
    from .audit_jobs import SOURCES as AUDIT
    from .training_jobs import SOURCES as TRAIN
    files = sorted(set(AUDIT)|set(TRAIN)|{'poc/mechanism.py','poc/mechanism_run.py','poc/milestones/m30.py','poc/milestones/mr.py'})
    record = {'protocol_version':'PTCO-1.0','statistical_family_size':280,'phase':phase,'created_utc':utc_now(),
              'generation_hashes':{'poc/generate.py':sha256_file(root/'poc/generate.py')},
              'source_hashes':{f:sha256_file(root/f) for f in files},
              'criteria_sha256':sha256_file(root/'config/criteria.json'),
              'checkpoint_hashes':[sha256_file(run_directory(root,r)/'weights.pt') for r in training],
              'training_manifests':[str((run_directory(root,r)/'manifest.json').relative_to(root)) for r in training]}
    if path.exists():
        prior = read_json(path)
        semantic = lambda x:{key:value for key,value in x.items() if key != 'created_utc'}
        if semantic(prior) != semantic(record):
            raise ValueError('A changed nominal freeze must be versioned before new outcomes')
        return prior
    atomic_json(path,record)
    return record
