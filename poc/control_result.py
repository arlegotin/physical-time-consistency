"""Numerical prerequisites and honest rendering for the limited utility design."""
from __future__ import annotations

from .control_confirmation import ACTUATORS
from .control_numerical import compare_refinement
from .control_utility import seed_costs, utility_bounds


def solver_effect_diagnostics(costs, cells):
    """Route large planned-cost discrepancies to a same-metric closed-loop audit."""
    import math
    indexed={(r['actuator'],r['model_id'],r['model_step_ns']):r['cost_by_seed'] for r in costs['rows']}
    result=[];seen=set()
    for cell in cells:
        if cell['cell_id'] in seen:
            raise ValueError('Duplicated solver effect diagnostic cell')
        seen.add(cell['cell_id'])
        a,h=cell['actuator'],cell['model_step_ns']
        baseline=indexed[(a,'node_generic',h)];method=indexed[(a,'power_correct',h)]
        if len(baseline)!=len(method) or len(baseline)!=len(costs['seed_ids']):
            raise ValueError('Complete paired solver effect clusters required')
        effect=math.fsum(b-m for b,m in zip(baseline,method))/len(baseline)
        shift=cell['maximum_absolute_cost_difference']
        if (not math.isfinite(effect) or shift is not None
                and (type(shift) not in (int,float) or not math.isfinite(shift) or shift<0)):
            raise ValueError('Finite solver cost discrepancy and paired effect required')
        result.append(dict(**cell,criterion_id=f'utility_control/{a}/node_generic/h{h}',
            paired_mean_episode_effect=effect,absolute_mean_episode_effect=abs(effect),
            diagnostic_cost_threshold=.1*abs(effect),
            requires_closed_loop_solver_audit=shift is None or shift>.1*abs(effect),
            interpretation='Saved 240 ms planned-cost discrepancy versus the matched mean realized episode effect is a conservative diagnostic trigger; it does not certify realized-cost convergence'))
    return result


def qualify_numerics(primary, refined, design, *, look_index, scope, solver_cells_required, solver_resolution=None):
    """Check a proposed look's sensitivity; no ledger reservation or sign decision."""
    if (not isinstance(solver_cells_required,list)
            or any(not isinstance(cell,str) or not cell for cell in solver_cells_required)
            or len(set(solver_cells_required))!=len(solver_cells_required)):
        raise ValueError('Explicit unique solver audit cell identities required')
    comparison=compare_refinement(primary,refined,design,scope=scope)
    envelopes={key:dict(value) for key,value in comparison['envelopes'].items()}
    solver_ready=False
    if solver_resolution is not None:
        validation=solver_resolution.get('validation',{})
        solver_ready=(solver_resolution.get('state')=='qualified'
            and solver_resolution.get('evidence_verified') is True
            and solver_resolution.get('required_actions')==[]
            and solver_resolution.get('parent_cells')==sorted(solver_cells_required)
            and validation.get('state')=='complete' and validation.get('n_fits')==10)
        if solver_ready:
            import math
            expected={f'utility_control/{a}/node_generic/h40000000' for a in ACTUATORS}
            values=solver_resolution.get('envelopes',{})
            if set(values)!=expected or any(set(value)!={'lower','upper'} or any(
                    type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in value.values())
                    for value in values.values()):
                raise ValueError('Complete finite outward solver envelopes required')
            for key,value in values.items():
                for side in ('lower','upper'):envelopes[key][side]+=value[side]
    bounds=utility_bounds(seed_costs(primary,design),design,look_index,envelopes)
    near=sorted({row['actuator'] for row in bounds['criteria']
                 if row['status'] not in ('lower_cost','higher_cost')})
    actions=[]
    affected=sorted(set(comparison['full_cells_required']) | (set(near) if scope=='subset' else set()))
    if affected:
        actions.append(dict(branch='physics_full' if scope=='subset' else 'physics_further_refinement',
                            actuators=affected))
    if solver_cells_required and not solver_ready:
        actions.append(dict(branch='solver_closed_loop',cells=sorted(solver_cells_required)))
    return dict(schema_version=1,state='additional_numerical_evidence_required' if actions else 'qualified',
        proposed_look_index=look_index,scope=scope,near_boundary_actuators=near,
        comparison=comparison,envelopes=envelopes,required_actions=actions,solver_resolution=solver_resolution,
        statistical_look_reserved=False,scientific_decision='not_evaluated',
        interpretation='Empirical physics discrepancy, not a certified error bound; solver changes require their own audit')


def render_result(result, costs, reference_counts):
    """Every signed comparison and every retained failure stays visible."""
    # The renderer checks the result coordinates without looking up measured files.
    criteria=result.get('criteria',[])
    expected={(a,m,h) for a in ACTUATORS for m in ('mlp_dt','node_generic') for h in (40_000_000,5_000_000)}
    coordinates=[(r['actuator'],r['competitor'],r['model_step_ns']) for r in criteria]
    allowed={'lower_cost','higher_cost','unresolved','unresolved_variance','unresolved_influence'}
    if (len(coordinates)!=8 or set(coordinates)!=expected
            or any(r['status'] not in allowed for r in criteria)
            or costs.get('n_episodes')!=1200 or costs.get('seed_ids')!=list(range(5))):
        raise ValueError('Complete registered signed control results required')
    rows=costs['rows']
    expected_costs={(a,m,h) for a in ACTUATORS for m in ('mlp_dt','node_generic','power_correct')
                    for h in (40_000_000,5_000_000)}
    if len(rows)!=12 or {(r['actuator'],r['model_id'],r['model_step_ns']) for r in rows}!=expected_costs:
        raise ValueError('Complete retained cost/failure/success groups required')
    fallback=sum(sum(r['fallback_by_seed']) for r in rows)
    success=sum(sum(r['success_by_seed']) for r in rows)
    def number(value):
        return 'unresolved' if value is None else f'{value:.9g}'
    lines=['# Limited control utility confirmation','',
        result['scope']+'.','',
        'This result uses 5 independent training seeds with 20 paired episodes per seed and condition: 1,200 complete episodes. '
        'Twenty episodes improve each seed mean; they do not create twenty independent training replications.','',
        '| Plant | Competitor | Model step (ms) | Evidence | Baseline mean cost | Head mean cost | Baseline minus head | Lower bound | Upper bound |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in criteria:
        bound=row['bounds']
        fields=[row['actuator'],row['competitor'],str(row['model_step_ns']//1_000_000),row['status'],
            number(row['baseline_mean_cost']),number(row['method_mean_cost']),number(row['point_difference']),
            number(bound['lower']),number(bound['upper'])]
        lines.append('| '+' | '.join(fields)+' |')
    lines.extend(['','Positive cost differences favor the head; negative differences favor the competitor. '
        'Ratios are descriptive. Signed bounds include empirical numerical discrepancies and paired training-seed uncertainty. '
        'Unresolved variance or influential clusters prevent a supported sign.','',
        f"Control look {result['look_index']}: alpha {result['look_alpha']:.9g}, per-direction tail {result['tail_alpha']:.9g}, "
        'eight comparisons and two directional tails. The original prediction ledger remains separate.','',
        f'Fallback intervals: {fallback}. Successful episodes: {success}. All episodes, including model-failure fallbacks, remain in the cost means.','',
        'Reference episodes: '+', '.join(f'{name}: {reference_counts.get(name+"_episodes", "not verified")}'
            for name in ('physics_fit','zero_command','sim_oracle'))+'. One fitted physical model per plant; task streams are not independent physics-model fits.','',
        'Original PTCO all-claims PASS is not established. This three-model clean fixed-clock comparison does not establish the untested full competitor suite, '
        'changed actuator rates, observation noise, external disturbances, or physical-robot behavior.',''])
    solver=result.get('solver_resolution')
    if solver is not None:
        lines.extend(['','The closed-loop NODE solver audit retains the original one-substep-trained discrete model and costs. '
            'The largest observed subset cost discrepancy expands nominal NODE uncertainty; it is an empirical envelope, not a certified bound.',''])
        if solver['validation']['material_nominal_fit_change']:
            lines.extend(['Solver refinement materially changes the nominal validation objective: the learned field is discretization-specific. '
                'Refined evaluation is diagnostic and does not replace the comparator. A continuous-field claim requires the separate registered solver-training interpretation.',''])
    return '\n'.join(lines)


def evaluate_evidence(root,evidence,*,report=False):
    """Consume independently closed evidence; never launch scientific jobs here."""
    import hashlib
    import math
    import copy
    from pathlib import Path
    from .io import atomic_json,atomic_bytes,job_lock,read_json,safe_path,sha256_file
    from .control_utility import reserve_control_look
    root=Path(root).resolve()
    evidence=copy.deepcopy(evidence)
    def verify_inputs():
        for name,digest in evidence['input_hashes'].items():
            if sha256_file(safe_path(root,name))!=digest:
                raise ValueError('Changed closed utility input: '+name)
    key=hashlib.sha256(b'limited utility result qualification and report').hexdigest()
    with job_lock(key,root):
        verify_inputs()
        design=evidence['design']
        qualification_path=root/'reports/control_confirmation_numerical.json'
        registry_hash=sha256_file(root/'config/control_confirmation_cells.json')
        batch_hash=sha256_file(root/'reports/control_confirmation_batch.json')
        ledgers=[read_json(p) for p in sorted((root/'decisions/control_utility_looks').glob('look-*.json'))]
        look_index=len(ledgers)+1
        matching=[]
        for record in ledgers:
            identity=record['evidence_identity']
            archived=root/f"decisions/control_utility_numerics/{identity['numerical_sha256']}.json"
            if not archived.exists() or sha256_file(archived)!=identity['numerical_sha256']:
                raise ValueError('Missing or changed immutable utility numerical evidence')
            previous=read_json(archived)
            if (identity['registry_sha256']==registry_hash and identity['batch_sha256']==batch_hash
                    and previous.get('input_hashes')==evidence['input_hashes']
                    and previous.get('scope')==evidence['scope']
                    and previous.get('evaluator_sha256')==sha256_file(Path(__file__))):
                matching.append(record)
        if len(matching)>1:
            raise ValueError('Duplicate registered utility evidence look')
        if matching:look_index=matching[0]['look_index']
        qualification=qualify_numerics(evidence['primary'],evidence['refined'],design,
            look_index=look_index,scope=evidence['scope'],solver_cells_required=evidence['solver_cells_required'],
            solver_resolution=evidence.get('solver_resolution'))
        qualification.update(input_hashes=evidence['input_hashes'],
            evaluator_sha256=sha256_file(Path(__file__)),
            solver_effect_diagnostics=evidence.get('solver_effect_diagnostics',[]))
        atomic_json(qualification_path,qualification)
        if not report or qualification['state']!='qualified':
            return qualification
        verify_inputs()
        identity=dict(registry_sha256=registry_hash,batch_sha256=batch_hash,
                      numerical_sha256=sha256_file(qualification_path))
        result=utility_bounds(evidence['seed_costs'],design,look_index,qualification['envelopes'])
        refs={name:sum(r['model_id']==name for r in evidence['references'])
              for name in ('physics_fit','zero_command','sim_oracle')}
        if refs!={'physics_fit':200,'zero_command':200,'sim_oracle':50}:
            raise ValueError('Complete registered same-task reference evidence required')
        resources={}
        for group,rows in evidence['resources'].items():
            for row in rows:
                if any(not math.isfinite(row[k]) or row[k]<0
                       for k in ('wall_seconds','cpu_seconds','peak_rss_bytes','disk_bytes')):
                    raise ValueError('Finite measured resource records required')
            resources[group]=dict(wall_seconds=math.fsum(r['wall_seconds'] for r in rows),
                cpu_seconds=math.fsum(r['cpu_seconds'] for r in rows),
                peak_rss_bytes=max(r['peak_rss_bytes'] for r in rows),
                artifact_bytes=sum(r['disk_bytes'] for r in rows),
                interpretation='Completed producer records; excludes training and failed/interrupted attempts')
        result.update(schema_version=1,evidence_identity=identity,input_hashes=evidence['input_hashes'],
            reference_counts=refs,seed_costs=evidence['seed_costs'],resources=resources,
            solver_effect_diagnostics=qualification['solver_effect_diagnostics'],
            solver_resolution=qualification.get('solver_resolution'),
            status='limited_scope_results',original_ptco_terminal_verdict=None,
            required_remaining_work='Reproduction, full-branch review and any registered unresolved-precision/solver branches')
        markdown=render_result(result,evidence['seed_costs'],{k+'_episodes':v for k,v in refs.items()})
        verify_inputs()
        # All coverage, resources and rendering must validate before spending a look.
        archived=root/f"decisions/control_utility_numerics/{identity['numerical_sha256']}.json"
        if archived.exists():
            if sha256_file(archived)!=identity['numerical_sha256']:
                raise ValueError('Changed immutable utility numerical evidence')
        else:
            atomic_bytes(archived,qualification_path.read_bytes())
        look=reserve_control_look(root,design,identity)
        if look['look_index']!=look_index:
            raise ValueError('Utility look changed during report qualification')
        path=root/'reports/control_confirmation_result.json'
        if path.exists() and read_json(path)!=result:
            # A changed look produces a new named immutable result, not an overwritten old observation.
            previous=read_json(path)
            archive=root/f"reports/control_confirmation_look{previous['look_index']}.json"
            if archive.exists() and read_json(archive)!=previous:
                raise ValueError('Conflicting archived utility report')
            atomic_json(archive,previous)
        atomic_json(path,result)
        atomic_bytes(root/'reports/control_confirmation_result.md',markdown.encode())
        return result


def main(argv=None):
    import argparse
    from pathlib import Path
    from .control_audit_closure import load_evidence
    from .control_solver_validation import load_closed_loop_evidence
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--scope',choices=('subset','full'),default='subset')
    parser.add_argument('--report',action='store_true',help='Reserve a look only after numerical/reference qualification')
    args=parser.parse_args(argv)
    evidence=load_closed_loop_evidence(args.root,load_evidence(args.root,scope=args.scope))
    result=evaluate_evidence(args.root,evidence,report=args.report)
    print({k:result[k] for k in ('state','required_actions','status','look_index') if k in result})
    return 0


if __name__=='__main__':
    raise SystemExit(main())
