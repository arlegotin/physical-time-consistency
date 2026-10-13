"""Prospective component claims require paired raw truth and pooled scorer calls."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from poc.calibration_models import HYBRID_MODEL_IDS
from poc.calibration_study import prospective_recipe


def report():
    assert importlib.util.find_spec('poc.hybrid_report'), 'Hybrid component gate is missing'
    from poc import hybrid_report
    return hybrid_report


def outcome(model='hybrid_affine_flow', session=1, **changes):
    result=dict(session_id=session, budget_s=30, initialization_id=0, model_id=model,
        state_id=0, trajectory_id=0, anchor_ns=320000000, query_step_ns=40000000,
        truth_receipt={'job_key':'shared'}, selected_index=1, selected_command_sha256='command',
        selected_truth_cost=1., selected_truth_envelope=.01, selected_unsafe=False,
        no_selection=False, no_true_feasible_bank=False, decision_failure=False,
        truth_feasibility_status='resolved', truth_ranking_status='unresolved')
    result.update(changes)
    return result


def paired(candidate=None, anchor=None):
    return report().paired_cost_comparison(candidate or [outcome()], anchor or [outcome('hybrid_accel_rk4')])


def test_same_command_cancels_shared_truth_even_with_different_candidate_indices():
    result=paired([outcome(selected_index=7)], [outcome('hybrid_accel_rk4', selected_index=1)])
    assert result['status']=='pass'
    assert result['mean_cost_difference']==result['mean_joint_envelope']==0.
    assert result['same_command_count']==1 and result['ranking_ties_disclosed']


@pytest.mark.parametrize('cost,status', [(1.02,'pass'),(1.020001,'fail')])
def test_distinct_commands_use_independent_numerical_intervals(cost,status):
    result=paired([outcome(selected_command_sha256='different', selected_truth_cost=cost)])
    assert result['status']==status and result['mean_joint_envelope']==.02


@pytest.mark.parametrize('change', [dict(selected_unsafe=True),dict(no_selection=True),
    dict(no_true_feasible_bank=True),dict(decision_failure=True),dict(selected_truth_cost=None),
    dict(selected_truth_envelope=None),dict(selected_command_sha256=None),
    dict(truth_feasibility_status='unresolved'),dict(truth_receipt={'job_key':'other'}),
    dict(budget_s=120),dict(initialization_id=1),dict(selected_truth_cost=float('nan'))])
def test_invalid_or_unmatched_decisions_cannot_pass(change):
    candidate=outcome();candidate.update(change)
    assert paired([candidate])['status']=='unresolved'


def test_missing_duplicate_empty_and_failed_anchor_cannot_pass():
    m=report()
    for left,right in [([],[]),([outcome()],[]),([outcome(),outcome()],[outcome()]),
                       ([outcome()],[outcome(decision_failure=True)])]:
        assert m.paired_cost_comparison(left,right)['status']=='unresolved'


def evidence():
    recipe=prospective_recipe(matrix='hybrid')
    rows=[]; outcomes=[]; timing=[]
    for model in HYBRID_MODEL_IDS:
        for session in (1,2,3):
            c=dict(model_id=model, session_id=session, budget_s=30, initialization_id=0)
            screen=dict(status='pass', prediction_failure_count=0, decision_failure_fraction=0.)
            rows.append({**c,'query_step_ns':40000000,'screen':screen,'optimization_limited':True,
                **{component+'_'+metric:.001 for component in ('q','v')
                   for metric in ('conservative_rmse_upper','conservative_p95_upper')}})
            outcomes.append(outcome(model,session))
            coordinate={**c,'query_step_ns':40000000,'category':'complete_scorer','batch_size':256}
            duration=.0075 if model=='hybrid_affine_flow' else .01
            timing.append({'coordinate':coordinate, 'failure_count':0,
                'samples':[dict(coordinate=coordinate,repetition=i,wall_s=duration,cpu_s=duration,status='ok')
                           for i in range(50)]})
    return dict(recipe=recipe, coverage={'complete':True,'failed':[]}, empirical_claims_allowed=True,
                replica_configurations=rows, decision_outcomes=outcomes, timing=timing)


def test_exact_primary_median_boundary_and_capped_fits_are_eligible():
    gate=report().hybrid_gate(evidence())
    assert gate['decision']=='component_advantage'
    assert gate['primary']['budget_s']==30 and gate['primary']['query_step_ns']==40000000
    assert gate['median_saving_fraction']==pytest.approx(.25)
    assert gate['primary_pool_stats']['hybrid_affine_flow']['sample_count']==150
    primary=[c for c in gate['configurations'] if c['budget_s']==30 and c['query_step_ns']==40000000]
    assert all(c['status']=='pass' for c in primary)
    assert all(s['optimization_limited'] for c in primary for s in c['sessions'])


@pytest.mark.parametrize('metric', ['q_conservative_rmse_upper','v_conservative_rmse_upper',
                                  'q_conservative_p95_upper','v_conservative_p95_upper'])
@pytest.mark.parametrize('ratio,expected', [(1.05,'component_advantage'),(1.050001,'no_component_advantage')])
def test_every_session_obeys_all_four_five_percent_bands(metric,ratio,expected):
    result=evidence();result['replica_configurations'][2][metric]*=ratio
    assert report().hybrid_gate(result)['decision']==expected


@pytest.mark.parametrize('failure', ['prediction_failure_count','decision_failure_fraction'])
def test_failure_rates_cannot_be_hidden_by_conditional_regret(failure):
    result=evidence();result['replica_configurations'][1]['screen'][failure]=1
    assert report().hybrid_gate(result)['decision']=='no_component_advantage'


def test_pool_uses_actual_calls_and_p95_guard():
    result=evidence()
    for row in result['replica_configurations']:
        if row['model_id'] in ('hybrid_affine_aware','hybrid_accel_aware'):row['screen']['status']='fail'
    # Averaging group medians (.008, .008, .004) would pass. The actual
    # pool has 76 slow calls and its median .008 misses the 25% saving bar.
    for i,group in enumerate(result['timing'][:3]):
        for sample in group['samples']:sample['wall_s']=.004
        for sample in group['samples'][:26 if i<2 else 24]:sample['wall_s']=.008
    assert report().hybrid_gate(result)['decision']=='no_component_advantage'
    result=evidence()
    for row in result['replica_configurations']:
        if row['model_id'] in ('hybrid_affine_aware','hybrid_accel_aware'):row['screen']['status']='fail'
    result['timing'][0]['samples'][:10]=[
        {**s,'wall_s':.02} for s in result['timing'][0]['samples'][:10]]
    assert report().hybrid_gate(result)['decision']=='no_component_advantage'


def test_cheap_acceptable_solver_prevents_gain_but_poor_quality_speed_is_excluded():
    result=evidence()
    for group in result['timing']:
        if group['coordinate']['model_id']=='hybrid_accel_aware':
            for sample in group['samples']: sample['wall_s']=.009
    gate=report().hybrid_gate(result)
    assert gate['decision']=='cheap_solver_matches'
    assert gate['fastest_acceptable_alternative']=='hybrid_accel_aware'
    for row in result['replica_configurations']:
        if row['model_id']=='hybrid_accel_aware':row['q_conservative_rmse_upper']=.1
    assert report().hybrid_gate(result)['decision']=='component_advantage'


@pytest.mark.parametrize('flow_p95,decision', [(.025,'component_advantage'),(.035,'no_component_advantage')])
def test_slower_cheap_tail_cannot_override_fastest_comparator_guards(flow_p95,decision):
    result=evidence()
    for group in result['timing']:
        model=group['coordinate']['model_id']
        if model=='hybrid_affine_flow':median,p95=.0075,flow_p95
        elif model=='hybrid_accel_rk4':median,p95=.010,.030
        else:median,p95=.020,.020
        # Ten tail calls per session set p95 while leaving median unchanged.
        for sample in group['samples']:
            sample['wall_s']=p95 if sample['repetition']<10 else median
    gate=report().hybrid_gate(result)
    assert gate['decision']==decision
    assert gate['fastest_acceptable_alternative']=='hybrid_accel_rk4'
    assert gate['median_ratio']==pytest.approx(.75)
    assert gate['median_saving_fraction']==pytest.approx(.25)
    assert gate['p95_ratio']==pytest.approx(flow_p95/.030)


@pytest.mark.parametrize('corruption', ['missing_session','duplicate_session','missing_timing','no_samples','duplicate_sample',
                                      'bad_sample_coordinate','bad_sample_status','failed_anchor','unit'])
def test_unresolved_evidence_cannot_make_primary_claim(corruption):
    result=evidence()
    if corruption=='missing_session':result['replica_configurations'].pop(1)
    if corruption=='duplicate_session':result['replica_configurations'].append(result['replica_configurations'][1])
    if corruption=='missing_timing':result['timing'].pop(1)
    if corruption=='no_samples':result['timing'][1].pop('samples')
    if corruption=='duplicate_sample':result['timing'][1]['samples'][0]['repetition']=1
    if corruption=='bad_sample_coordinate':result['timing'][1]['samples'][0]['coordinate']={}
    if corruption=='bad_sample_status':result['timing'][1]['samples'][0]['status']='fit_failed'
    if corruption=='failed_anchor':result['coverage']['failed']=[dict(model_id='hybrid_accel_rk4',session_id=1,budget_s=30,initialization_id=0)]
    if corruption=='unit':result['empirical_claims_allowed']=False
    assert report().hybrid_gate(result)['decision']=='unresolved'


def test_secondary_success_cannot_switch_fixed_primary_headline():
    result=evidence()
    for key in ('replica_configurations','decision_outcomes','timing'):
        extra=deepcopy(result[key])
        for row in extra:
            if key=='timing':
                row['coordinate']['budget_s']=120
                for s in row['samples']:s['coordinate']['budget_s']=120
            else:row['budget_s']=120
        result[key]+=extra
    result['replica_configurations'][0]['screen']['status']='fail'
    gate=report().hybrid_gate(result)
    assert gate['decision']=='no_component_advantage'
    assert any(c['budget_s']==120 and c['status']=='pass' for c in gate['configurations'])


def test_hybrid_reproduction_commands_are_portable_and_use_only_hybrid_matrix():
    from poc.calibration_report import _reproduce
    text=_reproduce({'recipe':prospective_recipe(matrix='hybrid'),'matrix':'hybrid'})
    assert '--matrix hybrid' in text and '--matrix pilot' not in text
    assert 'Complete the pilot' not in text


@pytest.fixture(scope='module')
def public_hybrid(tmp_path_factory):
    report()
    root=tmp_path_factory.mktemp('hybrid-public')/'checkout'
    source=Path(__file__).resolve().parents[1]
    shutil.copytree(source/'poc',root/'poc',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(source/'assets',root/'assets')
    output=root.parent/'unit output with spaces'
    def run(command,*args):
        p=subprocess.run([sys.executable,'-m','poc.calibration_study',command,'--root',str(root),
            '--output-root',str(output),'--matrix','hybrid',*args],cwd=root,capture_output=True,text=True)
        assert p.returncode==0,p.stderr
        return json.loads(p.stdout)
    run('prepare','--unit-fixture-segments','5','--unit-fixture-max-epochs','5',
        '--unit-fixture-evaluation-trajectories','2','--unit-fixture-evaluation-anchors-ns','320000000',
        '--unit-fixture-decision-candidates','2')
    for command in ('collect','train','predict','decisions'):run(command)
    subset_receipts=[run('time','--model',model)['timing_receipt']
                     for model in (*HYBRID_MODEL_IDS,'physics_nominal')]
    subset_coverage=run('verify')
    subset_paths=[Path(p) for p in run('report')['paths']]
    subset_result=json.loads(next(p for p in subset_paths if p.name=='short_calibration_result.json').read_text())
    run('time')
    return root,output,run,(subset_receipts,subset_coverage,subset_result)


def test_disjoint_hybrid_timing_retains_evidence_without_closing_component_coverage(public_hybrid):
    _,_,_,(receipts,coverage,result)=public_hybrid
    # The reduced recipe requires 15 fits plus three references, not a fixed 36.
    assert coverage['expected_fit_count']==15 and coverage['expected_nominal_count']==3
    assert not coverage['complete']
    assert coverage['invalid']==[]
    assert coverage['missing']==[{'stage':'timing','reason':'complete_pooled_request_required',
                                  'expected_coordinate_count':18}]
    assert coverage['counts']['timing']==504
    assert len(result['timing'])==504
    assert {r['receipt']['job_key'] for r in result['resources']['categories'] if r['stage']=='timing'}=={
        r['job_key'] for r in receipts}
    assert result['gate']['decision']=='unresolved'
    assert result['gate']['reasons']==['incomplete_or_invalid_evidence']
    # Even otherwise passing primary-quality/sample evidence cannot bypass coverage.
    passing=evidence();passing['coverage']=coverage
    assert report().hybrid_gate(passing)['decision']=='unresolved'


def test_complete_hybrid_timing_request_supersedes_retained_disjoint_subsets(public_hybrid):
    _,_,run,(receipts,_,_)=public_hybrid
    coverage=run('verify')
    assert coverage['complete'] and coverage['missing']==[] and coverage['invalid']==[]
    assert coverage['counts']['timing']==504
    assert {r['job_key'] for r in coverage['ignored_timing_receipts']}=={r['job_key'] for r in receipts}
    paths=[Path(p) for p in run('report')['paths']]
    result=json.loads(next(p for p in paths if p.name=='short_calibration_result.json').read_text())
    timings=[r for r in result['resources']['categories'] if r['stage']=='timing']
    assert len(timings)==1 and timings[0]['receipt']['job_key'] not in {r['job_key'] for r in receipts}
    assert len(result['timing'])==504
    assert result['gate']['reasons']==['unit_fixture_ineligible_for_empirical_claims']


def test_copied_public_hybrid_workflow_reports_verified_enriched_raw_evidence(public_hybrid):
    root,output,run,_=public_hybrid
    assert run('verify')['complete']
    paths=[Path(p) for p in run('report')['paths']]
    assert len(paths)==7
    result=json.loads(next(p for p in paths if p.name=='short_calibration_result.json').read_text())
    assert result['gate']['decision']=='unresolved'
    assert result['gate']['reasons']==['unit_fixture_ineligible_for_empirical_claims']
    assert {c['model_id'] for c in result['learning_curves']}==set((*HYBRID_MODEL_IDS,'physics_nominal'))
    assert {c['model_id'] for c in result['threshold_ratios']}==set((*HYBRID_MODEL_IDS[1:],'physics_nominal'))
    assert result['reproduction']['bindings']['source']
    for o in result['decision_outcomes']:
        assert 'truth_receipt' in o and 'selected_command_sha256' in o
        if o['decision_failure']:assert o['selected_truth_cost'] is None
    for g in result['timing']:
        if g['coordinate']['category']=='complete_scorer':
            assert [s['repetition'] for s in g['samples']]==list(range(50))
    csv=next(p for p in paths if p.suffix=='.csv').read_bytes()
    assert b'\r\n' not in csv
    assert b'component_eligibility' in csv and b'pooled_scorer_wall_median_s' in csv
    reproduction=next(p for p in paths if p.name=='short_calibration_reproduce.md').read_text()
    assert '--matrix hybrid' in reproduction and '--matrix pilot' not in reproduction
    assert 'Complete the pilot' not in reproduction
    svg=next(p for p in paths if p.suffix=='.svg').read_text()
    assert all(model in svg for model in HYBRID_MODEL_IDS)
    assert json.loads(next(p for p in paths if p.name=='component_gate.json').read_text())==result['gate']
    assert '30 seconds' in next(p for p in paths if p.name=='component_findings.md').read_text()
    raw_findings=next(p for p in paths if p.name=='short_calibration_findings.md').read_text()
    assert 'five sessions' not in raw_findings and 'all-session' in raw_findings
    # The copied checkout is the executing source, with a separately bound raw store.
    assert not (root/'study').exists() and run('verify')['complete']


def test_hybrid_enrichment_preserves_immutable_summaries_and_invalid_selected_nulls(public_hybrid):
    from poc.calibration_study import open_store
    from poc import calibration_report as reader
    from poc.io import read_json
    root,output,_,_=public_hybrid
    store=open_store(root,output,'hybrid')
    marker=next(output.glob('jobs/decisions/*/COMPLETED'))
    receipt={'job_key':marker.parent.name,'path':str(marker.parent.relative_to(output)),
             'manifest_sha256':read_json(marker)['manifest_sha256']}
    folder,manifest=store.read(receipt)
    saved=deepcopy(manifest['summary'])
    enriched=reader._decision_records(store,folder,manifest)
    assert manifest['summary']==saved
    assert all('selected_truth_cost' not in o for o in saved['outcomes'])
    assert all('selected_truth_cost' in o for o in enriched)
    manifest['summary']['outcomes'][0]['regret']=1000.
    with pytest.raises(ValueError,match='summary mismatch'):
        reader._decision_records(store,folder,manifest)
    marker=next(output.glob('jobs/timing/*/COMPLETED'))
    receipt={'job_key':marker.parent.name,'path':str(marker.parent.relative_to(output)),
             'manifest_sha256':read_json(marker)['manifest_sha256']}
    folder,manifest=store.read(receipt)
    saved=deepcopy(manifest['summary'])
    groups,_=reader._timing_records(store,receipt,folder,manifest)
    assert manifest['summary']==saved
    assert all('samples' not in g for g in saved['groups'])
    assert any('samples' in g for g in groups)
    manifest['summary']['groups'][0]['measured_calls']=1000
    with pytest.raises(ValueError,match='summary mismatch'):
        reader._timing_records(store,receipt,folder,manifest)
