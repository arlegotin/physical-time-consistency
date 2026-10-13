"""A complete tiny public-checkout workflow; phase=unit is never empirical evidence."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]


def report_module():
    assert importlib.util.find_spec('poc.calibration_report'), 'Reporting stage is missing'
    from poc import calibration_report
    return calibration_report


@pytest.fixture(scope='module')
def complete_unit(tmp_path_factory):
    report_module()
    tmp=tmp_path_factory.mktemp('public-report')
    root=tmp/'checkout'
    shutil.copytree(ROOT/'poc',root/'poc',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(ROOT/'assets',root/'assets')
    out=tmp/'output with spaces'
    def run(command,*args):
        p=subprocess.run([sys.executable,'-m','poc.calibration_study',command,'--root',str(root),
                          '--output-root',str(out),*args],cwd=root,capture_output=True,text=True)
        assert p.returncode==0,p.stderr
        return json.loads(p.stdout)
    run('prepare','--unit-fixture-segments','5','--unit-fixture-max-epochs','5',
        '--unit-fixture-physical-epochs','5','--unit-fixture-evaluation-trajectories','2',
        '--unit-fixture-evaluation-anchors-ns','320000000','--unit-fixture-decision-candidates','2')
    run('collect');run('train');run('predict');run('decisions');run('time')
    from poc.calibration_study import open_store
    return open_store(root,out),run


def test_public_full_workflow_verifies_raw_coverage_and_writes_all_formats(complete_unit,tmp_path):
    m=report_module();store,run=complete_unit
    verified=run('verify')
    assert verified['complete'] and verified['invalid']==[] and verified['missing']==[]
    assert verified['counts']['fit']==7 and verified['counts']['nominal']==1
    result=m.aggregate(store,'pilot')
    assert result['coverage']['complete'] and result['gate']['decision']=='unresolved'
    assert result['phase']=='unit' and result['empirical_claims_allowed'] is False
    assert len(result['configurations'])==32
    assert len(result['timing'])==224
    assert len(result['resources']['timing_finalization'])==1
    assert result['uncertainty']['independent_session_count']==1
    paths=m.write_report(result,tmp_path)
    assert {p.suffix for p in paths}=={'.md','.json','.csv','.svg'} and len(paths)==5
    assert json.loads((tmp_path/'short_calibration_result.json').read_text())==result
    assert len(run('report')['paths'])==5
    assert run('status')['coverage']['complete']
    assert not (store.output_root.parent/'checkout/study').exists()


def test_missing_fit_is_incomplete_and_tampered_raw_is_invalid(complete_unit,tmp_path):
    m=report_module();store,_=complete_unit
    # Receipt bindings include the original output path; read the original store
    # while temporarily moving/removing evidence, then always restore it.
    marker=next(store.output_root.glob('jobs/fit/*/COMPLETED'))
    saved=marker.read_bytes();marker.unlink()
    try:
        result=m.verify(store,'pilot')
        assert not result['complete'] and any(x['stage']=='fit' for x in result['missing'])
    finally: marker.write_bytes(saved)
    raw=next(store.output_root.glob('jobs/decisions/*/outcomes.jsonl'))
    saved=raw.read_bytes();raw.write_bytes(saved+b'{}\n')
    try:
        result=m.verify(store,'pilot')
        assert not result['complete'] and result['invalid']
        with pytest.raises(ValueError,match='Invalid'):
            m.aggregate(store,'pilot')
    finally: raw.write_bytes(saved)


def test_pilot_to_development_expansion_does_not_double_count_timing(complete_unit):
    m=report_module();store,run=complete_unit
    run('prepare','--matrix','development','--unit-fixture-segments','5','--unit-fixture-max-epochs','5',
        '--unit-fixture-physical-epochs','5','--unit-fixture-evaluation-trajectories','2',
        '--unit-fixture-evaluation-anchors-ns','320000000','--unit-fixture-decision-candidates','2')
    run('time','--matrix','development','--model','physics_nominal','--session','0')
    pilot=run('verify','--matrix','pilot')
    development=run('verify','--matrix','development')
    assert pilot['complete'] and len(pilot['ignored_timing_receipts'])==1
    assert not development['complete'] and development['invalid']==[]
    assert development['counts']['timing']==224
    assert len(m.aggregate(store,'pilot')['timing'])==224


def test_raw_outcome_duplicates_and_untruthful_summaries_rejected(complete_unit):
    m=report_module();store,_=complete_unit
    marker=next(store.output_root.glob('jobs/decisions/*/COMPLETED'))
    from poc.io import read_json
    receipt={'job_key':marker.parent.name,'path':str(marker.parent.relative_to(store.output_root)),
             'manifest_sha256':read_json(marker)['manifest_sha256']}
    folder,manifest=store.read(receipt)
    raw=folder/'outcomes.jsonl';saved=raw.read_bytes()
    raw.write_bytes(saved+saved.splitlines(keepends=True)[0])
    try:
        with pytest.raises(ValueError,match='Duplicate'):
            m._decision_records(store,folder,manifest)
    finally: raw.write_bytes(saved)
    manifest['summary']['outcomes'][0]['regret']=1000.
    with pytest.raises(ValueError,match='summary mismatch'):
        m._decision_records(store,folder,manifest)


def test_valid_failed_fit_is_retained_with_undefined_predictions(tmp_path,monkeypatch):
    m=report_module()
    from poc.calibration_study import prepare,open_store,collect_session
    from poc import calibration_training as training
    from poc.calibration_predictions import prepare_evaluation,prediction_truth,measure_prediction
    out=tmp_path/'failed-unit'
    prepare(ROOT,out,unit_fixture={'segment_count':5,'max_epochs_per_rate':5,
        'evaluation_trajectory_count':2,'evaluation_anchor_times_ns':[320_000_000],'decision_candidate_count':2})
    store=open_store(ROOT,out)
    session=collect_session(0,root=ROOT,output_root=out)
    def numerical_failure(*args,**kwargs):
        raise FloatingPointError('unit numerical method failure')
    monkeypatch.setattr(training,'batch_objective',numerical_failure)
    fitted=training.fit_model(store,session,'affine_flow',5,0)
    evaluation=prepare_evaluation(store,root=ROOT)
    truth=prediction_truth(store,evaluation,root=ROOT)
    measure_prediction(store,fitted,evaluation,truth)
    result=m.aggregate(store,'pilot')
    assert result['coverage']['invalid']==[] and len(result['coverage']['failed'])==1
    assert not result['coverage']['complete']
    own=[r for r in result['metrics'] if r['model_id']=='affine_flow']
    assert own and all(r['rmse'] is None and r['p95'] is None and r['failure_fraction']==1. for r in own)


def test_full_status_reports_invalid_raw_evidence_without_hiding_it(complete_unit):
    _,run=complete_unit;store,_=complete_unit
    raw=next(store.output_root.glob('jobs/predictions/*/rows.jsonl.gz'))
    saved=raw.read_bytes();raw.write_bytes(saved+b'{}\n')
    try:
        status=run('status')
        assert not status['coverage']['complete'] and status['coverage']['invalid']
    finally: raw.write_bytes(saved)


def test_malformed_observational_log_is_invalid_resource_evidence(complete_unit):
    m=report_module();store,_=complete_unit
    path=store.output_root/'events.jsonl';saved=path.read_bytes()
    path.write_bytes(saved+b'not-json\n')
    try:
        result=m.verify(store,'pilot')
        assert result['invalid'] and not result['complete']
    finally: path.write_bytes(saved)
