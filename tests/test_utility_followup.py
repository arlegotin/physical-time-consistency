"""One fixed follow-up queue runs audits only after primary control completes."""
from types import SimpleNamespace

import pytest
from poc.io import atomic_json,read_json


@pytest.fixture(autouse=True)
def private_lock_process_metadata(monkeypatch):
    # These tests stub child launch; lock metadata should not be captured as a child job.
    monkeypatch.setattr('poc.io._process_start',lambda pid:'private-fixture-process')


def primary_receipt(root,state='complete'):
    atomic_json(root/'reports/control_confirmation_batch.json',
                dict(state=state,expected_episodes=1200,completed_episodes=1200 if state=='complete' else 3,
                     episodes=[{'resources':dict(disk_bytes=1000,wall_seconds=2.)}]*1200 if state=='complete' else []))


def test_fixed_followup_runs_numerical_references_solver_then_reporting(tmp_path,monkeypatch):
    import poc.utility_followup as module
    primary_receipt(tmp_path);calls=[]
    def launch(argv,**kwargs):
        calls.append(argv[2:])
        if argv[2]=='poc.control_result' and '--report' not in argv:
            atomic_json(tmp_path/'reports/control_confirmation_numerical.json',
                        dict(state='qualified',required_actions=[]))
        return SimpleNamespace(returncode=0,stdout='completed\n',stderr='')
    monkeypatch.setattr(module.subprocess,'run',launch)
    result=module.run_followup(tmp_path,resume=True)
    assert calls==[
        ['poc.control_refinement','--scope','subset','--prepare'],
        ['poc.control_refinement','--scope','subset','--resume'],
        ['poc.control_references','--prepare'],['poc.control_references','--resume'],
        ['poc.confirmation_solver','--prepare'],['poc.confirmation_solver','--resume'],
        ['poc.control_result','--scope','subset'],['poc.control_result','--scope','subset','--report']]
    assert result['state']=='limited_result_ready_for_reproduction_and_review'
    assert result['scientific_verdict'] is None


def test_pending_primary_never_launches_followup_jobs(tmp_path,monkeypatch):
    import poc.utility_followup as module
    primary_receipt(tmp_path,state='running')
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**k:pytest.fail('Primary is incomplete'))
    with pytest.raises(ValueError,match='Complete primary'):
        module.run_followup(tmp_path,resume=True)


def test_required_full_physics_is_run_before_reporting(tmp_path,monkeypatch):
    import poc.utility_followup as module
    primary_receipt(tmp_path);calls=[]
    def launch(argv,**kwargs):
        calls.append(argv[2:])
        if argv[2]=='poc.control_result' and '--report' not in argv:
            scope=argv[argv.index('--scope')+1]
            atomic_json(tmp_path/'reports/control_confirmation_numerical.json',dict(
                state='qualified' if scope=='full' else 'additional_numerical_evidence_required',
                required_actions=[] if scope=='full' else [dict(branch='physics_full',actuators=['direct'])]))
        return SimpleNamespace(returncode=0,stdout='',stderr='')
    monkeypatch.setattr(module.subprocess,'run',launch)
    result=module.run_followup(tmp_path,resume=True)
    assert ['poc.control_refinement','--scope','full','--prepare'] in calls
    assert ['poc.control_refinement','--scope','full','--resume'] in calls
    assert calls[-1]==['poc.control_result','--scope','full','--report']
    assert result['scope']=='full'


def test_unresolved_solver_branch_is_recorded_without_claim_or_report(tmp_path,monkeypatch):
    import poc.utility_followup as module
    primary_receipt(tmp_path);calls=[]
    def launch(argv,**kwargs):
        calls.append(argv)
        if argv[2]=='poc.control_result':
            atomic_json(tmp_path/'reports/control_confirmation_numerical.json',dict(
                state='additional_numerical_evidence_required',
                required_actions=[dict(branch='solver_closed_loop',cells=['changed-node-bank'])]))
        return SimpleNamespace(returncode=0,stdout='',stderr='')
    monkeypatch.setattr(module.subprocess,'run',launch)
    result=module.run_followup(tmp_path,resume=True)
    assert result['state']=='named_diagnosis_required'
    assert result['required_actions']==[dict(branch='solver_closed_loop',cells=['changed-node-bank'])]
    assert not any('--report' in argv for argv in calls)


def test_failed_stage_preserves_diagnostics_and_does_not_run_later_stage(tmp_path,monkeypatch):
    import poc.utility_followup as module
    primary_receipt(tmp_path);calls=[]
    def launch(argv,**kwargs):
        calls.append(argv);return SimpleNamespace(returncode=1,stdout='partial diagnostics',stderr='named failure')
    monkeypatch.setattr(module.subprocess,'run',launch)
    with pytest.raises(RuntimeError,match='named failure'):
        module.run_followup(tmp_path,resume=True)
    receipt=read_json(tmp_path/'reports/utility_followup.json')
    assert receipt['state']=='interrupted' and receipt['scientific_verdict'] is None
    assert len(calls)==1
    assert 'named failure' in (tmp_path/receipt['attempts'][0]['stderr']).read_text()


def test_actual_disk_shortage_preserves_evidence_and_records_the_resume_prerequisite(tmp_path,monkeypatch):
    import poc.utility_followup as module
    primary_receipt(tmp_path)
    monkeypatch.setattr(module.shutil,'disk_usage',lambda root:SimpleNamespace(free=1))
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**k:pytest.fail('Insufficient actual disk'))
    with pytest.raises(RuntimeError,match='Actual disk prerequisite'):
        module.run_followup(tmp_path,resume=True)
    report=read_json(tmp_path/'reports/utility_followup.json')
    assert report['state']=='interrupted' and 'provision more disk' in report['reason']
    assert report['attempts']==[]


def test_resumption_reuses_completed_stage_runners_and_keeps_old_attempts(tmp_path,monkeypatch):
    import poc.utility_followup as module
    primary_receipt(tmp_path);calls=[]
    def launch(argv,**kwargs):
        calls.append(argv)
        if argv[2]=='poc.control_result':
            atomic_json(tmp_path/'reports/control_confirmation_numerical.json',dict(state='qualified',required_actions=[]))
        return SimpleNamespace(returncode=0,stdout='',stderr='')
    monkeypatch.setattr(module.subprocess,'run',launch)
    first=module.run_followup(tmp_path,resume=False)
    second=module.run_followup(tmp_path,resume=True)
    assert second['attempts'][:len(first['attempts'])]==first['attempts']
    assert len(second['attempts'])==2*len(first['attempts'])
    assert all('--resume' in call for call in calls if call[2] in ('poc.control_refinement','poc.control_references','poc.confirmation_solver')
               and '--prepare' not in call)


def test_resuming_full_followup_never_downgrades_numerical_scope(tmp_path,monkeypatch):
    import poc.utility_followup as module
    primary_receipt(tmp_path);calls=[]
    def launch(argv,**kwargs):
        calls.append(argv[2:])
        if argv[2]=='poc.control_result' and '--report' not in argv:
            scope=argv[argv.index('--scope')+1]
            atomic_json(tmp_path/'reports/control_confirmation_numerical.json',dict(
                state='qualified' if scope=='full' else 'additional_numerical_evidence_required',
                required_actions=[] if scope=='full' else [dict(branch='physics_full',actuators=['direct'])]))
        return SimpleNamespace(returncode=0,stdout='',stderr='')
    monkeypatch.setattr(module.subprocess,'run',launch)
    first=module.run_followup(tmp_path,resume=False)
    assert first['scope']=='full'
    calls.clear()
    second=module.run_followup(tmp_path,resume=True)
    assert second['scope']=='full'
    assert not any('--scope' in c and c[c.index('--scope')+1]=='subset' for c in calls)
    assert calls[-1]==['poc.control_result','--scope','full','--report']


def test_completed_children_need_no_new_episode_storage_for_reverification(tmp_path,monkeypatch):
    import poc.utility_followup as module
    from poc.io import sha256_file
    primary_receipt(tmp_path);calls=[]
    for registry,receipt,count,field in [
        ('config/control_refinement_subset.json','reports/control_refinement_subset_batch.json',300,'registry_sha256'),
        ('config/control_reference_jobs.json','reports/control_reference_batch.json',450,'registry_sha256'),
        ('config/control_confirmation_solver_inputs.json','reports/control_confirmation_solver.json',100,'input_registry_sha256')]:
        atomic_json(tmp_path/registry,dict(private_fixture=True))
        atomic_json(tmp_path/receipt,dict(state='complete',expected_episodes=count,completed_episodes=count,
            episodes=[{}]*count,**{field:sha256_file(tmp_path/registry)}))
    def launch(argv,**kwargs):
        calls.append(argv)
        if argv[2]=='poc.control_result':
            atomic_json(tmp_path/'reports/control_confirmation_numerical.json',dict(state='qualified',required_actions=[]))
        return SimpleNamespace(returncode=0,stdout='',stderr='')
    monkeypatch.setattr(module.shutil,'disk_usage',lambda root:SimpleNamespace(free=1))
    monkeypatch.setattr(module.subprocess,'run',launch)
    assert module.run_followup(tmp_path,resume=True)['state']=='limited_result_ready_for_reproduction_and_review'
    assert len(calls)==8


def test_solver_stage_checks_its_remaining_storage_before_running(tmp_path,monkeypatch):
    import poc.utility_followup as module
    primary_receipt(tmp_path);calls=[]
    # Initial refinement and reference checks have space; the next solver check does not.
    available=iter([1_000_000,1_000_000,1])
    monkeypatch.setattr(module.shutil,'disk_usage',lambda root:SimpleNamespace(free=next(available)))
    def launch(argv,**kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0,stdout='',stderr='')
    monkeypatch.setattr(module.subprocess,'run',launch)
    with pytest.raises(RuntimeError,match='Actual disk prerequisite'):
        module.run_followup(tmp_path,resume=True)
    assert not any(call[2]=='poc.confirmation_solver' for call in calls)
    assert read_json(tmp_path/'reports/utility_followup.json')['next_stage_episode_proxy']==100
