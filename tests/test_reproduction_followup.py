"""The final fixed queue cannot bypass numerical readiness or reproduction failures."""
from types import SimpleNamespace

import pytest
from poc.io import atomic_json,read_json,sha256_file


@pytest.fixture(autouse=True)
def private_lock_process_metadata(monkeypatch):
    monkeypatch.setattr('poc.io._process_start',lambda pid:'private-fixture-process')


def ready(root):
    atomic_json(root/'reports/control_confirmation_numerical.json',dict(state='qualified',required_actions=[]))
    atomic_json(root/'reports/control_confirmation_result.json',dict(status='limited_scope_results',
        original_ptco_terminal_verdict=None,input_hashes={},evidence_identity=dict(
            numerical_sha256=sha256_file(root/'reports/control_confirmation_numerical.json'))))
    atomic_json(root/'reports/utility_followup.json',dict(state='limited_result_ready_for_reproduction_and_review',
        required_actions=[],scientific_verdict=None))


def test_fixed_reproduction_queue_runs_replay_then_fresh_training_then_repeated_control(tmp_path,monkeypatch):
    import poc.reproduction_followup as module
    ready(tmp_path);calls=[]
    def child(argv,**kwargs):
        calls.append(argv[2:])
        if argv[2]=='poc.control_replay' and '--prepare' in argv:
            atomic_json(tmp_path/'config/control_replay_jobs.json',dict(resource_projection=dict(estimated_artifact_bytes=60_000)))
        elif argv[2]=='poc.control_retraining' and '--prepare' in argv:
            atomic_json(tmp_path/'config/control_retraining_jobs.json',dict(resource_projection=dict(context_copy_bytes=120_000)))
        elif argv[2]=='poc.control_retrained_policy' and '--prepare' in argv:
            atomic_json(tmp_path/'config/control_retrained_policy_jobs.json',dict(resource_projection=dict(estimated_artifact_bytes=480_000)))
        elif argv[2]=='poc.control_retrained_policy' and '--resume' in argv:
            atomic_json(tmp_path/'reports/control_retrained_policy_result.json',dict(status='pass',diagnosis_required=False,
                criteria_unchanged=True,alpha_spent=0,new_independent_training_clusters=0))
        return SimpleNamespace(returncode=0,stdout='completed\n',stderr='')
    monkeypatch.setattr(module.subprocess,'run',child)
    result=module.run_followup(tmp_path,resume=True)
    assert calls==[['poc.control_replay','--prepare'],['poc.control_replay','--resume'],
        ['poc.control_retraining','--prepare'],['poc.control_retraining','--resume'],
        ['poc.control_retrained_policy','--prepare'],['poc.control_retrained_policy','--resume']]
    assert result['state']=='reproduction_ready_for_whole_branch_review'
    assert result['scientific_verdict'] is None and result['whole_branch_review_completed'] is False


@pytest.mark.parametrize('state',['pending','running','named_diagnosis_required','interrupted'])
def test_unready_numerical_followup_cannot_launch_reproduction(tmp_path,monkeypatch,state):
    import poc.reproduction_followup as module
    atomic_json(tmp_path/'reports/utility_followup.json',dict(state=state,required_actions=[{'branch':'numerical'}]))
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**k:pytest.fail('Prerequisite is unready'))
    with pytest.raises(ValueError):module.run_followup(tmp_path,resume=True)


def test_stale_numerical_identity_cannot_be_used_as_reproduction_permission(tmp_path,monkeypatch):
    import poc.reproduction_followup as module
    ready(tmp_path);atomic_json(tmp_path/'reports/control_confirmation_numerical.json',dict(state='changed'))
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**k:pytest.fail('Numerical identity is stale'))
    with pytest.raises(ValueError):module.run_followup(tmp_path,resume=True)


def test_child_failure_retains_attempt_and_does_not_launch_next_stage(tmp_path,monkeypatch):
    import poc.reproduction_followup as module
    ready(tmp_path);calls=[]
    def child(argv,**kwargs):
        calls.append(argv);return SimpleNamespace(returncode=1,stdout='partial replay',stderr='reproduction mismatch')
    monkeypatch.setattr(module.subprocess,'run',child)
    with pytest.raises(RuntimeError,match='reproduction mismatch'):module.run_followup(tmp_path,resume=True)
    report=read_json(tmp_path/'reports/reproduction_followup.json')
    assert report['state']=='interrupted' and len(calls)==1
    assert (tmp_path/report['attempts'][0]['stderr']).read_text()=='reproduction mismatch'


def test_actual_disk_shortage_preserves_projected_stage_without_starting_it(tmp_path,monkeypatch):
    import poc.reproduction_followup as module
    ready(tmp_path);calls=[]
    def child(argv,**kwargs):
        calls.append(argv)
        atomic_json(tmp_path/'config/control_replay_jobs.json',dict(resource_projection=dict(estimated_artifact_bytes=60_000)))
        return SimpleNamespace(returncode=0,stdout='',stderr='')
    monkeypatch.setattr(module.subprocess,'run',child)
    monkeypatch.setattr(module.shutil,'disk_usage',lambda root:SimpleNamespace(free=1))
    with pytest.raises(RuntimeError,match='Actual disk prerequisite'):module.run_followup(tmp_path,resume=True)
    assert len(calls)==1 and '--prepare' in calls[0]
    report=read_json(tmp_path/'reports/reproduction_followup.json')
    assert report['actual_free_bytes']==1 and report['projected_new_bytes']==60_000


def test_completed_cached_stages_do_not_require_disk_for_recreating_their_artifacts(tmp_path,monkeypatch):
    import poc.reproduction_followup as module
    ready(tmp_path)
    specs={'poc.control_replay':('config/control_replay_jobs.json','reports/control_replay.json','estimated_artifact_bytes',60,'episodes'),
        'poc.control_retraining':('config/control_retraining_jobs.json','reports/control_retraining.json','context_copy_bytes',12,'fits'),
        'poc.control_retrained_policy':('config/control_retrained_policy_jobs.json','reports/control_retrained_policy_batch.json','estimated_artifact_bytes',480,'episodes')}
    def child(argv,**kwargs):
        registry,receipt,field,count,kind=specs[argv[2]]
        if '--prepare' in argv:
            atomic_json(tmp_path/registry,dict(resource_projection={field:1_000_000}))
            prefix='fits' if kind=='fits' else 'episodes'
            atomic_json(tmp_path/receipt,dict(state='complete',registry_sha256=sha256_file(tmp_path/registry),
                **{'expected_'+prefix:count,'completed_'+prefix:count,kind:[{}]*count}))
        if argv[2]=='poc.control_retrained_policy' and '--resume' in argv:
            atomic_json(tmp_path/'reports/control_retrained_policy_result.json',dict(status='pass',diagnosis_required=False,
                criteria_unchanged=True,alpha_spent=0,new_independent_training_clusters=0))
        return SimpleNamespace(returncode=0,stdout='',stderr='')
    monkeypatch.setattr(module.subprocess,'run',child)
    monkeypatch.setattr(module.shutil,'disk_usage',lambda root:SimpleNamespace(free=1))
    result=module.run_followup(tmp_path,resume=True)
    assert result['state']=='reproduction_ready_for_whole_branch_review' and result['projected_new_bytes']==0
