"""The new control cohort reuses ten audited fits and trains only twenty missing cells."""
import copy
from pathlib import Path
import pytest
from poc.io import read_json


def request():
    return read_json(Path('config/control_confirmation_training.json'))


def test_registered_training_reuses_original_qualified_fits_without_discovery_seeds():
    from poc.utility_training import validate_training_request
    design=read_json(Path('config/control_confirmation_design.json'))
    value=request()
    assert validate_training_request(value,design)==value
    assert len(value['reused_selected'])==10 and len(value['confirmation_cells_to_train'])==20


@pytest.mark.parametrize('mutation',['missing','duplicate','discovery','objective','phase','reused_model'])
def test_wrong_or_incomplete_nominal_cohort_fails_before_any_training(mutation):
    from poc.utility_training import validate_training_request
    design=read_json(Path('config/control_confirmation_design.json'));value=copy.deepcopy(request())
    if mutation=='missing':value['confirmation_cells_to_train'].pop()
    elif mutation=='duplicate':value['confirmation_cells_to_train'][-1]=copy.deepcopy(value['confirmation_cells_to_train'][0])
    elif mutation=='discovery':value['confirmation_cells_to_train'][0]['seed']=9000
    elif mutation=='objective':value['confirmation_cells_to_train'][0]['loss_mode']='one_step'
    elif mutation=='phase':value['phase']='exploratory'
    else:value['reused_selected'][0]['model_id']='mlp_dt'
    with pytest.raises(ValueError):validate_training_request(value,design)


def test_confirmation_training_cannot_overlap_unfinished_original_nominal_cohort(tmp_path):
    from poc.utility_training import require_completed_development
    from poc.io import atomic_json
    with pytest.raises(ValueError,match='Complete original nominal'):
        require_completed_development(tmp_path)
    atomic_json(tmp_path/'reports/control_nominal_completion.json',dict(state='running',cells=[]))
    with pytest.raises(ValueError,match='Complete original nominal'):
        require_completed_development(tmp_path)


def test_plumbing_requests_are_accepted_by_the_existing_research_command_parser(tmp_path):
    import subprocess,sys
    from poc.utility_training import plumbing_requests
    pairs=plumbing_requests(request())
    assert len(pairs)==4
    for command,parameters in pairs:
        assert set(parameters)=={'actuator','model_id','seed'}
        result=subprocess.run([sys.executable,'-m','poc.research_jobs',command,str(tmp_path/'absent.json')],
                              capture_output=True,text=True)
        assert 'invalid choice' not in result.stderr
        assert 'FileNotFoundError' in result.stderr


def test_reported_complete_development_cannot_hide_wrong_training_objective(tmp_path):
    from poc.utility_training import require_completed_development
    from poc.io import atomic_json,sha256_file
    original=read_json(Path('config/control_development.json'))
    atomic_json(tmp_path/'config/control_development.json',original)
    cells=[]
    for index,cell in enumerate(original['nominal_fits_to_complete']):
        path=tmp_path/'evidence/M30'/f'wrong-{index}'/'manifest.json'
        atomic_json(path,dict(phase='exploratory',milestone_id='M30',job_inputs={'parameters':
            {**cell,'condition':'F','loss_mode':'one_step'}}))
        (path.parent/'weights.pt').write_bytes(b'wrong nominal objective')
        cells.append(dict(cell=cell,manifest=str(path.relative_to(tmp_path)),manifest_sha256=sha256_file(path)))
    atomic_json(tmp_path/'reports/control_nominal_completion.json',dict(state='complete',cells=cells,
        request_sha256=sha256_file(tmp_path/'config/control_development.json')))
    with pytest.raises(ValueError,match='development fit identity'):
        require_completed_development(tmp_path)


def test_changed_reuse_audit_is_rejected_before_even_a_nominal_dry_run(tmp_path):
    import subprocess,sys
    from poc.io import atomic_json,sha256_file
    design=read_json(Path('config/control_confirmation_design.json'))
    atomic_json(tmp_path/'config/control_confirmation_design.json',design)
    value=request();value['design_sha256']=sha256_file(tmp_path/'config/control_confirmation_design.json')
    atomic_json(tmp_path/'config/control_confirmation_training.json',value)
    atomic_json(tmp_path/'reports/selection_integrity_forty.json',{'changed':True})
    result=subprocess.run([sys.executable,'-m','poc.utility_training','--root',str(tmp_path),'--dry-run'],
                          capture_output=True,text=True)
    assert result.returncode!=0 and 'Changed selected-checkpoint audit' in result.stderr
