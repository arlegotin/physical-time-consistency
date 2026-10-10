import pytest

from poc.milestones.mx import validate_request


def test_repair_request_needs_named_counterexample_and_all_twelve_contract_elements():
    with pytest.raises(ValueError,match='contract'):
        validate_request({'branch':'repair','contract':{}})
    request = {'branch':'repair','contract':{str(i):'S12/MX and S13.4/repair' for i in range(1,13)},
               'request_id':'MX-checkpoint-0001','return_milestone':'M30',
               'unresolved_question':'Does packed checkpoint serialization preserve exact continuation?',
               'counterexample':'4106 versus 18 storages for the larger corpus',
               'changed_files':['poc/train.py','tests/test_training.py'],
               'invalidated_run_ids':['interrupted-parent'],
               'regression_tests':['tests/test_training.py'],
               'resource_projection':{'wall_seconds':13},'protocol_version':'PTCO-1.0'}
    assert validate_request(request) == request
    with pytest.raises(ValueError,match='branch'):
        validate_request({**request,'branch':'architecture-search'})
    with pytest.raises(ValueError,match='file'):
        validate_request({**request,'changed_files':['unrelated.py']})
