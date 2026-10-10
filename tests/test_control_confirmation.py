"""The amended utility study cannot silently alter its prospective comparisons."""
import copy
from pathlib import Path
import pytest
from poc.io import read_json


def design():
    return copy.deepcopy(read_json(Path('config/control_confirmation_design.json')))


def test_registered_design_keeps_eight_signed_contrasts_and_thirty_checkpoints():
    from poc.control_confirmation import validate_design, expected_checkpoints
    value = design()
    assert validate_design(value) == value
    coordinates = expected_checkpoints(value)
    assert len(coordinates) == 30
    assert ('lagged', 'power_correct', 4) in coordinates
    assert len(value['criteria']) == 8


@pytest.mark.parametrize('mutation', ['clock', 'model', 'duplicate_seed', 'discovery_seed',
                                     'effect_floor', 'missing_contrast', 'statistical_family'])
def test_changed_scientific_design_fails_before_checkpoint_or_episode_access(mutation):
    from poc.control_confirmation import validate_design
    value = design()
    if mutation == 'clock': value['hardware_clocks_ns']['observation'] = 5_000_000
    elif mutation == 'model': value['models'].remove('node_generic')
    elif mutation == 'duplicate_seed': value['initial_training_seed_ids'][-1] = 0
    elif mutation == 'discovery_seed': value['initial_training_seed_ids'][-1] = 9000
    elif mutation == 'effect_floor': value['criteria'][0]['effect_floor'] = .2
    elif mutation == 'missing_contrast': value['criteria'].pop()
    else: value['statistical_design']['family_size'] = 4
    with pytest.raises(ValueError): validate_design(value)


def test_incomplete_or_duplicated_selected_fits_cannot_write_a_complete_freeze(tmp_path):
    from poc.control_confirmation import freeze_cells
    with pytest.raises(ValueError, match='checkpoint coverage'):
        freeze_cells(tmp_path, design(), [])
    selected = [dict(actuator='direct', model_id='power_correct', seed=0,
                     manifest='missing/manifest.json', manifest_sha256='a'*64,
                     checkpoint_sha256='b'*64)] * 30
    with pytest.raises(ValueError, match='checkpoint coverage'):
        freeze_cells(tmp_path, design(), selected)
    assert not (tmp_path/'config/control_confirmation_cells.json').exists()


def test_dry_run_reports_missing_prerequisites_without_running_science(tmp_path):
    import json
    import subprocess
    import sys
    from poc.io import atomic_json
    atomic_json(tmp_path/'config/control_confirmation_design.json', design())
    result = subprocess.run([sys.executable, '-m', 'poc.control_confirmation', '--root', str(tmp_path),
                             '--dry-run'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report['execute'] is False and report['expected_episodes'] == 1200
    assert report['checkpoint_freeze_complete'] is False
    assert report['missing_checkpoint_count'] == 30
    assert not (tmp_path/'reports').exists()


def test_unverified_complete_flag_cannot_make_confirmation_ready(tmp_path):
    import subprocess
    import sys
    from poc.io import atomic_json
    atomic_json(tmp_path/'config/control_confirmation_design.json', design())
    atomic_json(tmp_path/'config/control_confirmation_cells.json',
                {'state': 'complete', 'expected_episodes': 1200})
    result = subprocess.run([sys.executable, '-m', 'poc.control_confirmation', '--root', str(tmp_path),
                             '--dry-run'], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'checkpoint_freeze_complete' not in result.stdout


def test_changed_current_planner_cannot_make_an_existing_freeze_ready(tmp_path):
    from poc.control_confirmation import verify_freeze
    from poc.io import atomic_json, sha256_file
    value = design()
    atomic_json(tmp_path/'config/control_confirmation_design.json', value)
    atomic_json(tmp_path/'config/planner.json', {'settings': {'candidate_count': 64}})
    freeze = dict(state='complete', protocol_id=value['protocol_id'], expected_episodes=1200,
                  design_sha256=sha256_file(tmp_path/'config/control_confirmation_design.json'))
    with pytest.raises(ValueError, match='Changed common planner'):
        verify_freeze(tmp_path, value, freeze)


def test_even_hashed_unit_checkpoints_cannot_be_confirmation_fits(tmp_path):
    from poc.control_confirmation import verify_freeze, expected_checkpoints, REGISTRY_SOURCES
    from poc.control_jobs import SOURCES
    from poc.io import atomic_json, sha256_file
    import shutil
    value = design()
    atomic_json(tmp_path/'config/control_confirmation_design.json', value)
    selected=[]
    for actuator,model,seed in sorted(expected_checkpoints(value)):
        folder=tmp_path/'evidence/M30'/f'{actuator}-{model}-{seed}'
        folder.mkdir(parents=True)
        atomic_json(folder/'manifest.json', dict(phase='unit',milestone_id='M30',
            job_inputs={'parameters':dict(actuator=actuator,model_id=model,seed=seed,
                                         condition='A',loss_mode='shared')}))
        (folder/'weights.pt').write_bytes(b'not a qualified robot checkpoint')
        selected.append(dict(actuator=actuator,model_id=model,seed=seed,
            manifest=str((folder/'manifest.json').relative_to(tmp_path)),
            manifest_sha256=sha256_file(folder/'manifest.json'),checkpoint_sha256=sha256_file(folder/'weights.pt')))
    source_root=Path(__file__).resolve().parents[1]
    for name in ('config/planner.json', value['owner_objective_revision']):
        destination=tmp_path/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source_root/name,destination)
    hashes={}
    for name in (*SOURCES,*REGISTRY_SOURCES):
        destination=tmp_path/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source_root/name,destination);hashes[name]=sha256_file(destination)
    jobs=[]
    for item in selected:
        for h in (40_000_000,5_000_000):
            for episode in range(20):
                path=tmp_path/'config/jobs'/f"{item['actuator']}-{item['model_id']}-{item['seed']}-{h}-{episode}.json"
                atomic_json(path,dict(schema_version=1,phase='confirmatory',model_id=item['model_id'],
                    actuator=item['actuator'],seed=item['seed'],episode_id=episode,condition='clean',
                    model_partition_ns=[h]*(40_000_000//h),actuator_interval_ns=40_000_000,
                    physics_step_ns=250_000,power_mode=None,model_manifest=item['manifest'],
                    manifest_sha256=item['manifest_sha256'],checkpoint_sha256=item['checkpoint_sha256'],
                    planner_settings=dict(candidate_count=128,iterations=4,elite_count=16,initial_std=1.,std_floor=.1),
                    resource_projection={'reason':'synthetic complete-coverage mutation'}))
                jobs.append(dict(path=str(path.relative_to(tmp_path)),sha256=sha256_file(path)))
    freeze=dict(state='complete',protocol_id=value['protocol_id'],expected_episodes=1200,
        design_sha256=sha256_file(tmp_path/'config/control_confirmation_design.json'),selected=selected,
        evaluator_hashes=hashes,jobs=jobs)
    with pytest.raises(ValueError,match='fit identity'):
        verify_freeze(tmp_path,value,freeze)
