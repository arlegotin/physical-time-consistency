from pathlib import Path
import runpy

import numpy as np
import pytest

from poc.io import atomic_json, sha256_file

SCRIPT = Path(__file__).resolve().parents[1] / 'reports/audit_control_attribution_development.py'


def fixture(tmp_path, monkeypatch, mutation):
    import poc.contracts
    import poc.planner_qualification
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(poc.contracts, 'verify_run', lambda *a, **k: None)
    monkeypatch.setattr(poc.planner_qualification, 'training_recovery_compatibility', lambda root: ({}, []))
    jobs, episodes, index = [], [], 0
    for actuator in ('direct', 'lagged'):
        for episode in range(10):
            for model, mode in [('power_correct','low'), ('power_correct','direct'),
                                ('power_correct','lagged'), ('node_structured',None)]:
                for h in (40_000_000,5_000_000):
                    folder = tmp_path / 'evidence/M40' / str(index); folder.mkdir(parents=True)
                    target = [.1,.2]
                    # This mismatched-power cell is outside the six cells used in A/B.
                    if mutation == 'mismatched_task' and index == 5: target = [.3,.2]
                    row = dict(actuator=actuator,episode_id=episode,model_id=model,power_mode=mode,
                        model_partition_ns=[h] * (40_000_000 // h), seed=9000,phase='exploratory',
                        initial_state=[0.] * (4 if actuator == 'direct' else 6),target=target,
                        actuator_interval_ns=40_000_000,physics_step_ns=250_000,
                        realized_cost=(1.6+5)*sum(x*x for x in target), success=False,fallback_count=0)
                    truth = np.zeros((6401,len(row['initial_state'])))
                    if mutation == 'raw_initial' and index == 0: truth[0,0] = .5
                    raw_target = np.array(target)
                    if mutation == 'raw_target' and index == 0: raw_target[0] = .4
                    np.savez_compressed(folder/'true_trajectories.npz',anchors=truth[160::160],target=raw_target,
                        time_ns=np.arange(0,1_600_000_000+1,250_000,dtype=np.int64),state=truth)
                    atomic_json(folder/'manifest.json',dict(run_id=str(index),job_inputs={}))
                    atomic_json(folder/'episode.json',row);atomic_json(folder/'summary.json',row)
                    atomic_json(folder/'request.json',{'episode':index})
                    request_path=tmp_path/'config'/f'{index}.json';atomic_json(request_path,{'episode':index})
                    import json
                    (folder/'action_segments.jsonl').write_text('\n'.join(json.dumps(dict(
                        start_ns=k*40_000_000,duration_ns=40_000_000,command=[0.,0.])) for k in range(40))+'\n')
                    jobs.append(dict(cell_id=str(index),path=str(request_path.relative_to(tmp_path))))
                    episodes.append(dict(cell_id=str(index),manifest=str((folder/'manifest.json').relative_to(tmp_path)),
                        manifest_sha256=sha256_file(folder/'manifest.json'),summary=row))
                    index+=1
    registry=tmp_path/'config/control_attribution_development_jobs.json';atomic_json(registry,{'jobs':jobs})
    atomic_json(tmp_path/'reports/control_attribution_development.json',dict(state='complete',completed_episodes=160,
        registry_sha256=sha256_file(registry),episodes=episodes,
        contrasts=[dict(actuator=a,A=0.,B=0.,n_training_seeds=1,fraction_explained=None) for a in ('direct','lagged')]))


@pytest.mark.parametrize('mutation', ['mismatched_task','raw_initial','raw_target'])
def test_full_raw_audit_rejects_unpaired_mismatch_and_misreported_raw_task(tmp_path, monkeypatch, mutation):
    fixture(tmp_path,monkeypatch,mutation)
    with pytest.raises(ValueError,match='Raw initial state|Task pairing'):
        runpy.run_path(str(SCRIPT))


def test_full_raw_audit_reconciles_all_cells_without_fabricating_training_replications(tmp_path, monkeypatch):
    from poc.io import read_json
    fixture(tmp_path,monkeypatch,None)
    runpy.run_path(str(SCRIPT))
    result=read_json(tmp_path/'reports/control_attribution_development_audit.json')
    assert result['complete_episode_count']==160 and result['n_training_seeds']==1
    assert all(c['A']==0 and c['B']==0 for c in result['contrasts'])
