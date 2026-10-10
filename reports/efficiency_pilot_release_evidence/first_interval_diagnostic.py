"""Post-pilot diagnostic of saved selected-plan predictions, without new runs."""
import collections
import hashlib
import json
from pathlib import Path
import numpy as np

root = Path.cwd()
result_path = root/'reports/efficiency_pilot_result.json'
result = json.loads(result_path.read_text())
assert result['status'] == 'complete_exploratory' and len(result['episodes']) == 408

def rmse(delta):
    return float(np.sqrt(np.mean(np.square(delta))))

assert rmse(np.array([[-.2,.2],[.2,-.2]])) == .2
assert rmse(np.zeros((40,2))) == 0

episodes = []
for summary, entry in zip(result['episodes'],result['evidence']):
    if summary['model_id'] == 'sim_oracle':
        continue
    manifest_path = root/entry['path']
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == entry['sha256']
    manifest = json.loads(manifest_path.read_text())
    inputs = {}
    for name in ['decisions.jsonl','true_trajectories.npz']:
        path = manifest_path.parent/name
        expected = next(f['sha256'] for f in manifest['produced_files'] if root/f['path'] == path)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
        inputs[name] = dict(path=str(path.relative_to(root)),sha256=expected)
    decisions = [json.loads(s) for s in (manifest_path.parent/'decisions.jsonl').read_text().splitlines()]
    assert len(decisions) == 40
    with np.load(manifest_path.parent/'true_trajectories.npz',allow_pickle=False) as z:
        truth = z['anchors']
    predicted, gaps = [], []
    for index, decision in enumerate(decisions):
        assert decision['replanning_index'] == index and decision['time_ns'] == index*40_000_000
        assert decision['fallback'] is None and decision['status'] == 'ok'
        np.testing.assert_array_equal(decision['actual_first_interval_state'],truth[index])
        pred = np.asarray(decision['predicted_anchors'],dtype=np.float64)
        assert pred.shape == (6,truth.shape[1]) and np.isfinite(pred).all()
        predicted.append(pred[0])
        gaps.append(decision['predicted_first_interval_cost']-decision['actual_first_interval_cost'])
    delta = np.asarray(predicted)-truth
    episodes.append(dict(actuator=summary['actuator'],model=summary['model_id'],
        hold_ns=summary['actuator_interval_ns'],nominal_partition_ns=summary['model_partition_ns'][0],
        task=summary['episode_id'],q_endpoint_rmse_rad=rmse(delta[:,:2]),
        v_endpoint_rmse_rad_s=rmse(delta[:,2:4]),mean_absolute_first_cost_gap=float(np.mean(np.abs(gaps))),
        manifest=entry,inputs=inputs,n_intervals=40))
assert len(episodes)==400
grouped=collections.defaultdict(list)
for e in episodes:grouped[(e['actuator'],e['model'],e['hold_ns'],e['nominal_partition_ns'])].append(e)
rows=[]
for (actuator,model,hold,partition), items in sorted(grouped.items()):
    assert len(items)==10 and {e['task'] for e in items}==set(range(10))
    rows.append(dict(actuator=actuator,model=model,hold_ns=hold,actual_max_query_ns=min(hold,partition),
        nominal_partition_ns=partition,n_tasks=10,n_intervals_per_task=40,
        **{metric:float(np.mean([e[metric] for e in items])) for metric in
           ['q_endpoint_rmse_rad','v_endpoint_rmse_rad_s','mean_absolute_first_cost_gap']}))
out=dict(protocol_id='PTCO-EFF-0.1',analysis='post_pilot_descriptive_selected_plan_first_interval',
    source_result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),alpha_spent=0,n_training_seeds=1,
    interpretation='Mean of task-level prediction RMSE at the actually executed first 40 ms endpoint. '
                   'Each policy visits different states and chooses different commands; this is not a fixed-probe causal comparison. '
                   'Only the first planned interval is compared with its real outcome; later overlapping replans are excluded.',
    conditions=rows,episodes=episodes)
path=root/'reports/efficiency_pilot_first_interval_diagnostic.json'
path.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
for r in rows:print(r['actuator'],r['model'],r['hold_ns']//1_000_000,r['actual_max_query_ns']//1_000_000,
                   f"q {r['q_endpoint_rmse_rad']:.6g} rad; v {r['v_endpoint_rmse_rad_s']:.6g} rad/s")
