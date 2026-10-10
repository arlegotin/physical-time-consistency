"""Measured pilot profiles and complete, descriptive quality/compute comparisons."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from .io import atomic_bytes, atomic_json, read_json, safe_path, sha256_file, utc_now


def _coordinate(row):
    return (row['actuator'], row['model_id'], row['actuator_interval_ns'],
            row['model_partition_ns'][0], row['episode_id'])


def aggregate_rows(rows, expected_cells):
    identities = [_coordinate(row) for row in rows]
    expected = set(map(tuple, expected_cells))
    if len(identities) != len(expected) or len(set(identities)) != len(identities) or set(identities) != expected:
        raise ValueError('Incomplete or duplicated pilot episode coverage')
    groups = defaultdict(list)
    for row in rows:
        numbers = ('realized_cost', 'planner_latency_median_seconds', 'planner_latency_p95_seconds',
                   'q_rmse_rad', 'terminal_q_error_rad', 'neural_rows', 'fallback_count')
        if (row['complete'] is not True or row['phase'] != 'exploratory' or row['seed'] != 9000
                or any(not math.isfinite(row[k]) or row[k] < 0 for k in numbers)
                or row['planner_latency_median_seconds'] <= 0 or row['planner_latency_p95_seconds'] <= 0):
            raise ValueError('Invalid or missing measured pilot episode/timing')
        groups[_coordinate(row)[:4]].append(row)
    conditions = []
    for key, episodes in groups.items():
        actuator, model, rate, step = key
        conditions.append(dict(actuator=actuator, model_id=model, actuator_interval_ns=rate,
            nominal_model_partition_ns=step, actual_max_query_ns=min(rate, step), n_episodes=len(episodes),
            mean_cost=float(np.mean([e['realized_cost'] for e in episodes])),
            mean_episode_median_planner_seconds=float(np.mean([e['planner_latency_median_seconds'] for e in episodes])),
            mean_episode_p95_planner_seconds=float(np.mean([e['planner_latency_p95_seconds'] for e in episodes])),
            mean_q_rmse_rad=float(np.mean([e['q_rmse_rad'] for e in episodes])),
            mean_terminal_q_error_rad=float(np.mean([e['terminal_q_error_rad'] for e in episodes])),
            successes=sum(e['success'] for e in episodes), fallback_count=sum(e['fallback_count'] for e in episodes),
            mean_neural_rows=float(np.mean([e['neural_rows'] for e in episodes])),
            on_descriptive_frontier=None, inference='exploratory_descriptive'))
    for item in conditions:
        if item['model_id'] == 'sim_oracle':
            continue  # Two audit tasks cannot be plotted against ten-task means.
        alternatives = [other for other in conditions if other['actuator'] == item['actuator']
                        and other['actuator_interval_ns'] == item['actuator_interval_ns']
                        and other['model_id'] != 'sim_oracle']
        cost, latency = item['mean_cost'], item['mean_episode_median_planner_seconds']
        item['on_descriptive_frontier'] = not any(
            other['mean_cost'] <= cost and other['mean_episode_median_planner_seconds'] <= latency
            and (other['mean_cost'] < cost or other['mean_episode_median_planner_seconds'] < latency)
            for other in alternatives)
    comparisons = []
    for key, episodes in groups.items():
        actuator, model, rate, step = key
        if model != 'power_correct':
            continue
        methods = {e['episode_id']: e for e in episodes}
        for other_key, competitors in groups.items():
            a, competitor, r, s = other_key
            if (a, r, s) != (actuator, rate, step) or competitor in {'power_correct', 'sim_oracle'}:
                continue
            baselines = {e['episode_id']: e for e in competitors}
            if methods.keys() != baselines.keys():
                raise ValueError('A comparison needs exactly paired complete tasks')
            tasks = sorted(methods)
            m = np.array([methods[t]['realized_cost'] for t in tasks])
            b = np.array([baselines[t]['realized_cost'] for t in tasks])
            mt = np.array([methods[t]['planner_latency_median_seconds'] for t in tasks])
            bt = np.array([baselines[t]['planner_latency_median_seconds'] for t in tasks])
            comparisons.append(dict(actuator=actuator, competitor=competitor, actuator_interval_ns=rate,
                nominal_model_partition_ns=step, n_paired_tasks=len(tasks),
                cost_improvement=float(1-m.mean()/b.mean()) if b.mean() > 0 else None,
                planner_time_reduction=float(1-mt.mean()/bt.mean()),
                method_lower_cost_tasks=int((m < b).sum()),
                paired_cost_differences=(b-m).tolist(), inference='exploratory_descriptive'))
    return dict(conditions=conditions, comparisons=comparisons)


def profile_coordinates():
    from .efficiency_pilot import MODELS
    return [(a, m, variant, batch, h) for a in ('direct', 'lagged') for m in MODELS
            for variant in (('original', 'efficient') if m == 'power_correct' else ('original',))
            for batch in (1, 128) for h in (.005, .01, .04)]


def validate_profiles(profile, freeze_sha256, expected=None, sample_counts=None):
    expected = profile_coordinates() if expected is None else expected
    sample_counts = {1: 500, 128: 100} if sample_counts is None else sample_counts
    rows = profile['benchmark_rows']
    identities = [(r['actuator'], r['model_id'], r['variant'], r['batch_size'], r['h_s']) for r in rows]
    if (profile['protocol_id'] != 'PTCO-EFF-0.1' or profile['freeze_sha256'] != freeze_sha256
            or len(identities) != len(expected) or len(set(identities)) != len(identities)
            or set(identities) != set(expected)):
        raise ValueError('Missing or incompatible model profiles')
    for row in rows:
        values = np.asarray(row['samples_ns'], dtype=np.float64)
        if len(values) != sample_counts[row['batch_size']] or not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError('Every measured positive timing sample must be retained')
        if not math.isclose(row['median_ns'], float(np.median(values)), rel_tol=1e-12):
            raise ValueError('Profile median does not reconcile with raw samples')
        if not math.isclose(row['p95_ns'], float(np.quantile(values, .95, method='linear')), rel_tol=1e-12):
            raise ValueError('Profile p95 does not reconcile with raw samples')


def store_profile(root, profile):
    """Keep every timing observation; the ordinary report file is a pointer copy."""
    content = json.dumps(profile, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()+b'\n'
    digest = hashlib.sha256(content).hexdigest()
    relative = f'reports/efficiency_pilot_profile_runs/{digest}.json'
    path = root/relative
    if path.exists():
        if sha256_file(path) != digest:
            raise ValueError('An immutable timing profile was modified')
    else:
        atomic_bytes(path, content)
    atomic_bytes(root/'reports/efficiency_pilot_profiles.json', content)
    return dict(path=relative, sha256=digest)


def profile_models(root, registry):
    import torch
    from .efficiency_pilot import FREEZE, _verify_selected, verify_registry
    from .efficient_power import EfficientPowerHead, neural_work
    from .control_episode import episode_conditions
    from .execution import cpu_defaults
    freeze = verify_registry(root, registry)
    cpu_defaults(9000)
    source_sha = sha256_file(Path(__file__).resolve())
    result = dict(protocol_id='PTCO-EFF-0.1', freeze_sha256=sha256_file(root/FREEZE),
        source_sha256=source_sha, started_utc=utc_now(), dtype='float64', device='cpu',
        benchmark_rows=[], warmup_calls=50, sample_counts={'1': 500, '128': 100},
        interpretation='Repeated fixed input, no training or real-time robot claim', alpha_spent=0)
    overhead = []
    for _ in range(500):
        start = time.perf_counter_ns()
        overhead.append(time.perf_counter_ns()-start)
    result['timer_overhead_median_ns'] = float(np.median(overhead))
    with torch.no_grad():
        for actuator in ('direct', 'lagged'):
            conditions = episode_conditions(9000, 0, actuator, 'exploratory')
            state = torch.tensor(conditions['initial_state'], dtype=torch.float64)[None, :]
            command = torch.tensor([[.7, -1.1]], dtype=torch.float64)
            for model_id in ('power_correct', 'mlp_h', 'node_generic', 'node_structured', 'physics_fit'):
                original = _verify_selected(root, f'{actuator}:{model_id}', freeze['selections'][f'{actuator}:{model_id}'])
                models = [('original', original)]
                if model_id == 'power_correct':
                    models.append(('efficient', EfficientPowerHead.from_head(original)))
                for variant, model in models:
                    for batch in (1, 128):
                        x, u = state.repeat(batch, 1), command.repeat(batch, 1)
                        for h in (.005, .01, .04):
                            for _ in range(50):
                                model.predict(x, u, h)
                            samples = []
                            for _ in range(500 if batch == 1 else 100):
                                start = time.perf_counter_ns()
                                predicted = model.predict(x, u, h)
                                samples.append(time.perf_counter_ns()-start)
                            if any(s != 'ok' for s in predicted.status):
                                raise ValueError('Profiling input produced invalid predictions')
                            work = (neural_work(predicted, batch, actuator) if variant == 'efficient'
                                    else dict(network_dispatches=predicted.network_calls,
                                              vector_field_calls=predicted.vector_field_calls,
                                              neural_rows=batch*predicted.network_calls))
                            result['benchmark_rows'].append(dict(actuator=actuator, model_id=model_id, variant=variant,
                                batch_size=batch, h_s=h, samples_ns=samples, median_ns=float(np.median(samples)),
                                p95_ns=float(np.quantile(samples, .95, method='linear')), work=work,
                                parameter_count=sum(p.numel() for p in model.parameters()),
                                state=conditions['initial_state'].tolist(), command=[.7, -1.1]))
                    print(f'Profiled {actuator} {model_id} {variant}', flush=True)
    result['ended_utc'] = utc_now()
    validate_profiles(result, sha256_file(root/FREEZE))
    if source_sha != sha256_file(Path(__file__).resolve()):
        raise ValueError('Profile evaluator changed during measurement')
    store_profile(root, result)
    return result


def summarize(root, registry, completed, profiles):
    from .efficiency_pilot import FREEZE, REGISTRY, verify_completion, verify_registry
    from .action_path import ActionSegment
    from .control import anchor_components
    verify_registry(root, registry)
    validate_profiles(profiles, sha256_file(root/FREEZE))
    if profiles['source_sha256'] != sha256_file(Path(__file__).resolve()):
        raise ValueError('Profile consumer differs from measured source')
    if (completed['state'] != 'complete' or completed['registry_sha256'] != sha256_file(root/REGISTRY)
            or set(completed['completed']) != {j['cell_id'] for j in registry['jobs']}):
        raise ValueError('Every complete pilot episode is required before aggregation')
    rows, evidence = [], []
    for job in registry['jobs']:
        entry = completed['completed'][job['cell_id']]
        path = safe_path(root, entry['path'])
        if sha256_file(path) != entry['sha256']:
            raise ValueError('Modified pilot episode manifest')
        request = read_json(root/job['path'])
        verify_completion(root, path, request)
        summary = read_json(path.parent/'summary.json')
        import json
        segments = [ActionSegment(r['start_ns'], r['duration_ns'], np.array(r['command']))
                    for r in map(json.loads, (path.parent/'action_segments.jsonl').read_text().splitlines())]
        with np.load(path.parent/'true_trajectories.npz', allow_pickle=False) as arrays:
            recalculated = anchor_components(arrays['anchors'], arrays['target'], segments)['total']
        if not math.isclose(recalculated, summary['realized_cost'], abs_tol=1e-12, rel_tol=1e-12):
            raise ValueError('Pilot cost conflicts with raw truth and actual commands')
        if summary['protocol_id'] != 'PTCO-EFF-0.1' or summary['inference_variant'] != request['inference_variant']:
            raise ValueError('Episode uses an undeclared inference variant')
        rows.append(summary)
        evidence.append(entry)
    aggregate = aggregate_rows(rows, [job['coordinate'] for job in registry['jobs']])
    profile_hash = sha256_file(root/'reports/efficiency_pilot_profiles.json')
    profile_artifact = f'reports/efficiency_pilot_profile_runs/{profile_hash}.json'
    if sha256_file(root/profile_artifact) != profile_hash:
        raise ValueError('Immutable profile evidence is missing or modified')
    return dict(protocol_id='PTCO-EFF-0.1', status='complete_exploratory', alpha_spent=0, statistical_look=None,
                frozen_registry_sha256=sha256_file(root/REGISTRY), profiles_sha256=sha256_file(root/'reports/efficiency_pilot_profiles.json'),
                profiles_artifact=dict(path=profile_artifact, sha256=profile_hash),
                created_utc=utc_now(), **aggregate, episodes=rows, evidence=evidence,
                conclusion_scope='One development training seed. Cost and compute differences are descriptive; fresh independent confirmation is required.')


def _svg(result):
    colors = dict(power_correct='#087e8b', mlp_h='#d1495b', node_generic='#7b61a8',
                  node_structured='#bf8c00', physics_fit='#536d5b')
    pieces = ['<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="780" viewBox="0 0 1080 780">',
              '<rect width="1080" height="780" fill="white"/>',
              '<g font-family="sans-serif" fill="#222"><text x="40" y="30" font-size="20">Control quality versus planning time — exploratory, one training seed</text>']
    for panel, (actuator, rate) in enumerate([(a, r) for a in ('direct', 'lagged') for r in (40_000_000, 10_000_000)]):
        rows = [r for r in result['conditions'] if r['actuator'] == actuator and r['actuator_interval_ns'] == rate
                and r['model_id'] != 'sim_oracle']
        left, top = 65+(panel % 2)*520, 80+(panel//2)*325
        xs = [r['mean_episode_median_planner_seconds'] for r in rows]
        ys = [r['mean_cost'] for r in rows]
        xmin, xmax = 0., max(xs)*1.15
        ymin, ymax = min(ys)-.08*(max(ys)-min(ys)+.01), max(ys)+.12*(max(ys)-min(ys)+.01)
        pieces.append(f'<text x="{left}" y="{top-14}" font-size="16">{actuator}; commands every {rate//1000000} ms</text>')
        for i in range(5):
            y = top+230-i*57.5
            cost = ymin+(ymax-ymin)*i/4
            x = left+i*100
            pieces.append(f'<path d="M {left} {y} h 400" stroke="#ddd"/><text x="{left-8}" y="{y+4}" text-anchor="end" font-size="11">{cost:.3f}</text>')
            pieces.append(f'<text x="{x}" y="{top+248}" text-anchor="middle" font-size="11">{xmax*i/4:.2f}</text>')
        pieces.append(f'<path d="M {left} {top} v 230 h 400" stroke="#333" fill="none"/>')
        pieces.append(f'<text x="{left+200}" y="{top+273}" text-anchor="middle" font-size="12">Mean episode median planning seconds (lower is faster)</text>')
        for row in rows:
            x = left+400*row['mean_episode_median_planner_seconds']/xmax
            y = top+230-230*(row['mean_cost']-ymin)/(ymax-ymin)
            fill = colors[row['model_id']]
            stroke = '#222' if row['on_descriptive_frontier'] else fill
            label = f'{row["model_id"]}: cost {row["mean_cost"]:.5f}; seconds {row["mean_episode_median_planner_seconds"]:.3f}; query {row["actual_max_query_ns"]//1000000} ms'
            if row['nominal_model_partition_ns'] == 5_000_000:
                pieces.append(f'<rect x="{x-5}" y="{y-5}" width="10" height="10" fill="{fill}" stroke="{stroke}"><title>{label}</title></rect>')
            else:
                pieces.append(f'<circle cx="{x}" cy="{y}" r="5" fill="{fill}" stroke="{stroke}"><title>{label}</title></circle>')
    for i, (model, color) in enumerate(colors.items()):
        pieces.append(f'<circle cx="{55+i*202}" cy="747" r="5" fill="{color}"/><text x="{65+i*202}" y="751" font-size="12">{model}</text>')
    pieces.append('<text x="40" y="774" font-size="11">Cost: lower is better. Circle: coarsest legal query; square: 5 ms query. Frontier is descriptive, not statistically established.</text></g></svg>')
    return ''.join(pieces).encode()


def clip_data(time_ns, states, target):
    time_ns, states, target = np.asarray(time_ns), np.asarray(states), np.asarray(target)
    if (time_ns.ndim != 1 or len(time_ns) < 2 or time_ns[0] != 0 or time_ns[-1] != 1_600_000_000
            or (np.diff(time_ns) <= 0).any() or states.shape not in ((len(time_ns), 4), (len(time_ns), 6))
            or target.shape != (2,) or not np.isfinite(states).all() or not np.isfinite(target).all()):
        raise ValueError('Demo requires a complete finite physical trajectory in seconds/radians')
    indexes = np.unique(np.r_[np.searchsorted(time_ns, np.arange(0, 1_600_000_001, 10_000_000)), len(time_ns)-1])
    return dict(time_s=(time_ns[indexes]/1e9).tolist(), q=states[indexes, :2].tolist(), target=target.tolist())


def write_demo(root, result):
    import json
    clips = []
    for row, entry in zip(result['episodes'], result['evidence']):
        if row['model_id'] == 'sim_oracle':
            continue
        path = safe_path(root, entry['path']).parent/'true_trajectories.npz'
        with np.load(path, allow_pickle=False) as data:
            clip = clip_data(data['time_ns'], data['state'], data['target'])
        clips.append(dict(actuator=row['actuator'], model=row['model_id'],
            hold=row['actuator_interval_ns']//1_000_000, partition=row['model_partition_ns'][0]//1_000_000,
            task=row['episode_id'], cost=row['realized_cost'],
            planner_seconds=row['planner_latency_median_seconds'], **clip))
    payload = json.dumps(clips, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    template = '''<!doctype html><meta charset="utf-8"><title>Robot dynamics: measured pilot replay</title>
<style>body{font:16px system-ui;margin:28px;background:#f7f8fa;color:#202832}h1{font-size:26px}p{max-width:900px;line-height:1.5}select,button,input{font:inherit;margin:4px 10px 4px 0}#cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}.card{background:white;border:1px solid #ddd;border-radius:8px;padding:16px}canvas{width:100%;height:260px}.stats{font-size:14px;line-height:1.7}</style>
<h1>Which learned robot model controls better, and at what compute cost?</h1>
<p>This replays saved simulated motion from the exploratory pilot. All five controllers receive the same starting task and sensing rate. Each chooses its own torques. Ten tasks are available; one development training seed is not proof of general superiority.</p>
<label>Actuator <select id="plant"><option value="direct">Direct torque</option><option value="lagged">Torque with actuator lag</option></select></label>
<label>Command updates <select id="hold"><option value="40">40 ms</option><option value="10">10 ms</option></select></label>
<label>Model partition <select id="partition"><option value="40">Coarsest legal queries</option><option value="5">5 ms</option></select></label>
<label>Task <select id="task"></select></label><button id="play">Play / pause</button>
<p>Observations and replanning: always 40 ms. With 10 ms commands, coarse predictions split into 10 ms queries. Dashed red arms show the target joint angles. Playback follows physical simulation time; it does not show wall time spent planning.</p>
<input id="scrub" type="range" min="0" max="160" value="0" style="width:min(700px,95%)"><span id="time"></span><div id="cards"></div>
<script>
const clips=PAYLOAD;const names={power_correct:'Our power head',mlp_h:'Time-scaled neural model',node_generic:'Continuous neural model (NODE)',node_structured:'Mechanics-aware NODE',physics_fit:'Fitted mechanics reference'};
const models=Object.keys(names),byId=id=>document.getElementById(id);let chosen=[],playing=false,last=0,position=0;
for(let i=0;i<10;i++)byId('task').add(new Option(String(i+1),String(i)));
function choose(){chosen=models.map(m=>clips.find(c=>c.actuator===byId('plant').value&&c.model===m&&c.hold===+byId('hold').value&&c.partition===+byId('partition').value&&c.task===+byId('task').value));byId('cards').innerHTML=models.map((m,i)=>'<div class="card"><b>'+names[m]+'</b><canvas id="canvas'+i+'" width="300" height="260"></canvas><div class="stats" id="stats'+i+'"></div></div>').join('');position=0;byId('scrub').value=0;draw();}
function arm(ctx,q,color,dashed){const x=150,y=140,s=225;const x1=x+s*.30*Math.cos(q[0]),y1=y-s*.30*Math.sin(q[0]),x2=x1+s*.25*Math.cos(q[0]+q[1]),y2=y1-s*.25*Math.sin(q[0]+q[1]);ctx.beginPath();ctx.setLineDash(dashed?[6,5]:[]);ctx.lineWidth=dashed?3:8;ctx.strokeStyle=color;ctx.moveTo(x,y);ctx.lineTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();ctx.setLineDash([]);for(const [px,py]of [[x,y],[x1,y1],[x2,y2]]){ctx.beginPath();ctx.arc(px,py,5,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();}}
function draw(){const frame=Math.min(160,Math.round(position));byId('time').textContent=(frame/100).toFixed(2)+' s';chosen.forEach((c,i)=>{const canvas=byId('canvas'+i),ctx=canvas.getContext('2d');ctx.clearRect(0,0,300,260);arm(ctx,c.target,'#cf4b4b',true);arm(ctx,c.q[frame],'#087e8b',false);const error=Math.hypot(c.q[frame][0]-c.target[0],c.q[frame][1]-c.target[1]);byId('stats'+i).textContent='Episode cost: '+c.cost.toFixed(4)+' · lower is better. Median planning: '+c.planner_seconds.toFixed(3)+' s. Current joint error: '+error.toFixed(3)+' rad.';});}
for(const id of ['plant','hold','partition','task'])byId(id).onchange=choose;byId('scrub').oninput=()=>{position=+byId('scrub').value;draw();};byId('play').onclick=()=>{if(position>=160)position=0;playing=!playing;last=0;};
function animate(now){if(playing){if(last)position=Math.min(160,position+(now-last)/10);byId('scrub').value=position;draw();if(position>=160)playing=false;}last=now;requestAnimationFrame(animate);}choose();requestAnimationFrame(animate);
</script>'''
    atomic_bytes(root/'reports/efficiency_pilot_demo.html', template.replace('PAYLOAD', payload).encode())


def write_report(root, result):
    lines = ['# Targeted control and efficiency pilot', '',
        'Complete exploratory pilot on the two-joint robot. All results use one development training seed and ten paired tasks per main cell. These measurements do not establish population superiority, equivalence, real-time control, or a full PTCO PASS.', '',
        'Actual commands update every 40 or 10 ms; sensing and replanning stay at 40 ms. Every predictor splits at physical command changes. Both coarse and 5 ms queries use the same CEM 128/4/16 budget and six torque knots.', '',
        '## Quality and planning cost', '',
        'Planning seconds are the mean of per-episode median latency. Lower realized cost is better. All successful and failed-policy episodes are retained.', '',
        '| Plant | Command hold ms | Model | Maximum query ms | Mean cost | Planning s | Success / tasks | Fallbacks |',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    for row in result['conditions']:
        if row['model_id'] == 'sim_oracle':
            continue
        lines.append(f'| {row["actuator"]} | {row["actuator_interval_ns"]//1000000} | {row["model_id"]} | {row["actual_max_query_ns"]//1000000} | {row["mean_cost"]:.6f} | {row["mean_episode_median_planner_seconds"]:.3f} | {row["successes"]}/{row["n_episodes"]} | {row["fallback_count"]} |')
    lines += ['', '## Head versus each strong competitor', '',
        'Positive percentages favor the head; negative percentages favor the competitor. Task counts show the number with lower head cost, not independent trained-model replications.', '',
        '| Plant | Hold / query ms | Competitor | Cost reduction | Planning time reduction | Lower head cost / paired tasks |',
        '|---|---|---|---:|---:|---:|']
    for row in result['comparisons']:
        reduction = 'unresolved denominator' if row['cost_improvement'] is None else f'{100*row["cost_improvement"]:+.2f}%'
        lines.append(f'| {row["actuator"]} | {row["actuator_interval_ns"]//1000000} / {min(row["actuator_interval_ns"], row["nominal_model_partition_ns"])//1000000} | {row["competitor"]} | {reduction} | {100*row["planner_time_reduction"]:+.2f}% | {row["method_lower_cost_tasks"]}/{row["n_paired_tasks"]} |')
    profile_artifact = result['profiles_artifact']
    if sha256_file(root/profile_artifact['path']) != profile_artifact['sha256']:
        raise ValueError('Report timing evidence differs from the immutable measured profile')
    profiles = read_json(root/profile_artifact['path'])
    timings = {(r['actuator'], r['model_id'], r['variant'], r['batch_size'], r['h_s']): r
               for r in profiles['benchmark_rows']}
    lines += ['', '## Inference timing', '',
        'Median CPU float64 prediction time at a 10 ms query. Batch one represents a single prediction; batch 128 represents one CEM candidate bank. The complete 5/10/40 ms measurements, p95 values and raw samples are archived in the profile artifact. These measurements do not measure training speed.', '',
        '| Plant | Model / implementation | Batch one microseconds | Batch 128 microseconds | Neural rows / 128 candidates |',
        '|---|---|---:|---:|---:|']
    for actuator in ('direct', 'lagged'):
        for model_id, variant in [('power_correct', 'original'), ('power_correct', 'efficient'),
                                  ('mlp_h', 'original'), ('node_generic', 'original'),
                                  ('node_structured', 'original'), ('physics_fit', 'original')]:
            single = timings[actuator, model_id, variant, 1, .01]
            batch = timings[actuator, model_id, variant, 128, .01]
            lines.append(f'| {actuator} | {model_id} / {variant} | {single["median_ns"]/1000:.1f} | {batch["median_ns"]/1000:.1f} | {batch["work"]["neural_rows"]} |')
    lines += ['', '## Interpretation and limits', '',
        'The direct optimized head uses one neural pass. The lagged head batches two evaluations into one dispatch and still performs twice the neural row work. These are inference optimizations; no training-speed gain is claimed.', '',
        'The quality/time frontier compares both prediction resolutions within each physical condition. Its membership is descriptive. A convenient forced fine-step comparison does not establish that a model is faster than a NODE using its coarsest valid step.', '',
        'Eight true-plant audits cover only tasks 0–1 at each plant/rate. Their two-task means are excluded from the ten-task frontier. The calibrated mechanics reference has a stronger known-topology prior and one fitted model per plant. True-plant CEM is a reference, not a globally optimal controller.', '',
        'Fresh independent confirmation must freeze the chosen practical comparison before new outcomes. The four-joint robot, reduced-data training, moving targets, and new perturbation tests have not been run in this pilot.', '',
        '## Reproduce', '',
        'Selected checkpoints, requests, source hashes and all complete M40 manifests are bound by `config/efficiency_pilot_freeze.json`, `config/efficiency_pilot_cells.json` and `reports/efficiency_pilot_batch.json`. Raw costs are recomputed from true trajectories and actual commands.', '',
        'Run `.venv/bin/python -m poc.efficiency_result report` for verification/reaggregation. `profile` repeats the recorded timing procedure; `python -m poc.efficiency_pilot run` executes only missing compatible jobs. These commands do not spend a new statistical look.', '',
        'See `efficiency_pilot_result.json` for all condition metrics, paired task differences and evidence hashes; its `profiles_artifact` points to immutable inference timing samples; `efficiency_pilot_quality_compute.svg` shows the descriptive comparison. Open `efficiency_pilot_demo.html` locally for side-by-side replay of every main pilot task.', '']
    atomic_json(root/'reports/efficiency_pilot_result.json', result)
    atomic_bytes(root/'reports/efficiency_pilot_result.md', '\n'.join(lines).encode())
    atomic_bytes(root/'reports/efficiency_pilot_quality_compute.svg', _svg(result))
    write_demo(root, result)


def main(argv=None):
    from .efficiency_pilot import REGISTRY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('profile', 'report'))
    parser.add_argument('--root', type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.root.resolve()
    registry = read_json(root/REGISTRY)
    if args.command == 'profile':
        profile_models(root, registry)
    else:
        result = summarize(root, registry, read_json(root/'reports/efficiency_pilot_batch.json'),
                           read_json(root/'reports/efficiency_pilot_profiles.json'))
        write_report(root, result)
        print(root/'reports/efficiency_pilot_result.md', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
