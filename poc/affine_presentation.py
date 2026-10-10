"""Standalone replay from the complete, measured six-model refinement."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .io import atomic_bytes, atomic_json, read_json, safe_path, sha256_file

MODELS = ('affine_flow', 'power_correct', 'mlp_h', 'node_generic',
          'node_structured', 'physics_fit')


def prepare_clips(clips):
    expected = {(a, m, r, h, t) for a in ('direct', 'lagged') for m in MODELS
                for r in (40, 10) for h in (40, 5) for t in range(10)}
    coordinates = [(c['actuator'], c['model'], c['hold'], c['partition'], c['task']) for c in clips]
    if len(coordinates) != len(set(coordinates)) or set(coordinates) != expected:
        raise ValueError('Replay coordinate coverage mismatch')
    tasks = {}
    for clip in clips:
        times, angles, target = (np.asarray(clip[key], dtype=np.float64)
                                for key in ('time_s', 'q', 'target'))
        if (times.shape != (161,) or not np.allclose(times, np.arange(161)/100, atol=1e-12, rtol=0)
                or angles.shape != (161, 2) or target.shape != (2,)
                or not np.isfinite(angles).all() or not np.isfinite(target).all()
                or not all(math.isfinite(clip[key]) and clip[key] >= 0
                           for key in ('cost', 'planner_seconds'))):
            raise ValueError('Replay requires complete finite physical frames and metrics')
        key = clip['task']
        task = (angles[0].tolist(), target.tolist())
        if key in tasks and tasks[key] != task:
            raise ValueError('Replay paired task inputs differ')
        tasks[key] = task
    return sorted(clips, key=lambda c: (c['actuator'], c['hold'], c['partition'], c['task'], MODELS.index(c['model'])))


def render_document(clips):
    payload = json.dumps(prepare_clips(clips), separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    return TEMPLATE.replace('MEASURED_PAYLOAD', payload)


def validate_replay_episode(row, saved):
    fields=('model_id','actuator','seed','episode_id','initial_state','target','model_partition_ns',
            'actuator_interval_ns','realized_cost','planner_latency_median_seconds')
    if any(row[key] != saved[key] for key in fields):
        raise ValueError('Replay episode metrics/identity differ from immutable source summary')


def export(root):
    from .affine_measurement import MODELS as measured_models, recompute_episode_cost, summarize_control
    from .contracts import verify_run
    from .efficiency_result import clip_data
    root = Path(root).resolve()
    result = read_json(root/'reports/affine_control_result.json')
    if result['status'] != 'complete_exploratory' or tuple(measured_models) != MODELS:
        raise ValueError('Complete measured refinement required')
    expected = [(a, m, r, s, e) for a in ('direct', 'lagged') for m in MODELS
                for r in (40_000_000, 10_000_000) for s in (40_000_000, 5_000_000) for e in range(10)]
    recalculated = summarize_control(result['episodes'], expected)
    if any(result[key] != value for key, value in recalculated.items()):
        raise ValueError('Replay report disagrees with episode summaries')
    prior = read_json(root/'reports/efficiency_pilot_result.json')
    evidence = {(row['actuator'], row['model_id'], row['actuator_interval_ns'],
                 row['model_partition_ns'][0], row['episode_id']): entry
                for row, entry in zip(prior['episodes'], prior['evidence'])
                if row['model_id'] != 'sim_oracle'}
    jobs = read_json(root/'config/affine_measurement_freeze.json')['jobs']
    for job in jobs:
        evidence[(job['actuator'], 'affine_flow', job['actuator_interval_ns'],
                  job['model_partition_ns'][0], job['episode_id'])] = result['evidence'][job['cell_id']]
    clips, bindings = [], {}
    for row in result['episodes']:
        key = (row['actuator'], row['model_id'], row['actuator_interval_ns'],
               row['model_partition_ns'][0], row['episode_id'])
        entry = evidence[key]
        path = safe_path(root, entry['path'])
        if sha256_file(path) != entry['sha256']:
            raise ValueError('Replay evidence manifest hash mismatch')
        verify_run(root, read_json(path))
        validate_replay_episode(row,read_json(path.parent/'summary.json'))
        if not math.isclose(recompute_episode_cost(path.parent), row['realized_cost'], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('Replay summary differs from measured truth cost')
        with np.load(path.parent/'true_trajectories.npz', allow_pickle=False) as arrays:
            clip = clip_data(arrays['time_ns'], arrays['state'], arrays['target'])
        clips.append(dict(actuator=key[0], model=key[1], hold=key[2]//1_000_000,
                          partition=key[3]//1_000_000, task=key[4], cost=row['realized_cost'],
                          planner_seconds=row['planner_latency_median_seconds'], **clip))
        bindings[entry['path']] = entry['sha256']
    output = root/'reports/affine_refinement_demo.html'
    atomic_bytes(output, render_document(clips).encode())
    receipt = dict(status='complete_exploratory', clip_count=len(clips), frames_per_clip=161,
                   report_sha256=sha256_file(root/'reports/affine_control_result.json'),
                   output_path=str(output.relative_to(root)), output_sha256=sha256_file(output),
                   evidence=bindings, source_sha256=sha256_file(Path(__file__)))
    atomic_json(root/'reports/affine_replay_receipt.json', receipt)
    return receipt


TEMPLATE = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Measured robot model comparison</title>
<style>body{font:16px system-ui;margin:28px;background:#f7f8fa;color:#202832}h1{font-size:26px}p{max-width:950px;line-height:1.5}select,button,input{font:inherit;margin:4px 10px 4px 0}#cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}.card{background:white;border:1px solid #ddd;border-radius:8px;padding:16px}canvas{width:100%;height:260px}.stats{font-size:14px;line-height:1.7}</style>
<h1>Robot control: measured motion and planning cost</h1>
<p>Six predictors control the same simulated two-joint robot tasks. This replays 480 saved episodes, with ten paired tasks in each condition. The new affine-flow method couples motion and torque response. One development training seed establishes a pilot comparison; see the accompanying report for independent confirmation and its scope.</p>
<label>Actuation <select id="plant"><option value="direct">Direct torque</option><option value="lagged">Torque with actuator lag</option></select></label>
<label>Command updates <select id="hold"><option value="40">40 ms</option><option value="10">10 ms</option></select></label>
<label>Model queries <select id="partition"><option value="40">Coarsest legal queries</option><option value="5">5 ms</option></select></label>
<label>Task <select id="task"></select></label><button id="play">Play / pause</button>
<p>All controllers receive observations and replan every 40 ms. Coarse queries split at 10 ms command boundaries when commands update faster. Dashed red arms show the target. Playback follows physical motion time; the planning-time metric measures CPU work performed between observations.</p>
<input id="scrub" aria-label="Physical time" type="range" min="0" max="160" value="0" style="width:min(700px,95%)"><span id="time"></span><div id="cards"></div>
<script type="application/json" id="clips">MEASURED_PAYLOAD</script>
<script>
const clips=JSON.parse(document.getElementById('clips').textContent);
const names={affine_flow:'New affine-flow method',power_correct:'Original power head',mlp_h:'Time-scaled neural model',node_generic:'Continuous neural model (NODE)',node_structured:'Mechanics-aware NODE',physics_fit:'Fitted mechanics reference'};
const models=Object.keys(names),byId=id=>document.getElementById(id);let chosen=[],playing=false,last=0,position=0;
for(let i=0;i<10;i++)byId('task').add(new Option(String(i+1),String(i)));
function choose(){chosen=models.map(m=>clips.find(c=>c.actuator===byId('plant').value&&c.model===m&&c.hold===+byId('hold').value&&c.partition===+byId('partition').value&&c.task===+byId('task').value));byId('cards').innerHTML=models.map((m,i)=>'<div class="card"><b>'+names[m]+'</b><canvas id="canvas'+i+'" width="300" height="260"></canvas><div class="stats" id="stats'+i+'"></div></div>').join('');position=0;byId('scrub').value=0;draw();}
function arm(ctx,q,color,dashed){const x=150,y=140,s=225;const x1=x+s*.30*Math.cos(q[0]),y1=y-s*.30*Math.sin(q[0]),x2=x1+s*.25*Math.cos(q[0]+q[1]),y2=y1-s*.25*Math.sin(q[0]+q[1]);ctx.beginPath();ctx.setLineDash(dashed?[6,5]:[]);ctx.lineWidth=dashed?3:8;ctx.strokeStyle=color;ctx.moveTo(x,y);ctx.lineTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();ctx.setLineDash([]);for(const [px,py]of [[x,y],[x1,y1],[x2,y2]]){ctx.beginPath();ctx.arc(px,py,5,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();}}
function draw(){const frame=Math.min(160,Math.round(position));byId('time').textContent=chosen[0].time_s[frame].toFixed(2)+' s';chosen.forEach((c,i)=>{const ctx=byId('canvas'+i).getContext('2d');ctx.clearRect(0,0,300,260);arm(ctx,c.target,'#cf4b4b',true);arm(ctx,c.q[frame],'#087e8b',false);const error=Math.hypot(c.q[frame][0]-c.target[0],c.q[frame][1]-c.target[1]);byId('stats'+i).textContent='Episode cost: '+c.cost.toFixed(4)+' (lower is better). Median planning: '+c.planner_seconds.toFixed(3)+' s. Current joint error: '+error.toFixed(3)+' rad.';});}
for(const id of ['plant','hold','partition','task'])byId(id).onchange=choose;byId('scrub').oninput=()=>{position=+byId('scrub').value;draw();};byId('play').onclick=()=>{if(position>=160)position=0;playing=!playing;last=0;};
function animate(now){if(playing){if(last)position=Math.min(160,position+(now-last)/10);byId('scrub').value=position;draw();if(position>=160)playing=false;}last=now;requestAnimationFrame(animate);}choose();requestAnimationFrame(animate);
</script></html>'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    value = export(args.root)
    print(json.dumps({k:v for k,v in value.items() if k != 'evidence'}, sort_keys=True))


if __name__ == '__main__':
    main()
