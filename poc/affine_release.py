"""Shareable replay of the fresh paired fine/coarse NODE control comparison."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .io import atomic_bytes,atomic_json,read_json,sha256_file

MODES=('affine_flow_5','node_structured_5','node_structured_40','affine_flow_40')


def prepare_fresh_clips(clips,seeds,task_count):
    expected={(m,s,t) for m in MODES for s in seeds for t in range(task_count)}
    coordinates=[(c['model'],c['seed'],c['task']) for c in clips]
    if len(coordinates)!=len(set(coordinates)) or set(coordinates)!=expected:
        raise ValueError('Fresh replay coordinate coverage mismatch')
    paired={}
    for c in clips:
        times,q,target=(np.asarray(c[k],dtype=float) for k in ('time_s','q','target'))
        if (times.shape!=(161,) or not np.allclose(times,np.arange(161)/100,atol=1e-12,rtol=0)
                or q.shape!=(161,2) or target.shape!=(2,) or not np.isfinite(q).all()
                or not np.isfinite(target).all() or not all(math.isfinite(c[k]) and c[k]>=0
                    for k in ('cost','planner_seconds'))):
            raise ValueError('Fresh replay requires complete finite physical frames/metrics')
        key=(c['seed'],c['task']);initial=(q[0].tolist(),target.tolist())
        if key in paired and paired[key]!=initial:raise ValueError('Fresh replay paired tasks differ')
        paired[key]=initial
    return sorted(clips,key=lambda c:(c['seed'],c['task'],MODES.index(c['model'])))


def render_fresh_replay(clips,seeds,task_count):
    payload=json.dumps(prepare_fresh_clips(clips,seeds,task_count),separators=(',',':'),allow_nan=False).replace('<','\\u003c')
    return TEMPLATE.replace('MEASURED_PAYLOAD',payload)


def export(root):
    from .affine_confirmation import evaluation_freeze,_verify_entry
    from .affine_measurement import recompute_episode_cost
    from .efficiency_result import clip_data
    root=Path(root).resolve();freeze=evaluation_freeze(root)
    batch=read_json(root/'reports/affine_fresh_primary_batch.json')
    coarse=read_json(root/'reports/affine_fresh_coarse_primary_batch.json')
    if batch['state']!='complete' or coarse['state']!='complete':
        raise ValueError('Complete fresh fine/coarse controls required for replay')
    clips=[];bindings={}
    for entry in list(batch['completed'].values())+list(coarse['completed'].values()):
        folder=_verify_entry(root,entry);row=read_json(folder/'summary.json')
        cost=recompute_episode_cost(folder)
        if not math.isclose(cost,row['realized_cost'],rel_tol=1e-12,abs_tol=1e-12):
            raise ValueError('Fresh replay raw cost differs from saved summary')
        with np.load(folder/'true_trajectories.npz',allow_pickle=False) as a:
            clip=clip_data(a['time_ns'],a['state'],a['target'])
        clips.append(dict(model=row['model_id']+'_'+str(row['model_partition_ns'][0]//1000000),
                          seed=row['seed'],task=row['episode_id'],cost=cost,
                          planner_seconds=row['planner_latency_median_seconds'],**clip))
        bindings[entry['path']]=entry['sha256']
    output=root/'reports/affine_fresh_demo.html'
    atomic_bytes(output,render_fresh_replay(clips,freeze['seeds'],freeze['episodes_per_seed']).encode())
    receipt=dict(protocol_id='PTCO-AFF-0.2',clip_count=len(clips),output_path=str(output.relative_to(root)),
                 output_sha256=sha256_file(output),evidence=bindings,source_sha256=sha256_file(Path(__file__)))
    atomic_json(root/'reports/affine_fresh_demo_receipt.json',receipt);return receipt


TEMPLATE='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fresh robot model comparison</title><style>body{font:16px system-ui;background:#f7f8fa;color:#202832;margin:28px}p{max-width:960px;line-height:1.5}select,button,input{font:inherit;margin:6px}#cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}.card{background:white;border:1px solid #ddd;border-radius:8px;padding:16px}canvas{width:100%;height:260px}.stats{font-size:14px;line-height:1.6}</style>
<h1>Fresh independent robot-model comparison</h1><p>These are saved simulated episodes from independently trained affine-flow and mechanics-aware NODE models. Both models are tested with fine and coarse internal queries. All four controllers receive fresh observations and choose commands every40 ms. The actuator has a known80 ms lag. See the accompanying report for aggregate effects and uncertainty; a selected episode alone cannot establish superiority.</p>
<label>Training seed <select id="seed"></select></label><label>Paired task <select id="task"></select></label><button id="play">Play / pause</button>
<p>Dashed red arms show the target. Playback follows physical motion time. Planning time is CPU computation between observations, measured separately.</p>
<input id="scrub" aria-label="Physical time" type="range" min="0" max="160" value="0" style="width:min(700px,95%)"><span id="time"></span><div id="cards"></div>
<script type="application/json" id="clips">MEASURED_PAYLOAD</script><script>
const clips=JSON.parse(document.getElementById('clips').textContent),names={affine_flow_5:'Affine-flow,5 ms queries',node_structured_5:'Mechanics-aware NODE,5 ms',node_structured_40:'Same NODE,40 ms queries',affine_flow_40:'Same affine-flow,40 ms queries'},models=Object.keys(names),byId=id=>document.getElementById(id);let chosen=[],position=0,playing=false,last=0;
for(const s of [...new Set(clips.map(c=>c.seed))])byId('seed').add(new Option(String(s),String(s)));
for(const t of [...new Set(clips.map(c=>c.task))])byId('task').add(new Option(String(t+1),String(t)));
function choose(){chosen=models.map(m=>clips.find(c=>c.model===m&&c.seed===+byId('seed').value&&c.task===+byId('task').value));byId('cards').innerHTML=models.map((m,i)=>'<div class="card"><b>'+names[m]+'</b><canvas id="canvas'+i+'" width="300" height="260"></canvas><div class="stats" id="stats'+i+'"></div></div>').join('');position=0;byId('scrub').value=0;draw();}
function arm(ctx,q,color,dashed){const x=150,y=140,s=225,x1=x+s*.30*Math.cos(q[0]),y1=y-s*.30*Math.sin(q[0]),x2=x1+s*.25*Math.cos(q[0]+q[1]),y2=y1-s*.25*Math.sin(q[0]+q[1]);ctx.beginPath();ctx.setLineDash(dashed?[6,5]:[]);ctx.lineWidth=dashed?3:8;ctx.strokeStyle=color;ctx.moveTo(x,y);ctx.lineTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();ctx.setLineDash([]);for(const [px,py]of [[x,y],[x1,y1],[x2,y2]]){ctx.beginPath();ctx.arc(px,py,5,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();}}
function draw(){const f=Math.min(160,Math.round(position));byId('time').textContent=chosen[0].time_s[f].toFixed(2)+' s';chosen.forEach((c,i)=>{const ctx=byId('canvas'+i).getContext('2d');ctx.clearRect(0,0,300,260);arm(ctx,c.target,'#cf4b4b',true);arm(ctx,c.q[f],'#087e8b',false);byId('stats'+i).textContent='Realized episode cost: '+c.cost.toFixed(4)+' (lower is better). Median planning: '+c.planner_seconds.toFixed(3)+' s. Joint error: '+Math.hypot(c.q[f][0]-c.target[0],c.q[f][1]-c.target[1]).toFixed(3)+' rad.';});}
byId('seed').onchange=choose;byId('task').onchange=choose;byId('scrub').oninput=()=>{position=+byId('scrub').value;draw();};byId('play').onclick=()=>{if(position>=160)position=0;playing=!playing;last=0;};function animate(now){if(playing){if(last)position=Math.min(160,position+(now-last)/10);byId('scrub').value=position;draw();if(position>=160)playing=false;}last=now;requestAnimationFrame(animate);}choose();requestAnimationFrame(animate);
</script></html>'''


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path.cwd())
    args=parser.parse_args(argv);r=export(args.root);print({k:v for k,v in r.items() if k!='evidence'})


if __name__=='__main__':main()
