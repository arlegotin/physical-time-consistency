"""Audit a closed pilot and project unchanged primary development work."""
from pathlib import Path
import datetime,gzip,json,shutil,sys
raw=Path(sys.argv[1]); destination=Path(sys.argv[2]);destination.mkdir(parents=True,exist_ok=True)
ledgers=[];concerns=[]
for p in sorted(raw.glob('jobs/**/resources.json')):
    d=json.loads(p.read_text());attempts=d['attempts'];ledgers.append({'path':str(p.relative_to(raw)),'attempt_count':len(attempts),
        'statuses':[a['status'] for a in attempts],'unknown_tails':sum(a['unknown_tail'] for a in attempts)})
    for a in attempts:
        if a['status']!='completed' or a['unknown_tail'] or a.get('error_type'):
            concerns.append({'path':str(p.relative_to(raw)),'attempt':a})
events=[json.loads(line) for line in (raw/'events.jsonl').read_text().splitlines()]
failed_events=[e for e in events if e.get('event')=='failed']
logs=[]
for p in sorted((raw/'execution-logs').glob('*.stderr')):
    content=p.read_text();logs.append({'path':str(p.relative_to(raw)),'bytes':p.stat().st_size,'content':content})
for p in sorted((raw/'execution-logs').glob('*.execution.json')):
    d=json.loads(p.read_text())
    if d.get('returncode',0)!=0 or 'end_utc' not in d:concerns.append({'execution_log':str(p.relative_to(raw)),'record':d})
producers=[];compressed=decoded=0
for marker in sorted(raw.glob('jobs/*/*/COMPLETED')):
    p=marker.parent;m=json.loads((p/'manifest.json').read_text());r=m['resources'];stage=m['stage'];s=m['summary']
    factor=1;kind='common'
    if stage=='fit':
        for attempt in s.get('attempts',[]):
            if attempt.get('status')=='failed':concerns.append({'fit_attempt_failure':str(p.relative_to(raw)),'attempt':attempt})
    if stage=='collect':factor=5;kind='calibration_sessions'
    elif stage in {'fit','nominal'}:
        model=m['request']['model_id'];factor=5 if model in {'physics_fit','physics_nominal'} else 10;kind=model
    elif stage in {'predictions','propagation','decisions'}:
        model=json.loads((raw/m['request']['fit_receipt']['path']/'manifest.json').read_text())['request']['model_id']
        factor=5 if model in {'physics_fit','physics_nominal'} else 10;kind=model
    elif stage=='timing':factor=210/24;kind='pooled_timing_approximation'
    size=sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
    producers.append({'stage':stage,'kind':kind,'pilot_wall_s':r['wall_s'],'pilot_cpu_s':r['cpu_s'],
        'pilot_bytes':size,'development_multiplier':factor,'parent_attempts':[a['parent_producer_attempt'] for a in r['attempts']],
        'accounting_status':r['accounting_status'],'peak_rss_native':r['peak_rss_native']})
    for f in p.glob('rows.jsonl.gz'):
        compressed+=f.stat().st_size
        with gzip.open(f,'rb') as stream:
            while block:=stream.read(1024*1024):decoded+=len(block)
fits=[p for p in producers if p['stage']=='fit'];assert len(fits)==21,len(fits)
# Projection sums only top-level producer attempts. Nested supplemental truth is included by its parent.
retained=[p for p in producers if all(parent is None for parent in p['parent_attempts'])]
by_stage={}
for p in retained:
    row=by_stage.setdefault(p['stage'],{'pilot_wall_s':0.,'pilot_cpu_s':0.,'projected_development_total_wall_s':0.,'projected_development_total_cpu_s':0.})
    for field in ['wall','cpu']:
        row['pilot_'+field+'_s']+=p['pilot_'+field+'_s'];row['projected_development_total_'+field+'_s']+=p['pilot_'+field+'_s']*p['development_multiplier']
# Artifact byte projection counts every product's on-disk payload once, including nested products.
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'raw_output_root':str(raw),
 'resource_audit':{'ledger_count':len(ledgers),'ledgers':ledgers,'concerns':concerns,'failed_events':failed_events,'stderr_logs':logs,
   'known_parked_defect':'Caught interruption during post-commit resource observation may double-count persistence; no such interruption is inferred only when every ledger and command completed without failures.',
   'exact_primary_resource_totals_qualified':bool(concerns or failed_events)},
 'fit_count':21,'projected_development_fit_count':195,'producer_categories':producers,'nonoverlapping_stage_projection':by_stage,
 'projection_assumptions':['Each neural pilot coordinate is projected to ten session/initialization coordinates; physical and nominal coordinates to five sessions.',
   'The pilot selected epochs are observations, not assumed fixed in new draws; stopping/caps and throughput can vary substantially.',
   'Common evaluation/truth producers are reused unchanged; parent-inclusive nested producer spans are not added.',
   'One complete pooled timing request per matrix; pilot timing projection uses coordinate count ratio and is approximate.',
   'Primary commissioned fitting is separate from evaluator/truth/scoring/timing/reporting and all independent replication.',
   'These are workload estimates; no hardware commissioning or favorable-compute claim.'],
 'compressed_streams':{'compressed_bytes':compressed,'decoded_bytes':decoded,'decoded_to_compressed_ratio':decoded/compressed if compressed else None},
 'raw_bytes':sum(p.stat().st_size for p in raw.rglob('*') if p.is_file()),
 'projected_development_total_artifact_bytes':sum(p['pilot_bytes']*p['development_multiplier'] for p in producers),
 'raw_volume_free_bytes':shutil.disk_usage(raw).free,'repository_free_bytes':shutil.disk_usage(Path.cwd()).free,
 'observed_peak_rss_native':max(p['peak_rss_native'] for p in producers),'observed_peak_rss_native_unit':'bytes on macOS'}
(destination/'pilot_resource_projection.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['producer_categories','resource_audit']},indent=2))
print('audit_concerns',len(concerns),'failed_events',len(failed_events),'ledgers',len(ledgers))
