"""Archive resource attempts, exact commands, source manifests and artifact hashes."""
from pathlib import Path
import hashlib, json, shutil, sys

ROOT=Path(__file__).resolve().parents[2]
PRIMARY=Path('/private/tmp/physical-time-consistency-hybrid-component')
REPLICA=Path('/private/tmp/physical-time-consistency-hybrid-component-replication')
DEST=ROOT/'reports/hybrid_component'

def write(name, data):
    (DEST/name).write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')

def inspect(raw):
    commands=[json.loads(p.read_text()) for p in sorted((raw/'execution-logs').glob('*.execution.json'))]
    producers=[]; concerns=[]; counts={}; by_stage={}; attempts=[]
    for marker in sorted(raw.glob('jobs/*/*/COMPLETED')):
        manifest=json.loads((marker.parent/'manifest.json').read_text())
        stage=manifest['stage']; counts[stage]=counts.get(stage,0)+1
        resource=manifest['resources']
        producers.append({'stage':stage,'path':str(marker.parent.relative_to(raw)),
                          'request':manifest['request'],'resources':resource})
        for attempt in resource['attempts']:
            attempts.append({'producer':marker.parent.name,'stage':stage,**attempt})
            if attempt['status']!='completed' or attempt['unknown_tail']:
                concerns.append({'path':str(marker.parent.relative_to(raw)),'attempt':attempt})
            if attempt['parent_producer_attempt'] is None:
                total=by_stage.setdefault(stage,{'wall_s':0.,'cpu_s':0.,'attempt_count':0})
                total['attempt_count']+=1
                for kind in ('wall','cpu'):
                    total[kind+'_s']+=attempt['active_'+kind+'_s']+attempt['persistence_'+kind+'_s']
    for record in commands:
        if record.get('returncode')!=0 or 'end_utc' not in record: concerns.append({'command':record})
    ledgers=[]
    for path in sorted(raw.glob('jobs/**/resources.json')):
        data=json.loads(path.read_text()); ledgers.append({'path':str(path.relative_to(raw)),**data})
        for attempt in data['attempts']:
            if attempt['status']!='completed' or attempt['unknown_tail']: concerns.append({'ledger':str(path.relative_to(raw)),'attempt':attempt})
    events=[json.loads(line) for line in (raw/'events.jsonl').read_text().splitlines()]
    failed=[event for event in events if event.get('event')=='failed']
    return {'raw_root':str(raw),'counts':counts,'commands':commands,'producers':producers,
            'nonoverlapping_top_level_producer_totals_by_stage':by_stage,'ledger_records':ledgers,
            'concerns':concerns,'failed_events':failed,
            'raw_bytes':sum(p.stat().st_size for p in raw.rglob('*') if p.is_file()),
            'volume_free_bytes':shutil.disk_usage(raw).free,
            'scope':'Inclusive CLI child durations and top-level producer attempts are separate views. Nested ledgers and original finalization observations overlap their parents and are never added twice. Replica costs are separate.'}

def main():
    assert PRIMARY != REPLICA
    primary=inspect(PRIMARY); replica=inspect(REPLICA)
    write('execution_commands.json',{'primary':primary['commands'],'replica':replica['commands']})
    write('resource_audit.json',{'primary':primary,'replica':replica})
    for raw,label in ((PRIMARY,'primary'),(REPLICA,'replica')):
        shutil.copyfile(raw/'prepared.json',DEST/(label+'_prepared.json'))
        for record in sorted((raw/'execution-logs').glob('*')):
            if record.is_file(): shutil.copyfile(record,DEST/(label+'-'+record.name))
        for recipe in raw.glob('jobs/recipe/*/source_hashes.json'):
            for name in ('source_hashes.json','executing_source_hashes.json','asset_hashes.json'):
                shutil.copyfile(recipe.parent/name,DEST/(label+'_'+name))
    shutil.copyfile(REPLICA/'numerical-replication.json',DEST/'numerical_replication.json')
    shutil.copyfile(REPLICA/'execution-logs/numerical-replication-resources.json',DEST/'numerical_replication_resources.json')
    hashes={str(p.relative_to(DEST)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(DEST.iterdir()) if p.is_file() and p.name!='artifact_sha256.json'}
    write('artifact_sha256.json',hashes)
    print(json.dumps({'primary_counts':primary['counts'],'replica_counts':replica['counts'],
                      'primary_concerns':len(primary['concerns']),'replica_concerns':len(replica['concerns']),
                      'artifact_count':len(hashes)},indent=2))

if __name__=='__main__':main()
