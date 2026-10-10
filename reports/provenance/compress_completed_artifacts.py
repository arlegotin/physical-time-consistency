from pathlib import Path
import hashlib,json,os,stat,subprocess,shutil,uuid,time
from datetime import datetime,timezone
root=Path.cwd();scope=json.loads((root/'reports/storage_transparent_compression_scope.json').read_text());groups=scope['groups'];groups.sort(key=lambda g:(not g['source'].endswith('.csv'),-g['size_bytes']))
journal=root/'reports/storage_transparent_compression_003.jsonl';result_path=root/'reports/storage_transparent_compression_003.json';initial_free=shutil.disk_usage(root).free;start=time.perf_counter();seen_manifests=set();converted=skipped=0

def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def attrs(p):
 return subprocess.run(['/usr/bin/xattr','-lx',str(p)],capture_output=True,check=True).stdout

def has_compression(p):
 return bool(p.stat().st_flags&stat.UF_COMPRESSED) or 'com.apple.decmpfs' in subprocess.run(['/usr/bin/xattr',str(p)],capture_output=True,text=True,check=True).stdout.splitlines()

def complete(p,expected):
 folder=p.parent;m=folder/'manifest.json'
 if folder not in seen_manifests:
  marker=json.loads((folder/'COMPLETED').read_text());assert digest(m)==marker['manifest_sha256'],str(m)
  record=json.loads(m.read_text());assert record['completion_status']=='complete',str(m);seen_manifests.add(folder)
 else:record=json.loads(m.read_text())
 assert any(Path(f['path']).name==p.name and f['sha256']==expected for f in record['produced_files']),str(p)
 assert digest(p)==expected,str(p)

def log(row):
 with journal.open('a') as f:f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())

assert not journal.exists(),'New immutable operation receipt required'
try:
 for i,g in enumerate(groups):
  paths=[root/p for p in [g['source'],*g['copies']]];expected=g['sha256'];states={};metadata={}
  for p in paths:
   complete(p,expected);s=p.stat();assert not s.st_flags&(stat.UF_IMMUTABLE|stat.UF_APPEND|stat.SF_IMMUTABLE|stat.SF_APPEND),str(p)
   states[p]=s;metadata[p]=attrs(p)
  source=paths[0];baseline_attrs=metadata[source]
  source_already_compressed=bool(source.stat().st_flags&stat.UF_COMPRESSED)
  if any(v!=baseline_attrs for v in metadata.values()):
   log(dict(status='skipped_metadata_mismatch',source=g['source'],sha256=expected));skipped+=1;continue
  token=uuid.uuid4().hex[:12];compressed=source if source_already_compressed else source.with_name(source.name+'.ptco-compress-'+token+'.tmp')
  if not source_already_compressed:subprocess.run(['/usr/bin/ditto','--hfsCompression',str(source),str(compressed)],capture_output=True,check=True)
  compressed_flag=has_compression(compressed)
  if compressed_flag:os.chflags(compressed,states[source].st_flags|stat.UF_COMPRESSED)
  assert digest(compressed)==expected,str(compressed)
  if not compressed_flag or (not source_already_compressed and compressed.stat().st_blocks>=states[source].st_blocks):
   if compressed!=source:compressed.unlink()
   skipped+=1;continue
  source_replaced=False
  for p in paths:
   if p==source and source_already_compressed:continue
   s=states[p];assert digest(p)==expected,str(p)
   temp=compressed if p==source else p.with_name(p.name+'.ptco-compress-'+token+'.tmp')
   if p!=source:
    subprocess.run(['/bin/cp','-cp',str(source),str(temp)],capture_output=True,check=True)
    assert has_compression(temp),str(temp);os.chflags(temp,s.st_flags|stat.UF_COMPRESSED)
    assert temp.stat().st_ino!=source.stat().st_ino,str(temp)
   os.chmod(temp,stat.S_IMODE(s.st_mode));os.utime(temp,ns=(s.st_atime_ns,s.st_mtime_ns))
   assert digest(temp)==expected and temp.stat().st_size==s.st_size,str(temp)
   assert temp.stat().st_uid==s.st_uid and temp.stat().st_gid==s.st_gid,str(temp)
   log(dict(status='prepared',path=str(p.relative_to(root)),sha256=expected,size_bytes=s.st_size,old_flags=s.st_flags,new_flags=temp.stat().st_flags,mode=stat.S_IMODE(s.st_mode),mtime_ns=s.st_mtime_ns,original_xattr_sha256=hashlib.sha256(metadata[p]).hexdigest()))
   os.replace(temp,p)
   assert digest(p)==expected and p.stat().st_size==s.st_size,str(p)
   assert p.stat().st_mtime_ns==s.st_mtime_ns and stat.S_IMODE(p.stat().st_mode)==stat.S_IMODE(s.st_mode),str(p)
   log(dict(status='complete',path=str(p.relative_to(root)),sha256=expected,new_allocated_bytes=p.stat().st_blocks*512,new_flags=p.stat().st_flags))
   converted+=1
  if i%10==0:print(f'{i+1}/{len(groups)} groups; {converted} paths converted; free {shutil.disk_usage(root).free} bytes',flush=True)
 result=dict(status='complete',groups_considered=len(groups),converted_paths=converted,skipped_groups=skipped,initial_free_bytes=initial_free,final_free_bytes=shutil.disk_usage(root).free,elapsed_seconds=time.perf_counter()-start,recorded_utc=datetime.now(timezone.utc).isoformat(),method='Native ditto HFS/APFS compression plus explicit UF_COMPRESSED flag and independent COW clones; exact SHA256/mode/mtime/owner maintained',scope=scope['scope'],limitations='Actual free-space change includes concurrent volume activity; APFS storage method only, readable content unchanged')
 result['measured_free_change_bytes']=result['final_free_bytes']-initial_free;result_path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
except BaseException as e:
 result=dict(status='partial',error=repr(e),converted_paths=converted,skipped_groups=skipped,initial_free_bytes=initial_free,final_free_bytes=shutil.disk_usage(root).free,elapsed_seconds=time.perf_counter()-start)
 result_path.write_text(json.dumps(result,indent=2)+'\n');raise
