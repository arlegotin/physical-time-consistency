from pathlib import Path
import hashlib,json,os,stat,subprocess,shutil,uuid,time
from datetime import datetime,timezone
root=Path.cwd();scope=json.loads((root/'reports/storage_sealed_fit_compression_scope.json').read_text());groups=scope['groups'];groups.sort(key=lambda g:(not g['source'].endswith('.csv'),-g['size_bytes']))
journal=root/'reports/storage_sealed_fit_compression_004.jsonl';result_path=root/'reports/storage_sealed_fit_compression_004.json';initial_free=shutil.disk_usage(root).free;start=time.perf_counter();seen_manifests=set();converted=skipped=0

def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def attrs(p):
 return subprocess.run(['/usr/bin/xattr','-lx',str(p)],capture_output=True,check=True).stdout

def has_compression(p):
 return bool(p.stat().st_flags&stat.UF_COMPRESSED) or 'com.apple.decmpfs' in subprocess.run(['/usr/bin/xattr',str(p)],capture_output=True,text=True,check=True).stdout.splitlines()

context_canonical={'study/affine-fresh/f3c32d9a4ec2702dd414c8706fc0a9e6070c02b28fd8aa50962cdf1c083de7fe/lr0_001/training.csv': 'evidence/M30/20261008T092610-03a44692/training.csv', 'study/affine-fresh/f3c32d9a4ec2702dd414c8706fc0a9e6070c02b28fd8aa50962cdf1c083de7fe/lr0_001/resume.pt': 'evidence/M30/20261008T092610-03a44692/resume.pt', 'study/affine-fresh/2abdc5183e60aa02a05b414f11bed3a93ab71db5ba2f10467f4c3e7c88cae5e4/lr0_001/training.csv': 'evidence/M30/20261008T093734-37a86809/training.csv', 'study/affine-fresh/2abdc5183e60aa02a05b414f11bed3a93ab71db5ba2f10467f4c3e7c88cae5e4/lr0_001/resume.pt': 'evidence/M30/20261008T093734-37a86809/resume.pt', 'study/affine-fresh/4a6f226d84ff3c3da8fd3c85e36931c79b7b483961e06f73e851d1105485e2b2/lr0_001/training.csv': 'evidence/M30/20261008T100609-dff2768e/training.csv', 'study/affine-fresh/4a6f226d84ff3c3da8fd3c85e36931c79b7b483961e06f73e851d1105485e2b2/lr0_001/resume.pt': 'evidence/M30/20261008T100609-dff2768e/resume.pt', 'study/affine-fresh/ea796ec41cb895f7b16a2fcd47d0be7decc877cb6fc7e9a36d6d1a6c16c75d08/lr0_001/training.csv': 'evidence/M30/20261008T102208-8797a04d/training.csv', 'study/affine-fresh/ea796ec41cb895f7b16a2fcd47d0be7decc877cb6fc7e9a36d6d1a6c16c75d08/lr0_001/resume.pt': 'evidence/M30/20261008T102208-8797a04d/resume.pt', 'study/affine-fresh/bf1369070988c0b9490938a09aaee5ae164ff83cfaf394cd37eb34c28415a64c/lr0_001/training.csv': 'evidence/M30/20261008T103611-b18dd4b3/training.csv', 'study/affine-fresh/bf1369070988c0b9490938a09aaee5ae164ff83cfaf394cd37eb34c28415a64c/lr0_001/resume.pt': 'evidence/M30/20261008T103611-b18dd4b3/resume.pt', 'study/affine-fresh/f13f63b474d9dfd886059f1b569aa361f2dd3f1e3b9cccaf32a9ee64af6d0fe5/lr0_001/training.csv': 'evidence/M30/20261008T105848-6bcfc475/training.csv', 'study/affine-fresh/f13f63b474d9dfd886059f1b569aa361f2dd3f1e3b9cccaf32a9ee64af6d0fe5/lr0_001/resume.pt': 'evidence/M30/20261008T105848-6bcfc475/resume.pt', 'study/affine-fresh/ce915cd8b8605d2939de6853786f2a8d539869d8220a625c07b384f9bd396ec9/lr0_001/training.csv': 'evidence/M30/20261008T111044-b468d657/training.csv', 'study/affine-fresh/ce915cd8b8605d2939de6853786f2a8d539869d8220a625c07b384f9bd396ec9/lr0_001/resume.pt': 'evidence/M30/20261008T111044-b468d657/resume.pt', 'study/affine-fresh/d0508ca4ddd29643343fb2342822068dd6a43d2b3185c03d828674071879d4c2/lr0_001/training.csv': 'evidence/M30/20261008T113032-02a8274d/training.csv', 'study/affine-fresh/d0508ca4ddd29643343fb2342822068dd6a43d2b3185c03d828674071879d4c2/lr0_001/resume.pt': 'evidence/M30/20261008T113032-02a8274d/resume.pt', 'study/affine-fresh/897fff4843cedf69abedbeae33e5605a6660b7de6900ad0b8a377dd7d0daf9ea/lr0_001/training.csv': 'evidence/M30/20261008T115003-011739b4/training.csv', 'study/affine-fresh/897fff4843cedf69abedbeae33e5605a6660b7de6900ad0b8a377dd7d0daf9ea/lr0_001/resume.pt': 'evidence/M30/20261008T115003-011739b4/resume.pt', 'study/affine-fresh/bc4131af2f34230c1946b04bc4d635edf6638fd46d60a3dcc688745f7b50efc5/lr0_001/training.csv': 'evidence/M30/20261008T122142-0e973095/training.csv', 'study/affine-fresh/bc4131af2f34230c1946b04bc4d635edf6638fd46d60a3dcc688745f7b50efc5/lr0_001/resume.pt': 'evidence/M30/20261008T122142-0e973095/resume.pt', 'study/affine-fresh/93b95b738cfd69b9be3f377d001c7f3f87bd45b4770a9d75dc2a7309137b8a0a/lr0_001/training.csv': 'evidence/M30/20261008T124232-6ae4718b/training.csv', 'study/affine-fresh/93b95b738cfd69b9be3f377d001c7f3f87bd45b4770a9d75dc2a7309137b8a0a/lr0_001/resume.pt': 'evidence/M30/20261008T124232-6ae4718b/resume.pt', 'study/affine-fresh/3c135bde692c3da8f87b2e7bf58ad66bb450827a072636b99814a69908e179b5/lr0_001/training.csv': 'evidence/M30/20261008T093418-3e215d68/training.csv', 'study/affine-fresh/3c135bde692c3da8f87b2e7bf58ad66bb450827a072636b99814a69908e179b5/lr0_001/resume.pt': 'evidence/M30/20261008T093418-3e215d68/resume.pt', 'study/affine-fresh/3ddb0cb31c5b25c1031f7966e2967408e9209ea0b54f9dc01a5abd4f06f7ddfc/lr0_001/training.csv': 'evidence/M30/20261008T094409-db1bcb10/training.csv', 'study/affine-fresh/3ddb0cb31c5b25c1031f7966e2967408e9209ea0b54f9dc01a5abd4f06f7ddfc/lr0_001/resume.pt': 'evidence/M30/20261008T094409-db1bcb10/resume.pt', 'study/affine-fresh/2f688c4f59b2881c6f8cf69cf357ab7d86a28bacff8dd356195d8cdd94113554/lr0_001/training.csv': 'evidence/M30/20261008T100756-b02bcf79/training.csv', 'study/affine-fresh/2f688c4f59b2881c6f8cf69cf357ab7d86a28bacff8dd356195d8cdd94113554/lr0_001/resume.pt': 'evidence/M30/20261008T100756-b02bcf79/resume.pt', 'study/affine-fresh/9323d36a924cc422701fd920a623b163087a73b3e86ab9dad890f7ad35f6680b/lr0_001/training.csv': 'evidence/M30/20261008T102753-94cf2670/training.csv', 'study/affine-fresh/9323d36a924cc422701fd920a623b163087a73b3e86ab9dad890f7ad35f6680b/lr0_001/resume.pt': 'evidence/M30/20261008T102753-94cf2670/resume.pt', 'study/affine-fresh/29796ed05b10c82f5b36f4a2cd4271efb530ae43f510f9b63cb0eae259ec860c/lr0_001/training.csv': 'evidence/M30/20261008T103949-2ce43bd6/training.csv', 'study/affine-fresh/29796ed05b10c82f5b36f4a2cd4271efb530ae43f510f9b63cb0eae259ec860c/lr0_001/resume.pt': 'evidence/M30/20261008T103949-2ce43bd6/resume.pt', 'study/affine-fresh/57a8f2c56a09a194a58b8e3229cd91a6e0448d0efa91da747723f6eeb69165a9/lr0_001/training.csv': 'evidence/M30/20261008T110214-1a6e1110/training.csv', 'study/affine-fresh/57a8f2c56a09a194a58b8e3229cd91a6e0448d0efa91da747723f6eeb69165a9/lr0_001/resume.pt': 'evidence/M30/20261008T110214-1a6e1110/resume.pt', 'study/affine-fresh/2f76ee9fc1ac35d89d9f9ebb8c118eb5531b2f5fcc3caf81b535c39b769e8e58/lr0_001/training.csv': 'evidence/M30/20261008T112448-55042cd8/training.csv', 'study/affine-fresh/2f76ee9fc1ac35d89d9f9ebb8c118eb5531b2f5fcc3caf81b535c39b769e8e58/lr0_001/resume.pt': 'evidence/M30/20261008T112448-55042cd8/resume.pt', 'study/affine-fresh/629b95cb76810b270d48eea5558e8f124b8d50e2733ec4212af85f7f2dd049a2/lr0_001/training.csv': 'evidence/M30/20261008T113646-27662d79/training.csv', 'study/affine-fresh/629b95cb76810b270d48eea5558e8f124b8d50e2733ec4212af85f7f2dd049a2/lr0_001/resume.pt': 'evidence/M30/20261008T113646-27662d79/resume.pt', 'study/affine-fresh/059761f45d5b718e7af2da9205810768c73fdac2f6c36817c0cf50d9d2bcbe4d/lr0_001/training.csv': 'evidence/M30/20261008T120205-0e2231c5/training.csv', 'study/affine-fresh/059761f45d5b718e7af2da9205810768c73fdac2f6c36817c0cf50d9d2bcbe4d/lr0_001/resume.pt': 'evidence/M30/20261008T120205-0e2231c5/resume.pt', 'study/affine-fresh/67dd01924935977807b86bf3051bdb54b91f3c81329fe4a39326ebc9ae67843f/lr0_001/training.csv': 'evidence/M30/20261008T123332-f4dd0d1b/training.csv', 'study/affine-fresh/67dd01924935977807b86bf3051bdb54b91f3c81329fe4a39326ebc9ae67843f/lr0_001/resume.pt': 'evidence/M30/20261008T123332-f4dd0d1b/resume.pt', 'study/affine-fresh/ba226c9c22bcda7f0d8cd2fc5dd44e821e4305384c44fd4b558c9e6fbff28eee/lr0_001/training.csv': 'evidence/M30/20261008T124627-940bd37a/training.csv', 'study/affine-fresh/ba226c9c22bcda7f0d8cd2fc5dd44e821e4305384c44fd4b558c9e6fbff28eee/lr0_001/resume.pt': 'evidence/M30/20261008T124627-940bd37a/resume.pt'}

def complete(p,expected):
 original=p
 relative=str(p.relative_to(root))
 if relative in context_canonical:p=root/context_canonical[relative]
 folder=p.parent;m=folder/'manifest.json'
 if folder not in seen_manifests:
  marker=json.loads((folder/'COMPLETED').read_text());assert digest(m)==marker['manifest_sha256'],str(m)
  record=json.loads(m.read_text());assert record['completion_status']=='complete',str(m);seen_manifests.add(folder)
 else:record=json.loads(m.read_text())
 assert any(Path(f['path']).name==p.name and f['sha256']==expected for f in record['produced_files']),str(p)
 assert digest(p)==expected,str(p)
 assert digest(original)==expected,str(original)

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
