"""Read-only final export, source binding, hash, link and newline-aware range checks."""
from pathlib import Path
import csv, hashlib, io, itertools, json, re, sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from poc.calibration_study import prospective_recipe, study_bindings

ROOT=Path.cwd(); DEST=ROOT/'reports/hybrid_component'

def main():
    freeze=json.loads((DEST/'source-freeze.json').read_text())
    assert study_bindings(ROOT,prospective_recipe(matrix='hybrid')) == freeze['bindings']
    assert hashlib.sha256((ROOT/'docs/superpowers/specs/2026-10-12-hybrid-affine-component-design.md').read_bytes()).hexdigest()==freeze['spec_sha256']
    hashes=json.loads((DEST/'artifact_sha256.json').read_text())
    for name,digest in hashes.items():assert hashlib.sha256((DEST/name).read_bytes()).hexdigest()==digest,name
    result=json.loads((DEST/'short_calibration_result.json').read_text())
    gate=json.loads((DEST/'component_gate.json').read_text())
    assert gate==result['gate'] and result['coverage']['complete']
    assert result['coverage']['expected_fit_count']==30 and result['coverage']['expected_nominal_count']==6
    with (DEST/'short_calibration_metrics.csv').open(newline='') as stream:rows=list(csv.DictReader(stream))
    csv_text=(DEST/'short_calibration_metrics.csv').read_bytes().decode('utf-8')
    crlf_text=csv_text.replace('\r\n','\n').replace('\n','\r\n')
    assert list(csv.DictReader(io.StringIO(crlf_text,newline='')))==rows
    assert len(csv_text.splitlines())==len(crlf_text.splitlines())==len(rows)+1
    expected=set(itertools.product([*result['recipe']['models'],'physics_nominal'],[30,120],[5_000_000,10_000_000,20_000_000,40_000_000]))
    actual=[(r['model_id'],int(r['budget_s']),int(r['query_step_ns'])) for r in rows]
    assert set(actual)==expected and len(actual)==len(expected),(len(actual),len(expected))
    indexed={(r['model_id'],r['budget_s'],r['query_step_ns']):r for r in result['configurations']}
    for row,coordinate in zip(rows,actual):
        source=indexed[coordinate]
        assert row['screen_status']==source['screen']['status']
        for key in ('q_rmse','v_rmse','q_p95','v_p95','decision_failure_fraction','mean_valid_regret','scorer_wall_median_s','scorer_wall_p95_s'):
            assert (row[key]=='' and source[key] is None) or (row[key]!='' and float(row[key])==source[key]),(coordinate,key)
    checked_links=[]
    for path in [*DEST.glob('*.md'),ROOT/'docs/research-trajectory.md']:
        text=path.read_bytes().decode('utf-8')
        for match in re.finditer(r'\[[^\]]+\]\(([^)]+)\)',text):
            target=match.group(1).strip('<>')
            if target.startswith(('https://','http://','mailto:','#')):continue
            target=target.split('#',1)[0]
            suffix=re.search(r':(\d+)(?:-(\d+))?$',target)
            relative=target[:suffix.start()] if suffix else target
            linked=Path(relative) if relative.startswith('/') else path.parent/relative
            assert linked.exists(),(str(path),target)
            if suffix:
                # splitlines treats CRLF as one line and works for both Markdown and CSV artifacts.
                count=len(linked.read_bytes().decode('utf-8').splitlines())
                first=int(suffix.group(1));last=int(suffix.group(2) or first)
                assert 1<=first<=last<=count,(target,count)
            checked_links.append({'document':str(path.relative_to(ROOT)),'target':target})
    replica=json.loads((DEST/'numerical_replication.json').read_text())
    assert replica['all_equal'] and len(replica['fits'])==3
    print(json.dumps({'all_checks_passed':True,'artifact_hash_count':len(hashes),'csv_configuration_count':len(rows),
                      'lf_crlf_equivalent_csv_rows_and_line_ranges':True,'csv_line_count':len(csv_text.splitlines()),
                      'newline_handling':'csv newline=""; link ranges counted with bytes.decode().splitlines() (CRLF-aware)',
                      'local_link_count':len(checked_links),'checked_links':checked_links,
                      'source_runtime_recipe_and_spec_match_freeze':True},indent=2))

if __name__=='__main__':main()
