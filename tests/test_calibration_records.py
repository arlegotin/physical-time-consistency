"""Lossless raw-stream policy, including producer-facing integrity failures."""
import gzip
import importlib.util
import json
import pytest


def test_deterministic_lossless_gzip_and_tamper(tmp_path):
    assert importlib.util.find_spec('poc.calibration_records'), 'Compressed stream helper missing'
    from poc.calibration_records import write_records,read_records
    rows=[{'null':None,'bool':True,'integer':2**54,'float':.12345678901234567,'text':'μ','list':[False,-0.0]}]
    a,b=tmp_path/'a.jsonl.gz',tmp_path/'b.jsonl.gz'
    write_records(a,iter(rows));write_records(b,iter(rows))
    assert a.read_bytes()==b.read_bytes()
    assert json.loads(gzip.decompress(a.read_bytes()))==rows[0]
    assert read_records(a)==rows
    broken=bytearray(a.read_bytes());broken[-8]^=1;a.write_bytes(broken)
    with pytest.raises((OSError,ValueError)):
        read_records(a)
