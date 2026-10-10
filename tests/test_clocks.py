"""Physical paths are partitioned at events without creating observations."""
import importlib
import pytest


def api(name):
    assert importlib.util.find_spec('poc.clocks'), 'Integer physical schedules are not implemented'
    return getattr(importlib.import_module('poc.clocks'), name)


def test_seconds_to_exact_integer_nanoseconds():
    convert = api('to_ns')
    assert convert(.04) == 40_000_000
    assert convert(.000625) == 625_000
    for value in (-1, float('nan'), float('inf'), 1e-10, True):
        with pytest.raises(ValueError):
            convert(value)


def test_uniform_and_jitter_partitions_preserve_the_hold():
    partition, jitter = api('partition'), api('jitter_partition')
    assert partition(40_000_000, 5_000_000) == (5_000_000,)*8
    assert partition(12_000_000, 5_000_000) == (5_000_000, 5_000_000, 2_000_000)
    assert partition(0, 1) == ()
    for entropy in ((1201, 0, 0, 4), (1201, 1, 0, 4), (1201, 0, 1, 4)):
        durations = jitter(entropy)
        assert sum(durations) == 40_000_000
        assert sorted(durations) == [2_500_000]*4 + [7_500_000]*4
        assert jitter(entropy) == durations
    for duration, step in ((1, 0), (-1, 1), (1., 1), (True, 1)):
        with pytest.raises(ValueError):
            partition(duration, step)


def test_events_split_before_stepping_and_not_after():
    split = api('split_events')
    assert split(0, 40_000_000, (10_000_000, 20_000_000, 40_000_000)) == (
        (0, 10_000_000), (10_000_000, 10_000_000), (20_000_000, 20_000_000))
    assert split(640_000_000, 760_000_000, (640_000_000, 720_000_000)) == (
        (640_000_000, 80_000_000), (720_000_000, 40_000_000))
    with pytest.raises(ValueError):
        split(5, 4, ())
