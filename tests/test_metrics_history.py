from itertools import pairwise

import pytest

from metrics_history import history_position, history_span, summarize_history


def test_initial_readings_fill_linearly_before_compression():
    samples = [(i, 0, False) for i in range(31)]
    for count in (1, 10, 30, 31):
        summary = summarize_history(samples[:count], list(range(count)), 900, 30, 300)
        assert [item.x for item in summary] == list(range(300 - (count-1)*10, 301, 10))
        assert all(item.count == 1 for item in summary)


def test_compression_starts_continuously_after_full_width():
    assert history_span([0, 30], 900, 30) == 30
    assert history_span([0, 60], 900, 30) == 60
    assert history_position(20, 30.00001, 30) == pytest.approx(1-20/30, abs=1e-6)
    positions = [history_position(age, 900, 30) for age in range(901)]
    assert positions[0] == 1
    assert positions[-1] == 0
    assert all(a > b for a, b in pairwise(positions))
    assert positions[0] - positions[1] > positions[-2] - positions[-1]
