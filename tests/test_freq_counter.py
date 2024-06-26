import pytest

from vsk_dl_utils.utils.frequency_counter import FreqCounter


def test_freq_counter():
    counter = FreqCounter()
    assert sum(counter('test', 3) for _ in range(10)) == 3


def test_counter_create():
    counters = FreqCounter()
    counters('counter_0', 3)
    counter = counters.get_counter(3)
    assert sum(counter() for _ in range(10)) == 3
    # Test new names was created
    assert 'counter_1' in counters._call_count

    assert sum(counter(update_count=False) for _ in range(10)) == 0

def test_every_counter():
    counter = FreqCounter()
    assert sum(counter('test', 1) for _ in range(10)) == 10

@pytest.mark.parametrize('freq', [0, -1, -2])
def test_zero_freq_counter(freq):
    counter = FreqCounter()
    assert sum(counter('test', freq) for _ in range(10)) == 0
