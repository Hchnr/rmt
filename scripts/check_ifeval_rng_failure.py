"""Verify scoring RNG restoration even when the official checker raises."""
import random
import langdetect.detector_factory as factory_module
from rmt.evaluation import ifeval_rng
factory_module.init_factory()
factory = factory_module._factory
factory.set_seed(1234)
state = random.getstate()
old = ifeval_rng._ORIGINAL

def fail(doc, responses):
    random.random()
    raise ValueError('intentional checker failure')

try:
    ifeval_rng._ORIGINAL = fail
    try:
        ifeval_rng.score_ifeval({'key': 1}, ['answer'])
    except ValueError as error:
        assert str(error) == 'intentional checker failure'
    else:
        raise AssertionError('Expected checker error')
    assert random.getstate() == state
    assert factory.seed == 1234
finally:
    ifeval_rng._ORIGINAL = old
print('PASS: checker failure propagates and both RNG states are restored')
