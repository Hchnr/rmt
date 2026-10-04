"""Reproducible randomness around unmodified official IFEval checking rules."""
import random
import threading
from .prepare import digest

_LOCK = threading.RLock()
_ORIGINAL = None


def score_ifeval(doc, responses, master_seed=17):
    global _ORIGINAL
    from evalscope.benchmarks.ifeval import utils
    import langdetect.detector_factory as detector_factory
    with _LOCK:
        if _ORIGINAL is None:
            _ORIGINAL = utils.process_results
        seed = int(digest(['ifeval-score-v1', master_seed, doc])[:8], 16)
        state = random.getstate()
        detector_factory.init_factory()
        factory = detector_factory._factory
        previous_seed = factory.seed
        try:
            random.seed(seed)
            factory.set_seed(seed)
            return _ORIGINAL(doc, responses)
        finally:
            random.setstate(state)
            factory.set_seed(previous_seed)


def install(master_seed=17):
    global _ORIGINAL
    from evalscope.benchmarks.ifeval import utils
    with _LOCK:
        if _ORIGINAL is None:
            _ORIGINAL = utils.process_results
        utils.process_results = lambda doc, responses: score_ifeval(doc, responses, master_seed)
