from collections import namedtuple

OUTPUT_PATH = None
OUTPUT_PREFIX = None

TUNING_PARAMS = None
NCPU = None

VERBOSE=False
DEBUG=False
LOG_TIME = False

TuningParams = namedtuple('TuningParams', [
    'num_iter',
    'use_stumps', 'use_stable', 'use_corrected_loss',
    'eta_1', 'eta_2', 'bundle_max', 'epsilon'
])
