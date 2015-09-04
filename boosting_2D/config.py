from collections import namedtuple

OUTPUT_PATH = None
OUTPUT_PREFIX = None

TUNING_PARAMS = None
NCPU = None

VERBOSE = True
LOG_TIME = True

PLOT = True

TuningParams = namedtuple('TuningParams', [
    'num_iter',
    'use_stumps', 'use_stable', 'use_corrected_loss', 'use_prior',
    'eta_1', 'eta_2', 'bundle_max', 'epsilon'
])

BOOSTMODE = 'ADABOOST'
