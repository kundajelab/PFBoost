from collections import namedtuple

OUTPUT_PATH = None
OUTPUT_PREFIX = None

TUNING_PARAMS = None
SAVING_PARAMS = None
NCPU = None

VERBOSE = False
LOG_TIME = True

PLOT = True

PERF_METRICS = ['imbalanced_error']

TuningParams = namedtuple('TuningParams', [
    'num_iter',
    'use_stumps', 'use_stable', 'use_prior',
    'eta_1', 'eta_2', 'bundle_max', 'epsilon'
])

SavingParams = namedtuple('SavingParams', [
	'save_tree_only', 'save_complete_data',
	'save_for_post_processing'
])
