
import sys
import os
import random
import gzip

import numpy as np 
from scipy.sparse import *

import argparse
import pandas as pd
import multiprocessing
import ctypes

from functools import partial
import time
from collections import namedtuple
import pdb
import cPickle as pickle

from boosting_2D import config
from boosting_2D import util
from boosting_2D import plot
from boosting_2D import margin_score
from boosting_2D.data_class import *
from boosting_2D.find_rule import *
from boosting_2D import stabilize
from boosting_2D import prior
from boosting_2D import save_model

### Open log files
log = util.log

### Set constant parameters
# TuningParams = namedtuple('TuningParams', [
#     'num_iter',
#     'use_stumps', 'use_stable', 'use_corrected_loss', 'use_prior',
#     'eta_1', 'eta_2', 'bundle_max', 'epsilon'
# ])
# SavingParams = namedtuple('SavingParams', [
#     'save_tree_only', 'save_complete_data',
#     'save_for_post_processing'
# ])

### LOAD DATA
################################################################################################
################################################################################################

with gzip.open('test/test_results/2017_04_13_hema_test_adt_non_stable_10iter/saved_complete_model__2017_04_13_hema_test_adt_non_stable_10iter.gz','rb') as f: 
    model_dict = pickle.load(f)

# Assign data structures
x1 = model_dict['x1']
x2 = model_dict['x2']
y = model_dict['y']
tree = model_dict['tree']
if 'hierarchy' in model_dict:
    hierarchy = model_dict['hierarchy']

# Unpack config and prior dictionaries
for key in model_dict['config'].keys():
    exec('config.%s=model_dict["config"]["%s"]'%(key, key))
for key in model_dict['prior'].keys():
    exec('prior.%s=model_dict["prior"]["%s"]'%(key, key))

    