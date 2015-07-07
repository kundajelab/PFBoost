### TEMPORARY Function to load data only 
### Peyton Greenside
### 6/18/15

import sys
import os
import random

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
import pickle

from boosting_2D import config
from boosting_2D import util
from boosting_2D import plot
from boosting_2D import margin_score
from boosting_2D.data_class import *
from boosting_2D.find_rule import *
from boosting_2D import stabilize
from boosting_2D import prior

# Load config
TuningParams = namedtuple('TuningParams', [
    'num_iter',
    'use_stumps', 'use_stable', 'use_corrected_loss', 'use_prior',
    'eta_1', 'eta_2', 'bundle_max', 'epsilon'
])
config.OUTPUT_PATH = '/srv/persistent/pgreens/projects/boosting/results/'
config.OUTPUT_PREFIX = 'hematopoeisis_23K_stable_bindingTFsonly'
config.TUNING_PARAMS = TuningParams(
    100, 
    False, True, False,
    True,
    0.05, 0.01, 20, 1./holdout.n_train)
config.NCPU = 4

# Load y
y = TargetMatrix('/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/accessibilityMatrix_full_subset_CD34.txt', 
                 '/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/peak_headers_full_subset_CD34.txt', 
                 '/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/cell_types_pairwise.txt',
                 'triplet',
                 'sparse')
# Load x1
x1 = Motifs('/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/annotationMatrix_full_subset_CD34.txt', 
            '/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/annotationMatrix_headers_full.txt',
            '/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/peak_headers_full_subset_CD34.txt', 
            'triplet',
            'sparse')

# Load x2
x2 = Regulators('/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/regulatorExpression_bindingTFsonly.txt', 
                '/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/cell_types_pairwise.txt',
                '/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/regulator_names_bindingTFsonly.txt', 
                'triplet',
                'sparse')

# Load holdout
holdout = Holdout(y, 'sparse')

# Prior
params=prior.PriorParams(
    50, 0.998,
    'matrix', 
     '/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/prior_data/motifTFpriors.txt',\
      '/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/prior_data/motifTFpriors.rows.txt', \
     '/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/prior_data/motifTFpriors.columns_gene_only.txt', \
    None, None, None)
prior_mr, prior_rr = prior.parse_prior(params, x1, x2)

# Load tree
tree = pickle.load(open('/srv/persistent/pgreens/projects/boosting/results/saved_trees/hematopoeisis_23K_stable_bindingTFsonly_saved_tree_state_adt_stable_1000iter', 'rb'))

# XXX MAKE CODE PRETTY
# Calculate the margin score for each individual 
pool = multiprocessing.Pool(processes=config.NCPU) # create pool of processes

INDEX_PATH='/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/index_files/'
MARGIN_SCORE_PATH='/srv/persistent/pgreens/projects/boosting/results/margin_scores/'
all_comp = pd.read_table('/users/pgreens/git/boosting_2D/hema_data/index_files/hema_tree_cell_comparisons.txt', sep='\t', header=None)
for comp in all_comp.ix[:,0].tolist():
    print comp
    comp_reformat = comp.replace('v','_v_')
    cell_file = '{0}hema_{1}_cells.txt'.format(INDEX_PATH, comp_reformat)
    peak_file = '{0}hema_{1}_peaks.txt'.format(INDEX_PATH, comp_reformat)
    # XXX REFORMAT SO STRING IS SOMEWHERE USEFUL
    prefix = 'hema_{0}_1000iter_TFbindingonly'.format(comp_reformat)
    # Compute margin score for each of these
    margin_score.call_rank_by_margin_score(prefix=prefix,
      methods=['by_node'],
       y=y, x1=x1, x2=x2, tree=tree, pool=pool, 
       x1_feat_file=peak_file,
       x2_feat_file=cell_file)
    print comp

# # Close pool
pool.close() # stop adding processes
pool.join() # wait until all threads are done before going on





