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


### POST-PROCESSING
################################################################################################
################################################################################################

### Run margin score for each of the different cell types
################################################

# XXX MAKE CODE PRETTY
# Calculate the margin score for each individual 
# pool = multiprocessing.Pool(processes=config.NCPU) # create pool of processes
pool='serial'

INDEX_PATH='/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/index_files/'
MARGIN_SCORE_PATH='/srv/persistent/pgreens/projects/boosting/results/margin_scores/'
all_comp = pd.read_table('/users/pgreens/git/boosting_2D/hema_data/index_files/hema_tree_cell_comparisons.txt', sep='\t', header=None)
for comp in all_comp.ix[:,0].tolist():
    print comp
    comp_reformat = comp.replace('v','_v_')
    # cell_file = '{0}hema_{1}_cells.txt'.format(INDEX_PATH, comp_reformat)
    cell_file = '{0}hema_{1}_cells_direct_comp.txt'.format(INDEX_PATH, comp_reformat)
    peak_file = '{0}hema_{1}_peaks.txt'.format(INDEX_PATH, comp_reformat)
    # XXX REFORMAT SO STRING IS SOMEWHERE USEFUL
    prefix = 'hema_{0}_1000iter_TFbindingonly_direct_comp'.format(comp_reformat)
    # Compute margin score for each of these
    margin_score.call_rank_by_margin_score(prefix=prefix,
      methods=['by_x1', 'by_x2', 'by_node'],
       y=y, x1=x1, x2=x2, tree=tree, pool=pool, num_perm=10,
       x1_feat_file=peak_file,
       x2_feat_file=cell_file)  
    print comp


# Rank x1, x2, rule and node 
margin_score.call_rank_by_margin_score(prefix='hema_CMP_v_Mono_1000iter_TFbindingonly',
  methods=['by_x1', 'by_x2'],
   y=y, x1=x1, x2=x2, tree=tree, pool=pool, num_perm=100,
   x1_feat_file='/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/index_files/hema_CMP_v_Mono_peaks.txt',
   x2_feat_file='/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/index_files/hema_CMP_v_Mono_cells.txt')

margin_score.call_rank_by_margin_score(prefix='hema_MPP_HSC_v_pHSC_1000iter_TFbindingonly',
  methods=['by_x1', 'by_x2'],
   y=y, x1=x1, x2=x2, tree=tree, pool=pool, num_perm=100,
   x1_feat_file='/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/index_files/hema_MPP_HSC_v_pHSC_peaks.txt',
   x2_feat_file='/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/index_files/hema_MPP_HSC_v_pHSC_cells.txt')


# # Close pool
# pool.close() # stop adding processes
# pool.join() # wait until all threads are done before going on


### Print margin score for each of the different cell types
################################################

INDEX_PATH='/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/index_files/'
MARGIN_SCORE_PATH='/srv/persistent/pgreens/projects/boosting/results/margin_scores/'
all_comp = pd.read_table('/users/pgreens/git/boosting_2D/hema_data/index_files/hema_tree_cell_comparisons.txt', sep='\t', header=None)
for comp in all_comp.ix[:,0].tolist():
    print comp
    result = pd.read_table('{0}hema_{1}_1000iter_TFbindingonly_top_nodes_stable.txt'.format(MARGIN_SCORE_PATH, comp.replace('v','_v_')), sep="\t", header=None)
    print result.ix[0:10,[2,4,6]]


### Plot normalized margin scores for top regulators and motifs
################################################

conditions = [
"hema_HSC_v_MPP_1000iter_TFbindingonly", 
"hema_MPP_v_CMP_1000iter_TFbindingonly", 
"hema_CMP_v_GMP_1000iter_TFbindingonly", 
"hema_GMP_v_Mono_1000iter_TFbindingonly",
"hema_MPP_v_LMPP_1000iter_TFbindingonly",
"hema_CLP_v_Bcell_1000iter_TFbindingonly", 
"hema_CLP_v_CD4Tcell_1000iter_TFbindingonly", 
"hema_CLP_v_CD8Tcell_1000iter_TFbindingonly", 
"hema_CLP_v_NKcell_1000iter_TFbindingonly", 
"hema_CMP_v_MEP_1000iter_TFbindingonly", 
"hema_LMPP_v_CLP_1000iter_TFbindingonly", 
"hema_MEP_v_Ery_1000iter_TFbindingonly"]

# "hema_CMP_v_Mono_1000iter_TFbindingonly", 
# "hema_MPP_HSC_v_pHSC_1000iter_TFbindingonly"

### Re-write outputs to be "by_x1"
plot_label = '1000iter_TFbindingonly_all_comparisons'
num_feat = 20
method='x1_feat'
element_direction='ENH_UP'
margin_score.plot_norm_margin_score_across_conditions(conditions, method, plot_label, num_feat, element_direction)

result_path='/srv/persistent/pgreens/projects/boosting/results/margin_scores/'

### Compare UP TO DOWN  for enhancers and promoters
for cell_type in [el for el in os.listdir('/srv/persistent/pgreens/projects/boosting/results/margin_scores') if el != 'old']:
    if cell_type in ['hema_CMP_v_Mono_1000iter_TFbindingonly','hema_MPP_HSC_v_pHSC_1000iter_TFbindingonly']:
        continue
    for element_direction in ["ENH", "PROM"]:
        for method in ['x1_feat', 'x2_feat']:
            conditions = ['{0}{1}/{1}_{2}_UP_top_{3}.txt'.format(result_path, cell_type, element_direction, method),
            '{0}{1}/{1}_{2}_DOWN_top_{3}.txt'.format(result_path, cell_type, element_direction, method)]
            out_file='{0}{1}/{1}_{2}_UP_v_DOWN_top_{3}_discriminative.txt'.format(result_path, cell_type, element_direction, method)
            margin_score.find_discrimative_features(conditions=conditions, method=method, out_file=out_file)

# COMPARE increased to decreased accessibility
for cell_type in [el for el in os.listdir('/srv/persistent/pgreens/projects/boosting/results/margin_scores') if el != 'old']:
    if cell_type in ['hema_CMP_v_Mono_1000iter_TFbindingonly','hema_MPP_HSC_v_pHSC_1000iter_TFbindingonly']:
        continue
    # for element_direction in ["ENH_UP", "ENH_DOWN", "PROM_UP", "PROM_DOWN"]:
    for element_direction in ["UP", "DOWN"]:
        for method in ['x1_feat', 'x2_feat']:
            conditions = ['{0}{1}/{1}_ENH_{2}_top_{3}.txt'.format(result_path, cell_type, element_direction, method),
            '{0}{1}/{1}_PROM_{2}_top_{3}.txt'.format(result_path, cell_type, element_direction, method)]
            out_file='{0}{1}/{1}_ENH_v_PROM_{2}_top_{3}_discriminative.txt'.format(result_path, cell_type, element_direction, method)
            margin_score.find_discrimative_features(conditions=conditions, method=method, out_file=out_file)

#   

