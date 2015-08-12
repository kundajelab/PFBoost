### Script to Write out load data script in output
### Peyton Greenside
### 8/10/15
################################################################################################

from boosting_2D import config

def write_load_data_script(y, x1, x2, prior_params, tree_file_name):
	file_name = '{0}{1}/load_data_script.py'.format(config.OUTPUT_PATH, config.OUTPUT_PREFIX)
	f = open(file_name, 'w')
	f.write(
	"""
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


### LOAD DATA
################################################################################################
################################################################################################

# Load y
y = TargetMatrix('{0}', 
                 '{1}', 
                 '{2}',
                 '{3}',
                 '{4}')
# Load x1
x1 = Motifs('{5}', 
            '{6}', 
            '{7}',
            '{8}',
            '{9}')

# Load x2
x2 = Regulators('{10}', 
                '{11}', 
                '{12}',
                '{13}',
                '{14}')

# Load holdout
holdout = Holdout(y, 'sparse')

# Load config
TuningParams = namedtuple('TuningParams', [
    'num_iter',
    'use_stumps', 'use_stable', 'use_corrected_loss', 'use_prior',
    'eta_1', 'eta_2', 'bundle_max', 'epsilon'
])
config.OUTPUT_PATH = '{15}'
config.OUTPUT_PREFIX = '{16}'
config.TUNING_PARAMS = TuningParams(
    {17}, 
    {18}, {19}, {20},
    {21},
    {22}, {23}, {24}, {25})
config.NCPU = {26}
config.PLOT = {27}

# Prior
params=prior.PriorParams(
    {28}, {29},
    '{30}', 
    {31},
    {32}, 
    {33},
    {34},
    {35},
    {36})
prior_mr, prior_rr = prior.parse_prior(params, x1, x2)

# Load tree
tree = pickle.load(open('{37}', 'rb'))

	""".format(
		y.data_file,
		y.row_label_file,
		y.col_label_file,
		y.input_format,
		y.mult_format,
		x1.data_file,
		x1.row_label_file,
		x1.col_label_file,
		x1.input_format,
		x1.mult_format,
		x2.data_file,
		x2.row_label_file,
		x2.col_label_file,
		x2.input_format,
		x2.mult_format,
		config.OUTPUT_PATH,
		config.OUTPUT_PREFIX,
		config.TUNING_PARAMS.num_iter,
		config.TUNING_PARAMS.use_stumps,
		config.TUNING_PARAMS.use_stable,
		config.TUNING_PARAMS.use_corrected_loss,
		config.TUNING_PARAMS.use_prior,
		config.TUNING_PARAMS.eta_1,
		config.TUNING_PARAMS.eta_2,
		config.TUNING_PARAMS.bundle_max,
		config.TUNING_PARAMS.epsilon,
		config.NCPU,
		config.PLOT,
		prior_params.prior_constant,
		prior_params.prior_decay_rate,
		prior_params.prior_input_format,
		[prior_params.motif_reg_file if prior_params.motif_reg_file==None else "'{0}'".format(prior_params.motif_reg_file)][0],
		[prior_params.motif_reg_row_labels if prior_params.motif_reg_row_labels==None else "'{0}'".format(prior_params.motif_reg_row_labels)][0],
		[prior_params.motif_reg_col_labels if prior_params.motif_reg_col_labels==None else "'{0}'".format(prior_params.motif_reg_col_labels)][0],
		[prior_params.reg_reg_file if prior_params.reg_reg_file==None else "'{0}'".format(prior_params.reg_reg_file)][0],
		[prior_params.reg_reg_row_labels if prior_params.reg_reg_row_labels==None else "'{0}'".format(prior_params.reg_reg_row_labels)][0],
		[prior_params.reg_reg_col_labels if prior_params.reg_reg_col_labels==None else "'{0}'".format(prior_params.reg_reg_col_labels)][0],
		tree_file_name
		))
	f.close()

