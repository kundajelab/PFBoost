### Script to Write out load data script in output
### Peyton Greenside
### 8/10/15
################################################################################################

# import pickle
import cPickle as pickle
import gzip
import pdb

from boosting_2D import config
from boosting_2D import prior


### Store only paths in load file
################################################################################################

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


### Store and load data objects in pickle file
################################################################################################

### Store all model components in a dictionary and pickle that dictionary
def save_complete_model_state(pickle_file, x1, x2, y, tree):
    model_dict={}
    model_dict['x1']=x1
    model_dict['x2']=x2
    model_dict['y']=y
    model_dict['tree']=tree
    config_dict=store_module_in_dict(config)
    model_dict['config']=config_dict # also a module object
    prior_dict=store_module_in_dict(prior)
    model_dict['prior']=prior_dict # also a module object
    with gzip.open(pickle_file,'wb') as f: pickle.dump(obj=model_dict, file=f, protocol=2)

### In order to pickle a module object store the dictionary
def store_module_in_dict(module_object):
    module_dict={}
    for key in module_object.__dict__.keys():
        try: 
            tmp=pickle.dumps(module_object.__dict__[key])
            module_dict[key]=module_object.__dict__[key]
        except:
            pass
    return module_dict

### Write a script that loads the pickled objects
def write_load_pickle_data_script(pickle_file):
    file_name = '{0}{1}/load_pickle_data_script.py'.format(config.OUTPUT_PATH, config.OUTPUT_PREFIX)
    f = open(file_name, 'w')
    f.write(
    """
import sys
import os
os.chdir('/users/pgreens/git/boosting_2D')
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
TuningParams = namedtuple('TuningParams', [
    'num_iter',
    'use_stumps', 'use_stable', 'use_corrected_loss', 'use_prior',
    'eta_1', 'eta_2', 'bundle_max', 'epsilon'
])

### LOAD DATA
################################################################################################
################################################################################################

with gzip.open('{0}','rb') as f: 
    model_dict = pickle.load(f)

# Assign data structures
x1 = model_dict['x1']
x2 = model_dict['x2']
y = model_dict['y']
tree = model_dict['tree']

# Unpack config and prior dictionaries
for key in model_dict['config'].keys():
    exec('config.%s=model_dict["config"]["%s"]'%(key, key))
for key in model_dict['prior'].keys():
    exec('prior.%s=model_dict["prior"]["%s"]'%(key, key))

    """.format(pickle_file))
    print 'load data from {0}'.format(file_name)
    f.close()

