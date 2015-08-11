import sys
import config
import time
from datetime import datetime
import pickle
import copy

import numpy as np 
import sklearn.utils
from scipy.sparse import *


### Log 
##########################################

class Logger():
    def __init__(self, ofp=sys.stderr):
        self.ofp = ofp
    
    def __call__(self, msg, log_time=True, level=None):
        assert level in ('VERBOSE', 'QUIET', None)
        # if level == 'VERBOSE' or config.VERBOSE: 
        if level == 'QUIET' or not config.VERBOSE: return
        if log_time:
            time_stamp = datetime.fromtimestamp(time.time()).strftime(
                '%Y-%m-%d %H:%M:%S: ')
            msg = time_stamp + msg
        self.ofp.write(msg.strip() + "\n")

def log_progress(tree, i):
    msg = "\n".join([
        'imbalanced train error: {0}'.format(tree.imbal_train_err[i]),
        'imbalanced test error: {0}'.format(tree.imbal_test_err[i]),
        'x1 split feat {0}'.format(tree.split_x1[i]),
        'x2 split feat {0}'.format(tree.split_x2[i]),
        'rule score {0}'.format(tree.scores[i])])
    log(msg, log_time=False, level='VERBOSE')

### log prints to STDERR
log = Logger()

### Label Functions 
##########################################

# Get method label added to config.OUTPUT_PREFIX
def get_method_label():
    if config.TUNING_PARAMS.use_stable:
        stable_label='stable'
    else:
        stable_label='non_stable'
    if config.TUNING_PARAMS.use_stumps:
        method='stumps'
    else:
        method='adt'
    method_label = '{0}_{1}'.format(method, stable_label)
    return method_label


### Save Tree State 
##########################################

def save_tree_state(tree, pickle_file):
    with open(pickle_file,'wb') as f:
        pickle.dump(obj=tree, file=f)

def load_tree_state(pickle_file):
    with open(pickle_file,'rb') as f:
        pickle.load(f)

### Calculation Functions
##########################################

def calc_score(tree, rule_weights, rule_train_index):
    rule_score = 0.5*np.log((
        element_mult(rule_weights.w_pos, rule_train_index).sum()+config.TUNING_PARAMS.epsilon)/
        (element_mult(rule_weights.w_neg, rule_train_index).sum()+config.TUNING_PARAMS.epsilon))
    return rule_score

def calc_loss(wpos, wneg, wzero):
    loss = 2*np.sqrt(element_mult(wpos, wneg))+wzero
    return loss

def calc_margin(y, pred_test):
    # (Y * predicted value (h(x_i))
    margin = element_mult(y, pred_test)
    return margin.sum()


### Matrix Operations
##########################################

def element_mult(matrix1, matrix2):
    if isinstance(matrix1, csr.csr_matrix) and isinstance(matrix2, csr.csr_matrix):
        return matrix1.multiply(matrix2)
    elif isinstance(matrix1, np.ndarray) and isinstance(matrix2, np.ndarray):
        return np.multiply(matrix1, matrix2)
    else:
        assert False, "Inconsistent matrix formats '%s' '%s'" % (type(matrix1), type(matrix2))

def matrix_mult(matrix1, matrix2):
    if isinstance(matrix1, csr.csr_matrix) and isinstance(matrix2, csr.csr_matrix):
        return matrix1.dot(matrix2)
    elif isinstance(matrix1, np.ndarray) and isinstance(matrix2, np.ndarray):
        return np.dot(matrix1, matrix2)
    else:
        assert False, "Inconsistent matrix formats '%s' '%s'" % (type(matrix1), type(matrix2))

### Randomization Functions
##########################################

### Takes a data class object (y, x1, x2) and shuffle the data (in the same proportions)
def shuffle_data_object(obj):
    shuffle_obj = copy.deepcopy(obj)
    if obj.sparse:
        shuffle_obj.data = sklearn.utils.shuffle(obj.data, replace=False, random_state=1)
    else:
        random.seed(1)
        shuffle_obj.data = random.shuffle(shuffle_obj.data)
    return shuffle_obj
