import sys
import config
import time
from datetime import datetime
import pickle
import copy
import gzip
import random
import pdb

import numpy as np 
import sklearn.utils
from scipy.sparse import *


### Log 
##########################################

class Logger():
    def __init__(self, ofp=sys.stderr, verbose=config.VERBOSE):
        self.ofp = ofp
        self.verbose = verbose
    
    def __call__(self, msg, log_time=True, level=None):
        assert level in ('VERBOSE', 'QUIET', None)
        # if level == 'VERBOSE' or config.VERBOSE: 
        if level == 'QUIET' or not self.verbose: return
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
    with gzip.open(pickle_file,'wb') as f:
        pickle.dump(obj=tree, file=f)

def load_tree_state(pickle_file):
    with gzip.open(pickle_file,'rb') as f:
        tree = pickle.load(f)

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

# Element-wise  multiplication
def element_mult(matrix1, matrix2):
    if isinstance(matrix1, csr.csr_matrix) and isinstance(matrix2, csr.csr_matrix):
        return matrix1.multiply(matrix2)
    elif isinstance(matrix1, np.ndarray) and isinstance(matrix2, np.ndarray):
        return np.multiply(matrix1, matrix2)
    else:
        assert False, "Inconsistent matrix formats '%s' '%s'" % (type(matrix1), type(matrix2))

# Matrix multiplication
def matrix_mult(matrix1, matrix2):
    if isinstance(matrix1, csr.csr_matrix) and isinstance(matrix2, csr.csr_matrix):
        return matrix1.dot(matrix2)
    elif isinstance(matrix1, np.ndarray) and isinstance(matrix2, np.ndarray):
        return np.dot(matrix1, matrix2)
    else:
        assert False, "Inconsistent matrix formats '%s' '%s'" % (type(matrix1), type(matrix2))

# Convert type of matrix1 to match matrix2 if mismatch is between sparse and numpy array
def convert_type_to_match(matrix1, matrix2):
    if type(matrix1) == type(matrix2):
        return matrix1
    elif isinstance(matrix1,csr_matrix) and isinstance(matrix2, np.ndarray):
        matrix1_new = matrix1.toarray()
    elif isinstance(matrix1,np.ndarray) and isinstance(matrix2, csr_matrix):
        matrix1_new = csr_matrix(matrix1)
    return matrix1_new

### Randomization Functions
##########################################

### Takes a data class object (y, x1, x2) and shuffle the data (in the same proportions)
def shuffle_data_object(obj):
    shuffle_obj = copy.deepcopy(obj)
    if shuffle_obj.sparse:
        shuffle_obj.data = sklearn.utils.shuffle(shuffle_obj.data, replace=False, random_state=1)
    else:
        random.seed(1)
        shuffle_obj.data = np.random.permutation(shuffle_obj.data.ravel()).reshape(shuffle_obj.data.shape)
    return shuffle_obj

### Cluster Regulators
##########################################

import scipy.cluster.hierarchy as hier
import scipy.spatial.distance as dist

# Get cluster assignments based on euclidean distance + complete linkage
def get_data_clusters(data, max_distance=0):
    d = dist.pdist(data, 'euclidean') 
    l = hier.linkage(d, method='complete')
    ordered_data = data[hier.leaves_list(l),:]
    flat_clusters = hier.fcluster(l, t=max_distance, criterion='distance')
    print 'reduced {0} entries to {1} based on max distance {2}'.format(
        data.shape[0], len(np.unique(flat_clusters)), max_distance)
    return(flat_clusters)

# Compress data by taking average of elements in cluster
def regroup_data_by_clusters(data, clusters):
    new_data = np.zeros((len(np.unique(clusters)),data.shape[1]))
    for clust in np.unique(clusters):
        new_feat = np.apply_along_axis(np.mean, 0, data[clusters==clust,:])
        # clusters are 1-based, allocate into 0-based array
        new_data[clust-1,:]=new_feat
    return(new_data)

# Re-write labels by concatenating original labels
def regroup_labels_by_clusters(labels, clusters):
    new_labels = ['na']*len(np.unique(clusters))
    for clust in np.unique(clusters):
        new_label = '|'.join(labels[clusters==clust])
        # clusters are 1-based, allocate into 0-based array
        new_labels[clust-1]=new_label
    return(new_labels)

# Re=cast x2 object with compressed regulator data and labels
def compress_regulators(x2_obj):
    data = x2_obj.data.toarray().T if x2_obj.sparse else x2_obj.data.T
    labels = x2_obj.col_labels
    clusters = get_data_clusters(data, max_distance=0)
    new_data = regroup_data_by_clusters(data, clusters)
    new_labels = regroup_labels_by_clusters(labels, clusters)
    x2_obj.data = csr_matrix(new_data.T) if x2_obj.sparse else new_data.T
    x2_obj.col_labels = new_labels
    return(x2_obj)

# x2_obj = copy.deepcopy(x2)
# new_x2_obj = util.compress_regulators(x2_obj)



