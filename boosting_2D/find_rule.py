from grit.lib.multiprocessing_utils import fork_and_wait
import multiprocessing
import ctypes

import random
from collections import namedtuple

import numpy as np 
from scipy.sparse import *
import pdb

from boosting_2D import config
from boosting_2D import util

log = util.log

### Find best rule - NON-STABILIZED
##################################################################

ObjectStore = namedtuple("ObjectStore", [
    'w_up_regup', 'w_up_regdown', 
    'w_down_regup', 'w_down_regdown',
    'w_zero_regup', 'w_zero_regdown',
    'w_pos', 'w_neg'])

# Calc min loss with leaf training examples and current weights 
def find_min_loss(tree, leaf_training_examples, holdout, y, x1, x2):
    # log('start find_leaf_and_min_loss')
    example_weights=tree.weights
    ones_mat=tree.ones_mat
    # log('start calc_min_leaf_loss')
    (best_loss, regulator_sign) = calc_min_leaf_loss(
        leaf_training_examples, example_weights, ones_mat, holdout, y, x1, x2)
    # log('end calc_min_leaf_loss')
    return best_loss, regulator_sign

# For every leaf, get training examples and calculate loss
# Keep only leaf with best loss
def find_rule_process_worker(
        tree, holdout, y, x1, x2, leaf_index_cntr, (
            lock, best_leaf_loss, best_leaf_index, 
            shared_best_loss_mat, best_leaf_regulator_sign)):
    # until we have processed all of the leafs
    while True:
        # get the leaf node to work on
        with leaf_index_cntr.get_lock():
            leaf_index = leaf_index_cntr.value
            leaf_index_cntr.value += 1
        
        # if this isn't a valid leaf, then we are done
        if leaf_index >= tree.nsplit: 
            return
        
        leaf_training_examples = tree.ind_pred_train[leaf_index]

        # calculate the loss for this leaf  
        leaf_loss_mat, regulator_sign = find_min_loss(
            tree, leaf_training_examples, holdout, y, x1, x2)
        
        # if the loss does not beat the current best loss, then
        # we are done
        leaf_min_loss = leaf_loss_mat.min()
        with lock:
            if leaf_min_loss > best_leaf_loss:
                continue

            # otherwise we know this rule currently produces the smallest loss, 
            # so save it
            best_leaf_index.value = leaf_index
            best_leaf_loss.value = leaf_min_loss
            # we update the array being stored in shared memory 
            if tree.sparse:
                shared_best_loss_mat[:] = leaf_loss_mat.toarray().ravel()
            else:
                shared_best_loss_mat[:] = leaf_loss_mat.ravel()
            best_leaf_regulator_sign.value = regulator_sign
    
    return

# @profile
def find_rule_processes(tree, holdout, y, x1, x2):
    if config.TUNING_PARAMS.use_stumps:
        # since we aren't building a tree, we use all of the
        # training examples to choose a rule
        leaf_training_examples = tree.ind_pred_train[0]
        leaf_loss_mat, regulator_sign = find_min_loss(
            tree, leaf_training_examples, holdout, y, x1, x2)
        return 0, regulator_sign, leaf_loss_mat

    rule_processes = []

    # this should be an attribute of tree. Also, during the tree init,
    # accessing the global variables x1, x2, and y is really bad form. Since
    # you only need the dimensions you sohuld pass those as arguments into the 
    # init function. 
    nrow = x1.num_row
    ncol = x2.num_col
    
    # Initialize a lock to control access to the best rule objects. We use
    # rawvalues because all access is governed through the lock
    lock = multiprocessing.Lock()
    # initialize this to a large number, so that the first loss is chosen
    best_loss = multiprocessing.RawValue(ctypes.c_double, 1e100)
    best_leaf = multiprocessing.RawValue('i', -1)
    shared_best_loss_mat = multiprocessing.RawArray(
        ctypes.c_double, nrow*ncol)
    best_loss_reg = multiprocessing.RawValue('i', 0)

    # Store the value of the next leaf index that needs to be processed, so that
    # the workers know what leaf to work on
    leaf_index_cntr = multiprocessing.Value('i', 0)

    # pack arguments for the worker processes
    args = [tree, holdout, y, x1, x2, leaf_index_cntr, (
            lock, best_loss, best_leaf, shared_best_loss_mat, best_loss_reg)]
    
    # Fork worker processes, and wait for them to return
    fork_and_wait(config.NCPU, find_rule_process_worker, args)
    
    # Covert all of the shared types into standard python values
    best_leaf = int(best_leaf.value)
    best_loss_reg = int(best_loss_reg.value)
    # we convert the raw array into a numpy array
    best_loss_mat = np.reshape(np.array(shared_best_loss_mat), (nrow, ncol))
    
    # Return rule_processes
    return (best_leaf, best_loss_reg, best_loss_mat)

# Function - calc min loss with leaf training examples and current weights  
def calc_min_leaf_loss(leaf_training_examples, example_weights, ones_mat, holdout, y, x1, x2):
    # log('start find_rule_weights')
    rule_weights = find_rule_weights(leaf_training_examples, example_weights,
     ones_mat, holdout, y, x1, x2)
    # log('end find_rule_weights')
    
    ## Calculate Loss
    if config.TUNING_PARAMS.use_corrected_loss==True:
        loss_regup = util.corrected_loss(rule_weights.w_up_regup,
         rule_weights.w_down_regup, rule_weights.w_zero_regup)
        loss_regdown = util.corrected_loss(rule_weights.w_up_regdown,
         rule_weights.w_down_regdown, rule_weights.w_zero_regdown)
    else:
        loss_regup = util.calc_loss(rule_weights.w_up_regup,
         rule_weights.w_down_regup, rule_weights.w_zero_regup)
        loss_regdown = util.calc_loss(rule_weights.w_up_regdown,
         rule_weights.w_down_regdown, rule_weights.w_zero_regdown)

    ## Get loss matrix and regulator status
    loss_best_s = np.min([loss_regup.min(), loss_regdown.min()])
    loss_arg_min = np.argmin([loss_regup.min(), loss_regdown.min()])
    if loss_arg_min==0:
        reg_s=1
    else:
        reg_s=-1

    loss = [loss_regup, loss_regdown][loss_arg_min]
    return (loss, reg_s)

# Get rule weights of positive and negative examples
# @profile
def find_rule_weights(leaf_training_examples, example_weights, ones_mat, holdout, y, x1, x2):
    """
    Find rule weights, and return an object store containing them. 

    """
    # log('find_rule_weights start')
    w_temp = util.element_mult(example_weights, leaf_training_examples)
    # log('weights element-wise')
    w_pos = util.element_mult(w_temp, holdout.ind_train_up)
    # log('weights element-wise')
    w_neg = util.element_mult(w_temp, holdout.ind_train_down) 
    # log('weights element-wise')
    x2_pos = x2.element_mult(x2.data>0)
    # log('x2 element-wise')
    x2_neg = abs(x2.element_mult(x2.data<0))
    # log('x2 element-wise')
    x1wpos = x1.matrix_mult(w_pos)
    # log('x1 weights dot')
    x1wneg = x1.matrix_mult(w_neg)
    # log('x1 weights dot')
    w_up_regup = util.matrix_mult(x1wpos, x2_pos)
    # log('x1w x2 dot')
    w_up_regdown = util.matrix_mult(x1wpos, x2_neg)
    # log('x1w x2 dot')
    w_down_regup = util.matrix_mult(x1wneg, x2_pos)
    # log('x1w x2 dot')
    w_down_regdown = util.matrix_mult(x1wneg, x2_neg)
    # log('x1w x2 dot')
    w_zero_regup = ones_mat - w_up_regup - w_down_regup
    # log('weights subtraction')
    w_zero_regdown = ones_mat - w_up_regdown - w_down_regdown
    # log('weights subtraction')
    return ObjectStore(w_up_regup, w_up_regdown, 
                       w_down_regup, w_down_regdown, 
                       w_zero_regup, w_zero_regdown, 
                       w_pos, w_neg) 


# From the best split, get the current rule (motif, regulator, regulator sign and test/train indices)
def get_current_rule(tree, best_split, regulator_sign, loss_best, holdout, y, x1, x2):
    motif,regulator = np.where(loss_best == loss_best.min())
    # If multiple rules have the same loss, randomly select one
    if len(motif)>1:
        choice = random.sample(range(len(motif)), 1)
        motif = np.array(motif[choice])
        regulator = np.array(regulator[choice])
       
    # Convert to int
    if isinstance(motif,int)==False:
        motif=int(motif)
    if isinstance(regulator,int)==False:
        regulator=int(regulator)

    ## Find indices of where motif and regulator appear
    if x2.sparse:
        valid_m = np.nonzero(x1.data[motif,:])[1]
        valid_r = np.where(x2.data.toarray()[:,regulator]==regulator_sign)[0]
    else:
        valid_m = np.nonzero(x1.data[motif,:])[0]
        valid_r = np.where(x2.data[:,regulator]==regulator_sign)[0]
 
    ### Get joint motif-regulator index - training and testing
    if y.sparse:
        valid_mat = csr_matrix((y.num_row,y.num_col), dtype=np.bool)
    else:
        valid_mat = np.zeros((y.num_row,y.num_col), dtype=np.bool)
    valid_mat[np.ix_(valid_m, valid_r)]=1 # XX not efficient
    rule_train_index = util.element_mult(valid_mat, tree.ind_pred_train[best_split])
    rule_test_index = util.element_mult(valid_mat, tree.ind_pred_test[best_split])

    return motif, regulator, regulator_sign, rule_train_index, rule_test_index

