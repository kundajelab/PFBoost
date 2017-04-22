import sys
import os
import random

import numpy as np 
from scipy.sparse import *
import pdb

from collections import namedtuple
import multiprocessing
import ctypes

from boosting2D import util
from boosting2D import config
from boosting2D import find_rule
from boosting2D import hierarchy as h
from boosting2D.multiprocessing_utils import fork_and_wait

log = util.log

### Define bundle class - store  bundled motifs+regulators with min loss rule
class BundleStore(object):
    def __init__(self, 
                 rule_bundle_regup_motifs,
                 rule_bundle_regup_regs,
                 rule_bundle_regdown_motifs,
                 rule_bundle_regdown_regs):
        self.rule_bundle_regup_motifs = rule_bundle_regup_motifs
        self.rule_bundle_regup_regs = rule_bundle_regup_regs
        assert len(self.rule_bundle_regup_regs) == \
               len(self.rule_bundle_regup_motifs)

        self.rule_bundle_regdown_motifs = rule_bundle_regdown_motifs
        self.rule_bundle_regdown_regs = rule_bundle_regdown_regs
        assert len(self.rule_bundle_regdown_regs) == \
               len(self.rule_bundle_regdown_motifs)
        self.size = self.__get_bundle_size__()
        
    def __get_bundle_size__(self):
        return len(self.rule_bundle_regdown_motifs) + len(
            self.rule_bundle_regup_motifs)

### Stabilization Functions
# Get the stabilization threshold based on the current weights
def stable_boost_thresh(tree, y, weights_i): 
    if y.sparse:
       stable_thresh = np.sqrt(
        np.matrix.sum((np.square(weights_i[weights_i.nonzero()])/np.square(
                       np.matrix.sum(weights_i[weights_i.nonzero()])))))
    else:
       stable_thresh = np.sqrt(
        sum((np.square(weights_i[weights_i.nonzero()])/np.square(
             sum(weights_i[weights_i.nonzero()])))))
    return stable_thresh

# Calculate theta or score for the bundle 
def get_rule_score_and_indices(rule_bundle, training_examples, 
                               testing_examples, weights_i,
                               rule_weights, tree, y, x1, x2, holdout,
                               rule_train_index, rule_test_index):

    ### ADD IN 
    if rule_bundle.size==1:
        rule_score = util.calc_score(tree, rule_weights, rule_train_index)
        motif_bundle = []
        regulator_bundle = []
        return rule_score, rule_train_index, rule_test_index

    # Get a lock
    lock_stable = multiprocessing.Lock()

    # Initialize shared data objects
    theta_alphas = multiprocessing.RawArray(ctypes.c_double,rule_bundle.size)
    bundle_train_pred = multiprocessing.RawArray(ctypes.c_double,y.num_row*y.num_col)
    bundle_test_pred = multiprocessing.RawArray(ctypes.c_double,y.num_row*y.num_col)

    # Store the value of the next rule that needs to be worked on
    rule_index_cntr = multiprocessing.Value('i', 0)

    # Pack arguments
    stable_args = [y, x1, x2, rule_index_cntr, rule_bundle, \
                   training_examples, testing_examples, 
                   rule_weights.w_pos, rule_weights.w_neg,
                   (lock_stable, theta_alphas, bundle_train_pred, bundle_test_pred)]

    # Fork worker processes, and wait for them to return
    fork_and_wait(config.NCPU, return_rule_index, stable_args)

    ### Get results back into the right format ()
    theta_alphas =  np.array(theta_alphas)
    if y.sparse:
        bundle_train_pred = csr_matrix(np.reshape(
            np.array(bundle_train_pred), (y.data.shape)))
        bundle_test_pred = csr_matrix(np.reshape(
            np.array(bundle_test_pred), (y.data.shape)))
    else:
        bundle_train_pred = np.reshape(np.array(bundle_train_pred), (y.data.shape))
        bundle_test_pred = np.reshape(np.array(bundle_test_pred), (y.data.shape))

    # Calculate theta
    min_val = min([abs(a) for a in theta_alphas])
    theta = sum([abs(alph)-min_val for alph in theta_alphas])/2

    # new index is where absolute value greater than theta
    new_train_rule_ind = (abs(bundle_train_pred)>theta)
    new_test_rule_ind = (abs(bundle_test_pred)>theta)

    # calculate W+ and W- for new rule
    w_pos = util.element_mult(weights_i, holdout.ind_train_up)
    w_neg = util.element_mult(weights_i, holdout.ind_train_down)
    w_bundle_pos = util.element_mult(w_pos, new_train_rule_ind)
    w_bundle_neg = util.element_mult(w_neg, new_train_rule_ind) 

    # get score of new rule
    rule_bundle_score = 0.5*np.log(
                            (w_bundle_pos.sum()+config.TUNING_PARAMS.epsilon)/
                            (w_bundle_neg.sum()+config.TUNING_PARAMS.epsilon))

    return rule_bundle_score, new_train_rule_ind, new_test_rule_ind

def return_rule_index(y, x1, x2, rule_index_cntr, rule_bundle, 
                      best_split_train_index, best_split_test_index, w_pos, w_neg, 
                      (lock_stable, theta_alphas, bundle_train_pred, bundle_test_pred)):
    while True:
        # Get the leaf node to work on
        with rule_index_cntr.get_lock():
            rule_index = rule_index_cntr.value
            rule_index_cntr.value += 1
        
        # If this isn't a valid leaf, then we are done
        if rule_index >= len(rule_bundle.rule_bundle_regup_motifs) + \
                         len(rule_bundle.rule_bundle_regdown_motifs): 
            return
        
        # Allocate rule matrix to save memory (how to do that)
        if y.sparse:
            valid_mat_h = csr_matrix((y.num_row,y.num_col), dtype=bool)
        else:
            valid_mat_h = np.zeros((y.num_row,y.num_col))

        m_h = (rule_bundle.rule_bundle_regup_motifs+
               rule_bundle.rule_bundle_regdown_motifs)[rule_index]
        r_h = (rule_bundle.rule_bundle_regup_regs+
               rule_bundle.rule_bundle_regdown_regs)[rule_index]
        reg_h = ([+1]*len(rule_bundle.rule_bundle_regup_motifs)+
                 [-1]*len(rule_bundle.rule_bundle_regdown_motifs))[rule_index]

        if x1.sparse:
            valid_m_h = np.nonzero(x1.data[m_h,:])[1]
            valid_r_h = np.where(x2.data.toarray()[:,r_h]==reg_h)[0]
        else:
            valid_m_h = np.nonzero(x1.data[m_h,:])[0]
            valid_r_h = np.where(x2.data[:,r_h]==reg_h)[0]

        # Calculate the loss for this leaf  
        valid_mat_h[np.ix_(valid_m_h, valid_r_h)]=1
        rule_train_index_h = util.element_mult(valid_mat_h, best_split_train_index)
        rule_test_index_h = util.element_mult(valid_mat_h, best_split_test_index)
        rule_score_h = 0.5*np.log((util.element_mult(
                                 w_pos, rule_train_index_h).sum()+
                                 config.TUNING_PARAMS.epsilon)/(
                                 util.element_mult(w_neg,
                                 rule_train_index_h).sum()
                                 +config.TUNING_PARAMS.epsilon))

        # print rule_index

        # Update current predictions
        with lock_stable:

            # Add current rule to training and testing sets
            current_bundle_train_pred = np.reshape(
                np.array(bundle_train_pred), y.data.shape)
            updated_bundle_train_pred = current_bundle_train_pred + \
                                        rule_score_h*rule_train_index_h

            current_bundle_test_pred = np.reshape(
                np.array(bundle_test_pred), y.data.shape)
            updated_bundle_test_pred = current_bundle_test_pred + \
                                       rule_score_h*rule_test_index_h

            # Update shared object
            theta_alphas[rule_index]=rule_score_h
            if y.sparse:
                bundle_train_pred[:]=np.array(
                    updated_bundle_train_pred.ravel(), copy=False)[0]
                bundle_test_pred[:]=np.array(
                    updated_bundle_test_pred.ravel(), copy=False)[0]
            else:
                bundle_train_pred[:]=updated_bundle_train_pred.ravel()
                bundle_test_pred[:]=updated_bundle_test_pred.ravel()


# Get rules to average (give motif, regulator and index)
# @profile
def bundle_rules(tree, y, x1, x2, m, r, reg, best_split, 
                 rule_weights, hierarchy, hierarchy_node):
    level = 'VERBOSE' if config.VERBOSE else 'QUIET'

    log('starting bundle rules', level=level)
    log('best split is {0}'.format(best_split), level=level)
    log('calculate A', level=level)

    non_hier_training_examples = tree.ind_pred_train[best_split]
    training_examples = h.get_hierarchy_index(hierarchy_node, hierarchy, 
                                              non_hier_training_examples, tree)

    # Get weights based on current training examples (obeying hierarchy)
    # (formerly tree.ind_pred_train[best_split])
    weights_i = util.element_mult(tree.weights, training_examples) 

    # SYMM DIFF - calculate weights and weights squared of best loss_rule (A)
    if reg==1:
        a_val = rule_weights.w_up_regup[m,r]+ \
        rule_weights.w_down_regup[m,r]
    elif reg==-1:
        a_val = rule_weights.w_up_regdown[m,r]+ \
        rule_weights.w_down_regdown[m,r]

    # Allocate matrix of weight value
    if y.sparse:
        a_weights = csr_matrix(a_val*
            np.ones(shape=rule_weights.w_down_regup.shape))
    else:
        a_weights = a_val*np.ones(shape=rule_weights.w_down_regup.shape)

    ## Calculate weights and weights square of all the other rules (B)
    # W+ + W- from find_rule()
    log('calculate B', level=level)
    b_weights_regup = rule_weights.w_up_regup+ \
        rule_weights.w_down_regup 
    b_weights_regdown = rule_weights.w_up_regdown+ \
            rule_weights.w_down_regdown

    ## Calculate intersection of A and B (A union B)
    # Allocate matrix with best rule in repeated m matrix, and best rule in repeated r matrix
    log('calculate A+B', level=level)
    if y.sparse:
        reg_vec = (x2.data[:,r]==reg)
    else:
        reg_vec = np.reshape((x2.data[:,r]==reg), (x2.num_row,1))
    # Multiply best rule times all other rules
    log('best rule times others', level=level)
    x1_intersect = util.element_mult(x1.data[m,:], x1.data)
    x2_up = x2.element_mult(x2.data>0)
    x2_down = abs(x2.element_mult(x2.data<0))
    x2_intersect_regup = util.element_mult(reg_vec, x2_up)
    x2_intersect_regdown = util.element_mult(reg_vec, x2_down)

    # Get weights for intersection
    x1_intersect_weights = util.matrix_mult(x1_intersect, weights_i)
    log('intersection weights', level=level)
    ab_weights_regup = util.matrix_mult(
        x1_intersect_weights, x2_intersect_regup) # PRE-FILTER weights
    ab_weights_regdown = util.matrix_mult(
        x1_intersect_weights, x2_intersect_regdown) # PRE-FILTER weights

    # Get symmetric difference in weights
    log('symmetric diff', level=level)
    symm_diff_w_regup = a_weights + b_weights_regup - 2*ab_weights_regup
    symm_diff_w_regdown = a_weights + b_weights_regdown - 2*ab_weights_regdown

    ## Calculate threshold for stabilization
    log('get threshold', level=level)
    if y.sparse:
        bundle_thresh = util.calc_sqrt_sum_sqr_sqr_sums(weights_i.data)
    else:
        bundle_thresh = util.calc_sqrt_sum_sqr_sqr_sums(weights_i.ravel()) 
    
    ## If large bundle, but hard cap on number of rules in bundle:
    log('test bundle size', level=level)
    test_big_bundle = (symm_diff_w_regup < \
         config.TUNING_PARAMS.eta_1*bundle_thresh).sum() + \
         (symm_diff_w_regdown < \
         config.TUNING_PARAMS.eta_1*bundle_thresh).sum() \
          > config.TUNING_PARAMS.bundle_max
    # If large bundle cap at max bundle size
    if test_big_bundle:
        log('cap bundle', level=level)
        print "="*80
        print 'large bundle - cap at {0}'.format(config.TUNING_PARAMS.bundle_max)
        ### Get rule bundles
        if y.sparse: 
            rule_bundle_regup = np.where(symm_diff_w_regup.todense().ravel().argsort().argsort().reshape(symm_diff_w_regup.toarray().shape) < config.TUNING_PARAMS.bundle_max/2)
            rule_bundle_regdown = np.where(symm_diff_w_regdown.todense().ravel().argsort().argsort().reshape(symm_diff_w_regup.toarray().shape) < config.TUNING_PARAMS.bundle_max/2) 
        else:
            rule_bundle_regup = np.where(symm_diff_w_regup.ravel().argsort().argsort().reshape(symm_diff_w_regup.shape) < config.TUNING_PARAMS.bundle_max/2)
            rule_bundle_regdown = np.where(symm_diff_w_regdown.ravel().argsort().argsort().reshape(symm_diff_w_regup.shape) < config.TUNING_PARAMS.bundle_max/2) 

        rule_bundle_regup_motifs = rule_bundle_regup[0].tolist() # Keeping min loss rule
        rule_bundle_regup_regs = rule_bundle_regup[1].tolist()
        rule_bundle_regdown_motifs = rule_bundle_regdown[0].tolist()
        rule_bundle_regdown_regs = rule_bundle_regdown[1].tolist()


    # Otherwise take all bundled rules
    else:
        log('keep bundle', level=level)
        rule_bundle_regup = (symm_diff_w_regup < \
            config.TUNING_PARAMS.eta_1*bundle_thresh).nonzero()
        rule_bundle_regdown = (symm_diff_w_regdown < \
            config.TUNING_PARAMS.eta_1*bundle_thresh).nonzero()

        rule_bundle_regup_motifs = rule_bundle_regup[0].tolist() # Keeping min loss rule
        rule_bundle_regup_regs = rule_bundle_regup[1].tolist()
        rule_bundle_regdown_motifs = rule_bundle_regdown[0].tolist()
        rule_bundle_regdown_regs = rule_bundle_regdown[1].tolist()

    # # Investigate large bundles
    # if len(rule_bundle_regup_motifs)+len(rule_bundle_regdown_motifs) > 40:
    #     pdb.set_trace()

    # Print names of x1/x2 features that are bundled
    rule_bundle_motifs = x1.row_labels[ \
        rule_bundle_regup_motifs+rule_bundle_regdown_motifs]
    rule_bundle_regs = x2.col_labels[ \
        rule_bundle_regup_regs+rule_bundle_regdown_regs]

    # Return list where first element is bundle where reg_up and second is where reg_down
    return BundleStore(rule_bundle_regup_motifs, rule_bundle_regup_regs,
                         rule_bundle_regdown_motifs, rule_bundle_regdown_regs)

# Statistic to see whether stabilization applies
def stable_boost_test(tree, rule_train_index, holdout):
    w_pos = util.element_mult(tree.weights, holdout.ind_train_up)
    w_neg = util.element_mult(tree.weights, holdout.ind_train_down) 
    test = 0.5*abs(util.element_mult(w_pos, rule_train_index).sum() - \
           util.element_mult(w_neg, rule_train_index).sum())
    return test





