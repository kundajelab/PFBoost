import sys
import os
import random

import numpy as np 
from scipy.sparse import *
import pdb

from collections import namedtuple
import multiprocessing
import ctypes
from grit.lib.multiprocessing_utils import fork_and_wait

from boosting_2D import util
from boosting_2D import config
from boosting_2D import find_rule


## !! TEMP IN TERMS OF FORK NOT POOL
##########
####################
##############################

# Calculate theta or score for the bundle 
# !! give just 
# def calc_theta(rule_bundle, ind_pred_train, ind_pred_test, best_split, w_pos, w_neg, y, x1, x2):

#     # calculate alpha for each rule
#     motifs_up = rule_bundle.rule_bundle_regup_motifs
#     regs_up = rule_bundle.rule_bundle_regup_regs
#     motifs_down = rule_bundle.rule_bundle_regdown_motifs
#     regs_down = rule_bundle.rule_bundle_regdown_regs
#     bundle_size=len(motifs_up)+len(motifs_down)

#     # Get a lock
#     lock_stable = multiprocessing.Lock()

#     # Initialize shared data objects
#     theta_alphas = multiprocessing.RawArray(ctypes.c_double,bundle_size)
#     bundle_train_rule_indices = multiprocessing.RawArray(ctypes.c_double,bundle_size)
#     bundle_test_rule_indices = multiprocessing.RawArray(ctypes.c_double,bundle_size)
#     # bundle_train_rule_indices = (multiprocessing.RawArray(ctypes.c_double,bundle_size) for el in range(bundle_size))
#     # bundle_test_rule_indices = (multiprocessing.RawArray(ctypes.c_double,bundle_size) for el in range(bundle_size))


#     # Store the value of the next rule that needs to be worked on
#     rule_index_cntr = multiprocessing.Value('i', 0)

#     # Pack arguments
#     stable_args = [y, x1, x2, rule_index_cntr, rule_bundle, ind_pred_train[best_split], w_pos, w_neg, (
#         lock_stable, theta_alphas, bundle_train_rule_indices, bundle_test_rule_indices)]

#     pdb.set_trace()

#     # Fork worker processes, and wait for them to return
#     fork_and_wait(config.NCPU, return_rule_index, stable_args)

#     ### Get results back into the right format
#     # theta_alphas = [el[0] for el in bundle_rule_info]
#     # bundle_train_rule_indices = [el[1] for el in bundle_rule_info]
#     # bundle_test_rule_indices = [el[2] for el in bundle_rule_info]

#     # Calculate theta
#     theta = sum([abs(alph) for alph in theta_alphas]-min([abs(a) for a in theta_alphas]))/2

#     return [theta, theta_alphas, bundle_train_rule_indices, bundle_test_rule_indices]

# def return_rule_index(y, x1, x2, rule_index_cntr, rule_bundle, best_split_train_index, w_pos, w_neg, (
#             lock_stable, theta_alphas, bundle_train_rule_indices, bundle_test_rule_indices)):
#     while True:
#         # get the leaf node to work on
#         with rule_index_cntr.get_lock():
#             rule_index = rule_index_cntr.value
#             rule_index_cntr.value += 1
        
#         # if this isn't a valid leaf, then we are done
#         if rule_index >= len(rule_bundle.rule_bundle_regup_motifs)+len(rule_bundle.rule_bundle_regdown_motifs): 
#             return
        
#         # Allocate rule matrix to save memory (how to do that)
#         if y.sparse:
#             valid_mat_h = csr_matrix((y.num_row,y.num_col), dtype=bool)
#         else:
#             valid_mat_h = np.zeros((y.num_row,y.num_col))

#         m_h = (rule_bundle.rule_bundle_regup_motifs+rule_bundle.rule_bundle_regdown_motifs)[rule_index]
#         r_h = (rule_bundle.rule_bundle_regup_regs+rule_bundle.rule_bundle_regdown_regs)[rule_index]
#         reg_h = ([+1]*len(rule_bundle.rule_bundle_regup_motifs)+[-1]*len(rule_bundle.rule_bundle_regdown_motifs))[rule_index]

#         if x1.sparse:
#             valid_m_h = np.nonzero(x1.data[m_h,:])[1]
#             valid_r_h = np.where(x2.data.toarray()[:,r_h]==reg_h)[0]
#         else:
#             valid_m_h = np.nonzero(x1.data[m_h,:])[0]
#             valid_r_h = np.where(x2.data[:,r_h]==reg_h)[0]
#         valid_mat_h[np.ix_(valid_m_h, valid_r_h)]=1

#         # calculate the loss for this leaf  
#         valid_mat_h[np.ix_(valid_m_h, valid_r_h)]=1
#         rule_train_index_h = util.element_mult(valid_mat_h, best_split_train_index)
#         # pdb.set_trace()
#         rule_test_index_h = util.element_mult(valid_mat_h, best_split_train_index)
#         # rule_test_index_h = element_mult(valid_mat_h, holdout.ind_test_all)
#         rule_score_h = 0.5*np.log((util.element_mult(w_pos, rule_train_index_h).sum()+
#             config.TUNING_PARAMS.epsilon)/(util.element_mult(w_neg, rule_train_index_h).sum()+config.TUNING_PARAMS.epsilon))
        
#         print rule_index_cntr.value
#         print rule_index

#         # return the score and indices of this rule
#         with lock_stable:
#             ### !!! BUG HERE because cannot store array in array
#             theta_alphas[rule_index]=rule_score_h
#             bundle_train_rule_indices[rule_index]=rule_train_index_h
#             bundle_test_rule_indices[rule_index]=rule_test_index_h
    

##########
####################
##############################

### Stabilization Functions

### Return bundle
BundleStore = namedtuple("BundleStore", [
    'rule_bundle_regup_motifs', 'rule_bundle_regup_regs', 
    'rule_bundle_regdown_motifs', 'rule_bundle_regdown_regs'])

# Get the stabilizationt threshold based on the current weights
def stable_boost_thresh(tree, y, weights_i): 
    if y.sparse:
       stable_thresh = np.sqrt(
        np.matrix.sum((np.square(weights_i[weights_i.nonzero()])/np.square(np.matrix.sum(weights_i[weights_i.nonzero()])))))
    else:
       stable_thresh = np.sqrt(
        sum((np.square(weights_i[weights_i.nonzero()])/np.square(sum(weights_i[weights_i.nonzero()])))))
    return stable_thresh

# function to parallelize the getting  ( can also add this function to the main loop)
def return_rule_index(m_h, r_h, reg_h, valid_mat_h, ind_pred_train, ind_pred_test, best_split, w_pos, w_neg, y, x1, x2):
    if y.sparse:
        valid_m_h = np.nonzero(x1.data[m_h,:])[1]
        valid_r_h = np.where(x2.data.toarray()[:,r_h]==reg_h)[0]
    else:
        valid_m_h = np.nonzero(x1.data[m_h,:])[0]
        valid_r_h = np.where(x2.data[:,r_h]==reg_h)[0]
    valid_mat_h[np.ix_(valid_m_h, valid_r_h)]=1
    rule_train_index_h = util.element_mult(valid_mat_h, ind_pred_train[best_split])
    # pdb.set_trace()
    rule_test_index_h = util.element_mult(valid_mat_h, ind_pred_test[best_split])
    # rule_test_index_h = element_mult(valid_mat_h, holdout.ind_test_all)
    rule_score_h = 0.5*np.log((util.element_mult(w_pos, rule_train_index_h).sum()+ config.TUNING_PARAMS.epsilon)/(util.element_mult(w_neg, rule_train_index_h).sum()+config.TUNING_PARAMS.epsilon))
    return [rule_score_h, rule_train_index_h, rule_test_index_h]

def return_rule_index_wrapper(args):
    return return_rule_index(*args)

# Calculate theta or score for the bundle 
# !! give just 
def calc_theta(rule_bundle, ind_pred_train, ind_pred_test, best_split, w_pos, w_neg, y, x1, x2, pool):
    # Allocate rule matrix to save memory
    if y.sparse:
        valid_mat_h = csr_matrix((y.num_row,y.num_col), dtype=bool)
    else:
        valid_mat_h = np.zeros((y.num_row,y.num_col))
    # calculate alpha for each rule
    motifs_up = rule_bundle.rule_bundle_regup_motifs
    regs_up = rule_bundle.rule_bundle_regup_regs
    motifs_down = rule_bundle.rule_bundle_regdown_motifs
    regs_down = rule_bundle.rule_bundle_regdown_regs
    bundle_rule_info = pool.map(return_rule_index_wrapper,
         iterable=[(motifs_up[ind],regs_up[ind], 1, valid_mat_h, ind_pred_train, ind_pred_test, best_split, w_pos, w_neg, y, x1, x2)
          for ind in range(len(motifs_up))]+
          [(motifs_down[ind],regs_down[ind], -1, valid_mat_h, ind_pred_train, ind_pred_test, best_split, w_pos, w_neg, y, x1, x2) 
          for ind in range(len(motifs_down))]) 
    # Get scores, indices
    theta_alphas = [el[0] for el in bundle_rule_info]
    bundle_train_rule_indices = [el[1] for el in bundle_rule_info]
    bundle_test_rule_indices = [el[2] for el in bundle_rule_info]
    # Calculate theta
    theta = sum([abs(alph) for alph in theta_alphas]-min([abs(a) for a in theta_alphas]))/2
    return [theta, theta_alphas, bundle_train_rule_indices, bundle_test_rule_indices]


# Get rules to average (give motif, regulator and index)
def bundle_rules(tree, y, x1, x2, m, r, reg, best_split, rule_weights):
    print 'starting bundle rules'
    print 'best split is {0}'.format(best_split)
    weights_i = util.element_mult(tree.weights, tree.ind_pred_train[best_split])
    # SYMM DIFF - calculate weights and weights squared of best loss_rule (A)
    if reg==1:
        if y.sparse:
            a_val = rule_weights.w_up_regup[m,r].tolist()[0][0]+ \
            rule_weights.w_down_regup[m,r].tolist()[0][0]
        else:
            a_val = rule_weights.w_up_regup[m,r].tolist()[0]+ \
            rule_weights.w_down_regup[m,r].tolist()[0]
    elif reg==-1:
        if y.sparse:
            a_val = rule_weights.w_up_regdown[m,r].tolist()[0][0]+ \
            rule_weights.w_down_regdown[m,r].tolist()[0][0]
        else:
            a_val = rule_weights.w_up_regdown[m,r].tolist()[0]+ \
            rule_weights.w_down_regdown[m,r].tolist()[0]
    if y.sparse:
        a_weights = csr_matrix(a_val*np.ones(shape=rule_weights.w_down_regup.shape))
    else:
        a_weights = a_val*np.ones(shape=rule_weights.w_down_regup.shape)
    ## calculate weights and weights square of all the other rules (B)
    # W+ + W- from find_rule()
    b_weights_regup = rule_weights.w_up_regup+ \
        rule_weights.w_down_regup 
    b_weights_regdown = rule_weights.w_up_regdown+ \
            rule_weights.w_down_regdown
    ## Calculate intersection of A and B (A union B)
    # allocate matrix with best rule in repeated m matrix, and best rule in repeated r matrix
    if y.sparse:
        x1_best = vstack([x1.data[m,:] for el in range(x1.num_row)])
        reg_vec = (x2.data[:,r]==reg)
        x2_best = hstack([reg_vec for el in range(x2.num_col)], format='csr') 
    else:
        x1_best = np.vstack([x1.data[m,:] for el in range(x1.num_row)])
        reg_vec = (x2.data[:,r]==reg)
        x2_best = np.hstack([reg_vec for el in range(x2.num_col)]) 
    # Multiply best rule times all other rules
    x1_intersect = util.element_mult(x1_best, x1.data)
    x2_up = x2.element_mult(x2.data>0)
    x2_down = abs(x2.element_mult(x2.data<0))
    x2_intersect_regup = util.element_mult(x2_best, x2_up)
    x2_intersect_regdown = util.element_mult(x2_best, x2_down)
    # Get weights for intersection
    ab_weights_regup = util.matrix_mult(util.matrix_mult(x1_intersect, weights_i), x2_intersect_regup) # PRE-FILTER weights
    ab_weights_regdown = util.matrix_mult(util.matrix_mult(x1_intersect, weights_i), x2_intersect_regdown) # PRE-FILTER weights
    # Get symmetric difference weights
    symm_diff_w_regup = a_weights + b_weights_regup - 2*ab_weights_regup
    symm_diff_w_regdown = a_weights + b_weights_regdown - 2*ab_weights_regdown
    ## Calculate threshold for stabilization
    if y.sparse:
        bundle_thresh = np.sqrt(sum([np.square(w)
             for w in weights_i[weights_i.nonzero()].tolist()[0]])/
             np.square(sum(weights_i[weights_i.nonzero()].tolist()[0])))
    else:
        bundle_thresh = np.sqrt(sum([np.square(w)
             for w in weights_i[weights_i.nonzero()].tolist()])/
             np.square(sum(weights_i[weights_i.nonzero()].tolist())))
    ### check indices of where stabilization threshold reached
    ## If large bundle, but hard cap:
    if y.sparse:
        if sum(sum(symm_diff_w_regup.toarray() < config.TUNING_PARAMS.eta_1*bundle_thresh)) > config.TUNING_PARAMS.bundle_max \
            or sum(sum(symm_diff_w_regdown.toarray() < config.TUNING_PARAMS.eta_1*bundle_thresh)) > config.TUNING_PARAMS.bundle_max:
            print "="*80
            print 'large bundle - capping at {0}'.format(config.TUNING_PARAMS.bundle_max)
            # Rank all of the entries and get the top number
            rule_bundle_regup = np.where(symm_diff_w_regup.todense().ravel().argsort().argsort().reshape(symm_diff_w_regup.toarray().shape) < config.TUNING_PARAMS.bundle_max)
            rule_bundle_regdown = np.where(symm_diff_w_regdown.todense().ravel().argsort().argsort().reshape(symm_diff_w_regup.toarray().shape) < config.TUNING_PARAMS.bundle_max) 
        else:
            rule_bundle_regup = np.where(symm_diff_w_regup.todense() < config.TUNING_PARAMS.eta_1*bundle_thresh) 
            rule_bundle_regdown = np.where(symm_diff_w_regdown.todense() < config.TUNING_PARAMS.eta_1*bundle_thresh)
        rule_bundle_regup_motifs = rule_bundle_regup[0].tolist()[0] # Keeping min loss rule
        rule_bundle_regup_regs = rule_bundle_regup[1].tolist()[0]
        rule_bundle_regdown_motifs = rule_bundle_regdown[0].tolist()[0]
        rule_bundle_regdown_regs = rule_bundle_regdown[1].tolist()[0]

    else:
        if sum(sum(symm_diff_w_regup < config.TUNING_PARAMS.eta_1*bundle_thresh)) > config.TUNING_PARAMS.bundle_max \
            or sum(sum(symm_diff_w_regdown < config.TUNING_PARAMS.eta_1*bundle_thresh)) > config.TUNING_PARAMS.bundle_max:
            print "="*80
            print 'large bundle - capping at {0}'.format(config.TUNING_PARAMS.bundle_max)
            # Rank all of the entries and get the top number
            rule_bundle_regup = np.where(symm_diff_w_regup.ravel().argsort().argsort().reshape(symm_diff_w_regup.shape) < config.TUNING_PARAMS.bundle_max)
            rule_bundle_regdown = np.where(symm_diff_w_regdown.ravel().argsort().argsort().reshape(symm_diff_w_regup.shape) < config.TUNING_PARAMS.bundle_max) 
        else:
            rule_bundle_regup = np.where(symm_diff_w_regup < config.TUNING_PARAMS.eta_1*bundle_thresh) 
            rule_bundle_regdown = np.where(symm_diff_w_regdown < config.TUNING_PARAMS.eta_1*bundle_thresh)
        rule_bundle_regup_motifs = rule_bundle_regup[0].tolist() # Keeping min loss rule
        rule_bundle_regup_regs = rule_bundle_regup[1].tolist()
        rule_bundle_regdown_motifs = rule_bundle_regdown[0].tolist()
        rule_bundle_regdown_regs = rule_bundle_regdown[1].tolist()
    # Print names of x1/x2 features that are bundled
    rule_bundle_motifs = x1.col_labels[rule_bundle_regup_motifs+rule_bundle_regdown_motifs]
    rule_bundle_regs = x2.row_labels[rule_bundle_regup_regs+rule_bundle_regdown_regs]
    print rule_bundle_motifs
    print rule_bundle_regs
    # Return list where first element is bundle where reg_up and second is where reg_down
    return BundleStore(rule_bundle_regup_motifs,rule_bundle_regup_regs, rule_bundle_regdown_motifs,rule_bundle_regdown_regs)

# Get rule score
def get_bundle_rule(tree, rule_bundle, theta, best_split, theta_alphas, bundle_train_rule_indices, bundle_test_rule_indices, holdout, w_pos, w_neg):
    # Training - add score to all places where rule applies
    bundle_train_pred = bundle_train_rule_indices[0]*0
    for ind in range(len(theta_alphas)):
        bundle_train_pred = bundle_train_pred+theta_alphas[ind]*bundle_train_rule_indices[ind]  
    # new index is where absolute value greater than theta
    new_train_rule_ind = (abs(bundle_train_pred)>theta)
    # Testing - add score to all places where rule applies
    bundle_test_pred = bundle_test_rule_indices[0]*0
    for ind in range(len(theta_alphas)):
        bundle_test_pred = bundle_test_pred+theta_alphas[ind]*bundle_test_rule_indices[ind]
    # new index is where absolute value greater than theta
    new_test_rule_ind = (abs(bundle_test_pred)>theta)
    # calculate W+ and W- for new rule
    weights_i = util.element_mult(tree.weights, tree.ind_pred_train[best_split])
    w_pos = util.element_mult(weights_i, holdout.ind_train_up)
    w_neg = util.element_mult(weights_i, holdout.ind_train_down)
    w_bundle_pos = util.element_mult(w_pos, new_train_rule_ind)
    w_bundle_neg = util.element_mult(w_neg, new_train_rule_ind) 
    # get score of new rule
    rule_bundle_score = 0.5*np.log((w_bundle_pos.sum()+config.TUNING_PARAMS.epsilon)/
        (w_bundle_neg.sum()+config.TUNING_PARAMS.epsilon))
    return rule_bundle_score, new_train_rule_ind, new_test_rule_ind

# Statistic to see whether stabilization applies
def stable_boost_test(tree, rule_train_index, holdout):
    w_pos = util.element_mult(tree.weights, holdout.ind_train_up)
    w_neg = util.element_mult(tree.weights, holdout.ind_train_down) 
    test = 0.5*abs(util.element_mult(w_pos, rule_train_index).sum()-util.element_mult(w_neg, rule_train_index).sum())
    return test
