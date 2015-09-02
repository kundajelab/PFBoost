import sys
import os
import random
import pdb
import copy

import numpy as np 
from scipy.sparse import *
import scipy.stats
import pandas as pd
import time
from matplotlib import pyplot as plt
import random
import gzip

from boosting_2D import util
from boosting_2D import config

import sklearn.utils
import pickle

import multiprocessing
import multiprocessing.queues
from grit.lib.multiprocessing_utils import fork_and_wait
import ctypes

log = util.log

### XXX DEPENDENCIES
### Currently uses TSS file on nandi, need to add as an argument

###  MARGIN SCORE FUNCTIONS
#######################################################################################
#######################################################################################

def calc_margin_score_x1_wrapper(args):
    return calc_margin_score_x1(*args)

# calc margin_score for x1 features
def calc_margin_score_x1(tree, y, x1, x2, index_mat, x1_feat_index, by_example=False):
    # if x1_feat_index==18:
    #     pdb.set_trace()
    # from IPython import embed; embed()
    x1_feat_name = x1.row_labels[x1_feat_index]
    # All rules where x1 is in split or bundled
    x1_feat_nodes = [el for el in xrange(tree.nsplit) 
         if x1_feat_index in [tree.split_x1[el]]
         +tree.bundle_x1[el]]

    # # Write out bundles separately
    # x1_bundles = ['|'.join(x1.row_labels[[el for el in [tree.split_x1[node]]
    #      +tree.bundle_x1[node] if el != x1_feat_index]]) for node in x1_feat_nodes]
    # x1_bundle_string = '--'.join([el if len(el)>0 else "none" for el in x1_bundles])
    # Write all unique terms bundled (separated by pipe) or None of no motifs bundled
    unique_bundle_terms=np.unique([x1.row_labels[el] for node in x1_feat_nodes
      for el in [tree.split_x1[node]]+tree.bundle_x1[node]
      if el != x1_feat_index]).tolist()
    x1_bundle_string = '|'.join(unique_bundle_terms) if len(
        unique_bundle_terms)>0 else "None"

    # All rules where the motif is above it, used in the split, or bundled in the split
    rules_w_x1_feat = [el for el in xrange(tree.nsplit)
         if x1_feat_index in tree.above_motifs[el]
         +[tree.split_x1[el]]
         +tree.bundle_x1[el]]

    # New predictions without rule (adjusted prediction = pred_adj)
    pred_adj = copy.deepcopy(tree.pred_train)
    for rule in rules_w_x1_feat:
        pred_adj -= tree.scores[rule]*tree.ind_pred_train[rule]
    if by_example==True:
        margin_score = util.element_mult(y.element_mult(tree.pred_train-pred_adj), index_mat)
        margin_score_array = margin_score.toarray().flatten() if y.sparse else margin_score.flatten()
        return margin_score_array
    else:
        margin_score = util.element_mult(y.element_mult(tree.pred_train-pred_adj), index_mat).sum()

    margin_score_norm = margin_score/index_mat.sum()

    # Get all rules where x1 feat is above or in rule or bundle
    if len(rules_w_x1_feat)>0: # can calculate rule fraction only when feature is used
        rule_index_mat = tree.ind_pred_train[rules_w_x1_feat[0]]
        for r in rules_w_x1_feat:
            rule_index_mat += tree.ind_pred_train[r]
        # Index where x1 feat is used 
        rule_index_mat = (rule_index_mat>0)
        # Index where x1 feat is used and examples of interest
        if type(index_mat) != type(rule_index_mat):
            pdb.set_trace()
        rule_index_joint = util.element_mult(index_mat, rule_index_mat)
        # Fraction of examples of interest where x1 feat used
        if index_mat.sum()==0:
            pdb.set_trace()
        rule_index_fraction = float(rule_index_joint.sum())/index_mat.sum()
    else:
        rule_index_fraction=0
    # print rule_index_fraction
    return [x1_feat_name, x1_bundle_string, margin_score, margin_score_norm, rule_index_fraction]

def calc_margin_score_x1_worker(tree, y, x1, x2, index_mat, by_example, (lock, index_cntr, matrix_or_list)):

    while True:
        # get the motif index to work on
        with index_cntr.get_lock():
            index = index_cntr.value
            index_cntr.value += 1
        
        # if this isn't a valid motif, then we are done
        if index >= len(x1.row_labels): 
            return

        # calculate the margin score for this feature
        if by_example==True:  
            b = calc_margin_score_x1(
                tree, y, x1, x2, index_mat,
                 x1_feat_index=index, by_example=True).reshape(
                 (1,  y.data.shape[0]*y.data.shape[1]))
            
            # add the margin score
            with lock:
                new_y = x1.num_row
                new_x = len(matrix_or_list)/new_y
                new_matrix = np.frombuffer(matrix_or_list.get_obj()).reshape(
                (new_x, new_y)) # reshapes view of data, not true data object
                new_matrix[np.where(b!=0),index]=b[b!=0]  
                print index
        elif by_example==False:
            b = calc_margin_score_x1(
                tree, y, x1, x2, index_mat,
                 x1_feat_index=index, by_example=False)
            print b
            with lock:
                matrix_or_list.put(b)
                print index
    return

def calc_margin_score_x2_wrapper(args):
    return calc_margin_score_x2(*args)

# calc_margin_score_x2(tree, y, x1, x2, csr_matrix(np.ones((y.num_row, y.num_col))), 'YDR085C')
def calc_margin_score_x2(tree, y, x1, x2, index_mat, x2_feat_index, by_example=False):
    x2_feat_name = x2.col_labels[x2_feat_index]
    # All rules where x1 is in split or bundled
    x2_feat_nodes = [el for el in xrange(tree.nsplit) 
         if x2_feat_index in [tree.split_x2[el]]
         +tree.bundle_x2[el]]

    # # Write out bundles separately
    # x2_bundles = ['|'.join(x2.col_labels[[el for el in [tree.split_x2[node]]
    #      +tree.bundle_x2[node] if el != x2_feat_index]]) for node in x2_feat_nodes]
    # x2_bundle_string = '--'.join([el if len(el)>0 else "none" for el in x2_bundles])
    # Write all unique terms bundled (separated by pipe) or None of no motifs bundled
    unique_bundle_terms=np.unique([x2.col_labels[el] for node in x2_feat_nodes for el in [tree.split_x2[node]]
          +tree.bundle_x2[node] if el != x2_feat_index]).tolist()
    x2_bundle_string = '|'.join(unique_bundle_terms) if len(unique_bundle_terms)>0 else "None"

    # All rules where the motif is above it, used in the split, or bundled in the split
    rules_w_x2_feat = [el for el in xrange(tree.nsplit)
         if x2_feat_index in tree.above_regs[el]
         +[tree.split_x2[el]]
         +tree.bundle_x2[el]]

    # New predictions without rule (adjusted prediction = pred_adj)
    pred_adj = tree.pred_train
    for rule in rules_w_x2_feat:
        pred_adj = pred_adj - tree.scores[rule]*tree.ind_pred_train[rule]

    if by_example==True:
        margin_score = util.element_mult(y.element_mult(tree.pred_train-pred_adj), index_mat)
        margin_score_array = margin_score.toarray().flatten() if y.sparse else margin_score.flatten()
        return margin_score_array
    else:
        margin_score = util.element_mult(y.element_mult(tree.pred_train-pred_adj), index_mat).sum()
    margin_score_norm = margin_score/index_mat.sum()

    # Get all rules where x1 feat is above or in rule or bundle
    if len(rules_w_x2_feat)>0:
        rule_index_mat = tree.ind_pred_train[rules_w_x2_feat[0]]
        for r in rules_w_x2_feat:
            rule_index_mat = rule_index_mat + tree.ind_pred_train[r]
        # Index where x1 feat is used 
        rule_index_mat = (rule_index_mat>0)
        # Index where x1 feat is used and examples of interest
        rule_index_joint = util.element_mult(index_mat, rule_index_mat)
        # Fraction of examples of interest where x1 feat used
        if index_mat.sum()==0:
            pdb.set_trace()
        rule_index_fraction = float(rule_index_joint.sum())/index_mat.sum()
    else:
        rule_index_fraction=0
    # print rule_index_fraction
    return [x2_feat_name, x2_bundle_string, margin_score, margin_score_norm, rule_index_fraction]

def calc_margin_score_rule_wrapper(args):
    return calc_margin_score_rule(*args)

### Multiprocessing worker
def calc_margin_score_x2_worker(tree, y, x1, x2, index_mat, by_example, (lock, index_cntr, matrix_or_list)):

    while True:
        # get the motif index to work on
        with index_cntr.get_lock():
            index = index_cntr.value
            index_cntr.value += 1
        
        # if this isn't a valid motif, then we are done
        if index >= len(x2.col_labels): 
            return

        # calculate the margin score for this node  
        if by_example==True:  
            b = calc_margin_score_x2(
                tree, y, x1, x2, index_mat,
                 x2_feat_index=index, by_example=True).reshape(
                 (1,  y.data.shape[0]*y.data.shape[1]))
        
            # add the margin score
            with lock:
                new_y = x2.num_col
                new_x = len(matrix_or_list)/new_y
                new_matrix = np.frombuffer(matrix_or_list.get_obj()).reshape(
                (new_x, new_y)) # reshapes view of data, not true data object
                new_matrix[np.where(b!=0),index]=b[b!=0]  
                print index
        elif by_example==False:
            b = calc_margin_score_x2(
                tree, y, x1, x2, index_mat,
                 x2_feat_index=index, by_example=False)
            with lock:
                matrix_or_list.put(b)
                print index
    return

### CALCULATE MARGIN SCORE BY ANY JOINT APPEARANCE OF MOTIF-REGULATOR (MULTIPLE NODES)
def calc_margin_score_rule(tree, y, x1, x2, index_mat, x1_feat_index, x2_feat_index, by_example=False):
    # Feature names
    x1_feat_name = x1.row_labels[x1_feat_index]
    x2_feat_name = x2.col_labels[x2_feat_index]

    # All the nodes that contain x1 and x2 in bundle
    pair_rules = [el for el in xrange(tree.nsplit) 
         if x1_feat_index in [tree.split_x1[el]]
         +tree.bundle_x1[el] and x2_feat_index in 
         [tree.split_x2[el]]+tree.bundle_x2[el]]

    ### WRITE out bundles
    x1_unique_bundle_terms=np.unique([x1.row_labels[el] for node in pair_rules for el in [tree.split_x1[node]]
          +tree.bundle_x1[node] if el != x1_feat_index]).tolist()
    x1_bundle_string = '|'.join(x1_unique_bundle_terms) if len(x1_unique_bundle_terms)>0 else "None"
    x2_unique_bundle_terms=np.unique([x2.col_labels[el] for node in pair_rules for el in [tree.split_x2[node]]
          +tree.bundle_x2[node] if el != x2_feat_index]).tolist()
    x2_bundle_string = '|'.join(x2_unique_bundle_terms) if len(x2_unique_bundle_terms)>0 else "None"

    # Allocate Prediction Matrix
    pred_adj = tree.pred_train
    for rule in pair_rules:
        pred_adj = pred_adj - tree.scores[rule]*tree.ind_pred_train[rule]

    if by_example==True:
        margin_score = util.element_mult(y.element_mult(tree.pred_train-pred_adj), index_mat)
        margin_score_array = margin_score.toarray().flatten() if y.sparse else margin_score.flatten()
        return margin_score_array
    else:
        margin_score = util.element_mult(y.element_mult(tree.pred_train-pred_adj), index_mat).sum()
    margin_score_norm = margin_score/index_mat.sum()

    ### If considering specific rule only (rule with both motif and reg, but any node that is it as first split or bundle)
    rules_w_x1_feat_and_x2_feat = list(
        set([el for el in xrange(tree.nsplit)
         if x2_feat_index in tree.above_regs[el]+
         [tree.split_x2[el]]
         +tree.bundle_x2[el]])
        &
        set([el for el in xrange(tree.nsplit)
         if x1_feat_index in tree.above_motifs[el]+
         [tree.split_x1[el]]
         +tree.bundle_x1[el]]))
    # Add up where every rule is used
    if len(rules_w_x1_feat_and_x2_feat)>0:
        rule_index_mat = tree.ind_pred_train[rules_w_x1_feat_and_x2_feat[0]]
        for r in rules_w_x1_feat_and_x2_feat:
            rule_index_mat = rule_index_mat + tree.ind_pred_train[r]
        # Index where rule  is used 
        rule_index_mat = (rule_index_mat>0)
        # Index where rules is used and examples of interest
        rule_index_joint = util.element_mult(index_mat, rule_index_mat)
        # Fraction of examples of interest where rule used
        rule_index_fraction = float(rule_index_joint.sum())/index_mat.sum()
    else:
        rule_index_fraction=0
    # print rule_index_fraction
    return [x1_feat_name, x1_bundle_string, x2_feat_name, x2_bundle_string, margin_score, margin_score_norm, rule_index_fraction]


def calc_margin_score_node_wrapper(args):
    return calc_margin_score_node(*args)

## Calculate margin score for each individual node
def calc_margin_score_node(tree, y, x1, x2, index_mat, node, by_example=True):
    # Feature names
    x1_feat_index = tree.split_x1[node]
    x2_feat_index = tree.split_x2[node]
    x1_feat_name = x1.row_labels[x1_feat_index]
    x2_feat_name = x2.col_labels[x2_feat_index]
    # stringify the bundle
    x1_bundle_string = '|'.join([x1.row_labels[el] for el in tree.bundle_x1[node]])
    x2_bundle_string = '|'.join([x2.col_labels[el] for el in tree.bundle_x2[node]])

    # Prediction of more or less accessible
    direction = np.sign(tree.scores[node])

    # All rules where the chosen node is above it or the node itself
    subtract_rules = [el for el in xrange(tree.nsplit)
         if node in tree.above_nodes[el] if el!=node]+[node]

    # Get Prediction Matrix
    pred_adj = tree.pred_train
    for rule in subtract_rules:
        pred_adj = pred_adj - tree.scores[rule]*tree.ind_pred_train[rule]

    if by_example==True:
        margin_score = util.element_mult(y.element_mult(tree.pred_train-pred_adj), index_mat)
        margin_score_array = margin_score.toarray().flatten() if y.sparse else margin_score.flatten()
        return margin_score_array
    else:
        margin_score = util.element_mult(y.element_mult(tree.pred_train-pred_adj), index_mat).sum()
    margin_score_norm = margin_score/index_mat.sum()

    ### Chosen node and all nodes added below 
    rules_w_node = [el for el in xrange(tree.nsplit)
         if node in tree.above_nodes[el] or el==node]
    if len(rules_w_node)>0:
        rule_index_mat = tree.ind_pred_train[rules_w_node[0]]
        for r in rules_w_node:
            rule_index_mat = rule_index_mat + tree.ind_pred_train[r]

        # Index where rule  is used 
        rule_index_mat = (rule_index_mat>0)
        # Index where rules is used and examples of interest
        rule_index_joint = util.element_mult(index_mat, rule_index_mat)
        # Fraction of examples of interest where rule used
        rule_index_fraction = float(rule_index_joint.sum())/index_mat.sum()
    else:
        rule_index_fraction=0

    # print rule_index_fraction
    return [node, x1_feat_name, x1_bundle_string, x2_feat_name, x2_bundle_string, margin_score, margin_score_norm, rule_index_fraction, direction]


### Multiprocessing worker
def calc_margin_score_node_worker(tree, y, x1, x2, index_mat, by_example, (lock, index_cntr, matrix_or_list)):

    while True:
        # get the motif index to work on
        with index_cntr.get_lock():
            index = index_cntr.value
            index_cntr.value += 1
        # if this isn't a valid motif, then we are done
        if index >= len(tree.nsplit): 
            return

        # calculate the margin score for this node  
        if by_example==True:  
            b = calc_margin_score_node(
                tree, y, x1, x2, index_mat,
                 node=index, by_example=True).reshape(
                 (1,  y.data.shape[0]*y.data.shape[1]))
            
            # add the margin score
            with lock:
                new_y = tree.nsplit
                new_x = len(matrix_or_list)/new_y
                new_matrix = np.frombuffer(matrix_or_list.get_obj()).reshape(
                (new_x, new_y)) # reshapes view of data, not true data object
                new_matrix[np.where(b!=0),index]=b[b!=0]  
                print index

        elif by_example==False:
            b = calc_margin_score_node(
                tree, y, x1, x2, index_mat,
                 node=index, by_example=False)
            with lock:
                matrix_or_list.put(b)
                print index
    return

## Calculate margin score for a given path
def calc_margin_score_path(tree, y, x1, x2, index_mat, node, by_example=False):
    # Prediction of more or less accessible at the end of the path
    direction = np.sign(tree.scores[node])

    # All rules where the node is not above it or the node itself except root
    nodes_in_path = [el for el in [node]+tree.above_nodes[node] if el !=0]

    # Index of examples going to end of path
    path_index = tree.ind_pred_train[node]

    # Get Prediction Matrix (remove score of all nodes in path for just the index of rules that get to end of path)
    pred_adj = tree.pred_train
    for rule in nodes_in_path:
        pred_adj = pred_adj - tree.scores[rule]*path_index
 
    if by_example==True:
        margin_score = util.element_mult(y.element_mult(tree.pred_train-pred_adj), index_mat)
        margin_score_array = margin_score.toarray().flatten() if y.sparse else margin_score.flatten()
        return margin_score_array
    else:
        margin_score = util.element_mult(y.element_mult(tree.pred_train-pred_adj), index_mat).sum()
    margin_score_norm = margin_score/index_mat.sum()

    ### Chosen node and all nodes above in path 
    rule_index_mat = tree.ind_pred_train[nodes_in_path[0]]
    for n in nodes_in_path:
        rule_index_mat = rule_index_mat + tree.ind_pred_train[n]

    # Index where rule  is used 
    rule_index_mat = (rule_index_mat>0)
    # Index where rules is used and examples of interest
    rule_index_joint = util.element_mult(index_mat, rule_index_mat)
    # Fraction of examples of interest where rule used
    rule_index_fraction = float(rule_index_joint.sum())/index_mat.sum()

    # Label path string (can do better)
    # path_string='path_{0}'.format(node)
    path_string='{0}_{1}'.format(node,
        '|'.join([','.join([x1.row_labels[tree.split_x1[n]], x2.col_labels[tree.split_x2[n]]]) for n in nodes_in_path]))

    # print rule_index_fraction
    return [node, path_string, len(nodes_in_path), margin_score, margin_score_norm, rule_index_fraction, direction]

### Multiprocessing worker
def calc_margin_score_path_worker(tree, y, x1, x2, index_mat, by_example, (lock, index_cntr, matrix_or_list)):

    while True:
        # get the motif index to work on
        with index_cntr.get_lock():
            index = index_cntr.value
            index_cntr.value += 1
        # if this isn't a valid motif, then we are done
        if index >= len(tree.nsplit): 
            return

        # calculate the margin score for this node  
        if by_example==True:  
            b = calc_margin_score_path(
                tree, y, x1, x2, index_mat,
                 node=index, by_example=True).reshape(
                 (1,  y.data.shape[0]*y.data.shape[1]))
        
            # add the margin score
            with lock:
                new_y = tree.nsplit
                new_x = len(matrix_or_list)/new_y
                new_matrix = np.frombuffer(matrix_or_list.get_obj()).reshape(
                (new_x, new_y)) # reshapes view of data, not true data object
                new_matrix[np.where(b!=0),index]=b[b!=0]  
                print index

        elif by_example==False:
            b = calc_margin_score_path(
                tree, y, x1, x2, index_mat,
                 node=index, by_example=True)
            print b
            with lock:
                matrix_or_list.put(b)
                print index
    return


def rank_by_margin_score(tree, y, x1, x2, index_mat, pool, method):
    assert method in ('x1', 'x2', 'x1_and_x2', 'node', 'path')
    # Rank x1 features only
    if method=='x1':
        print 'computing margin score for x1'
        # All x1 features used in a tree, not equal to root
        used_x1_feats = np.unique([el for el in tree.split_x1 if el != 'root']+ \
            [el for listy in tree.bundle_x1 for el in listy if el != 'root']).tolist()
        ### SERIAL VERSION
        # rule_processes = []    
        # for feat in used_x1_feats:
        #     result=calc_margin_score_x1(tree, y, x1, x2, index_mat, feat)
        #     rule_processes.append(result)
        ### PARALLEL version
        lock = multiprocessing.Lock()
        rule_processes_mp = multiprocessing.queues.SimpleQueue()
        index_cntr = multiprocessing.Value('i', 0)
        args = [tree, y, x1, x2, index_mat, False, (lock, index_cntr, rule_processes_mp)]
        fork_and_wait(config.NCPU, calc_margin_score_x1_worker, args)
        rule_processes = []
        while rule_processes_mp.empty()==False: 
            rule_processes.append(rule_processes_mp.get())
        # Report data frame with feature 
        ranked_score_df = pd.DataFrame({'x1_feat':[el[0] for el in rule_processes], \
            'x1_feat_bundles':[el[1] for el in rule_processes], \
            'margin_score':[el[2] for el in rule_processes], \
            'margin_score_norm':[el[3] for el in rule_processes], \
            'rule_index_fraction':[el[4] for el in rule_processes]}).sort(
            columns=['margin_score'], ascending=False)
        pdb.set_trace()


    # Rank x2 features only
    if method=='x2':
        print 'computing margin score for x2'
        # All x2 features used in a treem, not equal to root
        used_x2_feats = np.unique([el for el in tree.split_x2 if el != 'root']+ \
                [el for listy in tree.bundle_x2 for el in listy if el != 'root']).tolist()
        ### SERIAL VERSION
        # rule_processes = []    
        # for feat in used_x2_feats:
        #     result=calc_margin_score_x2(tree, y, x1, x2, index_mat, feat)
        #     rule_processes.append(result)
        # PARALLEL VERSION
        lock = multiprocessing.Lock()
        rule_processes_mp = multiprocessing.queues.SimpleQueue()
        index_cntr = multiprocessing.Value('i', 0)
        args = [tree, y, x1, x2, index_mat, False, (lock, index_cntr, rule_processes_mp)]
        fork_and_wait(config.NCPU, calc_margin_score_x2_worker, args)
        rule_processes = []
        while rule_processes_mp.empty()==False: 
            rule_processes.append(rule_processes_mp.get())
        # Report data frame with feature 
        ranked_score_df = pd.DataFrame({'x2_feat':[el[0] for el in rule_processes], \
            'x2_feat_bundles':[el[1] for el in rule_processes], \
            'margin_score':[el[2] for el in rule_processes], \
            'margin_score_norm':[el[3] for el in rule_processes], \
            'rule_index_fraction':[el[4] for el in rule_processes]}).sort(
            columns=['margin_score'], ascending=False)
    # Rank by rules 
    if method=='x1_and_x2':
        print 'computing margin score for x1_and_x2 jointly'
        # unlike previous take non-unique sets
        used_x1_feats = [el for el in tree.split_x1 if el != 'root']+ \
                [el for listy in tree.bundle_x1 for el in listy if el != 'root']
        used_x2_feats = [el for el in tree.split_x2 if el != 'root']+ \
                [el for listy in tree.bundle_x2 for el in listy if el != 'root']
        # Serial version
        uniq_x1_x2_pairs = {d[1:]:d for d in zip(used_x1_feats, used_x2_feats)}
        rule_processes = []    
        for (x1_feat,x2_feat) in uniq_x1_x2_pairs.values():
            result=calc_margin_score_rule(tree, y, x1, x2, index_mat, x1_feat, x2_feat)
            rule_processes.append(result)
        # Report data frame with feature 
        ranked_score_df = pd.DataFrame({'x1_feat':[el[0] for el in rule_processes], \
            'x1_feat_bundles':[el[1] for el in rule_processes], \
            'x2_feat':[el[2] for el in rule_processes], \
            'x2_feat_bundles':[el[3] for el in rule_processes], \
            'margin_score':[el[4] for el in rule_processes], \
            'margin_score_norm':[el[5] for el in rule_processes], \
            'rule_index_fraction':[el[6] for el in rule_processes]}).sort(
            columns=['margin_score'], ascending=False)
    if method=='node':
        print 'computing margin score for node'
        # SERIAL VERSION
        # rule_processes = []   ### Make dictionary with keys equal to dataframe
        # for node in xrange(1,tree.nsplit):
        #     result=calc_margin_score_node(tree, y, x1, x2, index_mat, node)
        #     rule_processes.append(result)
        # PARALLEL VERSION
        lock = multiprocessing.Lock()
        rule_processes_mp = multiprocessing.queues.SimpleQueue()
        index_cntr = multiprocessing.Value('i', 0)
        args = [tree, y, x1, x2, index_mat, False, (lock, index_cntr, rule_processes_mp)]
        fork_and_wait(config.NCPU, calc_margin_score_node_worker, args)
        rule_processes = []
        while rule_processes_mp.empty()==False: 
            rule_processes.append(rule_processes_mp.get())
        # Report data frame with feature 
        ranked_score_df = pd.DataFrame({'node':[el[0] for el in rule_processes], \
            'x1_feat':[el[1] for el in rule_processes], \
            'x1_feat_bundles':[el[2] for el in rule_processes], \
            'x2_feat':[el[3] for el in rule_processes], \
            'x2_feat_bundles':[el[4] for el in rule_processes], \
            'margin_score':[el[5] for el in rule_processes], \
            'margin_score_norm':[el[6] for el in rule_processes], \
            'rule_index_fraction':[el[7] for el in rule_processes], \
            'direction':[el[8] for el in rule_processes]}).sort(
            columns=['margin_score'], ascending=False)
    if method=='path':
        print 'computing margin score for path'
        ### SERIAL VERSION
        # rule_processes = []    
        # for node in xrange(1,tree.nsplit):
        #     result=calc_margin_score_path(tree, y, x1, x2, index_mat, node)
        #     rule_processes.append(result)
        ### PARALLEL VERSION
        lock = multiprocessing.Lock()
        rule_processes_mp = multiprocessing.queues.SimpleQueue()
        index_cntr = multiprocessing.Value('i', 0)
        args = [tree, y, x1, x2, index_mat, False, (lock, index_cntr, rule_processes_mp)]
        fork_and_wait(config.NCPU, calc_margin_score_path_worker, args)
        rule_processes = []
        while rule_processes_mp.empty()==False: 
            rule_processes.append(rule_processes_mp.get())
        # Report data frame with feature 
        ranked_score_df = pd.DataFrame({'node':[el[0] for el in rule_processes], \
            'path_name':[el[1] for el in rule_processes], \
            'path_length':[el[2] for el in rule_processes], \
            'margin_score':[el[3] for el in rule_processes], \
            'margin_score_norm':[el[4] for el in rule_processes], \
            'rule_index_fraction':[el[5] for el in rule_processes], \
            'direction':[el[6] for el in rule_processes]}).sort(
            columns=['margin_score'], ascending=False)
    ranked_score_df.drop_duplicates()
    # Return matrix
    return ranked_score_df

    return [node, path_string, len(nodes_in_path), margin_score, margin_score_norm, rule_index_fraction, direction]


###  FUNCTIONS TO GET INDEX
#######################################################################################
#######################################################################################

### Get the index with a text file or document of of interest
### Takes either index or the names of the features y.row_labels or y.col_labels
def get_index(y, x1, x2, tree, condition_feat_file=None, region_feat_file=None):
    if region_feat_file!=None:
        x1_file = pd.read_table(region_feat_file, header=None)
        # if providing index numbers
        if x1_file.applymap(lambda x: isinstance(x, (int, float))).sum().tolist()[0]==x1_file.shape[0]:
            # ASSUMING INPUT IS 1 BASED
            x1_index = [el-1 for el in x1_file.ix[:,0].tolist()]
        # if providing labels
        else:
            x1_index = [el for el in xrange(y.data.shape[0]) if y.row_labels[el] in x1_file.ix[:,0].tolist()]
    # else allow all
    else:
        x1_index = range(x1.num_col)
    if condition_feat_file!=None:
        x2_file = pd.read_table(condition_feat_file, header=None)
        # if providing index numbers
        if x2_file.applymap(lambda x: isinstance(x, (int, float))).sum().tolist()[0]==x2_file.shape[0]: # this is terrible fix
            # ASSUMING INPUT IS 1 BASED
            x2_index = [el-1 for el in x2_file.ix[:,0].tolist()]
        # if providing labels
        else:
            x2_index = [el for el in xrange(y.data.shape[1]) if y.col_labels[el] in x2_file.ix[:,0].tolist()]
    else:
        x2_index = range(x2.num_row)
    index_mat = np.zeros((y.num_row, y.num_col), dtype=bool)
    index_mat[np.ix_(x1_index, x2_index)]=1
    if y.sparse:
        index_mat = csr_matrix(index_mat)
    return index_mat

### Function takes the index matrix and splits it into promoter and enhancer matrices
### Based on coordinates in y.row_labels
def split_index_mat_prom_enh(index_mat, y, tss_file):
    # Threshold for distance to TSSS
    prom_thresh=100
    # Convert row labels to bed file and write to temporary file
    temp_bed_file = os.path.splitext(y.data_file)[0]+'_row_coords_TEMP.txt'
    temp_intersect_file = os.path.splitext(y.data_file)[0]+'_row_coords_tss_intersect_TEMP.txt'
    row_coords=[el.split(';')[0].replace(':', '\t').replace('-','\t') for el in y.row_labels]
    f = open(temp_bed_file, 'w')
    for coord in row_coords:
        f.write("%s\n"%coord)
    f.close()
    # Get the distances of the peaks from the TSS 
    command="zcat %s |  awk '$14!=\".\"' | awk -v OFS=\"\t\" '{print $1, $4, $5, $14}' | \
         bedtools closest -a %s -b - -d | bedtools groupby -g 1,2,3,4,5,6,8 -c 7 -o collapse | \
         awk -v OFS='\t' '{print $1,$2,$3,$4,$5,$6,$8,$7}' \
          > %s"%(tss_file, temp_bed_file, temp_intersect_file)
    os.system(command)
    result = pd.read_table(temp_intersect_file, sep='\t',header=None)
    # Get promoter regions less the threshold and enhancer greater than equal to threshold
    prom_index = pd.DataFrame(result.ix[:,7]<prom_thresh)
    if y.sparse:
        prom_mat = csr_matrix(np.hstack([prom_index for i in range(index_mat.shape[1])]))
    else:
        prom_mat = np.hstack([prom_index for i in range(index_mat.shape[1])])
    enh_index = pd.DataFrame(result.ix[:,7]>=prom_thresh)
    if y.sparse:
        enh_mat = csr_matrix(np.hstack([enh_index for i in range(index_mat.shape[1])]))
    else:
        enh_mat = np.hstack([enh_index for i in range(index_mat.shape[1])])
    index_mat_prom = util.element_mult(index_mat, prom_mat)
    index_mat_enh = util.element_mult(index_mat, enh_mat)
    # Remove the coordinate and intersection files
    os.system('rm {0}'.format(temp_bed_file))
    os.system('rm {0}'.format(temp_intersect_file))
    # Reurn index mats for promoter and enhancer regions
    return (index_mat_prom, index_mat_enh)

###  FUNCTIONS TO CALL MARGIN SCORE
#######################################################################################
#######################################################################################

### Call rank by margin score
def call_rank_by_margin_score(index_mat, key, method, prefix, y, x1, x2, tree, pool, num_perm=100, null_tree_file=None):
    # Make margin score directory in output directory
    margin_outdir = '{0}{1}/margin_scores/'.format(config.OUTPUT_PATH, config.OUTPUT_PREFIX)
    if not os.path.exists(margin_outdir):
        os.makedirs(margin_outdir)
    # Assign y-value
    y_value = +1 if 'up' in key else -1
    # Calculate real margin score
    rank_score_df = rank_by_margin_score(tree, y, x1, x2, index_mat, pool, method=method)
    # If getting p-value, compute permutations
    if num_perm>0:
        if null_tree_file==None:
        # Compute p-value by shuffling y values
            rank_score_df = calculate_null_margin_score_dist_by_shuffling_target(rank_score_df, 
                index_mat, method, pool, num_perm, tree, y, x1, x2, y_value)
        # Compute p-value based on separate NULL model
        else:
            rank_score_df = calculate_null_margin_score_dist_from_NULL_tree(rank_score_df, 
                index_mat, method, pool, num_perm, tree, y, x1, x2, y_value, null_tree_file)                        
    # Write margin score to output_file 
    rank_score_df.to_csv('{0}{1}_{2}_{3}_margin_score.txt'.format(margin_outdir, prefix, method, key), 
            sep="\t", index=None, header=True)
    return 0



### MARGIN SCORE NULL MODEL 
#######################################################################################
#######################################################################################
### Sample entries with same value within the same column or row or matrix (maybe useful for specificity score later)
def sample_values_from_axis(y, index_mat, method, value):
    new_mat = index_mat*False
    if y.sparse:
        ymat = y.data.toarray()
        indmat = index_mat.toarray()
    else:
        ymat = y.data
        indmat = index_mat
    # If want to permute rows (motifs), iterate over columns
    if method=='by_x1':
        for i in xrange(indmat.shape[1]):
            new_ind = np.random.choice(np.where(ymat[:,i]==value)[0], np.sum(indmat[:,i]))
            new_mat[new_ind,i] = True
    # If you want to permute columns (regulators), iterate over rows
    if method=='by_x2':
        for i in xrange(indmat.shape[0]):
            new_ind = np.random.choice(np.where(ymat[i,:]==value)[0], np.sum(indmat[i,:]))
            new_mat[new_ind,i] = True
    # If you want to permute columns and rows (nodes), iterate first over rows then nodes
    if method=='by_node' or method=='by_x1_and_x2' or method=='path':
        value_vec = np.where(ymat==value)
        sample_ind = np.random.choice(range(len(value_vec[0])), np.sum(indmat))
        new_mat[value_vec[0][sample_ind], value_vec[1][sample_ind]]=True        
    return new_mat

### Sample same value from anywhere in matrix
def sample_values_from_data_class(y, index_mat, method, value):
    if y.sparse:
        new_mat = csr_matrix(index_mat.shape, dtype=bool)
        ymat = y.data.toarray()
        indmat = index_mat.toarray()
    else:
        new_mat = np.zeros(index_mat.shape, dtype=bool)
        ymat = y.data
        indmat = index_mat
    # Sample same value from anywhere
    value_vec = np.where(ymat==value)
    # For a given y-value, want
    sample_ind = np.random.choice(range(len(value_vec[0])), np.sum(indmat), replace=False)
    new_mat[value_vec[0][sample_ind], value_vec[1][sample_ind]]=True        
    return new_mat

### Calculate permutations of the index matrix and re-calculate margin scores
def calculate_null_margin_score_dist_by_shuffling_target(rank_score_df, index_mat, method, pool, num_perm, tree, y, x1, x2, y_value):
    # Initialize dictionary of margin scores
    dict_names = ['perm{0}'.format(el) for el in xrange(num_perm)]
    margin_score_dict = {}
    for name in dict_names: margin_score_dict[name]=0
    # Set seed for permutations
    random.seed(1)
    # Shuffle Y matrix 
    y_null = util.shuffle_data_object(obj=y)
    # Given the index matrix, randomly sample the same number of + or - examples
    if method=='x1':
        # For each permutation calculate margin scores and add to dictionary
        for i in xrange(num_perm):
            # Permute rows
            new_index=sample_values_from_data_class(y=y_null, index_mat=index_mat, method=method, value=y_value)
            margin_score_dict['perm{0}'.format(i)]=rank_by_margin_score(tree, y_null, x1, x2, new_index, pool, method=method)
    elif method=='x2':
        # For each permutation calculate margin scores and add to dictionary
        for i in xrange(num_perm):
            # Permute columns
            new_index=sample_values_from_data_class(y=y_null, index_mat=index_mat, method=method, value=y_value)
            margin_score_dict['perm{0}'.format(i)]=rank_by_margin_score(tree, y_null, x1, x2, new_index, pool, method=method)
    elif method=='node' or method=='x1_and_x2' or method=='path':
        # For each permutation calculate margin scores and add to dictionary
        for i in xrange(num_perm):
            # Permute rows and columns
            new_index=sample_values_from_data_class(y=y_null, index_mat=index_mat, method=method, value=y_value)
            margin_score_dict['perm{0}'.format(i)]=rank_by_margin_score(tree, y_null, x1, x2, new_index, pool, method=method)
    else:
        assert False, "provide method in ['x1', 'x2', 'x1_and_x2', 'node', 'path']"
    # calculate p-values for each margin score 
    all_margin_scores_ranked = np.sort([el for df in margin_score_dict.values() for el in df.ix[:,'margin_score'].tolist()])
    print all_margin_scores_ranked
    pvalues = [float(sum(all_margin_scores_ranked>el))/len(all_margin_scores_ranked) for el in rank_score_df.ix[:,'margin_score'].tolist()]
    rank_score_df['pvalue']=pvalues
    # qvalues = stats.p_adjust(FloatVector(pvalues), method = 'BH')
    return rank_score_df


def calculate_null_margin_score_dist_from_NULL_tree(rank_score_df, index_mat, method,
 pool, num_perm, tree, y, x1, x2, y_value, null_tree_file):
    # Read in NULL tree
    tree_null = pickle.load(gzip.open(null_tree_file, 'rb'))
    # If the two formats are not compatible
    if type(tree_null.pred_train)!=type(y.data):
        print 'NULL model and true model are incompatible types. Use models of same sparse/dense type.'
        return 1
    # tree_null.pred_train = util.convert_type_to_match(tree_null.pred_train, y.data)
    # Initialize dictionary of margin scores
    dict_names = ['perm{0}'.format(el) for el in xrange(num_perm)]
    margin_score_dict = {}
    for name in dict_names: margin_score_dict[name]=0
    ### Re-create y from permutations 
    y_null = util.shuffle_data_object(obj=y)
    # Given the index matrix, randomly sample the same number of + or - examples
    if method=='x1':
        # For each permutation calculate margin scores and add to dictionary
        for i in xrange(num_perm):
            # Permute rows
            new_index=sample_values_from_data_class(y=y_null, index_mat=index_mat,
             method=method, value=y_value)
            margin_score_dict['perm{0}'.format(i)]=rank_by_margin_score(tree_null,
             y_null, x1, x2, new_index, pool, method=method)
    elif method=='x2':
        # For each permutation calculate margin scores and add to dictionary
        for i in xrange(num_perm):
            # Permute columns
            new_index=sample_values_from_data_class(y=y_null, index_mat=index_mat,
             method=method, value=y_value)
            margin_score_dict['perm{0}'.format(i)]=rank_by_margin_score(tree_null,
             y_null, x1, x2, new_index, pool, method=method)
    elif method=='node' or method=='x1_and_x2':
        # For each permutation calculate margin scores and add to dictionary
        for i in xrange(num_perm):
            # Permute rows and columns
            new_index=sample_values_from_data_class(y=y_null, index_mat=index_mat,
             method=method, value=y_value)
            margin_score_dict['perm{0}'.format(i)]=rank_by_margin_score(tree_null,
             y_null, x1, x2, new_index, pool, method=method)
    else:
        assert False, "provide method in ['x1', 'x2', 'x1_and_x2', 'node', 'path']"
    # calculate p-values for each margin score 
    all_margin_scores_ranked = np.sort([el for df in margin_score_dict.values()
     for el in df.ix[:,'margin_score'].tolist()])
    print all_margin_scores_ranked
    pvalues = [float(sum(all_margin_scores_ranked>el))/len(all_margin_scores_ranked)
     for el in rank_score_df.ix[:,'margin_score'].tolist()]
    rank_score_df['pvalue']=pvalues
    # qvalues = stats.p_adjust(FloatVector(pvalues), method = 'BH')
    return rank_score_df




### Plot the normalized margin scores over all the conditions
###########################################################################################################################

### element direction in ["ENH_UP", "ENH_DOWN", "PROM_UP", "PROM_DOWN"]
### Currently just works with x1_feat and x2_feat
def plot_norm_margin_score_across_conditions(conditions, method, plot_label, num_feat, element_direction):
    # Read in all the files
    result_path='/srv/persistent/pgreens/projects/boosting/results/margin_scores/'
    result_dfs = {}
    for condition in conditions:
        print condition
        result_dfs[condition] = pd.read_table('{0}{1}/{1}_{2}_top_{3}.txt'.format(result_path, condition, element_direction, method))
        result_dfs[condition]['condition']=[condition]*result_dfs[condition].shape[0]
    result_df = pd.concat(result_dfs.values()).sort(columns=['margin_score_norm'], ascending=False)
    result_df.index=range(result_df.shape[0])
    # Get top 10 regulators
    top_reg = []
    index = 0
    while len(top_reg)<num_feat:
        ### XXX METHOD must match column label ['x1_feat, x2_feat']
        while result_df.ix[index,method] in top_reg:
            index+=1
        top_reg = top_reg+[result_df.ix[index,method]]
    # Allocate plot matrix
    plot_df = pd.DataFrame(index=conditions, columns=top_reg)
    for condition in plot_df.index:
        plot_df.ix[condition,:]=[result_dfs[condition].ix[result_dfs[condition].ix[:,method]==reg,'margin_score_norm'].tolist()[0] for reg in top_reg]
    plt.figure(figsize=(12, 9))
    plt.pcolor(plot_df)
    plt.yticks(np.arange(0.5, len(plot_df.index), 1), plot_df.index)
    plt.xticks(np.arange(0.5, len(plot_df.columns), 1), plot_df.columns, rotation=90)
    # plt.jet()
    plt.colorbar()
    plt.show()
    plt.xlabel('Feature')
    plt.ylabel('Experiment')
    plt.title('Normalized Margin Scores Across Conditions {0} \n {1}'.format(element_direction, plot_label))
    plt.legend(loc=1)
    # plt.savefig('/users/pgreens/temp_heatmap.png', bbox_inches='tight')
    plt.savefig('{0}/plots/margin_score_plots/{1}_top{2}_{3}_{4}.png'.format(config.OUTPUT_PATH, plot_label, num_feat, method, element_direction), bbox_inches='tight')


### Find discriminative motifs and enhancers between conditions
###############################################################

method='x1_feat'
label = 'hema_MPP_v_LMPP_against_MPP_v_CMP_1000iter_TFbindingonly_ENH_DOWN'
conditions = ['/srv/persistent/pgreens/projects/boosting/results/margin_scores/hema_MPP_v_LMPP_1000iter_TFbindingonly/hema_MPP_v_LMPP_1000iter_TFbindingonly_ENH_DOWN_top_x1_feat.txt',
 '/srv/persistent/pgreens/projects/boosting/results/margin_scores/hema_MPP_v_CMP_1000iter_TFbindingonly/hema_MPP_v_CMP_1000iter_TFbindingonly_ENH_DOWN_top_x1_feat.txt']
def find_discrimative_features(conditions, method, out_file):
    ### For each feature calculate the difference in normalized margin score
    df1 = pd.read_table(conditions[0], header=0)
    df2 = pd.read_table(conditions[1], header=0)
    joint_feat = list(set(df1[method]) & set(df2[method]))
    discrim_df = pd.DataFrame(index=range(len(joint_feat)), columns=['norm_margin_score_diff', method])
    for i in discrim_df.index.tolist():
        diff = df1.ix[df1[method]==joint_feat[i],'margin_score_norm'].tolist()[0]-df2.ix[df2[method]==joint_feat[i],'margin_score_norm'].tolist()[0]
        discrim_df.ix[i,:]=[diff, joint_feat[i]]
    discrim_df['sort'] = discrim_df['norm_margin_score_diff'].abs()
    discrim_df = discrim_df.sort(columns='sort', ascending=False).drop('sort', axis=1)
    discrim_df.to_csv(out_file, header=True, index=False, sep="\t")







