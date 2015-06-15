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
from boosting_2D.plot import *
from boosting_2D import margin_score
from boosting_2D.data_class import *
from boosting_2D.find_rule import *
from boosting_2D import stabilize

log = util.log

TuningParams = namedtuple('TuningParams', [
    'num_iter',
    'use_stumps', 'use_stable', 'use_corrected_loss',
    'eta_1', 'eta_2', 'bundle_max', 'epsilon'
])

def parse_args():
    # Get arguments
    parser = argparse.ArgumentParser(description='Extract Chromatin States')

    parser.add_argument('--output-prefix', 
                        help='Analysis name for output plots')
    parser.add_argument('--output-path', 
                        help='path to write the results to', 
                        default='/users/pgreens/projects/boosting/results/')

    parser.add_argument('--input-format', help='options are: matrix, triplet')
    parser.add_argument('--mult-format', help='options are: matrix, dense')

    parser.add_argument('-y', '--target-file', 
                        help='target matrix - dimensionality GxE')
    parser.add_argument('-g', '--target-row-labels', 
                        help='row labels for y matrix (dimension G)')
    parser.add_argument('-e', '--target-col-labels', 
                        help='column labels for y matrix (dimension E)')

    parser.add_argument('-x', '--motifs-file', 
                        help='x1 features - dimensionality MxG')
    parser.add_argument('-m', '--m-col-labels', 
                        help='column labels for x1 matrix (dimension M)')

    parser.add_argument('-z', '--regulators-file', 
                        help='x2 features - dimensionality ExR')
    parser.add_argument('-r', '--r-row-labels', 
                        help='row labels for x2 matrix (dimension R)')

    parser.add_argument('-n', '--num-iter', 
                        help='Number of iterations', default=500, type=int)

    parser.add_argument('--eta1', help='stabilization threshold 1', type=float)
    parser.add_argument('--eta2', help='stabilization threshold 2', type=float)

    parser.add_argument('--stumps', 
                        help='specify to do stumps instead of adt', 
                        action='store_true')
    parser.add_argument('--stable', 
                        help='bundle rules/implement stabilized boosting', 
                        action='store_true')
    parser.add_argument('--corrected-loss', 
                        action='store_true', help='For corrected Loss')

    parser.add_argument('--ncpu', 
                        help='number of cores to run on', type=int)

    parser.add_argument('--holdout-file', 
                        help='Specify holdout matrix, same as y dimensions', default=None)
    parser.add_argument('--holdout-format', 
                        help='format for holdout matrix', 
                        default=None)

    # Parse arguments
    args = parser.parse_args()
    
    log('load y start ')
    y = TargetMatrix(args.target_file, 
                     args.target_row_labels, 
                     args.target_col_labels,
                     args.input_format,
                     args.mult_format)
    log('load y stop')

    log('load x1 start')
    x1 = Motifs(args.motifs_file, 
                args.target_row_labels, 
                args.m_col_labels,
                args.input_format,
                args.mult_format)
    log('load x1 stop')
    
    log('load x2 start')
    x2 = Regulators(args.regulators_file, 
                    args.r_row_labels, 
                    args.target_col_labels,
                    args.input_format,
                    args.mult_format)
    log('load x2 stop')
   
    # model_state = ModelState()
    log('load holdout start')
    holdout = Holdout(y, args.mult_format, args.holdout_file, args.holdout_format)
    log('load holdout stop')

    config.OUTPUT_PATH = args.output_path
    config.OUTPUT_PREFIX = args.output_prefix
    config.TUNING_PARAMS = TuningParams(
        args.num_iter, 
        args.stumps, args.stable, args.corrected_loss,
        args.eta1, args.eta2, 20, 1./holdout.n_train)
    config.NCPU = args.ncpu

    return (x1, x2, y, holdout)

def find_next_decision_node(tree, holdout, y, x1, x2):
    ## Calculate loss at all search nodes
    # log('start rule_processes')
    best_split, regulator_sign, loss_best = find_rule_processes(
        tree, holdout, y, x1, x2) 
    # log('end rule_processes')

    # Get rule weights for the best split
    # log('start find_rule_weights')
    rule_weights = find_rule_weights(
        tree.ind_pred_train[best_split], tree.weights, tree.ones_mat, 
        holdout, y, x1, x2)
    # log('end find_rule_weights')

    # Get current rule, no stabilization
    # log('start get_current_rule')
    motif,regulator,reg_sign,rule_train_index,rule_test_index = get_current_rule(
        tree, best_split, regulator_sign, loss_best, holdout, y, x1, x2)
    # log('end get_current_rule')

    ## Update score without stabilization,  if stabilization results 
    ## in one rule or if stabilization criterion not met
    rule_score = calc_score(tree, rule_weights, rule_train_index)
    motif_bundle = []
    regulator_bundle = []

    ### Store motifs/regulators above this node (for margin score)
    above_motifs = tree.above_motifs[best_split]+tree.split_x1[best_split].tolist()
    above_regs = tree.above_regs[best_split]+tree.split_x2[best_split].tolist()

    return (motif, regulator, best_split, 
            motif_bundle, regulator_bundle, 
            rule_train_index, rule_test_index, rule_score, 
            above_motifs, above_regs)


def find_next_decision_node_stable(tree, holdout, y, x1, x2, pool):
    ## Calculate loss at all search nodes
    # log('start rule_processes')
    best_split, regulator_sign, loss_best = find_rule_processes(
        tree, holdout, y, x1, x2) 
    # log('end rule_processes')

    # Get rule weights for the best split
    # log('start find_rule_weights')
    rule_weights = find_rule_weights(
        tree.ind_pred_train[best_split], tree.weights, tree.ones_mat, 
        holdout, y, x1, x2)
    # log('end find_rule_weights')

    # Get current rule, no stabilization
    # log('start get_current_rule')
    motif, regulator, regulator_sign, rule_train_index, rule_test_index = get_current_rule(
        tree, best_split, regulator_sign, loss_best, holdout, y, x1, x2)
    # log('end get_current_rule')

    # Store current weights
    weights_i = util.element_mult(tree.weights, tree.ind_pred_train[best_split])

    # Test if stabilization criterion is met
    stable_test = stabilize.stable_boost_test(tree, rule_train_index, holdout)
    stable_thresh = stabilize.stable_boost_thresh(tree, y, weights_i)

    # If stabilization criterion met
    if stable_test >= config.TUNING_PARAMS.eta_2*stable_thresh:
        print 'stabilization criterion applies'
        # Get rules that are bundled together
        # log('start bundle_rules')
        bundle = stabilize.bundle_rules(tree, y, x1, x2, motif, regulator, regulator_sign, best_split, rule_weights)
        # log('end bundle_rules')

        # Store bundle size
        bundle_size=len(bundle.rule_bundle_regup_motifs)+len(bundle.rule_bundle_regdown_motifs)

        # If bundle size > 1
        if bundle_size>1:

            # Get theta and indices/score for each individual rule in bundle
            # log('start calc_theta')
            ( theta, theta_alphas, 
              bundle_train_rule_indices, 
              bundle_test_rule_indices
              ) = stabilize.calc_theta(bundle, tree.ind_pred_train, 
              tree.ind_pred_test, best_split, 
              rule_weights.w_pos, rule_weights.w_neg, y, x1, x2, pool)
            # log('end calc_theta')
            
            # Get new index for joint bundled rule
            # log('start get_bundle_rule')
            ( rule_score, rule_train_index, rule_test_index 
              ) = stabilize.get_bundle_rule(
                  tree, bundle, theta, best_split, theta_alphas, 
                  bundle_train_rule_indices, 
                  bundle_test_rule_indices, holdout, rule_weights.w_pos, rule_weights.w_neg)
            # log('end get_bundle_rule')

            # Add bundled rules to bundle
            motif_bundle = bundle.rule_bundle_regup_motifs+bundle.rule_bundle_regdown_motifs
            regulator_bundle = bundle.rule_bundle_regup_regs+bundle.rule_bundle_regdown_regs

    # If stabilization criterion not met, default bundle size 1
    else:
        bundle_size=1

    # Stabilization criterion not met, continue with best rule
    if bundle_size==1:
        rule_score = calc_score(tree, rule_weights, rule_train_index)
        motif_bundle = []
        regulator_bundle = []

    above_motifs = tree.above_motifs[best_split]+np.unique(tree.bundle_x1[best_split]+tree.split_x1[best_split].tolist()).tolist()
    above_regs = tree.above_regs[best_split]+np.unique(tree.bundle_x2[best_split]+tree.split_x2[best_split].tolist()).tolist()

    return (motif, regulator, best_split, 
            motif_bundle, regulator_bundle, 
            rule_train_index, rule_test_index, rule_score, 
            above_motifs, above_regs)


def main():
    print 'starting main loop'

    level='QUIET'
    ### Parse arguments
    log('parse args start', level=level)
    (x1, x2, y, holdout) = parse_args()
    log('parse args end', level=level)

    ### Create tree object
    log('make tree start', level=level)
    tree = DecisionTree(holdout, y, x1, x2)
    log('make tree stop', level=level)

    # Make pool
    pool = multiprocessing.Pool(processes=config.NCPU) # create pool of processes

    ### Main Loop
    for i in xrange(1,config.TUNING_PARAMS.num_iter):
        log('iteration {0}'.format(i), level=level)
        
        (motif, regulator, best_split, 
         motif_bundle, regulator_bundle, 
         rule_train_index, rule_test_index, rule_score, 
         above_motifs, above_regs) = find_next_decision_node_stable(
             tree, holdout, y, x1, x2, pool)
        
        ### Add the rule with best loss
        tree.add_rule(motif, regulator, best_split, 
                      motif_bundle, regulator_bundle, 
                      rule_train_index, rule_test_index, rule_score, 
                      above_motifs, above_regs, holdout, y)

        ### Return default to bundle
        bundle_set = 1

        ### Print progress
        log_progress(tree, i)

    ### Get plot label so plot label uses parameters used
    method_label=get_plot_label()

    ### Write out rules
    out_file_name='{0}global_rules/{1}_tree_rules_{2}_{3}iter.txt'.format(
        config.OUTPUT_PATH, config.OUTPUT_PREFIX, 
        method_label, config.TUNING_PARAMS.num_iter)
    tree.write_out_rules(tree, x1, x2, config.TUNING_PARAMS, method_label, out_file=out_file_name)

    ### Make plots
    plot_margin(tree, method_label, config.TUNING_PARAMS.num_iter)
    plot_balanced_error(tree, method_label, config.TUNING_PARAMS.num_iter)
    plot_imbalanced_error(tree, method_label, config.TUNING_PARAMS.num_iter)

    # Close pool
    pool.close() # stop adding processes
    pool.join() # wait until all threads are done before going on



### Main
if __name__ == "__main__":
    main()
