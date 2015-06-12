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
from boosting_2D.margin_score import *
from boosting_2D.data_class import *
from boosting_2D.find_rule import *

log = util.log

TuningParams = namedtuple('TuningParams', [
    'num_iter',
    'use_stumps', 'use_stable', 'use_corrected_loss',
    'eta_1', 'eta_2', 'bundle_max'
])

def parse_args():
    # Get arguments
    parser = argparse.ArgumentParser(description='Extract Chromatin States')

    parser.add_argument('--output-prefix', 
                        help='Analysis name for output plots')
    parser.add_argument('--output-path', 
                        help='path to write the results to', 
                        default='/users/pgreens/projects/boosting/results/')

    parser.add_argument('--input_format', help='options are: matrix, triplet')
    parser.add_argument('--mult_format', help='options are: matrix, dense')

    parser.add_argument('-y', '--target-file', 
                        help='target matrix - dimensionality GxE')
    parser.add_argument('-g', '--target_row_labels', 
                        help='row labels for y matrix (dimension G)')
    parser.add_argument('-e', '--target_col_labels', 
                        help='column labels for y matrix (dimension E)')

    parser.add_argument('-x', '--motifs-file', 
                        help='x1 features - dimensionality MxG')
    parser.add_argument('-m', '--m-col-labels', 
                        help='column labels for x1 matrix (dimension M)')

    parser.add_argument('-z', '--regulators-file', 
                        help='x2 features - dimensionality ExR')
    parser.add_argument('-r', '--r-row-labels', 
                        help='row labels for x2 matrix (dimension R)')

    parser.add_argument('-n', '--num_iter', 
                        help='Number of iterations', default=500, type=int)

    parser.add_argument('--eta1', help='stabilization threshold 1', type=float)
    parser.add_argument('--eta2', help='stabilization threshold 2', type=float)

    parser.add_argument('-s', '--stumps', 
                        help='specify to do stumps instead of adt', 
                        action='store_true')
    parser.add_argument('-d', '--stable', 
                        help='bundle rules/implement stabilized boosting', 
                        action='store_true')
    parser.add_argument('-c', '--corrected-loss', 
                        action='store_true', help='For corrected Loss')

    parser.add_argument('-u', '--ncpu', 
                        help='number of cores to run on', type=int)

    parser.add_argument('--holdout-file', 
                        help='Specify holdout matrix, same as y dimensions', default=None)
    parser.add_argument('--holdout-format', 
                        help='format for holdout matrix', 
                        default=None)

    # Parse arguments
    args = parser.parse_args()
    
    config.OUTPUT_PATH = args.output_path
    config.OUTPUT_PREFIX = args.output_prefix
    config.TUNING_PARAMS = TuningParams(
        args.num_iter, 
        args.stumps, args.stable, args.corrected_loss,
        args.eta1, args.eta2, 20)
    config.NCPU = args.ncpu

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
    
    return (x1, x2, y, holdout)

def find_next_decision_node(tree, holdout, y, x1, x2):
    # State number of parameters to search
    if config.TUNING_PARAMS.use_stumps:
        tree.nsearch = 1
    else:
        tree.nsearch = tree.npred

    ## Calculate loss at all search nodes
    log('start rule_processes')
    best_split, regulator_sign, loss_best = find_rule_processes(tree, holdout, y, x1, x2) # find rules with class call
    log('end rule_processes')

    # Get rule weights for the best split
    log('start find_rule_weights')
    rule_weights = find_rule_weights(
        tree.ind_pred_train[best_split], tree.weights, tree.ones_mat, holdout, y, x1, x2)
    log('end find_rule_weights')

    ### get_bundled_rules (returns the current rule if no bundling)  
    # Get current rule, no stabilization
    log('start get_current_rule')
    motif,regulator,reg_sign,rule_train_index,rule_test_index = get_current_rule(
        tree, best_split, regulator_sign, loss_best, holdout, y, x1, x2)
    log('end get_current_rule')

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

def main():
    print 'starting main loop'

    ### Parse arguments
    log('parse args start')
    (x1, x2, y, holdout) = parse_args()
    log('parse args end')

    ### Create tree object
    log('make tree start')
    tree = DecisionTree(holdout, y, x1, x2)
    log('make tree stop')

    ### Keeps track of if there are any terms to bundle
    bundle_set=1

    ### Main Loop
    for i in range(1,config.TUNING_PARAMS.num_iter):
        log('iteration {0}'.format(i), level='VERBOSE')
        
        (motif, regulator, best_split, 
         motif_bundle, regulator_bundle, 
         rule_train_index, rule_test_index, rule_score, 
         above_motifs, above_regs) = find_next_decision_node(tree, holdout, y, x1, x2)
        
        ### Add the rule with best loss
        tree.add_rule(motif, regulator, best_split, 
                      motif_bundle, regulator_bundle, 
                      rule_train_index, rule_test_index, rule_score, 
                      above_motifs, above_regs, holdout, y)

        ### Return default to bundle
        bundle_set = 1

        ### Print progress
        log_progress(tree, i)

    ### Get rid of this, add a method to the tree:
    ## Write out rules
    # Get label (MOVE)
    if config.TUNING_PARAMS.use_stable:
        stable_label='stable'
    else:
        stable_label='non_stable'
    if config.TUNING_PARAMS.use_stumps:
        method='stumps'
    else:
        method='adt'
    method_label = '{0}_{1}'.format(method, stable_label)

    out_file='{0}global_rules/{1}_tree_rules_{2}_{3}iter.txt'.format(config.OUTPUT_PATH, config.OUTPUT_PREFIX, method_label, config.TUNING_PARAMS.num_iter)
    tree.write_out_rules(config.TUNING_PARAMS, method_label, out_file=out_file)

    ### Make plots
    plot_margin(train_margins, test_margins, method, niter)
    plot_balanced_error(loss_train, loss_test, method, niter)
    plot_imbalanced_error(imbal_train, imbal_test, method, niter)


if __name__ == "__main__":
    main()
