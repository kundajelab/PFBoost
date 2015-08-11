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
from boosting_2D import write_load_data_script

### Open log files
log = util.log

### Set constant parameters
TuningParams = namedtuple('TuningParams', [
    'num_iter',
    'use_stumps', 'use_stable', 'use_corrected_loss', 'use_prior',
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
    parser.add_argument('--mult-format', help='options are: dense, sparse')

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
    parser.add_argument('--plot', 
                        action='store_true', help='Plot imbalanced & balanced loss and margins')

    parser.add_argument('--use_prior', 
                        action='store_true', help='Use prior',)
    parser.add_argument('--prior_input_format', 
                        help='options are: matrix, triplet',)
    parser.add_argument('--motif_reg_file', 
                        default=None, help='motif-regulator priors [0,1] real-valued',)
    parser.add_argument('--motif_reg_row_labels', 
                        default=None, help='motif labels for motif-regulator prior',)
    parser.add_argument('--motif_reg_col_labels', 
                        default=None, help='regulator labels for motif-regulator prior',)
    parser.add_argument('--reg_reg_file', 
                        default=None, help='regulator-regulator priors [0,1] real-valued',) 
    parser.add_argument('--reg_reg_row_labels', 
                        default=None, help='motif labels for regulator-regulator prior',)    
    parser.add_argument('--reg_reg_col_labels', 
                        default=None, help='regulator labels for regulator-regulator prior',)
    

    parser.add_argument('--ncpu', 
                        help='number of cores to run on', type=int)

    parser.add_argument('--holdout-file', 
                        help='Specify holdout matrix, same as y dimensions', default=None)
    parser.add_argument('--holdout-format', 
                        help='format for holdout matrix', 
                        default=None)

    parser.add_argument('--shuffle_y', 
                        help='flag to shuffle the contents of y matrix', 
                        action='store_true')
    parser.add_argument('--shuffle_x1', 
                        help='flag to shuffle the contents of y matrix', 
                        action='store_true')
    parser.add_argument('--shuffle_x2', 
                        help='flag to shuffle the contents of y matrix', 
                        action='store_true')

    # Parse arguments
    args = parser.parse_args()

    # Load the three feature matrices
    log('load y start ')
    y = TargetMatrix(args.target_file, 
                     args.target_row_labels, 
                     args.target_col_labels,
                     args.input_format,
                     args.mult_format)
    log('load y stop')

    log('load x1 start')
    x1 = Motifs(args.motifs_file, 
                args.m_col_labels,
                args.target_row_labels, 
                args.input_format,
                args.mult_format)
    log('load x1 stop')
    
    log('load x2 start')
    x2 = Regulators(args.regulators_file, 
                    args.target_col_labels,
                    args.r_row_labels, 
                    args.input_format,
                    args.mult_format)
    log('load x2 stop')
   
    # Shuffle data
    if args.shuffle_y:
        y = util.shuffle_data_object(y)
    if args.shuffle_x1:
        x1 = util.shuffle_data_object(x1)
    if args.shuffle_x2:
        x2 = util.shuffle_data_object(x1)

    # Load holdout
    log('load holdout start')
    holdout = Holdout(y, args.mult_format, args.holdout_file, args.holdout_format)
    log('load holdout stop')

    # Configure tuning tarameters
    config.TUNING_PARAMS = TuningParams(
        args.num_iter, 
        args.stumps, args.stable, args.corrected_loss,
        args.use_prior,
        args.eta1, args.eta2, 20, 1./holdout.n_train)

    # Get method label so plot label uses parameters used
    method_label=util.get_method_label()
    # get time stamp
    time_stamp = time.strftime("%Y_%m_%d")
    # Configure output directory - date-stamped directory in output path
    config.OUTPUT_PATH = args.output_path
    config.OUTPUT_PREFIX = time_stamp+'_'+args.output_prefix+'_'+method_label+'_'+str(config.TUNING_PARAMS.num_iter)+'iter'
    if not os.path.exists(config.OUTPUT_PATH+config.OUTPUT_PREFIX):
        os.makedirs(config.OUTPUT_PATH+config.OUTPUT_PREFIX)


    # Configure prior matrix
    if config.TUNING_PARAMS.use_prior:
        prior.PRIOR_PARAMS=prior.PriorParams(
            50, 0.998,
            args.prior_input_format, 
            args.motif_reg_file, args.motif_reg_row_labels, args.motif_reg_col_labels,
            args.reg_reg_file, args.reg_reg_row_labels, args.reg_reg_col_labels)
        prior.prior_motifreg, prior.prior_regreg = prior.parse_prior(prior.PRIOR_PARAMS, x1, x2)

    config.NCPU = args.ncpu
    config.PLOT = args.plot

    return (x1, x2, y, holdout)


### Find next decision node given current state of tree
def find_next_decision_node(tree, holdout, y, x1, x2, iteration):
    level='QUIET'
    ## Calculate loss at all search nodes
    log('find rule process', level=level)
    best_split, regulator_sign, loss_best = find_rule_processes(
        tree, holdout, y, x1, x2) 

    log('update loss with prior', level=level)
    if config.TUNING_PARAMS.use_prior:
        loss_best = prior.update_loss_with_prior(loss_best, prior.PRIOR_PARAMS, prior.prior_motifreg, prior.prior_regreg, iteration)

    # Get rule weights for the best split
    log('find rule weights', level=level)
    rule_weights = find_rule_weights(
        tree.ind_pred_train[best_split], tree.weights, tree.ones_mat, 
        holdout, y, x1, x2)

    # Get current rule, no stabilization
    log('get current rule', level=level)
    (motif, regulator, regulator_sign, rule_train_index, rule_test_index 
     ) = get_current_rule(
         tree, best_split, regulator_sign, loss_best, holdout, y, x1, x2)

    if config.TUNING_PARAMS.use_stable:
        log('starting stabilization', level=level)
        # Store current training weights
        weights_i = util.element_mult(tree.weights, tree.ind_pred_train[best_split])

        # Test if stabilization criterion is met
        log('stabilization test', level=level)
        stable_test = stabilize.stable_boost_test(tree, rule_train_index, holdout)
        stable_thresh = stabilize.stable_boost_thresh(tree, y, weights_i)

        # If stabilization criterion met, then we want to find a bundle of 
        # correlated rules to use as a single node  
        if stable_test >= config.TUNING_PARAMS.eta_2*stable_thresh:
            print 'stabilization criterion applies'
            # Get rules that are bundled together
            log('getting rule bundle', level=level)
            bundle = stabilize.bundle_rules(
                tree, y, x1, x2, 
                motif, 
                regulator, regulator_sign, 
                best_split, rule_weights)

            # rule score is the direction and magnitude of the prediciton update
            # for the rule given by rule_weights and rule_train_index
            log('updating scores and indices with bundle', level=level)
            ( rule_score, rule_train_index, rule_test_index 
              ) = stabilize.get_rule_score_and_indices(bundle, 
              tree.ind_pred_train, tree.ind_pred_test, 
              best_split, weights_i, rule_weights,
              tree, y, x1, x2, holdout,
              rule_train_index, rule_test_index)

            # Add bundled rules to bundle
            log('adding bundles to rule', level=level)
            motif_bundle = bundle.rule_bundle_regup_motifs+bundle.rule_bundle_regdown_motifs
            regulator_bundle = bundle.rule_bundle_regup_regs+bundle.rule_bundle_regdown_regs

        else:
            # rule score is the direction and magnitude of the prediciton update
            # for the rule given by rule_weights and rule_train_index
            log('updating rule without stabilization', level=level)
            rule_score = calc_score(tree, rule_weights, rule_train_index)
            motif_bundle = []
            regulator_bundle = []

    # If no stabilization
    else:
        # rule score is the direction and magnitude of the prediciton update
        # for the rule given by rule_weights and rule_train_index
        log('updating rule without stabilization', level=level)
        rule_score = calc_score(tree, rule_weights, rule_train_index)
        motif_bundle = []
        regulator_bundle = []

        
    log('adding above motifs/regs', level=level)
    above_motifs = tree.above_motifs[best_split]+np.unique(
        tree.bundle_x1[best_split]+[tree.split_x1[best_split]]).tolist()
    above_regs = tree.above_regs[best_split]+np.unique(
        tree.bundle_x2[best_split]+[tree.split_x2[best_split]]).tolist()

    return (motif, regulator, best_split, 
            motif_bundle, regulator_bundle, 
            rule_train_index, rule_test_index, rule_score, 
            above_motifs, above_regs)


def main():
    print 'starting main loop'

    ### Parse arguments
    level='VERBOSE'
    log('parse args start', level=level)
    (x1, x2, y, holdout) = parse_args()
    log('parse args end', level=level)

    ### logfile saves output to file
    logfile_name='{0}{1}/LOG_FILE.txt'.format(
            config.OUTPUT_PATH, config.OUTPUT_PREFIX)
    if not os.path.exists('{0}{1}'.format(config.OUTPUT_PATH, config.OUTPUT_PREFIX)):
        os.makedirs('{0}{1}'.format(config.OUTPUT_PATH, config.OUTPUT_PREFIX))
    f = open(logfile_name, 'w')
    logfile = Logger(ofp=f)
    logfile("Command run:\n {0} \n \n ".format(' '.join(sys.argv)), log_time=False)

    ### Print time to output
    t0 = time.time()
    logfile('starting main loop: {0}'.format(t0), log_time=True)

    ### Create tree object
    log('make tree start', level=level)
    tree = DecisionTree(holdout, y, x1, x2)
    log('make tree stop', level=level)

    ### Main Loop
    for i in xrange(1,config.TUNING_PARAMS.num_iter):

        log('iteration {0}'.format(i))
        
        log('find next node', level=level)
        (motif, regulator, best_split, 
         motif_bundle, regulator_bundle, 
         rule_train_index, rule_test_index, rule_score, 
         above_motifs, above_regs) = find_next_decision_node(
             tree, holdout, y, x1, x2, i)
        
        ### Add the rule with best loss
        log('adding next rule', level=level)
        tree.add_rule(motif, regulator, best_split, 
                      motif_bundle, regulator_bundle, 
                      rule_train_index, rule_test_index, rule_score, 
                      above_motifs, above_regs, holdout, y)

        ### Print progress
        util.log_progress(tree, i)

    # Save tree state
    tree_file_name='{0}{1}/saved_tree_state__{1}'.format(
        config.OUTPUT_PATH, config.OUTPUT_PREFIX)
    save_tree_state(tree, pickle_file=tree_file_name)

    ### Write out rules
    rule_file_name='{0}{1}/global_rules__{1}.txt'.format(
        config.OUTPUT_PATH, config.OUTPUT_PREFIX)
    tree.write_out_rules(tree, x1, x2, config.TUNING_PARAMS, out_file=rule_file_name)

    ### Write out load data file
    write_load_data_script.write_load_data_script(y, x1, x2, prior.PRIOR_PARAMS, tree_file_name)

    ### Make plots
    if config.PLOT:
        plot.configure_plot_dir(tree, config.TUNING_PARAMS.num_iter)
        plot.plot_margin(tree, config.TUNING_PARAMS.num_iter)
        plot.plot_balanced_error(tree, config.TUNING_PARAMS.num_iter)
        plot.plot_imbalanced_error(tree, config.TUNING_PARAMS.num_iter)

    ### Print end time and close logfile pointer
    t = time.time()
    logfile('ending main loop: {0}'.format(t), log_time=True)
    logfile('total time: {0}'.format(t-t0), log_time=True)
    f.close()

### Main
if __name__ == "__main__":
    main()
