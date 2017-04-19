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
from boosting_2D import save_model
from boosting_2D import hierarchy as h

### Open log files
log = util.log

def parse_args():
    # Get arguments
    parser = argparse.ArgumentParser(description='Run boosting2D')

    parser.add_argument('--output-prefix', 
                        help='Analysis name for output plots')
    parser.add_argument('--output-path', 
                        help='path to write the results to', 
                        default='/users/pgreens/projects/boosting/results/')

    parser.add_argument('--input-format', help='options are: matrix, triplet', default='matrix')
    parser.add_argument('--mult-format', help='options are: dense, sparse', default='sparse')

    parser.add_argument('-y', '--target-file', 
                        help='target matrix - dimensionality GxE')
    parser.add_argument('-g', '--target-row-labels', 
                        help='row labels for y matrix (dimension G)')
    parser.add_argument('-e', '--target-col-labels', 
                        help='column labels for y matrix (dimension E)')

    parser.add_argument('-x', '--motifs-file', 
                        help='x1 features - dimensionality MxG')
    parser.add_argument('-m', '--motif-labels', 
                        help='column labels for x1 matrix (dimension M)')

    parser.add_argument('-z', '--regulators-file', 
                        help='x2 features - dimensionality ExR')
    parser.add_argument('-r', '--regulator-labels', 
                        help='row labels for x2 matrix (dimension R)')

    parser.add_argument('-n', '--num-iter', 
                        help='Number of iterations', default=500, type=int)

    parser.add_argument('--eta1', help='stabilization threshold 1', type=float, default=0.05)
    parser.add_argument('--eta2', help='stabilization threshold 2', type=float, default=0.01)

    parser.add_argument('--verbose', help='print all logs', default=False, action="store_true")

    parser.add_argument('--stumps', 
                        help='specify to do stumps instead of adt', 
                        action='store_true')
    parser.add_argument('--stable', 
                        help='bundle rules/implement stabilized boosting', 
                        action='store_true')
    parser.add_argument('--max-bundle-size', 
                        help='maximum allowed size for bundle', type=int, default=20)
    parser.add_argument('--corrected-loss', 
                        action='store_true', help='For corrected Loss')

    parser.add_argument('--plot', 
                        action='store_true', help='Plot imbalanced & balanced loss and margins')

    parser.add_argument('--use-prior', 
                        action='store_true', help='Use prior', default=False)
    parser.add_argument('--motif-reg-file', 
                        default=None, help='motif-regulator priors [0,1] real-valued',)
    parser.add_argument('--motif-reg-row-labels', 
                        default=None, help='motif labels for motif-regulator prior',)
    parser.add_argument('--motif-reg-col-labels', 
                        default=None, help='regulator labels for motif-regulator prior',)
    parser.add_argument('--reg-reg-file', 
                        default=None, help='regulator-regulator priors [0,1] real-valued',) 
    parser.add_argument('--reg-reg-labels', 
                        default=None, help='labels for regulator-regulator prior, used for cols + rows',)    

    parser.add_argument('--ncpu', 
                        help='number of cores to run on', type=int, default=1)

    parser.add_argument('--holdout-file', 
                        help='Specify holdout matrix, same as y dimensions', default=None)
    parser.add_argument('--holdout-format', 
                        help='(matrix or triplet) format for holdout matrix', 
                        default='matrix')
    parser.add_argument('--train-fraction', 
                        help='fraction of data used for training. validation is 1 - train_fraction',
                        type=float, default=0.8)    

    parser.add_argument('--compress-regulators', 
                        help='combine regulators with same pattern across conditions', 
                        action='store_true', default=False)
    parser.add_argument('--shuffle-y', 
                        help='flag to shuffle the contents of y matrix', 
                        action='store_true')
    parser.add_argument('--shuffle-x1', 
                        help='flag to shuffle the contents of y matrix', 
                        action='store_true')
    parser.add_argument('--shuffle-x2', 
                        help='flag to shuffle the contents of y matrix', 
                        action='store_true')

    parser.add_argument('--save-tree-only', 
                        help='Pickle the tree only', default=False, action="store_true")
    parser.add_argument('--save-complete-data', 
                        help='Pickle every single matrix (x1, x2, y, holdout, prior, tree, etc.)',
                        default=False, action="store_true")
    parser.add_argument('--save-for-post-processing', 
                        help='Generate script to load results', default=True, action="store_true")

    parser.add_argument('--hierarchy-name', 
                        help='Reference for hierarchy encoding in hierarchy.py', default=None)

    # Parse arguments
    args = parser.parse_args()

    # First set level of verbosity
    config.VERBOSE = args.verbose

    # Load the three feature matrices
    log('load y start')
    y = TargetMatrix(args.target_file, 
                     args.target_row_labels, 
                     args.target_col_labels,
                     args.input_format,
                     args.mult_format)
    log('load y stop')

    log('load x1 start')
    x1 = Motifs(args.motifs_file, 
                args.m_row_labels,
                args.target_row_labels, 
                args.input_format,
                args.mult_format)
    log('load x1 stop')
    
    log('load x2 start')
    x2 = Regulators(args.regulators_file, 
                    args.target_col_labels,
                    args.r_col_labels, 
                    args.input_format,
                    args.mult_format)
    log('load x2 stop')

    # Shuffle data
    if args.shuffle_y:
        y = util.shuffle_data_object(y)
    if args.shuffle_x1:
        x1 = util.shuffle_data_object(x1)
    if args.shuffle_x2:
        x2 = util.shuffle_data_object(x2)

    # Compress regulators
    if args.compress_regulators:
        x2 = util.compress_regulators(x2)

    # Load holdout
    log('load holdout start')
    holdout = Holdout(y, args.mult_format,
                      args.holdout_file, args.holdout_format,
                      args.train_fraction)
    log('load holdout stop')

    # Configure hierarchy
    hierarchy = h.get_hierarchy(name=args.hierarchy_name)
    if hierarchy is not None:
        log('Applying hierarchy: %s'%hierarchy.name, level='VERBOSE')        

    # Configure tuning tarameters
    config.TUNING_PARAMS = config.TuningParams(
        args.num_iter, 
        args.stumps, args.stable, args.corrected_loss,
        args.use_prior,
        args.eta1, args.eta2, args.max_bundle_size, 1./holdout.n_train)
    config.SAVING_PARAMS = config.SavingParams(
        args.save_tree_only,
        args.save_complete_data,
        args.save_for_post_processing)

    # Get method label so plot label uses parameters used
    method_label = util.get_method_label()
    # get time stamp
    time_stamp = time.strftime("%Y_%m_%d")
    # Configure output directory - date-stamped directory in output path
    config.OUTPUT_PATH = args.output_path if args.output_path \
                         is not None else os.getcwd()
    config.OUTPUT_PREFIX = time_stamp + '_' + args.output_prefix + '_' + method_label + \
                           '_'+ str(config.TUNING_PARAMS.num_iter) + 'iter'
    if hierarchy is not None:
        config.OUTPUT_PREFIX = config.OUTPUT_PREFIX + '_hierarchy_%s'%hierarchy.name

    if not os.path.exists(config.OUTPUT_PATH+config.OUTPUT_PREFIX):
        os.makedirs(config.OUTPUT_PATH+config.OUTPUT_PREFIX)

    # Configure prior matrix
    if config.TUNING_PARAMS.use_prior:
        log('Applying prior', level='VERBOSE') 
        prior.PRIOR_PARAMS=prior.PriorParams(
            50, 0.998,
            args.motif_reg_file, 
            args.motif_reg_row_labels, args.motif_reg_col_labels,
            args.reg_reg_file, 
            args.reg_reg_labels)
        prior.prior_motifreg, prior.prior_regreg = prior.parse_prior(
            prior.PRIOR_PARAMS, x1, x2)
    else:
        prior.PRIOR_PARAMS=None

    config.NCPU = args.ncpu
    config.PLOT = args.plot

    return (x1, x2, y, holdout, hierarchy)


### Find next decision node given current state of tree
# @profile
def find_next_decision_node(tree, holdout, y, x1, x2, hierarchy, iteration):
    level = 'VERBOSE' if config.VERBOSE else 'QUIET'

    ## Calculate loss at all search nodes 
    ## (will search across current hier node and direct children to find best hier node)
    log('find rule process', level=level)
    best_split, regulator_sign, hierarchy_node, loss_best = find_rule_processes(
        tree, holdout, y, x1, x2, hierarchy) 

    # Update loss with prior
    if config.TUNING_PARAMS.use_prior:
        log('update loss with prior', level=level)
        best_split_regulator = util.get_best_split_regulator(tree, x2, best_split)
        loss_best = prior.update_loss_with_prior(loss_best, prior.PRIOR_PARAMS,
         prior.prior_motifreg, prior.prior_regreg, iteration, best_split_regulator)

    log('find rule weights', level=level)

    # Mask training/testing examples by using only hierarchy children
    non_hier_training_examples = tree.ind_pred_train[best_split]
    training_examples = h.get_hierarchy_index(hierarchy_node, hierarchy, 
                                              non_hier_training_examples, tree)

    non_hier_testing_examples = tree.ind_pred_test[best_split]
    testing_examples = h.get_hierarchy_index(hierarchy_node, hierarchy, 
                                             non_hier_testing_examples, tree)

    # Get rule weights for the best split
    rule_weights = find_rule_weights(
        training_examples, tree.weights, tree.ones_mat, 
        holdout, y, x1, x2)

    # Get current rule, no stabilization
    # rule_train_index/rule_test_index restricted to hierarchy
    log('get current rule', level=level)
    (motif, regulator, regulator_sign, rule_train_index, rule_test_index 
     ) = get_current_rule(
         tree, best_split, regulator_sign, loss_best, 
         holdout, y, x1, x2, hierarchy, hierarchy_node)

    if config.TUNING_PARAMS.use_stable:
        log('starting stabilization', level=level)
        # Store current training weights
        weights_i = util.element_mult(tree.weights, training_examples)

        # Test if stabilization criterion is met
        log('stabilization test', level=level)
        stable_test = stabilize.stable_boost_test(tree, rule_train_index, holdout)
        stable_thresh = stabilize.stable_boost_thresh(tree, y, weights_i)

        # If stabilization criterion met, then we want to find a bundle of 
        # correlated rules to use as a single node  
        if stable_test >= config.TUNING_PARAMS.eta_2 * stable_thresh:
            log('stabilization criterion applies', level='VERBOSE')
            # Get rules that are bundled together
            log('getting rule bundle', level=level)
            bundle = stabilize.bundle_rules(tree, y, x1, x2, 
                                            motif, 
                                            regulator, regulator_sign, 
                                            best_split, rule_weights,
                                            hierarchy, hierarchy_node)

            # rule score is the direction and magnitude of the prediciton update
            # for the rule given by rule_weights and rule_train_index
            log('updating scores and indices with bundle', level=level)
            (rule_score, rule_train_index, rule_test_index 
            ) = stabilize.get_rule_score_and_indices(bundle, 
                          training_examples, testing_examples,
                          weights_i, rule_weights,
                          tree, y, x1, x2, holdout,
                          rule_train_index, rule_test_index)

            # Add bundled rules to bundle
            log('adding bundles to rule', level=level)
            motif_bundle = bundle.rule_bundle_regup_motifs + bundle.rule_bundle_regdown_motifs
            regulator_bundle = bundle.rule_bundle_regup_regs + bundle.rule_bundle_regdown_regs

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
    above_motifs = tree.above_motifs[best_split] + np.unique(
        tree.bundle_x1[best_split] + [tree.split_x1[best_split]]).tolist()
    above_regs = tree.above_regs[best_split] + np.unique(
        tree.bundle_x2[best_split] + [tree.split_x2[best_split]]).tolist()

    return (motif, regulator, best_split, hierarchy_node,
            motif_bundle, regulator_bundle, 
            rule_train_index, rule_test_index, rule_score, 
            above_motifs, above_regs)


def main():
    ### Parse arguments
    log('parse args start', level='VERBOSE')
    (x1, x2, y, holdout, hierarchy) = parse_args() # Sets up output path and prefix
    log('parse args end', level='VERBOSE')

    ### Set up output files
    out_file_prefix = '{0}{1}'.format(config.OUTPUT_PATH, config.OUTPUT_PREFIX)
    logfile_name = '{0}/LOG_FILE.txt'.format(out_file_prefix)
    pickle_file = '{0}/saved_complete_model__{1}.gz'.format(out_file_prefix, config.OUTPUT_PREFIX)
    pickle_script_file = '{0}/load_pickle_data_script.py'.format(out_file_prefix)
    load_complete_data_script_file = '{0}/load_complete_data_script.py'.format(out_file_prefix)

    log('Log File: {0}'.format(logfile_name), level='VERBOSE')
    if not os.path.exists('{0}{1}'.format(config.OUTPUT_PATH, config.OUTPUT_PREFIX)):
        os.makedirs('{0}{1}'.format(config.OUTPUT_PATH, config.OUTPUT_PREFIX))
    f = open(logfile_name, 'w')
    logfile = Logger(ofp=f, verbose=True)
    logfile("Command run:\n {0} \n \n ".format(' '.join(sys.argv)), 
            log_time=False, level='VERBOSE')

    ### Print time to output
    t0 = time.time()
    logfile('starting main loop: {0}'.format(t0), 
            log_time=True, level='VERBOSE')

    ### Create tree object
    log('make tree start')
    tree = DecisionTree(holdout, y, x1, x2)
    log('make tree stop')

    log('starting main loop', level='VERBOSE') 

    ### Main Loop
    for i in xrange(1, config.TUNING_PARAMS.num_iter + 1):

        # log('iteration {0}'.format(i))
        
        log('find next node')
        (motif, regulator, best_split, hierarchy_node,
         motif_bundle, regulator_bundle, 
         rule_train_index, rule_test_index, rule_score, 
         above_motifs, above_regs) = find_next_decision_node(
             tree, holdout, y, x1, x2, hierarchy, i)
        
        ### Add the rule with best loss
        log('adding next rule')
        tree.add_rule(motif, regulator, best_split, hierarchy_node,
                      motif_bundle, regulator_bundle, 
                      rule_train_index, rule_test_index, rule_score, 
                      above_motifs, above_regs, holdout, y)

        ### Print progress
        util.log_progress(tree, i, x1, x2, hierarchy, ofp=f, verbose=True)

    ### Print end time and close logfile pointer
    t1 = time.time()
    logfile('ending main loop: {0}'.format(t1), log_time=True, level='VERBOSE')
    logfile('total time: {0}'.format(t1 - t0), log_time=True, level='VERBOSE')


    ### Write out rules
    rule_file_name = '{0}{1}/global_rules__{1}.txt'.format(
        config.OUTPUT_PATH, config.OUTPUT_PREFIX)
    tree.write_out_rules(tree, x1, x2, config.TUNING_PARAMS,
                         out_file=rule_file_name, logfile_pointer=f)

    # Save tree state
    if config.SAVING_PARAMS.save_tree_only:
        tree_file_name = '{0}{1}/saved_tree_state__{1}.gz'.format(
            config.OUTPUT_PATH, config.OUTPUT_PREFIX)
        save_tree_state(tree, pickle_file=tree_file_name)
        log('Pickled tree written to: {0}'.format(tree_file_name), level='VERBOSE')

    ### Write out load data file
    if config.SAVING_PARAMS.save_complete_data:
        save_model.write_load_complete_data_script(y, x1, x2, holdout, hierarchy, prior.PRIOR_PARAMS,
                                                   tree_file_name, load_complete_data_script_file,
                                                   logfile_pointer=f)

    ### Store model objects and script to load iteration
    if config.SAVING_PARAMS.save_for_post_processing:
        save_model.save_complete_model_state(pickle_file, x1, x2, y, hierarchy, tree)
        save_model.write_postprocesssing_load_script(pickle_file, 
                                                     pickle_script_file=pickle_script_file, 
                                                     logfile_pointer=f)

    ### Print pickling time and close logfile pointer
    t2 = time.time()
    logfile('save model time: {0}'.format(t2 - t1), log_time=True, level='VERBOSE')
    f.close()

    ### Make plots
    if config.PLOT:
        plot.configure_plot_dir(tree, config.TUNING_PARAMS.num_iter)
        plot.plot_margin(tree, config.TUNING_PARAMS.num_iter)
        plot.plot_balanced_error(tree, config.TUNING_PARAMS.num_iter)
        plot.plot_imbalanced_error(tree, config.TUNING_PARAMS.num_iter)
        log('Plots generated in: {0}/plots/'.format(out_file_prefix), log_time=False, level='VERBOSE')


### Main
if __name__ == "__main__":
    main()
