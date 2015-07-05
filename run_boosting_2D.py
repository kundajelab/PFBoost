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

log = util.log

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
   
    # model_state = ModelState()
    log('load holdout start')
    holdout = Holdout(y, args.mult_format, args.holdout_file, args.holdout_format)
    log('load holdout stop')

    config.OUTPUT_PATH = args.output_path
    config.OUTPUT_PREFIX = args.output_prefix
    config.TUNING_PARAMS = TuningParams(
        args.num_iter, 
        args.stumps, args.stable, args.corrected_loss,
        args.use_prior,
        args.eta1, args.eta2, 20, 1./holdout.n_train)

    if config.TUNING_PARAMS.use_prior:
        prior.PRIOR_PARAMS=prior.PriorParams(
            50, 0.998,
            args.prior_input_format, 
            args.motif_reg_file, args.motif_reg_row_labels, args.motif_reg_col_labels,
            args.reg_reg_file, args.reg_reg_row_labels, args.reg_reg_col_labels)
        prior.prior_motifreg, prior.prior_regreg = prior.parse_prior(prior.PRIOR_PARAMS, x1, x2)

    config.NCPU = args.ncpu

    return (x1, x2, y, holdout)

# def find_next_decision_node(tree, holdout, y, x1, x2):
#     ## Calculate loss at all search nodes
#     # log('start rule_processes')
#     best_split, regulator_sign, loss_best = find_rule_processes(
#         tree, holdout, y, x1, x2) 
#     # log('end rule_processes')

#     # Get rule weights for the best split
#     # log('start find_rule_weights')
#     rule_weights = find_rule_weights(
#         tree.ind_pred_train[best_split], tree.weights, tree.ones_mat, 
#         holdout, y, x1, x2)
#     # log('end find_rule_weights')

#     # Get current rule, no stabilization
#     # log('start get_current_rule')
#     motif,regulator,reg_sign,rule_train_index,rule_test_index = get_current_rule(
#         tree, best_split, regulator_sign, loss_best, holdout, y, x1, x2)
#     # log('end get_current_rule')

#     ## Update score without stabilization,  if stabilization results 
#     ## in one rule or if stabilization criterion not met
#     rule_score = calc_score(tree, rule_weights, rule_train_index)
#     motif_bundle = []
#     regulator_bundle = []

#     ### Store motifs/regulators above this node (for margin score)
#     above_motifs = tree.above_motifs[best_split]+tree.split_x1[best_split].tolist()
#     above_regs = tree.above_regs[best_split]+tree.split_x2[best_split].tolist()

#     return (motif, regulator, best_split, 
#             motif_bundle, regulator_bundle, 
#             rule_train_index, rule_test_index, rule_score, 
#             above_motifs, above_regs)


def find_next_decision_node_stable(tree, holdout, y, x1, x2, iteration):
    ## Calculate loss at all search nodes
    # log('start rule_processes')    
    best_split, regulator_sign, loss_best = find_rule_processes(
        tree, holdout, y, x1, x2) 
    # log('end rule_processes')

    # log('update with prior')
    if config.TUNING_PARAMS.use_prior:
        loss_best = prior.update_loss_with_prior(loss_best, prior.PRIOR_PARAMS, prior.prior_motifreg, prior.prior_regreg, iteration)

    # log('finish with prior')

    # Get rule weights for the best split
    # log('start find_rule_weights')
    rule_weights = find_rule_weights(
        tree.ind_pred_train[best_split], tree.weights, tree.ones_mat, 
        holdout, y, x1, x2)
    # log('end find_rule_weights')

    # Get current rule, no stabilization
    # log('start get_current_rule')
    (motif, regulator, regulator_sign, rule_train_index, rule_test_index 
     ) = get_current_rule(
         tree, best_split, regulator_sign, loss_best, holdout, y, x1, x2)
    # log('end get_current_rule')

    # Store current training weights
    weights_i = util.element_mult(tree.weights, tree.ind_pred_train[best_split])

    # Test if stabilization criterion is met
    stable_test = stabilize.stable_boost_test(tree, rule_train_index, holdout)
    stable_thresh = stabilize.stable_boost_thresh(tree, y, weights_i)

    # If stabilization criterion met, then we want to find a bundle of 
    # correlated rules to use as a single node  
    if stable_test >= config.TUNING_PARAMS.eta_2*stable_thresh:
        print 'stabilization criterion applies'
        # Get rules that are bundled together
        # log('start bundle_rules')
        bundle = stabilize.bundle_rules(
            tree, y, x1, x2, 
            motif, 
            regulator, regulator_sign, 
            best_split, rule_weights)
        # log('end bundle_rules')

        # rule score is the direction and magnitude of the prediciton update
        # for the rule given by rule_weights and rule_train_index
        ( rule_score, rule_train_index, rule_test_index 
          ) = stabilize.get_rule_score_and_indices(bundle, 
          tree.ind_pred_train, tree.ind_pred_test, 
          best_split, weights_i, rule_weights,
          tree, y, x1, x2, holdout,
          rule_train_index, rule_test_index)

        # Add bundled rules to bundle
        motif_bundle = bundle.rule_bundle_regup_motifs+bundle.rule_bundle_regdown_motifs
        regulator_bundle = bundle.rule_bundle_regup_regs+bundle.rule_bundle_regdown_regs

    else:
        # rule score is the direction and magnitude of the prediciton update
        # for the rule given by rule_weights and rule_train_index
        rule_score = calc_score(tree, rule_weights, rule_train_index)
        motif_bundle = []
        regulator_bundle = []
        
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

    level='QUIET'
    ### Parse arguments
    log('parse args start', level=level)
    (x1, x2, y, holdout) = parse_args()
    log('parse args end', level=level)

    ### Create tree object
    log('make tree start', level=level)
    tree = DecisionTree(holdout, y, x1, x2)
    log('make tree stop', level=level)

    ### Main Loop
    for i in xrange(1,config.TUNING_PARAMS.num_iter):

        log('iteration {0}'.format(i))
        
        (motif, regulator, best_split, 
         motif_bundle, regulator_bundle, 
         rule_train_index, rule_test_index, rule_score, 
         above_motifs, above_regs) = find_next_decision_node_stable(
             tree, holdout, y, x1, x2, i)
        
        ### Add the rule with best loss
        tree.add_rule(motif, regulator, best_split, 
                      motif_bundle, regulator_bundle, 
                      rule_train_index, rule_test_index, rule_score, 
                      above_motifs, above_regs, holdout, y)

        ### Print progress
        util.log_progress(tree, i)

    pdb.set_trace()

    ### Get plot label so plot label uses parameters used
    method_label=plot.get_plot_label()

    # Save tree state
    save_tree_state(tree, pickle_file='{0}saved_trees/{1}_saved_tree_state_{2}_{3}iter'.format(
        config.OUTPUT_PATH, config.OUTPUT_PREFIX, method_label, config.TUNING_PARAMS.num_iter))

    ### Write out rules
    out_file_name='{0}global_rules/{1}_tree_rules_{2}_{3}iter.txt'.format(
        config.OUTPUT_PATH, config.OUTPUT_PREFIX, 
        method_label, config.TUNING_PARAMS.num_iter)
    tree.write_out_rules(tree, x1, x2, config.TUNING_PARAMS, method_label, out_file=out_file_name)

    # XX Re-factor based on 
    # Make pool
    pool = multiprocessing.Pool(processes=config.NCPU) # create pool of processes

    # Rank x1, x2, rule and node 
    margin_score.call_rank_by_margin_score(prefix='hema_CMP_v_Mono_1000iter_TFbindingonly',
      methods=['by_node'],
       y=y, x1=x1, x2=x2, tree=tree, pool=pool, 
       x1_feat_file='/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/index_files/hema_CMP_v_Mono_peaks.txt',
       x2_feat_file='/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/index_files/hema_CMP_v_Mono_cells.txt')

    margin_score.call_rank_by_margin_score(prefix='hema_MPP_HSC_v_pHSC_1000iter_TFbindingonly',
      methods=['by_node'],
       y=y, x1=x1, x2=x2, tree=tree, pool=pool, 
       x1_feat_file='/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/index_files/hema_MPP_HSC_v_pHSC_peaks.txt',
       x2_feat_file='/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/index_files/hema_MPP_HSC_v_pHSC_cell_types.txt')

    # # Close pool
    pool.close() # stop adding processes
    pool.join() # wait until all threads are done before going on

    ### Make plots
    plot.plot_margin(tree, method_label, config.TUNING_PARAMS.num_iter)
    plot.plot_balanced_error(tree, method_label, config.TUNING_PARAMS.num_iter)
    plot.plot_imbalanced_error(tree, method_label, config.TUNING_PARAMS.num_iter)



### Main
if __name__ == "__main__":
    main()
