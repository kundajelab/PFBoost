### Peyton Greenside
### Script to run desired post processing modules (replaces load_data.py)
### 8/15/15

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

log = util.log

TuningParams = namedtuple('TuningParams', [
    'num_iter',
    'use_stumps', 'use_stable', 'use_corrected_loss', 'use_prior',
    'eta_1', 'eta_2', 'bundle_max', 'epsilon'
])

PostProcess_Params = namedtuple('PostProcess_Params', [
    'model_path', 'margin_score_prefix',
    'run_margin_score', 'margin_score_methods',
    'condition_feat_file', 'region_feat_file', 
    'num_perm', 'split_prom_enh_dist',
    'null_tree_model'
])


def parse_args():
    # Get arguments
    parser = argparse.ArgumentParser(description='Arguments for Post-Processing')

	# Model files 
    parser.add_argument('--model-path', 
                        help='path to stored data model')
    parser.add_argument('--margin-score-prefix', 
                        help='path to stored data model')

    # Margin score arguments
    parser.add_argument('--run-margin-score', 
                        help='Flag to run margin sccore', action='store_true')
    parser.add_argument('--condition-feat-file', 
                        help='set of conditions to calculate margin score over', default=None)
    parser.add_argument('--region-feat-file', 
                        help='set of regions (genes, peaks, etc.) to calculate margin score over', default=None)
    parser.add_argument('--margin-score-methods', 
                        help='choices of: x1, x2, node, path. can combine several comma-separated.', default=None)
    parser.add_argument('--num-perm', 
                        help='number of permutations when computing margin score', default=100, type=int)
    parser.add_argument('--split-prom-enh-dist', 
                        help='distance of element to promoter to split on', default=None, type=int)
    parser.add_argument('--null-tree-model', 
                        help='pickled tree file of NULL model', default=None)
    # parser.add_argument('--tss_file', 
    #                     help='file containing promoter start sites for splitting index matrix', default=None) #ADD


    # Parse arguments
    args = parser.parse_args()

    # If no subset files, name consistently for generating unsupervised learning
    prefix = 'full_model' if args.condition_feat_file==None and args.region_feat_file==None else args.margin_score_prefix 

    # Store arguments in a named tuple
    PARAMS = PostProcess_Params(
        args.model_path, prefix,
        args.run_margin_score, args.margin_score_methods.split(','),
        args.condition_feat_file, args.region_feat_file,
        args.num_perm, args.split_prom_enh_dist,
        args.null_tree_model
        )

    return PARAMS


# Run main loop
def main():
    print 'starting main loop'

    # Parse arguments
    log('parse args start')
    PARAMS = parse_args()
    log('parse args end')
    
    # Load tree file and update globals with file state
    locals_dict = {}
    execfile(PARAMS.model_path, {}, locals_dict)
    globals().update(locals_dict)

    # Stop
    from IPython import embed; embed()
    # pdb.set_trace()
    
    ### Run margin score
    if PARAMS.run_margin_score:

        # Get index matrix
        index_mat = margin_score.get_index(y, x1, x2, tree,
         condition_feat_file=PARAMS.condition_feat_file, region_feat_file=PARAMS.region_feat_file)

        # Keep promoters and enhancers together 
        if PARAMS.split_prom_enh_dist == None:
            index_mat_dict = {}
            index_mat_dict['all_up'] = util.element_mult(index_mat, y.data==1)
            index_mat_dict['all_down'] = util.element_mult(index_mat, y.data==-1)

        # Split promoters and enhancers 
        if PARAMS.split_prom_enh_dist != None:
            tss_file="""/mnt/data/annotations/by_release/hg19.GRCh37/GENCODE_ann/
                gencodeTSS/v19/TSS_human_strict_with_gencodetss_notlow_ext50eachside
            _merged_withgenctsscoord_andgnlist.gff.gz""".replace('\n','').replace(' ', '')
            (index_mat_prom, index_mat_enh) = margin_score.split_index_mat_prom_enh(
                index_mat, y, tss_file)
            index_mat_dict = {}
            index_mat_dict['enh_up'] = util.element_mult(index_mat_enh, y.data==1)
            index_mat_dict['enh_down'] = util.element_mult(index_mat_enh, y.data==-1)
            index_mat_dict['prom_up'] = util.element_mult(index_mat_prom, y.data==1)
            index_mat_dict['prom_down'] = util.element_mult(index_mat_prom, y.data==-1)

        # Create a pool to calculate all margin scores efficiently
        # pool = multiprocessing.Pool(processes=config.NCPU) # create pool of processes
        pool='serial'

        # Iterate through all methods and index matrices
        for method in PARAMS.margin_score_methods:
            for key in index_mat_dict.keys():
                margin_score.call_rank_by_margin_score(index_mat_dict[key], key, method, PARAMS.margin_score_prefix,
                y, x1, x2, tree, pool,
                num_perm=PARAMS.num_perm, null_tree_file=PARAMS.null_tree_model)

        # Close pool
        if pool!='serial':
            pool.join()
            pool.close()

        print 'DONE: margin scores in {0}{1}/margin_scores/'.format(config.OUTPUT_PATH, config.OUTPUT_PREFIX)



### Main
if __name__ == "__main__":
    main()


