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
from boosting_2D import post_processing_unsupervised

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
                        help='Flag to run margin score', action='store_true')
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

    # Unsupervised Learning Arguments
    parser.add_argument('--run-unsupervised-clustering', 
                        help='Flag to run unsupervised clustering', action='store_true')
    parser.add_argument('--n-clusters-start', 
                        help='Number of k-means clusters to start with sofiaML', default=5000, type=int)
    parser.add_argument('--features-for-kmeans', 
                        help='comma separated list with options: motif,reg,node,path', default='motif')
    parser.add_argument('--run-knn-with-examples', 
                        help='Flag to run unsupervised clustering', action='store_true')
    parser.add_argument('--examples-to-track', 
                        help='comma separated text files with 2 columns tab-separated \
                         files contain sets of peaks and conditions expected to functionally relate \
                         col1: feature label or index, col2: condition label or index', default=None)
    parser.add_argument('--number-knneighbors', 
                        help='number of k-nearest neighbors to return for every example', default=None)


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
        args.null_tree_model,
        args.run_unsupervised_clustering, args.n_clusters_start,
        args.features_for_kmeans.split(','),
        args.run_knn_with_examples,
        args.examples_to_track.split(','), args.number_knneighbors 
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
    config.NCPU=8

    # from IPython import embed; embed()
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
            # Return error if enh matrix is empty
            if index_mat_dict['enh_up'].sum()==0 or index_mat_dict['enh_down'].sum()==0:
                print "No enhancers found - empty index matrix. Try without separating enh/prom."
                return 0
            if index_mat_dict['prom_up'].sum()==0 or index_mat_dict['prom_down'].sum()==0:
                print "No promoters found - empty index matrix. Try without separating enh/prom."
                return 0

        # Create a pool to calculate all margin scores efficiently
        pool='serial' # REMOVE

        # Iterate through all methods and index matrices
        for method in PARAMS.margin_score_methods:
            for key in index_mat_dict.keys():
                margin_score.call_rank_by_margin_score(index_mat_dict[key], 
                    key, method, PARAMS.margin_score_prefix,
                    y, x1, x2, tree, pool, 
                    num_perm=PARAMS.num_perm, 
                    null_tree_file=PARAMS.null_tree_model)

        print 'DONE: margin scores in {0}{1}/margin_scores/'.format(
            config.OUTPUT_PATH, config.OUTPUT_PREFIX)


    ### Run unsupervised clustering
    if PARAMS.run_unsupervised_clustering:
        # Get clusters
        print ('Beginning clustering, this may take up to several hours '
        'depending on the size of the matrix')
        (cluster_file, new_clusters) = post_processing_unsupervised.cluster_examples_kmeans(
         y, x1, x2, tree, n_clusters_start=PARAMS.n_clusters_start,
          mat_features=PARAMS.features_for_kmeans)
        # Write out bed files with each cluster
        write_out_cluster(y, cluster_file, new_clusters,
         clusters_to_write='all', create_match_null=True)
        # Track examples
        for f in PARAMS.examples_to_track:
            post_processing_unsupervised.knn(f)
        
        print 'DONE: clusters in {0}{1}/clustering/'.format(
            config.OUTPUT_PATH, config.OUTPUT_PREFIX)

    ### Run KNN with the provided example
    if PARAMS.run_knn_with_examples:
        # get KNN for every example provided by users
        knn_dict = post_processing_unsupervised.knn(
            ex_file, y, x1, x2, tree, ex_by_feat_mat)
        # Write out KNN to file
        knn_path='{0}{1}/knn/'.format(
            config.OUTPUT_PATH, config.OUTPUT_PREFIX)
        if not os.path.exists(knn_path):
            os.makedirs(knn_path)
        post_processing_unsupervised.write_knn(ex_file=ex_file, 
            knn_dict=knn_dict, output_path=knn_path)

        print 'DONE: k-nearest neighbors in {0}{1}/knn/'.format(
            config.OUTPUT_PATH, config.OUTPUT_PREFIX)        


### Main
if __name__ == "__main__":
    main()


