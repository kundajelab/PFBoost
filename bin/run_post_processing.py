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
from boosting_2D import post_process_unsupervised

log = util.log

TuningParams = namedtuple('TuningParams', [
    'num_iter',
    'use_stumps', 'use_stable', 'use_corrected_loss', 'use_prior',
    'eta_1', 'eta_2', 'bundle_max', 'epsilon'
])

PostProcess_Params = namedtuple('PostProcess_Params', [
    'model_path', 'analysis_label',
    'run_margin_score', 'run_disc_margin_score', 
    'return_ex_by_feat_matrix', 'margin_score_methods',
    'condition_feat_file', 'region_feat_file', 
    'condition_feat_file2', 'region_feat_file2', 
    'num_perm', 'split_prom_enh_dist',
    'null_tree_model',
    'run_unsupervised_clustering', 'n_clusters_start',
    'features_to_use', 'clusters_to_write',
    'run_knn_with_examples', 'examples_to_track',
    'number_knneighbors'
])

def parse_args():
    # Get arguments
    parser = argparse.ArgumentParser(description='Arguments for Post-Processing')

    # Model files 
    parser.add_argument('--model-path', 
                        help='path to stored data model')
    parser.add_argument('--analysis-label', 
                        help='label of current analysis (e.g. label for cell/peak subset')

    # Margin score arguments
    parser.add_argument('--run-margin-score', 
                        help='Flag to run margin score', action='store_true')
    parser.add_argument('--run-disc-margin-score', 
                        help='Flag to run discriminative margin score between two index matrices', action='store_true')
    parser.add_argument('--return-ex-by-feat-matrix', 
                        help='Flag to run discriminative margin score between two index matrices', action='store_true')
    parser.add_argument('--condition-feat-file', 
                        help='set of conditions to calculate margin score over', default=None)
    parser.add_argument('--region-feat-file', 
                        help='set of regions (genes, peaks, etc.) to calculate margin score over', default=None)
    parser.add_argument('--condition-feat-file2', 
                        help='set of conditions to calculate margin score over', default=None)
    parser.add_argument('--region-feat-file2', 
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
    parser.add_argument('--features-to-use', 
                        help='comma separated list with options: motif,reg,node,path', default='motif')
    parser.add_argument('--clusters-to-write', 
                        help='options: "none", "all", or a comma separated string of clusters', default='none')

    # K-nearest neighbors Arguments
    parser.add_argument('--run-knn-with-examples', 
                        help='Flag to run unsupervised clustering', action='store_true')
    parser.add_argument('--examples-to-track', 
                        help='comma separated text files with 2 columns tab-separated \
                         files contain sets of peaks and conditions expected to functionally relate \
                         col1: feature label or index, col2: condition label or index', default=None)
    parser.add_argument('--number-knneighbors', 
                        help='number of k-nearest neighbors to return for every example', default=None, type=int)


    # Parse arguments
    args = parser.parse_args()

    # If no subset files, name consistently for generating unsupervised learning
    prefix = 'full_model' if args.condition_feat_file==None and args.region_feat_file==None else args.analysis_label 

    margin_score_methods = args.margin_score_methods.split(',') if args.margin_score_methods != None else None
    features_to_use = args.features_to_use.split(',') if args.features_to_use != None else None
    examples_to_track = args.examples_to_track.split(',') if args.examples_to_track != None else None

    # Store arguments in a named tuple
    PARAMS = PostProcess_Params(
        args.model_path, prefix,
        args.run_margin_score, args.run_disc_margin_score,
        args.return_ex_by_feat_matrix, margin_score_methods,
        args.condition_feat_file, args.region_feat_file,
        args.condition_feat_file2, args.region_feat_file2,
        args.num_perm, args.split_prom_enh_dist,
        args.null_tree_model,
        args.run_unsupervised_clustering, args.n_clusters_start,
        features_to_use, args.clusters_to_write,
        args.run_knn_with_examples,
        examples_to_track, args.number_knneighbors 
        )

    return PARAMS

# Get a dictionary of all index matrices to evaluate
def get_index_mat_dict(index_mat):
        index_mat_dict = {}
        index_mat_dict['all_up'] = util.element_mult(index_mat, y.data==1)
        index_mat_dict['all_down'] = util.element_mult(index_mat, y.data==-1)
        return index_mat_dict

# Get a dictionary of all index matrices to evaluate with enh and prom
def get_index_mat_dict_enh_prom(index_mat_enh, index_math_prom):
        index_mat_dict = {}
        index_mat_dict['enh_up'] = util.element_mult(index_mat_enh, y.data==1)
        index_mat_dict['enh_down'] = util.element_mult(index_mat_enh, y.data==-1)
        index_mat_dict['prom_up'] = util.element_mult(index_mat_prom, y.data==1)
        index_mat_dict['prom_down'] = util.element_mult(index_mat_prom, y.data==-1)
        return index_mat_dict

# Check if any index matrices are empty
def check_empty_index_mat(index_mat_dict):
    empty_mat=False
    for key in index_mat_dict.keys():
        if index_mat_dict[key].sum()==0:
            empty_mat=True
    return empty_mat


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
    config.NCPU=4

    # from IPython import embed; embed()
    # pdb.set_trace()
    
    ### Run margin score
    if PARAMS.run_margin_score:

        # Get index matrix
        index_mat = margin_score.get_index(y, x1, x2, tree,
         condition_feat_file=PARAMS.condition_feat_file, region_feat_file=PARAMS.region_feat_file)

        # Keep promoters and enhancers together 
        if PARAMS.split_prom_enh_dist == None:
            index_mat_dict = get_index_mat_dict(index_mat)

        # Split promoters and enhancers 
        if PARAMS.split_prom_enh_dist != None:
            tss_file="""/mnt/data/annotations/by_release/hg19.GRCh37/GENCODE_ann/
                gencodeTSS/v19/TSS_human_strict_with_gencodetss_notlow_ext50eachside
            _merged_withgenctsscoord_andgnlist.gff.gz""".replace('\n','').replace(' ', '')
            (index_mat_prom, index_mat_enh) = margin_score.split_index_mat_prom_enh(
                index_mat, y, tss_file)
            index_mat_dict = get_index_mat_dict_enh_prom(index_mat_enh, index_math_prom)
            # Return error if matrix is empty
            if check_empty_index_mat(index_mat_dict):
                print "No enhancers found - empty index matrix. Try without separating enh/prom."
                return 0

        # Create a pool to calculate all margin scores efficiently
        pool='serial' # REMOVE

        # Iterate through all methods and index matrices
        for method in PARAMS.margin_score_methods:
            for key in index_mat_dict.keys():
                margin_score.call_rank_by_margin_score(index_mat_dict[key], 
                    key, method, PARAMS.analysis_label,
                    y, x1, x2, tree, pool, 
                    num_perm=PARAMS.num_perm, 
                    null_tree_file=PARAMS.null_tree_model)

        print 'DONE: margin scores in {0}{1}/margin_scores/'.format(
            config.OUTPUT_PATH, config.OUTPUT_PREFIX)

    # Find discriminative features between two sets of conditions
    if PARAMS.run_disc_margin_score:

        # Get index matrix
        index_mat1 = margin_score.get_index(y, x1, x2, tree,
         condition_feat_file=PARAMS.condition_feat_file, region_feat_file=PARAMS.region_feat_file)
        index_mat2 = margin_score.get_index(y, x1, x2, tree,
         condition_feat_file=PARAMS.condition_feat_file2, region_feat_file=PARAMS.region_feat_file2)

        # Keep promoters and enhancers together 
        if PARAMS.split_prom_enh_dist == None:
            index_mat_dict1 = get_index_mat_dict(index_mat1)
            index_mat_dict2 = get_index_mat_dict(index_mat2)

        # Split promoters and enhancers 
        if PARAMS.split_prom_enh_dist != None:
            tss_file="""/mnt/data/annotations/by_release/hg19.GRCh37/GENCODE_ann/
                gencodeTSS/v19/TSS_human_strict_with_gencodetss_notlow_ext50eachside
            _merged_withgenctsscoord_andgnlist.gff.gz""".replace('\n','').replace(' ', '')
            (index_mat_prom1, index_mat_enh1) = margin_score.split_index_mat_prom_enh(
                index_mat1, y, tss_file)
            (index_mat_prom2, index_mat_enh2) = margin_score.split_index_mat_prom_enh(
                index_mat2, y, tss_file)
            index_mat_dict1 = get_index_mat_dict_enh_prom(index_mat_enh1, index_math_prom1)
            index_mat_dict2 = get_index_mat_dict_enh_prom(index_mat_enh2, index_math_prom2)
            # Return error if matrix is empty
            if check_empty_index_mat(index_mat_dict1):
                print "No enhancers found - empty index matrix 1. Try without separating enh/prom."
                return 0
            if check_empty_index_mat(index_mat_dict2):
                print "No enhancers found - empty index matrix 2. Try without separating enh/prom."
                return 0

        # Create a pool to calculate all margin scores efficiently
        pool='serial' # REMOVE

        # Iterate through all methods and index matrices
        for method in PARAMS.margin_score_methods:
            for key in index_mat_dict1.keys():
                margin_score.call_discriminate_margin_score(index_mat_dict1[key], index_mat_dict2[key],
                    key, method, PARAMS.analysis_label,
                    y, x1, x2, tree, pool, 
                    num_perm=PARAMS.num_perm, 
                    null_tree_file=PARAMS.null_tree_model)

        print 'DONE: margin scores in {0}{1}/disc_margin_scores/'.format(
            config.OUTPUT_PATH, config.OUTPUT_PREFIX)

    ### Run unsupervised clustering
    if PARAMS.run_unsupervised_clustering:
        # Get clusters
        print ('Beginning clustering, this may take up to several hours '
        'depending on the size of the matrix')
        (cluster_file, new_clusters) = post_process_unsupervised.cluster_examples_kmeans(
         y, x1, x2, tree, n_clusters_start=PARAMS.n_clusters_start,
          mat_features=PARAMS.features_to_use)
        pdb.set_trace()
        # Write out bed files with each cluster
        if PARAMS.clusters_to_write!='none':
            write_out_cluster(y, cluster_file, new_clusters,
             clusters_to_write=PARAMS.clusters_to_write, create_match_null=True)
        # Track examples
        for f in PARAMS.examples_to_track:
            post_process_unsupervised.knn(f)
        
        print 'DONE: clusters in {0}{1}/clustering/'.format(
            config.OUTPUT_PATH, config.OUTPUT_PREFIX)

    ### Run KNN with the provided example
    if PARAMS.run_knn_with_examples:
        # get KNN for every example provided by users
        for ex_file in PARAMS.examples_to_track:
            knn_dict = post_process_unsupervised.knn(
                ex_file, y, x1, x2, tree, num_neighbors=PARAMS.number_knneighbors)
            # Write out KNN to file
            knn_path='{0}{1}/knn/'.format(
                config.OUTPUT_PATH, config.OUTPUT_PREFIX)
            if not os.path.exists(knn_path):
                os.makedirs(knn_path)
            post_process_unsupervised.write_knn(y=y, ex_file=ex_file, 
                knn_dict=knn_dict, output_path=knn_path)
            print 'Finished file {0}'.format(ex_file)

        print 'DONE: k-nearest neighbors in {0}{1}/knn/'.format(
            config.OUTPUT_PATH, config.OUTPUT_PREFIX)        

    ### Return matrix of normalized margin scores for desired index
    if PARAMS.return_ex_by_feat_matrix:

        ### Set up output folder
        mat_outdir = '{0}{1}/ex_by_feat_matrix/'.format(
            config.OUTPUT_PATH, config.OUTPUT_PREFIX)
        if not os.path.exists(mat_outdir):
            os.makedirs(mat_outdir)

        print "getting example-by-feature matrix"
        # get example by feature matrix
        ex_by_feat_mat = post_process_unsupervised.gen_ex_by_feature_matrix(
            y, x1, x2, tree, feat=PARAMS.features_to_use)

        print "subsetting example-by-feature matrix"
        pdb.set_trace()
        # subset matrix to relevant features
        sub_ex_by_feat_df = post_process_unsupervised.subset_ex_by_feature_matrix(
            ex_by_feat_mat, y, x1, condition_feat_file=PARAMS.condition_feat_file, 
            region_feat_file=PARAMS.region_feat_file, feat=PARAMS.features_to_use,
            remove_zeros=True)

        print "writing example-by-feature matrix"
        # Write matrix out
        sub_ex_by_feat_df.to_csv('{0}{1}_example_by_feature_matrix.txt'.format(
            mat_outdir, PARAMS.analysis_label), 
            sep="\t", index=True, header=True)

        print 'DONE: example by feature matrix in {0}'.format(mat_outdir)        

### Main
if __name__ == "__main__":
    main()


