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

from boosting_2D import util
from boosting_2D import config

import sklearn.utils
import pickle

### XXX DEPENDENCIES
### Currently uses TSS file on nandi, can add as an argument


###  MARGIN SCORE FUNCTIONS
#######################################################################################

def calc_margin_score_x1_wrapper(args):
    return calc_margin_score_x1(*args)

def calc_margin_score_x1(tree, y, x1, x2, index_mat, x1_feat_index):
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
    unique_bundle_terms=np.unique([x1.row_labels[el] for node in x1_feat_nodes for el in [tree.split_x1[node]]
          +tree.bundle_x1[node] if el != x1_feat_index]).tolist()
    x1_bundle_string = '|'.join(unique_bundle_terms) if len(unique_bundle_terms)>0 else "None"

    # All rules where the motif is above it, used in the split, or bundled in the split
    rules_w_x1_feat = [el for el in xrange(tree.nsplit)
         if x1_feat_index in tree.above_motifs[el]
         +[tree.split_x1[el]]
         +tree.bundle_x1[el]]

    # For each cell type comparison, print the rules that apply 
    # for ind in range(y.num_col): print ind; print np.nonzero([np.apply_along_axis(np.sum, 0, tree.ind_pred_train[rule].toarray())[ind] for rule in range(1,tree.nsplit)])
    # for rule in range(tree.nsplit): a = np.apply_along_axis(np.sum, 0, tree.ind_pred_train[rule].toarray()); print a[1]

    # New predictions without rule (adjusted prediction = pred_adj)
    pred_adj = tree.pred_train
    for rule in rules_w_x1_feat:
        pred_adj = pred_adj - tree.scores[rule]*tree.ind_pred_train[rule]
    margin_score = util.element_mult(y.element_mult(tree.pred_train-pred_adj), index_mat).sum()
    margin_score_norm = margin_score/index_mat.sum()

    # Get all rules where x1 feat is above or in rule or bundle
    rule_index_mat = tree.ind_pred_train[rules_w_x1_feat[0]]
    for r in rules_w_x1_feat:
        rule_index_mat = rule_index_mat + tree.ind_pred_train[r]
    # Index where x1 feat is used 
    rule_index_mat = (rule_index_mat>0)
    # Index where x1 feat is used and examples of interest
    rule_index_joint = util.element_mult(index_mat, rule_index_mat)
    # Fraction of examples of interest where x1 feat used
    rule_index_fraction = float(rule_index_joint.sum())/index_mat.sum()
    # print rule_index_fraction
    return [x1_feat_name, x1_bundle_string, margin_score, margin_score_norm, rule_index_fraction]

def calc_margin_score_x2_wrapper(args):
    return calc_margin_score_x2(*args)

# calc_margin_score_x2(tree, y, x1, x2, csr_matrix(np.ones((y.num_row, y.num_col))), 'YDR085C')
def calc_margin_score_x2(tree, y, x1, x2, index_mat, x2_feat_index):
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
    margin_score = util.element_mult(y.element_mult(tree.pred_train-pred_adj), index_mat).sum()
    margin_score_norm = margin_score/index_mat.sum()

    # Get all rules where x1 feat is above or in rule or bundle
    rule_index_mat = tree.ind_pred_train[rules_w_x2_feat[0]]
    for r in rules_w_x2_feat:
        rule_index_mat = rule_index_mat + tree.ind_pred_train[r]
    # Index where x1 feat is used 
    rule_index_mat = (rule_index_mat>0)
    # Index where x1 feat is used and examples of interest
    rule_index_joint = util.element_mult(index_mat, rule_index_mat)
    # Fraction of examples of interest where x1 feat used
    rule_index_fraction = float(rule_index_joint.sum())/index_mat.sum()
    # print rule_index_fraction
    return [x2_feat_name, x2_bundle_string, margin_score, margin_score_norm, rule_index_fraction]

def calc_margin_score_rule_wrapper(args):
    return calc_margin_score_rule(*args)

### CALCULATE MARGIN SCORE BY ANY JOINT APPEARANCE OF MOTIF-REGULATOR (MULTIPLE NODES)
def calc_margin_score_rule(tree, y, x1, x2, index_mat, x1_feat_index, x2_feat_index):
    # Feature names
    x1_feat_name = x1.row_labels[x1_feat_index]
    x2_feat_name = x2.col_labels[x2_feat_index]

    # # All rules where the x1/x2 feat is not above it, used in the split, or bundled in the split
    # rules_w_x1_feat = [el for el in xrange(tree.nsplit)
    #      if x1_feat_index in tree.above_motifs[el]
    #      +[tree.split_x1[el]]
    #      +tree.bundle_x1[el]]
    # rules_w_x2_feat = [el for el in xrange(tree.nsplit)
    #      if x2_feat_index in tree.above_regs[el]
    #      +[tree.split_x2[el]]
    #      +tree.bundle_x2[el]]
    # pair_rules = np.unique(rules_w_x1_feat+rules_w_x2_feat).tolist()
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
    rule_index_mat = tree.ind_pred_train[rules_w_x1_feat_and_x2_feat[0]]
    for r in rules_w_x1_feat_and_x2_feat:
        rule_index_mat = rule_index_mat + tree.ind_pred_train[r]
    # Index where rule  is used 
    rule_index_mat = (rule_index_mat>0)
    # Index where rules is used and examples of interest
    rule_index_joint = util.element_mult(index_mat, rule_index_mat)
    # Fraction of examples of interest where rule used
    rule_index_fraction = float(rule_index_joint.sum())/index_mat.sum()
    # print rule_index_fraction
    return [x1_feat_name, x1_bundle_string, x2_feat_name, x2_bundle_string, margin_score, margin_score_norm, rule_index_fraction]


def calc_margin_score_node_wrapper(args):
    return calc_margin_score_node(*args)

## Calculate margin score for each individual node
def calc_margin_score_node(tree, y, x1, x2, index_mat, node):
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

    # All rules where the node is not above it or the node itself
    subtract_rules = [el for el in xrange(tree.nsplit)
         if node in tree.above_nodes[el] if el!=node]+[node]

    # Get Prediction Matrix
    pred_adj = tree.pred_train
    for rule in subtract_rules:
        pred_adj = pred_adj - tree.scores[rule]*tree.ind_pred_train[rule]
    margin_score = util.element_mult(y.element_mult(tree.pred_train-pred_adj), index_mat).sum()
    margin_score_norm = margin_score/index_mat.sum()

    ### Chosen node and all nodes added below 
    rules_w_node = [el for el in xrange(tree.nsplit)
         if node in tree.above_nodes[el] or el==node]
    rule_index_mat = tree.ind_pred_train[rules_w_node[0]]
    for r in rules_w_node:
        rule_index_mat = rule_index_mat + tree.ind_pred_train[r]

    # Index where rule  is used 
    rule_index_mat = (rule_index_mat>0)
    # Index where rules is used and examples of interest
    rule_index_joint = util.element_mult(index_mat, rule_index_mat)
    # Fraction of examples of interest where rule used
    rule_index_fraction = float(rule_index_joint.sum())/index_mat.sum()

    # print rule_index_fraction
    return [node, x1_feat_name, x1_bundle_string, x2_feat_name, x2_bundle_string, margin_score, margin_score_norm, rule_index_fraction, direction]

def rank_by_margin_score(tree, y, x1, x2, index_mat, pool, method):
    assert method in ('by_x1', 'by_x2', 'by_x1_and_x2', 'by_node')
    # Rank x1 features only
    if method=='by_x1':
        print 'by_x1'
        # All x1 features used in a tree, not equal to root
        used_x1_feats = np.unique([el for el in tree.split_x1 if el != 'root']+ \
            [el for listy in tree.bundle_x1 for el in listy if el != 'root']).tolist()
        ### SERIAL VERSION
        rule_processes = []    
        for feat in used_x1_feats:
            result=calc_margin_score_x1(tree, y, x1, x2, index_mat, feat)
            rule_processes.append(result)
        # Get margin score for each x1 features
        # rule_processes = pool.map(calc_margin_score_x1_wrapper, iterable=[ \
        #     (tree, y, x1, x2, index_mat, m) \
        #     for m in used_x1_feats])
        # Report data frame with feature 
        ranked_score_df = pd.DataFrame({'x1_feat':[el[0] for el in rule_processes], \
            'x1_feat_bundles':[el[1] for el in rule_processes], \
            'margin_score':[el[2] for el in rule_processes], \
            'margin_score_norm':[el[3] for el in rule_processes], \
            'rule_index_fraction':[el[4] for el in rule_processes]}).sort(columns=['margin_score'], ascending=False)
    # Rank x2 features only
    if method=='by_x2':
        print 'by_x2'
        # All x2 features used in a treem, not equal to root
        used_x2_feats = np.unique([el for el in tree.split_x2 if el != 'root']+ \
                [el for listy in tree.bundle_x2 for el in listy if el != 'root']).tolist()
        rule_processes = []    
        for feat in used_x2_feats:
            result=calc_margin_score_x2(tree, y, x1, x2, index_mat, feat)
            rule_processes.append(result)
        # Get margin score for each x2 feature
        # rule_processes = pool.map(calc_margin_score_x2_wrapper, iterable=[ \
        #     (tree, y, x1, x2, index_mat, r) \
        #     for r in used_x2_feats])
        # Report data frame with feature 
        ranked_score_df = pd.DataFrame({'x2_feat':[el[0] for el in rule_processes], \
            'x2_feat_bundles':[el[1] for el in rule_processes], \
            'margin_score':[el[2] for el in rule_processes], \
            'margin_score_norm':[el[3] for el in rule_processes], \
            'rule_index_fraction':[el[4] for el in rule_processes]}).sort(columns=['margin_score'], ascending=False)
    # Rank by rules 
    if method=='by_x1_and_x2':
        print 'by_x1_and_x2'
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
        # Get margin score for each rule
        # rule_processes = pool.map(calc_margin_score_rule_wrapper, iterable=[ \
        #     (tree, y, x1, x2, index_mat, used_x1_feats[i], used_x2_feats[i])  \
        #     for i in xrange(len(used_x2_feats))])

        # Report data frame with feature 
        ranked_score_df = pd.DataFrame({'x1_feat':[el[0] for el in rule_processes], \
            'x1_feat_bundles':[el[1] for el in rule_processes], \
            'x2_feat':[el[2] for el in rule_processes], \
            'x2_feat_bundles':[el[3] for el in rule_processes], \
            'margin_score':[el[4] for el in rule_processes], \
            'margin_score_norm':[el[5] for el in rule_processes], \
            'rule_index_fraction':[el[6] for el in rule_processes]}).sort(columns=['margin_score'], ascending=False)
    if method=='by_node':
        print 'by_node'
        # Get margin score for each rule
        # rule_processes = pool.map(calc_margin_score_node_wrapper, iterable=[ \
        #     (tree, y, x1, x2, index_mat, node)  \
        #     for node in xrange(1,tree.nsplit)])
        # SERIAL VERSION
        rule_processes = []    
        for node in xrange(1,tree.nsplit):
            result=calc_margin_score_node(tree, y, x1, x2, index_mat, node)
            rule_processes.append(result)
        # Report data frame with feature 
        ranked_score_df = pd.DataFrame({'node':[el[0] for el in rule_processes], \
            'x1_feat':[el[1] for el in rule_processes], \
            'x1_feat_bundles':[el[2] for el in rule_processes], \
            'x2_feat':[el[3] for el in rule_processes], \
            'x2_feat_bundles':[el[4] for el in rule_processes], \
            'margin_score':[el[5] for el in rule_processes], \
            'margin_score_norm':[el[6] for el in rule_processes], \
            'rule_index_fraction':[el[7] for el in rule_processes], \
            'direction':[el[8] for el in rule_processes]}).sort(columns=['margin_score'], ascending=False)
    ranked_score_df.drop_duplicates()
    # Return matrix
    return ranked_score_df


###  FUNCTIONS TO GET INDEX
#######################################################################################

### Get the index with a text file or document of of interest
### Takes either index or the names of the features y.row_labels or y.col_labels
def get_index(y, x1, x2, tree, x1_feat_file=None, x2_feat_file=None):
    if x1_feat_file!=None:
        x1_file = pd.read_table(x1_feat_file, header=None)
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
    if x2_feat_file!=None:
        x2_file = pd.read_table(x2_feat_file, header=None)
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
    index_mat_prom = index_mat.multiply(prom_mat)
    index_mat_enh = index_mat.multiply(enh_mat)
    # Remove the coordinate and intersection files
    os.system('rm {0}'.format(temp_bed_file))
    os.system('rm {0}'.format(temp_intersect_file))
    # Reurn index mats for promoter and enhancer regions
    return (index_mat_prom, index_mat_enh)

### Call feature ranking
def call_rank_by_margin_score(prefix, methods, y, x1, x2, tree, pool, num_perm=100, x1_feat_file=None, x2_feat_file=None, split_prom_enh=True, null_tree_file=None
):
    # Create directory for this analysis
    margin_outdir = '{0}/margin_scores/{1}/'.format(config.OUTPUT_PATH, prefix)
    if not os.path.exists(margin_outdir):
        os.makedirs(margin_outdir)
    # get index matrix from feature files
    index_mat = get_index(y, x1, x2, tree, x1_feat_file=x1_feat_file, x2_feat_file=x2_feat_file)
    # Initialize dictionary of data frames
    dict_names = ['up', 'down']
    rank_score_df_dict = {}
    for name in dict_names: rank_score_df_dict[name]=0
    # keep promoters and enhancers together
    if split_prom_enh==False:
        print 'calculating margin score with promoters and enhancers together'
        index_mat_up = util.element_mult(index_mat, y.data==1)
        index_mat_down = util.element_mult(index_mat, y.data==-1)
        if 'by_x1' in methods:
            for key in rank_score_df_dict.keys():
                rank_score_df_dict[key] = rank_by_margin_score(tree, y, x1, x2, eval('index_mat_{0}'.format(key)), pool, method='by_x1')
                rank_score_df_dict[key].to_csv('{0}{1}_{2}_top_x1_feat.txt'.format(margin_outdir, prefix, key.upper()), 
                    sep="\t", index=None, header=True)
        if 'by_x2' in methods:
            for key in rank_score_df_dict.keys():
                rank_score_df_dict[key] = rank_by_margin_score(tree, y, x1, x2, eval('index_mat_{0}'.format(key)), pool, method='by_x2')
                rank_score_df_dict[key].to_csv('{0}{1}_{2}_top_x2_feat.txt'.format(margin_outdir, prefix, key.upper()), 
                    sep="\t", index=None, header=True)
        if 'by_x1_and_x2' in methods:
            for key in rank_score_df_dict.keys():
                rank_score_df_dict[key] = rank_by_margin_score(tree, y, x1, x2, eval('index_mat_{0}'.format(key)), pool, method='by_x1_and_x2')
                rank_score_df_dict[key].to_csv('{0}{1}_{2}_top_x1_and_x2_joint_feat.txt'.format(margin_outdir, prefix, key.upper()), 
                    sep="\t", index=None, header=True)
        if 'by_node' in methods:
            for key in rank_score_df_dict.keys():
                rank_score_df_dict[key] = rank_by_margin_score(tree, y, x1, x2, eval('index_mat_{0}'.format(key)), pool, method='by_node')
                rank_score_df_dict[key].to_csv('{0}{1}_{2}_top_nodes.txt'.format(margin_outdir, prefix, key.upper()), 
                    sep="\t", index=None, header=True)
    # separate promoters and enhancers 
    if split_prom_enh==True:
        print 'separating index matrix into promoters and enhancers'
        ### SOFT CODE
        tss_file="""/mnt/data/annotations/by_release/hg19.GRCh37/GENCODE_ann/
        gencodeTSS/v19/TSS_human_strict_with_gencodetss_notlow_ext50eachside
        _merged_withgenctsscoord_andgnlist.gff.gz""".replace('\n','').replace(' ', '')
        (index_mat_prom, index_mat_enh) = split_index_mat_prom_enh(index_mat, y, tss_file)
        index_mat_prom_up = util.element_mult(index_mat_prom, y.data==1)
        index_mat_prom_down = util.element_mult(index_mat_prom, y.data==-1)
        index_mat_enh_up = util.element_mult(index_mat_enh, y.data==1)
        index_mat_enh_down = util.element_mult(index_mat_enh, y.data==-1)
        # initialize dictionary of all rank score matrices
        dict_names = ['prom_up', 'prom_down', 'enh_up', 'enh_down']
        rank_score_df_dict = {}
        for name in dict_names: rank_score_df_dict[name]=0
        # For each method, get the margin score matrix and write to file
        if 'by_x1' in methods:
            for key in rank_score_df_dict.keys():
                print key
                y_value = +1 if 'up' in key else -1
                rank_score_df_dict[key] = rank_by_margin_score(tree, y, x1, x2, eval('index_mat_{0}'.format(key)), pool, method='by_x1')
                if num_perm>0:
                    if null_tree_file==None:
                        rank_score_df_w_perm = calculate_null_margin_score_dist_by_shuffling_target(rank_score_df_dict[key], 
                            eval('index_mat_{0}'.format(key)), 'by_x1', pool, num_perm, tree, y, x1, x2, y_value)
                    else:
                        rank_score_df_w_perm = calculate_null_margin_score_dist_from_NULL_tree(rank_score_df_dict[key], 
                            eval('index_mat_{0}'.format(key)), 'by_x1', pool, num_perm, tree, y, x1, x2, y_value, null_tree_file)                        
                    rank_score_df_w_perm.to_csv('{0}{1}_{2}_top_x1_feat_{3}_permutations.txt'.format(margin_outdir, prefix, key.upper(), num_perm), 
                        sep="\t", index=None, header=True)
                else:
                    rank_score_df_dict[key].to_csv('{0}{1}_{2}_top_x1_feat.txt'.format(margin_outdir, prefix, key.upper()), 
                        sep="\t", index=None, header=True)

        if 'by_x2' in methods:
            for key in rank_score_df_dict.keys():
                y_value = +1 if 'up' in key else -1
                rank_score_df_dict[key] = rank_by_margin_score(tree, y, x1, x2, eval('index_mat_{0}'.format(key)), pool, method='by_x2')
                if num_perm>0:
                    if null_tree_file==None:
                        rank_score_df_w_perm = calculate_null_margin_score_dist_by_shuffling_target(rank_score_df_dict[key], 
                            eval('index_mat_{0}'.format(key)), 'by_x2', pool, num_perm, tree, y, x1, x2, y_value)
                    else:
                        rank_score_df_w_perm = calculate_null_margin_score_dist_from_NULL_tree(rank_score_df_dict[key], 
                            eval('index_mat_{0}'.format(key)), 'by_x2', pool, num_perm, tree, y, x1, x2, y_value, null_tree_file)
                    rank_score_df_w_perm.to_csv('{0}{1}_{2}_top_x2_feat_{3}_permutations.txt'.format(margin_outdir, prefix, key.upper(), num_perm), 
                        sep="\t", index=None, header=True)
                else:
                    rank_score_df_dict[key].to_csv('{0}{1}_{2}_top_x2_feat.txt'.format(margin_outdir, prefix, key.upper()), 
                        sep="\t", index=None, header=True)

        if 'by_x1_and_x2' in methods:
            for key in rank_score_df_dict.keys():
                y_value = +1 if 'up' in key else -1
                rank_score_df_dict[key] = rank_by_margin_score(tree, y, x1, x2, eval('index_mat_{0}'.format(key)), pool, method='by_x1_and_x2')
                if num_perm>0:
                    if null_tree_file==None:
                        rank_score_df_w_perm = calculate_null_margin_score_dist_by_shuffling_target(rank_score_df_dict[key], 
                            eval('index_mat_{0}'.format(key)), 'by_x1_and_x2', pool, num_perm, tree, y, x1, x2, y_value)
                    else:
                        rank_score_df_w_perm = calculate_null_margin_score_dist_from_NULL_tree(rank_score_df_dict[key], 
                            eval('index_mat_{0}'.format(key)), 'by_x1_and_x2', pool, num_perm, tree, y, x1, x2, y_value, null_tree_file)
                    rank_score_df_w_perm.to_csv('{0}{1}_{2}_top_x1_and_x2_joint_feat_{3}_permutations.txt'.format(margin_outdir, prefix, key.upper(), num_perm), 
                        sep="\t", index=None, header=True)
                else:
                    rank_score_df_dict[key].to_csv('{0}{1}_{2}_top_x1_and_x2_joint_feat.txt'.format(margin_outdir, prefix, key.upper()), 
                        sep="\t", index=None, header=True)
        if 'by_node' in methods:
            for key in rank_score_df_dict.keys():
                y_value = +1 if 'up' in key else -1
                rank_score_df_dict[key] = rank_by_margin_score(tree, y, x1, x2, eval('index_mat_{0}'.format(key)), pool, method='by_node')
                if num_perm>0:
                    if null_tree_file==None:
                        rank_score_df_w_perm = calculate_null_margin_score_dist_by_shuffling_target(rank_score_df_dict[key], 
                            eval('index_mat_{0}'.format(key)), 'by_node', pool, num_perm, tree, y, x1, x2, y_value)
                    else:
                        rank_score_df_w_perm = calculate_null_margin_score_dist_from_NULL_tree(rank_score_df_dict[key], 
                            eval('index_mat_{0}'.format(key)), 'by_node', pool, num_perm, tree, y, x1, x2, y_value, null_tree_file)
                    rank_score_df_w_perm.to_csv('{0}{1}_{2}_top_nodes_{3}_permutations.txt'.format(margin_outdir, prefix, key.upper(), num_perm), 
                        sep="\t", index=None, header=True)
                else:
                    rank_score_df_dict[key].to_csv('{0}{1}_{2}_top_nodes.txt'.format(margin_outdir, prefix, key.upper()), 
                    sep="\t", index=None, header=True)

    return 0

### MARGIN SCORE NULL MODEL 
###########################################################################################################################

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
    if method=='by_node' or method=='by_x1_and_x2':
        value_vec = np.where(ymat==value)
        sample_ind = np.random.choice(range(len(value_vec[0])), np.sum(indmat))
        new_mat[value_vec[0][sample_ind], value_vec[1][sample_ind]]=True        
    return new_mat

### Sample same value from anywhere in matrix
def sample_values_from_data_class(y, index_mat, method, value):
    new_mat = csr_matrix(index_mat.shape, dtype=bool)
    if y.sparse:
        ymat = y.data.toarray()
        indmat = index_mat.toarray()
    else:
        ymat = y.data
        indmat = index_mat
    # Sample same value from anywhere
    value_vec = np.where(ymat==value)
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
    if method=='by_x1':
        # For each permutation calculate margin scores and add to dictionary
        for i in xrange(num_perm):
            # Permute rows
            new_index=sample_values_from_data_class(y=y_null, index_mat=index_mat, method=method, value=y_value)
            margin_score_dict['perm{0}'.format(i)]=rank_by_margin_score(tree, y_null, x1, x2, new_index, pool, method=method)
    elif method=='by_x2':
        # For each permutation calculate margin scores and add to dictionary
        for i in xrange(num_perm):
            # Permute columns
            new_index=sample_values_from_data_class(y=y_null, index_mat=index_mat, method=method, value=y_value)
            margin_score_dict['perm{0}'.format(i)]=rank_by_margin_score(tree, y_null, x1, x2, new_index, pool, method=method)
    elif method=='by_node' or method=='by_x1_and_x2':
        # For each permutation calculate margin scores and add to dictionary
        for i in xrange(num_perm):
            # Permute rows and columns
            new_index=sample_values_from_data_class(y=y_null, index_mat=index_mat, method=method, value=y_value)
            margin_score_dict['perm{0}'.format(i)]=rank_by_margin_score(tree, y_null, x1, x2, new_index, pool, method=method)
    else:
        assert False, "provide method in ['by_x1', 'by_x2', 'by_x1_and_x2', 'by_node']"
    # calculate p-values for each margin score 
    all_margin_scores_ranked = np.sort([el for df in margin_score_dict.values() for el in df.ix[:,'margin_score'].tolist()])
    print all_margin_scores_ranked
    pvalues = [float(sum(all_margin_scores_ranked>el))/len(all_margin_scores_ranked) for el in rank_score_df.ix[:,'margin_score'].tolist()]
    rank_score_df['pvalue']=pvalues
    # qvalues = stats.p_adjust(FloatVector(pvalues), method = 'BH')
    return rank_score_df


def calculate_null_margin_score_dist_from_NULL_tree(rank_score_df, index_mat, method, pool, num_perm, tree, y, x1, x2, y_value, null_tree_file):
    # Read in NULL tree
    tree_null = pickle.load(open(null_tree_file, 'rb'))
    # Initialize dictionary of margin scores
    dict_names = ['perm{0}'.format(el) for el in xrange(num_perm)]
    margin_score_dict = {}
    for name in dict_names: margin_score_dict[name]=0
    ### Re-create y from permutations 
    y_null = util.shuffle_data_object(obj=y)
    # Given the index matrix, randomly sample the same number of + or - examples
    if method=='by_x1':
        # For each permutation calculate margin scores and add to dictionary
        for i in xrange(num_perm):
            # Permute rows
            new_index=sample_values_from_data_class(y=y_null, index_mat=index_mat, method=method, value=y_value)
            margin_score_dict['perm{0}'.format(i)]=rank_by_margin_score(tree_null, y_null, x1, x2, new_index, pool, method=method)
    elif method=='by_x2':
        # For each permutation calculate margin scores and add to dictionary
        for i in xrange(num_perm):
            # Permute columns
            new_index=sample_values_from_data_class(y=y_null, index_mat=index_mat, method=method, value=y_value)
            margin_score_dict['perm{0}'.format(i)]=rank_by_margin_score(tree_null, y_null, x1, x2, new_index, pool, method=method)
    elif method=='by_node' or method=='by_x1_and_x2':
        # For each permutation calculate margin scores and add to dictionary
        for i in xrange(num_perm):
            # Permute rows and columns
            new_index=sample_values_from_data_class(y=y_null, index_mat=index_mat, method=method, value=y_value)
            margin_score_dict['perm{0}'.format(i)]=rank_by_margin_score(tree_null, y_null, x1, x2, new_index, pool, method=method)
    else:
        assert False, "provide method in ['by_x1', 'by_x2', 'by_x1_and_x2', 'by_node']"
    # calculate p-values for each margin score 
    all_margin_scores_ranked = np.sort([el for df in margin_score_dict.values() for el in df.ix[:,'margin_score'].tolist()])
    print all_margin_scores_ranked
    pvalues = [float(sum(all_margin_scores_ranked>el))/len(all_margin_scores_ranked) for el in rank_score_df.ix[:,'margin_score'].tolist()]
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


### Calculate discriminative motifs over all of these things
# for pair in itertools.combinations(iterable, r)
#     for element_direction in ["ENH_UP", "ENH_DOWN", "PROM_UP", "PROM_DOWN"]:
#         for method in ['x1_feat', 'x2_feat']:







