import sys
import os
import random
import pdb

import numpy as np 
from scipy.sparse import *
import scipy.stats
import pandas as pd
import time

from boosting_2D import util
from boosting_2D import config

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
    print rule_index_fraction
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
    print rule_index_fraction
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
    print rule_index_fraction
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

    print rule_index_fraction
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
def call_rank_by_margin_score(prefix, methods, y, x1, x2, tree, pool, num_perm=10, x1_feat_file=None, x2_feat_file=None,split_prom_enh=True):
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
                rank_score_df_dict[key] = rank_by_margin_score(tree, y, x1, x2, eval('index_mat_{0}'.format(key)), pool, method='by_x1')
                if num_perm>0:
                    rank_score_df_w_perm = calculate_null_margin_score_dist(rank_score_df_dict[key], 
                        eval('index_mat_{0}'.format(key)), 'by_x1', pool, num_perm, tree, y, x1, x2)
                    rank_score_df_w_perm.to_csv('{0}{1}_{2}_top_x1_feat_permuted.txt'.format(margin_outdir, prefix, key.upper()), 
                        sep="\t", index=None, header=True)
                else:
                    rank_score_df_dict[key].to_csv('{0}{1}_{2}_top_x1_feat.txt'.format(margin_outdir, prefix, key.upper()), 
                        sep="\t", index=None, header=True)

        if 'by_x2' in methods:
            for key in rank_score_df_dict.keys():
                rank_score_df_dict[key] = rank_by_margin_score(tree, y, x1, x2, eval('index_mat_{0}'.format(key)), pool, method='by_x2')
                if num_perm>0:
                    rank_score_df_w_perm = calculate_null_margin_score_dist(rank_score_df_dict[key], 
                        eval('index_mat_{0}'.format(key)), 'by_x2', pool, num_perm, tree, y, x1, x2)
                    rank_score_df_w_perm.to_csv('{0}{1}_{2}_top_x2_feat_permuted.txt'.format(margin_outdir, prefix, key.upper()), 
                        sep="\t", index=None, header=True)
                else:
                    rank_score_df_dict[key].to_csv('{0}{1}_{2}_top_x2_feat.txt'.format(margin_outdir, prefix, key.upper()), 
                        sep="\t", index=None, header=True)

        if 'by_x1_and_x2' in methods:
            for key in rank_score_df_dict.keys():
                rank_score_df_dict[key] = rank_by_margin_score(tree, y, x1, x2, eval('index_mat_{0}'.format(key)), pool, method='by_x1_and_x2')
                if num_perm>0:
                    rank_score_df_w_perm = calculate_null_margin_score_dist(rank_score_df_dict[key], 
                        eval('index_mat_{0}'.format(key)), 'by_x1_and_x2', pool, num_perm, tree, y, x1, x2)
                    rank_score_df_w_perm.to_csv('{0}{1}_{2}_top_x1_and_x2_joint_feat_permuted.txt'.format(margin_outdir, prefix, key.upper()), 
                        sep="\t", index=None, header=True)
                else:
                    rank_score_df_dict[key].to_csv('{0}{1}_{2}_top_x1_and_x2_joint_feat.txt'.format(margin_outdir, prefix, key.upper()), 
                        sep="\t", index=None, header=True)
        if 'by_node' in methods:
            for key in rank_score_df_dict.keys():
                rank_score_df_dict[key] = rank_by_margin_score(tree, y, x1, x2, eval('index_mat_{0}'.format(key)), pool, method='by_node')
                if num_perm>0:
                    rank_score_df_w_perm = calculate_null_margin_score_dist(rank_score_df_dict[key], 
                        eval('index_mat_{0}'.format(key)), 'by_node', pool, num_perm, tree, y, x1, x2)
                    rank_score_df_w_perm.to_csv('{0}{1}_{2}_top_nodes_permuted.txt'.format(margin_outdir, prefix, key.upper()), 
                        sep="\t", index=None, header=True)
                else:
                    rank_score_df_dict[key].to_csv('{0}{1}_{2}_top_nodes.txt'.format(margin_outdir, prefix, key.upper()), 
                    sep="\t", index=None, header=True)

    return 0


### Calculate permutations of the index matrix and re-calculate margin scores
def calculate_null_margin_score_dist(rank_score_df, index_mat, method, pool, num_perm, tree, y, x1, x2):
    # Initialize dictionary of margin scores
    dict_names = ['perm{0}'.format(el) for el in xrange(num_perm)]
    margin_score_dict = {}
    for name in dict_names: margin_score_dict[name]=0
    # Set seed for permutations
    random.seed(1)
    # Given the index matrix, randomly sample the same number of + or - examples
    if method=='by_x1':
        # For each permutation calculate margin scores and add to dictionary
        for i in xrange(num_perm):
            # Permute rows
            if y.sparse:
                new_index=csr_matrix(np.apply_along_axis(np.random.permutation, 0, index_mat.toarray())) ### MAKE SPARSE
            else:
                new_index=np.apply_along_axis(np.random.permutation, 0, index_mat)
            margin_score_dict['perm{0}'.format(i)]=rank_by_margin_score(tree, y, x1, x2, new_index, pool, method=method)
    elif method=='by_x2':
        # For each permutation calculate margin scores and add to dictionary
        for i in xrange(num_perm):
            # Permute columns
            if y.sparse:
                new_index=csr_matrix(np.apply_along_axis(np.random.permutation, 1, index_mat.toarray()))
            else:
                new_index=np.apply_along_axis(np.random.permutation, 1, index_mat)
            margin_score_dict['perm{0}'.format(i)]=rank_by_margin_score(tree, y, x1, x2, new_index, pool, method=method)
    elif method=='by_node' or method=='by_x1_and_x2':
        # For each permutation calculate margin scores and add to dictionary
        for i in xrange(num_perm):
            # Permute rows and columns
            if y.sparse:
                new_index=csr_matrix(np.random.permutation(index_mat.toarray().flat).reshape(index_mat.shape))
            else:
                new_index=np.random.permutation(index_mat.flat).reshape(index_mat.shape)
            margin_score_dict['perm{0}'.format(i)]=rank_by_margin_score(tree, y, x1, x2, new_index, pool, method=method)
    else:
        assert False, "provide method in ['by_x1', 'by_x2', 'by_x1_and_x2', 'by_node']"
    # calculate p-values for each margin score 
    all_margin_scores_ranked = np.sort([el for df in margin_score_dict.values() for el in df.ix[:,'margin_score'].tolist()])
    pvalues = [float(sum(all_margin_scores_ranked>el))/len(all_margin_scores_ranked) for el in rank_score_df.ix[:,'margin_score'].tolist()]
    rank_score_df['pvalue']=pvalues
    # qvalues = stats.p_adjust(FloatVector(pvalues), method = 'BH')
    return rank_score_df



### Plot
# import matplotlib.pyplot as plt
# plt.figure()
# plt.hist(all_margin_scores_ranked[all_margin_scores_ranked!=0], bins=100)
# plt.title("Ranked Margin Scores")
# plt.xlabel("Value")
# plt.ylabel("Frequency")
# plt.savefig('/users/pgreens/all_margin_scores_ranked_hist_no_zero.png')






