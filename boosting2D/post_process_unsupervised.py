### Peyton Greenside
### 8/11/15
### Script to find unsupervised modules 
###############################################################

import os
import sys
import pdb

from matplotlib import pyplot as plt
from sklearn.datasets import dump_svmlight_file
import scipy.cluster.hierarchy as hier
import scipy.spatial.distance as dist
from sklearn.cluster.bicluster import SpectralCoclustering

import pandas as pd
import numpy as np
from scipy.sparse import *
import random

import multiprocessing
import ctypes

from boosting2D import margin_score
from boosting2D import util
from boosting2D import config
from boosting2D.multiprocessing_utils import fork_and_wait


### K-Means Clustering
###############################################################
###############################################################

# Run k-means clustering using sofia_ml and then join similar clusters
def cluster_examples_kmeans(y, x1, x2, tree, n_clusters_start=5000,
                            mat_features=['motif']):

    ### Set up output folder
    cluster_outdir = '{0}{1}/clustering/'.format(
        config.OUTPUT_PATH, config.OUTPUT_PREFIX)
    if not os.path.exists(cluster_outdir):
        os.makedirs(cluster_outdir)

    # Generate matrix of peaks-by-conditions by regulatory programs
    ex_by_feat_mat = gen_ex_by_feature_matrix(y, x1, x2, tree, feat=mat_features)
    # Keep only peaks that are non-zero in y
    ex_by_feat_mat = ex_by_feat_mat[y.data.toarray().ravel()!=0,]

    # Dump marix in SVM Light format
    sofiaml_file = convert_matrix_to_svmlight_for_sofiaml(ex_by_feat_mat,
     label=config.OUTPUT_PREFIX, out_dir=cluster_outdir, mat_features=mat_features)

    # Initial Kmeans clustering
    (cluster_file, assignment_file) = cluster_matrix_w_sofiaml_kmeans(
        ex_by_feat_mat=ex_by_feat_mat,
        sofiaml_file=sofiaml_file, out_dir=cluster_outdir,
         n_clusters_start=n_clusters_start)

    # Join  Kmeans clusters that are too similar using hierarchical Clustering
    new_clusters = join_similar_kmeans_cluster(cluster_file, assignment_file,
     n_clusters_start)
    
    return (cluster_file, new_clusters)

### Post clustering with SofiaML
###############################################################

# Convert format 
def convert_matrix_to_svmlight_for_sofiaml(ex_by_feat_mat, label, 
                                           out_dir, mat_features):
    sofiaml_file='{0}sofiaml_input_{1}.txt'.format(out_dir,
        '_'.join(mat_features))
    dump_svmlight_file(X=ex_by_feat_mat, y=[0]*ex_by_feat_mat.shape[0],
     f=sofiaml_file)
    return sofiaml_file

def cluster_matrix_w_sofiaml_kmeans(ex_by_feat_mat, sofiaml_file, out_dir, 
                                    n_clusters_start, SOFIA_ML_PATH):
    ### Run SOFIA ML Kmeans
    cluster_file = '{0}/sofiaml_clusters_n{1}.txt'.format(
        out_dir, n_clusters_start)
    assignment_file = '{0}/sofiaml_assignments_n{1}.txt'.format(
        out_dir, n_clusters_start)

    # Get cluster centers
    sofia_train_command="""
    {0}/sofia-kmeans --k {1} --init_type random
     --opt_type mini_batch_kmeans --mini_batch_size 1000
     --dimensionality {2}
      --iterations 500 --objective_after_init --objective_after_training
       --training_file {3} --model_out {4}
    """.format(SOFIA_ML_PATH, n_clusters_start, ex_by_feat_mat.shape[1],
     sofiaml_file, cluster_file).replace('\n', '')

    os.system(sofia_train_command)

    # Assign clusters
    sofia_test_command="""
    {0}/sofia-kmeans --model_in {1}
     --test_file {2} --objective_on_test
      --cluster_assignments_out {3}
    """.format(SOFIA_ML_PATH, cluster_file, sofiaml_file,
     assignment_file).replace('\n', '')

    os.system(sofia_test_command)

    return (cluster_file, assignment_file)

# Take k-means cluster files from sofiaML and join similar clusters
def join_similar_kmeans_cluster(cluster_file, assignment_file, max_distance=0.5):

    # Read in clusters and assignments from assignment
    clusters = np.loadtxt(cluster_file)
    assigns = np.loadtxt(assignment_file)

    # Do hierarchical clustering on the clusters and group similar clusters
    d = dist.pdist(clusters, 'euclidean')  
    l = hier.linkage(d, method='average')
    ordered_data = clusters[hier.leaves_list(l),:]
    max_distance=0.5
    flat_clusters = hier.fcluster(l, t=max_distance, criterion='distance')

    # Reassign clusters by collapsing similar clusters
    new_clusters = [flat_clusters[assigns[i,0]] for i in range(assigns.shape[0])]

    return new_clusters


def write_out_cluster(y, cluster_file, new_clusters,
                      clusters_to_write='all', create_match_null=False):

    ### Set up output folder
    cluster_bed_dir = '{0}{1}/clustering/cluster_bed_files/'.format(
        config.OUTPUT_PATH, config.OUTPUT_PREFIX)
    if not os.path.exists(cluster_outdir):
        os.makedirs(cluster_outdir)

    # Write background out once
    bkgrd_coords = [y.row_labels[i].split(';')[0].replace(':', '\t').replace('-', '\t')
     for i in xrange(y.num_row)]
    bed_file='{0}BACKGROUND.bed'.format(cluster_bed_dir)
    with open(bed_file, 'w') as f:
        for item in bkgrd_coords:
            f.write("{0}\n".format(item))

    ### Get column labels for GREAT analysis
    if clusters_to_write=='all':
        clusters_for_iter=np.unique(new_clusters)
    else:
        clusters_for_iter=[int(el) for el in clusters_to_write.split(',')]

    # Iterate over all clusters and write out bed files of coordinates
    # !! Current assumes labels are the coordinates
    for clust in clusters_to_write:
        # clust=3
        print clust
        clust_ex=[i for i in xrange(len(new_clusters)) if new_clusters[i]==clust]

        # Flatten concatenates rows [row1, row2, etc.]
        conditions = [val%y.num_col for val in clust_ex]
        peaks = [np.floor(np.divide(val, y.num_col)) for val in clust_ex]
        peak_coords = [y.row_labels[i].split(';')[0].replace(':', '\t').replace('-', '\t')
         for i in np.unique(peaks)]
        print 'number of peaks {0}'.format(len(np.unique(peaks)))

        # Write to bed file for great
        bed_file='{0}clust{1}.bed'.format(cluster_bed_dir, clust)
        with open(bed_file, 'w') as f:
            for item in peak_coords:
                f.write("{0}\n".format(item))

        # If we want to create matched sets to compare GREAT enrichments
        if create_match_null==True:
            ### Randomly sample coordinates to compare enrichment
            print 'getting random of same size'
            rand_ind = random.sample(range(ex_by_feat_mat.shape[0]), len(clust_ex))
            # rand_ind = random.sample(range(2179548), len(clust_ex))

            conditions = [val%y.num_col for val in rand_ind]
            peaks = [np.floor(np.divide(val, y.num_col)) for val in rand_ind]
            peak_coords = [y.row_labels[i].split(';')[0].replace(':', '\t').replace('-', '\t')
             for i in np.unique(peaks)]
            bkgrd_coords = [y.row_labels[i].split(';')[0].replace(':', '\t').replace('-', '\t')
             for i in xrange(y.num_row) if i not in np.unique(peaks)]

            # Write to bed file for great
            bed_file='{0}clust{1}_RAND.bed'.format(cluster_bed_dir, clust)
            with open(bed_file, 'w') as f:
                for item in peak_coords:
                    f.write("{0}\n".format(item))

    print 'DONE: Find clusters in: {0}'.format(cluster_bed_dir)


### Plot margin scores across conditions
###############################################################
###############################################################

# get contradictory motifs and regulators
# List the files where you want to get it from (or a directory)
# List the motif name you want

def get_margin_score_across_conditions(list_of_margin_scores, margin_score_path,
 feat_name, feat_col, margin_score_col):
    margin_scores = {}
    with open(list_of_margin_scores) as f: all_files = f.read().splitlines() 
    all_files.pop()
    f.close()
    for filey in all_files:
        name=filey.split('.')[0]
        df = pd.read_table('{0}{1}'.format(margin_score_path, filey), sep="\t", header=0)
        np.where(df.ix[:,feat_col-1]==feat_name)
        if feat_name in df.ix[:,feat_col-1].tolist():
            val = df.ix[df.ix[:,feat_col-1]==feat_name,margin_score_col].tolist()[0]
            margin_scores[name]=val
        else:
            print 'feature not present in {0}'.format(filey)
    return margin_scores



### K-NN Clustering to track examples
###############################################################
###############################################################

### K-nearest neighbors - take in set of examples, find indices
### and return a dictionary of elements 
def knn(ex_file, y, x1, x2, tree, num_neighbors):
    ### Read in examples and get indices
    ex_df = pd.read_table(ex_file, header=None)
    peaks = ex_df.ix[:,0].tolist()
    conditions = ex_df.ix[:,1].tolist()
    ### Generate feature matrix
    ex_by_feat_mat = gen_ex_by_feature_matrix(y, x1, x2, tree, feat=['motif'])
    # if providing index numbers, subtract 1, else get index of labels
    # if peaks.applymap(lambda x: isinstance(x, (int, float))).sum().tolist()[0]==len(peaks):
    if np.sum([isinstance(el, (int, float)) for el in peaks])==len(peaks):
        # ASSUMING INPUT IS 1 BASED
        peak_index_list = [el-1 for el in peaks]
    else:
        # peak_index_list = [el for el in xrange(y.data.shape[0]) if y.row_labels[el] in peaks]
        if set(peaks).issubset(y.row_labels)==False:
            assert False, "Your examples don't match target peak (row) labels. \
            Provide index numbers or valid peak labels"
        peak_index_list = [np.where(y.row_labels==el)[0][0] for el in peaks]
    # if providing index numbers, subtract 1, else get index of labels
    # if conditions.applymap(lambda x: isinstance(x, (int, float))).sum().tolist()[0]==len(conditions):
    if np.sum([isinstance(el, (int, float)) for el in conditions])==len(conditions):
        # ASSUMING INPUT IS 1 BASED
        condition_index_list = [el-1 for el in peaks]
    else:
        # condition_index_list = [el for el in xrange(y.data.shape[1]) if y.col_labels[el] in conditions]
        if set(conditions).issubset(y.col_labels)==False:
            assert False, "Your examples don't match target condition (column) labels \
            Provide index numbers or valid condition labels"
        condition_index_list = [np.where(y.col_labels==el)[0][0] for el in conditions]
    # Get dictionary of all features applying to each example
    feat_dict = {}
    for ind in xrange(len(peak_index_list)):
        if y.data[peak_index_list[ind],condition_index_list[ind]]==0:
            print 'There is no change at this feature in this condition.'
            continue
        feat_dict[ind] = extract_feat_affecting_example_set(y, tree, 
            peak_index_list[ind], condition_index_list[ind], ex_by_feat_mat)
    # Get examples to search over for each the 
    ex_dict = {}
    for ind in feat_dict.keys():
        if len(feat_dict[ind])==0:
            ex_dict[ind]=[]
        else:
            ex_dict[ind] = find_examples_affected_by_same_features(tree, 
            feat_dict[ind])
    # get k nearest neighbors based on example_index and other examples
    knn_dict = {}
    for ind in feat_dict.keys():
        knn_dict[ind] = get_k_nearest_neighbors(y, 
            peak_index_list[ind], condition_index_list[ind],
            ex_dict[ind], ex_by_feat_mat, num_neighbors)
    # return the KNN dictionary of nearest neighbors
    return knn_dict



# Find the indices of the features affecting example index given
def extract_feat_affecting_example_set(y, tree, peak_index, condition_index, ex_by_feat_mat):
    # ! check
    example_index=peak_index*y.num_col+condition_index
    if y.data[peak_index,condition_index]==0:
        print 'There is no change at this feature in this condition. STOP.'
        return None
    # Find the features that have non-zero margin score
    # feat_list = (ex_by_feat_mat[example_index,:]!=0).nonzero()[1].tolist()
    # Find the features where the index actually overlaps the place
    df = csr_matrix(y.data.shape)
    # Make one TRUE entry at yo
    df[peak_index, condition_index]=True
    feat_list=[]
    for ind in xrange(tree.nsplit):
        if util.element_mult(tree.ind_pred_test[ind]+tree.ind_pred_train[ind], df).sum()!=0:
            feat_list.append(ind)
    return feat_list

# Get all examples affected by any of the same feature sets
def find_examples_affected_by_same_features(tree, feat_list):
    # Concatenate the index of 
    index_vec = np.sum([tree.ind_pred_train[node]+tree.ind_pred_test[node]
       for node in feat_list if node != 0]).toarray().flatten()
    expanded_example_list = np.where(index_vec!=0)[0].tolist()
    return expanded_example_list

# Give an example index, the neighbor list and feature matrix, get KNN
def get_k_nearest_neighbors(y, peak_index, condition_index, 
    expanded_example_list, ex_by_feat_mat, num_neighbors=100):
    # Subset example by feature matrix
    if len(expanded_example_list)==0:
        print 'no examples are affectd by the same features'
        return []
    else:
        example_mat = ex_by_feat_mat[expanded_example_list,:]
        example_index=peak_index*y.num_col+condition_index
        # Run KNN with closest examples
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=num_neighbors,
         algorithm='ball_tree').fit(example_mat)
        distances, indices = nbrs.kneighbors(ex_by_feat_mat[example_index,])
        # Return the indices of the number of neighbors - reconvert into peaks/conditions
        neighbor_index=indices[0]
        return neighbor_index

# Write out KNN as columns
# Format: peak|condition peak|condition peak|condition (as original examples)
# k+1 rows, first row is the original example and then the k lines after are the 
# k nearest neighbors found
def write_knn(y, ex_file, knn_dict, output_path):
    file_name=output_path+os.path.basename(ex_file).split('.')[0]+ \
        '_with_knneighbors.txt'
    ex_df = pd.read_table(ex_file, header=None)
    peaks = ex_df.ix[:,0].tolist()
    conditions = ex_df.ix[:,1].tolist()
    # Write out headers as the examples provided
    headers = ['|'.join([peaks[ind], conditions[ind]]) for ind in knn_dict.keys()]
    # Below each one add in 
    rows = {}
    for ind in knn_dict.keys():
        # write out the peak and condition names based on the indices 
        examples = knn_dict[ind]
        peak_names = [y.row_labels[np.floor(np.divide(val, y.num_col))] for val in examples]
        cond_names = [y.col_labels[val%y.num_col] for val in examples]
        row = ['|'.join([peak_names[i], cond_names[i]]) \
         for i in xrange(len(examples))]
        rows[ind] = row 
    # Write out each example
    knn_file = open(file_name, 'w')
    knn_file.write("%s\n" % '\t'.join(headers))
    for ind in range(len(row)):
        knn_file.write("%s\n" % '\t'.join(
            [rows[col][ind] for col in rows.keys()]))
    knn_file.close()
    return 0


### Function to generate ex by feature matrix
###############################################################
###############################################################

def gen_ex_by_feature_matrix(y, x1, x2, tree, feat=[
    'motif', 'reg', 'node', 'path']):

    example_labels = ['|'.join([col_lab, row_lab]) for row_lab in y.row_labels
     for col_lab in y.col_labels]
    # Features for motifs
    motif_labels = x1.row_labels
    # Features for regulators
    regulator_labels = x2.col_labels
    # Features for nodes
    node_labels = ['|'.join([x1.row_labels[tree.split_x1[ind]],
     x2.col_labels[tree.split_x2[ind]]]) 
        for ind in range(1,tree.nsplit)]
    # Features for paths
    path_dict = enumerate_paths(tree)
    path_labels = path_dict.values()
    # get index matrix
    index_mat = margin_score.get_index(y, x1, x2, tree,
     condition_feat_file=None, region_feat_file=None)

    ex_by_feat_mat = csr_matrix((len(example_labels), 1))
    ############### MULTIPROCESS ###############
    # Get lock
    lock = multiprocessing.Lock()
    ### Motif Matrix
    if 'motif' in feat:
        motif_matrix_mp = multiprocessing.Array(
            ctypes.c_double, len(example_labels)*len(motif_labels))
        index_cntr = multiprocessing.Value('i', 0)

        # pack arguments for the worker processes
        args = [tree, y, x1, x2, index_mat, True, (
                lock, index_cntr, motif_matrix_mp)]
        
        # Fork worker processes, and wait for them to return
        fork_and_wait(config.NCPU, margin_score.calc_margin_score_x1_worker, args)

        motif_matrix = np.frombuffer(motif_matrix_mp.get_obj()).reshape(
            (len(example_labels), len(motif_labels)))
        # motif_matrix = np.frombuffer(motif_matrix_mp.get_obj()).reshape(
            # (len(motif_labels), len(example_labels))).T # try T
        ex_by_feat_mat = hstack([ex_by_feat_mat, csr_matrix(motif_matrix)])
        # compare array to y.data.toarray().flatten()
        # (motif_matrix[y.data.toarray().flatten()==0,]).sum()
    ### Regulator Matrix
    if 'reg' in feat:
        reg_matrix_mp = multiprocessing.Array(
            ctypes.c_double, len(example_labels)*len(regulator_labels))
        index_cntr = multiprocessing.Value('i', 0)

        # pack arguments for the worker processes
        args = [tree, y, x1, x2, index_mat, True, (
                lock, index_cntr, reg_matrix_mp)]
        
        # Fork worker processes, and wait for them to return
        fork_and_wait(config.NCPU, margin_score.calc_margin_score_x2_worker, args)

        reg_matrix = np.frombuffer(reg_matrix_mp.get_obj()).reshape(
           (len(example_labels), len(regulator_labels)))
        ex_by_feat_mat = hstack([ex_by_feat_mat, csr_matrix(reg_matrix)])
    ### Node Matrix
    if 'node' in feat:
        node_matrix_mp = multiprocessing.Array(
            ctypes.c_double, len(example_labels)*len(node_labels))
        index_cntr = multiprocessing.Value('i', 0)

        # pack arguments for the worker processes
        args = [tree, y, x1, x2, index_mat, True, (
                lock, index_cntr, node_matrix_mp)]
        
        # Fork worker processes, and wait for them to return
        fork_and_wait(config.NCPU, margin_score.calc_margin_score_node_worker, args)

        node_matrix = np.frombuffer(node_matrix_mp.get_obj()).reshape(
           (len(example_labels), len(node_labels)))
        ex_by_feat_mat = hstack([ex_by_feat_mat, csr_matrix(node_matrix)])
    ### Path Matrix
    if 'path' in feat:
        path_matrix_mp = multiprocessing.Array(
            ctypes.c_double, len(example_labels)*len(path_labels))
        index_cntr = multiprocessing.Value('i', 0)

        # pack arguments for the worker processes
        args = [tree, y, x1, x2, index_mat, True, (
                lock, index_cntr, path_matrix_mp)]
        
        # Fork worker processes, and wait for them to return
        fork_and_wait(config.NCPU, margin_score.calc_margin_score_path_worker, args)

        path_matrix = np.frombuffer(path_matrix_mp.get_obj()).reshape(
           (len(example_labels), len(path_labels)))
        ex_by_feat_mat = hstack([ex_by_feat_mat, csr_matrix(path_matrix)])

    # Remove dummy column and convert to csr_matrix
    ex_by_feat_mat = csr_matrix(ex_by_feat_mat)[:,1::]
    ############### MULTIPROCESS ###############

    return ex_by_feat_mat

# Function to return subset of ex_by_feat_mat 
def subset_ex_by_feature_matrix(ex_by_feat_mat, y, x1, x2, condition_feat_file, region_feat_file, feat, remove_zeros=True):
    # If no need to subset, then just return the original
    if condition_feat_file==None and region_feat_file==None:
        return ex_by_feat_mat
    # Read in label names to include
    x1_file = pd.read_table(region_feat_file, header=None)
    x2_file = pd.read_table(condition_feat_file, header=None)
    # Create a list of all label names
    example_labels = ['|'.join([col_lab, row_lab]) for row_lab in y.row_labels
     for col_lab in y.col_labels]
    ### Done in terms of data frame to keep labels (probably a better way to do this)
    ex_by_feat_df = pd.DataFrame(ex_by_feat_mat.toarray())
    ex_by_feat_df.index = example_labels
    if 'motif' in feat and 'reg' in feat:
        ex_by_feat_df.columns = x1.row_labels.tolist()+x2.col_labels.tolist()
    elif 'motif' in feat:
        ex_by_feat_df.columns = x1.row_labels
    elif 'reg' in feat:
        ex_by_feat_df.columns = x2.col_labels
    subset_labels = ['|'.join([cond, peak]) for peak in x1_file.ix[:,0] for cond in x2_file.ix[:,0]]
    index_dict = dict((value, idx) for idx,value in enumerate(ex_by_feat_df.index))
    subset_index = [index_dict[x] for x in subset_labels]
    sub_ex_by_feat_df = ex_by_feat_df.ix[subset_index,:]
    if remove_zeros:
        sub_ex_by_feat_df0 = sub_ex_by_feat_df.ix[y.data.toarray().ravel()[subset_index]!=0,]
        sub_ex_by_feat_df1 = sub_ex_by_feat_df0.ix[:,sub_ex_by_feat_df0.apply(np.sum, 0)!=0]
        sub_ex_by_feat_df2 = sub_ex_by_feat_df1.ix[sub_ex_by_feat_df1.apply(np.sum, 1)!=0,:]
        return sub_ex_by_feat_df2
    else:
        return sub_ex_by_feat_df


# Function to cluster feature by example matrix
def cluster_ex_by_feature_matrix(sub_ex_by_feat_mat, plot_file):
    if sub_ex_by_feat_mat.shape[0]>50000:
        print "Matrix too large to be efficient, pleased reduce number of examples"

    # Subset down to motifs that are used
    plot_df = sub_ex_by_feat_mat[:,np.apply_along_axis(np.max, 0, sub_ex_by_feat_mat.toarray())!=0]
    # for numpy array
    plot_df = sub_ex_by_feat_mat_1[np.apply_along_axis(lambda row: (row!=0).sum(), 1, sub_ex_by_feat_mat_1.toarray())>10,:]
    plot_df = plot_df[:,np.apply_along_axis(lambda column: (column!=0).sum(), 0, sub_ex_by_feat_mat_1.toarray())>50]
    # for pandas
    plot_df = sub_ex_by_feat_df2.ix[sub_ex_by_feat_df2.apply(lambda row: (row!=0).sum(), 1)>10,:]
    plot_df = plot_df.ix[:,plot_df.apply(lambda row: (row!=0).sum(), 0)>50]
    plot_df = sub_ex_by_feat_df2

    np.apply_along_axis(lambda column: (column!=0).sum(), 0, sub_ex_by_feat_mat_1.toarray())

    model = SpectralCoclustering(n_clusters=50)
    model.fit(plot_df) # fits for 50K
    fit_data = plot_df.ix[np.argsort(model.row_labels_)]
    fit_data = fit_data.ix[:, np.argsort(model.column_labels_)]

    plt.figure()
    plt.matshow(fit_data.ix[0:500,], cmap=plt.cm.YlGnBu, aspect='auto')
    plt.savefig(plot_file)

    print "DONE: biclustering plot here: {0}".format(plot_file)

    return "pretty picture"

# Enumerate paths for a tree by listing all nodes in path
###############################################################
def enumerate_paths(tree):
    path_dict = {}
    for node in range(1,tree.nsplit):
        path_dict[node]='|'.join([str(node)]+[
            str(el) for el in tree.above_nodes[node] if el!=0])
    return path_dict


