### Peyton Greenside
### 8/11/15
### Script to find unsupervised modules 
###############################################################

from matplotlib import pyplot as plt

### Load data
###############################################################
RESULT_PATH='/srv/persistent/pgreens/projects/boosting/results/'
# execfile('{0}2015_08_14_hematopoeisis_23K_bindingTFsonly_adt_stable_5iter/load_pickle_data_script.py'.format(RESULT_PATH))
execfile('{0}2015_08_15_hematopoeisis_23K_bindingTFsonly_adt_stable_1000iter/load_pickle_data_script.py'.format(RESULT_PATH))

### Generate matrix of peaks-by-conditions by regulatory programs
###############################################################
ex_by_feat_mat = gen_ex_by_feature_matrix(y, x1, x2, tree, feat=['motif', 'reg'])


### Post clustering with SofiaML
###############################################################
from sklearn.datasets import dump_svmlight_file

SCRIPT_PATH='/users/pgreens/svn/sofia-ml-read-only/'
SOFIAML_PATH='/srv/persistent/pgreens/projects/boosting/results/sofiaml_files/'
sofiaml_file=SOFIAML_PATH+'sofia_input.txt'
dump_svmlight_file(X=ex_by_feat_mat, y=[0]*ex_by_feat_mat.shape[0], f=sofiaml_file)

### Run SOFIA

sofia_train_command="""
{0}/sofia-kmeans --k 5000 --init_type random
 --opt_type mini_batch_kmeans --mini_batch_size 100
 --dimensionality {1}
  --iterations 500 --objective_after_init --objective_after_training
   --training_file {2}/sofia_input.txt --model_out {2}/clusters2.txt
""".format(SCRIPT_PATH, ex_by_feat_mat.shape[1], SOFIAML_PATH).replace('\n', '')

os.system(sofia_train_command)


sofia_test_command="""
{0}/sofia-kmeans --model_in {1}/clusters2.txt
 --test_file {1}/sofia_input.txt --objective_on_test
  --cluster_assignments_out {1}/assignments.txt
""".format(SCRIPT_PATH, SOFIAML_PATH).replace('\n', '')

os.system(sofia_test_command)

### Read in results

clusters = pd.read_table('{0}/clusters2.txt'.format(SOFIAML_PATH),
 sep=" ", header=None)
assigns = pd.read_table('{0}/assignments.txt'.format(SOFIAML_PATH),
    header=None)

### Get column labels for GREAT analysis
clust=3
clust_ex=[i for i in xrange(assigns.shape[0]) if assigns.ix[i,0]==clust]

# Flatten concatenates rows [row1, row2, etc.]
conditions = [val%y.num_col for val in clust_ex]
peaks = [np.floor(np.divide(val, y.num_col)) for val in clust_ex]
peak_coords = [y.row_labels[i].split(';')[0].replace(':', '\t').replace('-', '\t')
 for i in np.unique(peaks)]

# Write to bed file for great
bed_file='{0}/bed_file_clust{1}.txt'.format(SOFIAML_PATH, clust)
with open(bed_file, 'w') as f:
    for item in peak_coords:
        f.write("{0}\n".format(item))

### Randomly sample coordinates to compare enrichment
# rand_ind = random.sample(range(ex_by_feat_mat.shape[0]), len(clust_ex))
rand_ind = random.sample(range(2179548), len(clust_ex))

conditions = [val%y.num_col for val in rand_ind]
peaks = [np.floor(np.divide(val, y.num_col)) for val in rand_ind]
peak_coords = [y.row_labels[i].split(';')[0].replace(':', '\t').replace('-', '\t')
 for i in np.unique(peaks)]

# Write to bed file for great
bed_file='{0}/bed_file_clust{1}_RAND.txt'.format(SOFIAML_PATH, clust)
with open(bed_file, 'w') as f:
    for item in peak_coords:
        f.write("{0}\n".format(item))



### Find coordinated elements
def find_module_by_peak_and_condition(peak, condition):


### Read in clusters 

### Clustering methods
###############################################################
# NMF
# Biclustering
# PCA/SVD
# LDA

# from sklearn.decomposition import ProjectedGradientNMF
# model = ProjectedGradientNMF(n_components=50, init='random', random_state=0)
from sklearn.decomposition import FactorAnalysis
model = FactorAnalysis(n_components=50, random_state=0)
model.fit(ex_by_feat_mat.toarray())

from sklearn.decomposition import RandomizedPCA
pca = RandomizedPCA(n_components=50)
pca.fit(ex_by_feat_mat)                 
RandomizedPCA(copy=True, iterated_power=3, n_components=2,
       random_state=None, whiten=False)

### PCA
from sklearn.decomposition import TruncatedSVD
pca = TruncatedSVD(n_components=150)
pca.fit(ex_by_feat_mat)       
# DATA = SCORES x LOADINGS
scores = pca.transform(ex_by_feat_mat)    
loadings = pca.components_

# Divide examples into components
# Just assign the component max score (very biased toward first component)
ex_cluster_assign = np.apply_along_axis(np.argmax, 0, loadings)
feat_cluster_assign = np.apply_along_axis(np.argmax, 1, scores)
# K-means clustering based on components

### Biclustering (MEMORY ERROR)
from sklearn.cluster.bicluster import SpectralCoclustering
test_mat = ex_by_feat_mat[0:5000,:].toarray()
test_mat = ex_by_feat_mat[0:1000,:].toarray()
# model = SpectralCoclustering(n_clusters=50)
model = SpectralCoclustering(n_clusters=50)
model.fit(test_mat) # fits for 50K
fit_data = test_mat[np.argsort(model.row_labels_)]
fit_data = fit_data[:, np.argsort(model.column_labels_)]

plt.figure()
plt.matshow(fit_data, cmap=plt.cm.Blues)
plt.savefig('/users/pgreens/cluster.png')


### LDA
test_mat = ex_by_feat_mat.toarray()[0:500,:]
test_mat = ex_by_feat_mat[0:10000,:].toarray()
import lda
model = lda.LDA(n_topics=20, n_iter=500, random_state=1)
model.fit(test_mat)

### Hierarchical Clustering (MEMORY ERROR)
import scipy.cluster.hierarchy as hier
import scipy.spatial.distance as dist

# Convert to nujmpy array
data = ex_by_feat_mat.toarray()
# Get read of zero entries
data = data[np.apply_along_axis(sum, 1, data)!=0,]
# Get cluster assignments based on euclidean distance + complete linkage
d = dist.pdist(data[1:50000,], 'euclidean') 
l = hier.linkage(d, method='average')
ordered_data = data[hier.leaves_list(l),:]
flat_clusters = hier.fcluster(l, t=max_distance, criterion='distance')

### K-means with sparse matrices (takes forever)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

labeler = KMeans(n_clusters=200) 
labeler.fit(ex_by_feat_mat[1:50000,])  
for (row, label) in enumerate(labeler.labels_):   
  print "row %d has label %d"%(row, label)

data = ex_by_feat_mat[np.array(ex_by_feat_mat.sum(axis=1)!=0).reshape(-1,),:]
data = ex_by_feat_mat[1:20000,]
range_n_clusters=[50,100,150,200,250,300,350,400,450,500,600,700,800,900,1000]
sil_avgs={}
for n_clusters in range_n_clusters:

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(data)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(data, cluster_labels)
    sil_avgs[n_clusters]=silhouette_avg
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(data, cluster_labels)



### Fast cluster (MEMORY ERROR with full data set)
data = ex_by_feat_mat[0:20000,:].toarray()
import fastcluster
from scipy import spatial

distance = spatial.distance.pdist(data)
c = fastcluster.linkage(distance, method='single', metric='euclidean', preserve_input=False)
ordered_data = data[hier.leaves_list(c),:]
max_distance=0.3
flat_clusters = hier.fcluster(c, t=max_distance, criterion='distance')
max([len(np.where(flat_clusters==i)[0]) for i in xrange(len(np.unique(flat_clusters)))])

### Try SOFIA ML

# ### Evaluate clusters through GO term enrichments
# gwas_path = '/srv/gsfs0/projects/kundaje/users/pgreens/projects/enh_gene_link_gwas/results/'
# out_path = '/srv/gsfs0/projects/kundaje/users/pgreens/projects/enh_gene_link_gwas/DAVID_output/'
# plot_path = '/srv/gsfs0/projects/kundaje/users/pgreens/projects/enh_gene_link_gwas/plots/'

# ### Run DAVID QUERY for every GWAS (takes a while)
# gwas=list.files(gwas_path)
# for (g in gwas){
#     # Run for links
#     input = sprintf('%s%s/%s_best_linked_genes.txt', gwas_path, g, strsplit(g, 'roadmap_')[[1]][2])
#     output = sprintf('%s%s_linked_GO_enrichment.txt', out_path, strsplit(g, 'roadmap_')[[1]][2])
#     system(sprintf('/srv/gs1/software/R/R-3.0.1/bin/Rscript /srv/gsfs0/projects/kundaje/users/pgreens/scripts/DAVID_R_access.R -f %s -c 5 -o %s', input, output), intern=TRUE)
#     # Run for nearest
#     input = sprintf('%s%s/%s_nearest_genes.txt', gwas_path, g, strsplit(g, 'roadmap_')[[1]][2])
#     output = sprintf('%s%s_nearest_GO_enrichment.txt', out_path, strsplit(g, 'roadmap_')[[1]][2])
#     system(sprintf('/srv/gs1/software/R/R-3.0.1/bin/Rscript /srv/gsfs0/projects/kundaje/users/pgreens/scripts/DAVID_R_access.R -f %s -c 6 -o %s', input, output), intern=TRUE)
# }


### Plot
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


def plot_margin_score_across_conditions(cond_margin_scores, feat_name, label):
    ### Then plot them
    PLOT_PATH='/srv/persistent/pgreens/projects/boosting/plots/margin_score_by_feat/'
    import matplotlib.pyplot as plt
    # order = [el-1 for el in [8, 12, 13, 9, 5, 6, 1, 2, 3, 4, 7, 11, 14, 10]]
    # order = [el-1 for el in [7, 10, 11, 5, 1, 13, 8, 9, 12, 14, 2, 4, 3, 6]] # Specific to HEMA
    order = get_order_vec(cond_margin_scores, order_file)
    labels0 = [el.split('hema_')[1].split('_x')[0] for el in cond_margin_scores.keys()]
    labels = [labels0[el] for el in order]
    plt.figure(figsize=(20,10))
    plt.plot(range(len(cond_margin_scores.values())),
     [cond_margin_scores.values()[el] for el in order],
     color='green', lw=5)
    plt.ylabel('Normalized Margin Score', fontsize=22)
    plt.xticks(range(len(labels)), labels, rotation=70, fontsize=18, ha='center')
    plt.title('Margin Scores for: {0}'.format(feat_name),
        fontsize=24)
    # plt.savefig('/users/pgreens/test_plot.pdf', bbox_inches='tight', format='pdf')
    plot_name='{0}{1}_{2}_across_hema.pdf'.format(PLOT_PATH, feat_name, label)
    plt.savefig(plot_name, bbox_inches='tight', format='pdf')
    print 'DONE: plot here {0}'.format(plot_name)

order_file='/srv/persistent/pgreens/projects/boosting/results/margin_score_files/hema_order_file.txt'
def get_order_vec(cond_margin_scores, order_file):
    with open(order_file) as f: order_labels = f.read().splitlines() 
    f.close()
    order_vec = [key for el in range(len(order_labels))
     for key in range(len(cond_margin_scores))
      if order_labels[el] in cond_margin_scores.keys()[key]]
    return order_vec



feat_name='CEBPA-MA0102.3'
feat_name='Spi1-MA0080.3'
list_of_margin_scores=('/srv/persistent/pgreens/projects/boosting/results/'
    'margin_score_files/hema_margin_score_files_x1_all_down.txt')
# list_of_margin_scores=('/srv/persistent/pgreens/projects/boosting/results/'
#     'margin_score_files/hema_margin_score_files_x1_all_up.txt')
cond_margin_scores = get_margin_score_across_conditions(list_of_margin_scores,
 MARGIN_SCORE_PATH, feat_name, feat_col, margin_score_col)
feat_list=['CEBPA-MA0102.3', 'Gata1-MA0035.3', 'CEBPB-MA0466.1',
 'EBF1-MA0154.2', 'FOXP1-MA0481.1', 'Hoxc9-MA0485.1', 
'Hoxa9-MA0594.1', 'Spi1-MA0080.3', 'TAL1-GATA1-MA0140.2']

feat_col=4
margin_score_col=2
MARGIN_SCORE_PATH=('/srv/persistent/pgreens/projects/boosting/results/2015_'
    '08_15_hematopoeisis_23K_bindingTFsonly_adt_stable_1000iter/margin_scores/')

### Iterate over motif/reg/node up/down to make all trajectories
list_dict = {}
list_dict['motif_down']=('/srv/persistent/pgreens/projects/boosting/results/'
    'margin_score_files/hema_margin_score_files_x1_all_down.txt')
list_dict['motif_up']=('/srv/persistent/pgreens/projects/boosting/results/'
    'margin_score_files/hema_margin_score_files_x1_all_up.txt')
list_dict['reg_down']=('/srv/persistent/pgreens/projects/boosting/results/'
    'margin_score_files/hema_margin_score_files_x2_all_down.txt')
list_dict['reg_up']=('/srv/persistent/pgreens/projects/boosting/results/'
    'margin_score_files/hema_margin_score_files_x2_all_up.txt')
list_dict['node_down']=('/srv/persistent/pgreens/projects/boosting/results/'
    'margin_score_files/hema_margin_score_files_node_all_down.txt')
list_dict['node_up']=('/srv/persistent/pgreens/projects/boosting/results/'
    'margin_score_files/hema_margin_score_files_node_all_up.txt')
feat_dict = {}
feat_dict['motif_down']=x1.row_labels
feat_dict['motif_up']=x1.row_labels
feat_dict['reg_down']=x2.col_labels
feat_dict['reg_up']=x2.col_labels
feat_dict['node_down']=range(1,tree.nsplit)
feat_dict['node_up']=range(1,tree.nsplit)

for key in list_dict.keys():
    if 'node' in key:
        continue
    for feat in feat_dict[key]:
        cond_margin_scores = get_margin_score_across_conditions(
            list_of_margin_scores=list_dict[key], 
            margin_score_path=MARGIN_SCORE_PATH, feat_name=feat,
             feat_col=feat_col, margin_score_col=margin_score_col)
        if len(cond_margin_scores)==14:
            print feat
            plot_margin_score_across_conditions(
            cond_margin_scores=cond_margin_scores, feat_name=feat, label=key)


### Function to generate ex by feature matrix
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

    # Return only peaks that are non-zero in y
    ex_by_feat_mat = ex_by_feat_mat[y.data.toarray().ravel()!=0,]

    return ex_by_feat_mat

    ############### SERIAL ###############
    ### MOTIF MATRIX
 #    if 'motif' in feat:
 #        motif_matrix = csr_matrix((len(example_labels), len(motif_labels)))
 #        # Fill in matrix with index of where motif applies
 #        # for ind in range(len(motif_labels)):
 #        #     motif_matrix.ix[:,ind]=tree.ind_pred_train[1].toarray().flatten()
 #        # Fill in matrix with margin score of motif - SERIAL
 #        for motif in range(len(motif_labels)):
 #            b=margin_score.calc_margin_score_x1(
 #                tree, y, x1, x2, index_mat,
 #                 x1_feat_index=motif, by_example=True).reshape(
 #                 (1, motif_matrix.shape[0]))
 #            motif_matrix[np.where(b!=0),motif]=b[b!=0]
 #    ### REG MATRIX
 #    if 'reg' in feat:
 #        reg_matrix = csr_matrix((len(example_labels), len(regulator_labels)))
 #        # Fill in matrix with margin score of motif - SERIAL
 #        for reg in range(len(regulator_labels)):
 #            b=margin_score.calc_margin_score_x2(
 #                tree, y, x1, x2, index_mat,
 #                 x2_feat_index=reg, by_example=True).reshape(
 #                  (1, reg_matrix.shape[0]))
 #            reg_matrix[np.where(b!=0),reg]=b[b!=0]
 #    ### NODE MATRIX
 #    node_matrix = csr_matrix((len(example_labels), len(node_labels)))
    # # Fill in matrix with margin score of motif - SERIAL
 #    if 'node' in feat:
 #      for node in range(len(node_labels)):
 #          b=margin_score.calc_margin_score_node(
 #              tree, y, x1, x2, index_mat,
 #               node=node, by_example=True).reshape(
 #               (1, node_matrix.shape[0]))
 #          node_matrix[np.where(b!=0),reg]=b[b!=0]
    # ### PATH MATRIX
 #    if 'path' in feat:
 #      path_matrix = csr_matrix((len(example_labels), len(path_labels)))
 #      # Fill in matrix with margin score of motif - SERIAL
 #      for node in range(len(node_labels)):
 #          b=margin_score.calc_margin_score_node(
 #              tree, y, x1, x2, index_mat,
 #               node=node, by_example=True).reshape(
 #               (1, node_matrix.shape[0]))
 #          path_matrix[np.where(b!=0),reg]=b[b!=0]
    ############### SERIAL ###############


# Create matrix of peaks by peaks
###############################################################
def gen_ex_by_ex_matrix():


# Enumerate paths for a tree by listing all nodes in path
###############################################################
def enumerate_paths(tree):
	path_dict = {}
	for node in range(1,tree.nsplit):
		path_dict[node]='|'.join([str(node)]+[
			str(el) for el in tree.above_nodes[node] if el!=0])
	return path_dict






### TEMP LOCATION (make GWAS DF for hema project) 
#################################################################################

peak_file='/srv/persistent/pgreens/projects/hema_gwas/data/hema_peaks_full_n590650.bed'
gwas_dir='/mnt/lab_data/kundaje/users/pgreens/gwas/expanded_LD_geno/rsq_0.8'
out_path='/srv/persistent/pgreens/projects/hema_gwas/results/gwas_matrices/'

thresh = 1e-5
num_peaks = int(os.popen('cat {0} | wc -l'.format(peak_file)).read().split('\n')[0])
all_gwas = os.popen('find {0} -type f'.format(gwas_dir)).read().split('\n')
all_gwas.pop()
num_gwas =  int(os.popen('find {0} -type f | wc -l'.format(gwas_dir)).read().split('\n')[0])


gwas_df = pd.DataFrame(index=range(num_peaks), columns=range(num_gwas))
for ind in xrange(len(all_gwas)):
	gwas_expanded = all_gwas[ind]
	gwas_pruned = '/mnt/lab_data/kundaje/users/pgreens/gwas/pruned_LD_geno/rsq_0.8/'+gwas_expanded.split('_expanded')[0].split('rsq_0.8/')[1]+'.bed'
	col = os.popen("cat {0} | awk '$5>{1}' | cut -f4 | sort | uniq | grep -w -F -f /dev/stdin {2} |bedtools intersect -a {3} -b - -c | cut -f4".format(gwas_pruned, thresh, gwas_expanded, peak_file)).read().split('\n')
	col.pop()
	entry = [int(el) for el in col]
	gwas_df.ix[:,ind]=entry
	print ind

gwas_df.columns=[el.split('/')[-1].split('_pruned')[0] for el in all_gwas]
# gwas_df.to_csv('{0}hema_peaks_w_SIG_gwas_hits_below_thresh{1}.txt'.format(out_path, thresh), sep="\t", header=True, index=False)
gwas_df.to_csv('{0}hema_peaks_w_NONSIG_gwas_hits_above_thresh{1}.txt'.format(out_path, thresh), sep="\t", header=True, index=False)

