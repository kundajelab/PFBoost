# boosting2D

### pyBoost: a software package to learn gene regulatory programs

Contact: Peyton Greenside (pgreens@stanford.edu), Kundaje Lab @ Stanford

Github Repository: https://github.com/kundajelab/boosting2D 

### When to use:  
Learn the regulatory programs - transcriptional regulators and their corresponding sequence motifs - that govern dynamic patterns of chromatin accessibility, gene expression, or other phenotypes across conditions. Conditions can be time courses, different cell types, or other experimental perturbations.

### A short description of the method:  
The method consists of two parts: 1) Build an alternating decision tree model to predict output (accessibility/expression) based on sequence motifs and expressed regulators; 2) Interpret this tree to learn which motifs and regulatory proteins are controlling the regulation of which examples (genes/non-coding regions).  

1) At every iteration, the method pairs a motif and regulator that best predict changes in the target expression/accessibility matrix. This motif-regulator pair with its corresponding index into the target matrix and corresponding prediction (up/down regulated) is known as one “rule.” For 1000 iterations we have 1000 such rules that form an alternating decision tree. After each iteration the data is re-weighted to up-weight examples that are incorrectly predicted and the next rule is chosen based on the current weighting of the training examples. 

2) After generating a model to learn the rules that predict over the entire target matrix, we can use the learned alternating decision tree model to look at any subset of the matrix (any set of conditions or peaks or combination thereof) to learn which regulatory programs govern the data subset of interest. This occurs in the post-processing steps in a separate run from training the main model. It's usually best to train the model across all conditions together with all captured dynamics, and then look at each condition or set of peaks/genes individually in the post-processing steps. 

### The how-to-use summary: 
In order to use this code you will need 3 matrices: one each for the motifs/x1 matrix (0/1 if a motif is absent/present), regulators/x2 matrix (-1/0/1 for decreased/no change/increased or 0/1 for not expressed/expressed) and target expression or chromatin accessibility/y matrix (-1/0/1 for decreased/no change/increased or -1/1 for not expressed/expressed) and 4 sets of labels for the four unique dimensions of the three matrices. Provide an output path and a relevant label for your analysis folder and you’re pretty much good to go. 

# Installation

I strongly recommend first installing anaconda for Python 2.7 for easy installation: https://www.continuum.io/downloads

Clone the repository here: https://github.com/kundajelab/boosting2D.git. For access, email pgreens@stanford.edu.
 
Install the software with “python setup.py install” in the boosting2D repository.  

# Main Model

How to run vanilla main model:

python run_boosting_2D.py --output-prefix name_of_analysis --motifs-file MOTIF_matrix.txt --regulator-file RNA_matrix.txt --target-file ATAC_matrix.txt --target-row-labels LABEL_peaks.txt --target-col-labels LABEL_conditions.txt --m-row-labels LABEL_motifs.txt --r-col-labels LABEL_regulators.txt --ncpu 1 --output-path /path/to/result/folder/ --stable --num-iter 1000 

## Arguments:

### Output Location  

--output-prefix   
The label associated with the analysis (e.g. hematopoiesis_data)

--output-path  
Output path for results folder (e.g. /path/to/results/)

### Matrices   

--target-file 
What you want to predict (typically chromatin accessibility or gene expression). Takes -1/0/1 or -1/1. Dimension (peaks x conditions).

--motifs-file 
The motifs associated with the genomic regions in the target matrix. Takes 0/1. Dimension (motifs x peaks).

--regulator-file 
 The transcriptional regulators (TF binding proteins or genes with GO term “transcriptional regulation” or similar). Takes -1/0/1 or 0/1. Dimension (conditions x regulators).

### Labels:  

--target-row-labels 
Peak or gene names associated with rows of target matrix and columns of motif matrix. NOTE: If you plan to use the current implementation of unsupervised post processing, peaks should be labeled as “chrX:START-END”. Any other annotations must be separated with a colon such as  “chrX:START-END;other_annotations”. Example peak label: “chr5:200-700;BRAF”. If you are just training a model, any format is fine.

--target-col-labels 
Condition labels associated with columns of target matrix and rows of regulator matrix

--m-row-labels
Motif labels. Matches number of rows in motif matrix

--r-col-labels 
Regulator labels. Matches number of columns in regulator matrix.

### Input Parameters  

--input-format 
“matrix or triplet” - Triplet is 3 columns: row, column, value. Matrix is typical dense matrix. Default is matrix.

--mult-format 
“dense or sparse”  - Format of matrices for algorithm and for matrix multiplication. Currently sparse is faster. Default is sparse.

### Tuning parameters:  

--num-iter 
Number of iterations to boost for. At each iteration one rule (motif+regulator) is added. Default 500.

--eta1 
Stabilization threshold for deciding if stabilization is used. Default 0.05.

--eta2  
Stabilization threshold for deciding bundle size. Default 0.01.

--ncpu 
Number of CPUs to parallelize over. Default is 1.

### Optional Tuning parameters:  

--stable  
Stabilize by joining correlated features. Recommended. 

--max-bundle-size  
Maximum bundle size for stabilized boosting. Default is 20.

--plot  
Produce plots of balanced and imbalanced training error and the margin

--stumps  
Use stumps instead of trees. Rarely used, generally worse performance.

### Prior Matrices: 

--use-prior 
Add flag if using a prior matrix

--prior-input-format 
“matrix or triplet” - Triplet is 3 row, column, value. Matrix is typical dense matrix. Default is matrix.

--motif-reg-file 
Prior matrix for motifs associated with certain regulators. Real valued [0,1].

--motif-reg-row-labels 
Motif labels for prior matrix. BE SURE these match the labels of the motif matrix.

--motif-reg-col-labels 
Regulator labels for prior matrix. BE SURE these match the labels of the regulator matrix.

--reg-reg-file 
Prior matrix for which regulators interact. Such as from PPI. Real valued [0,1]. Symmetric.

--reg-reg-labels 
Regulator row and column labels (should be the same) for prior matrix. BE SURE these match the labels of the regulator matrix.


### Holdout Matrix    

--holdout-file 
Path to a file that contains a binary holdout matrix. Default is to randomly generate a holdout matrix with each run, but to reproduce results one can use the a pre-specified holdout matrix

--holdout-format 
“matrix or triplet” - Triplet is 3 row, column, value. Matrix is typical dense matrix. Default is matrix.

--train-fraction 
Specifies the fraction of data used for training. Validation set will be the remainder. Default is 0.8. If a holdout matrix is supplied then this option is ignored.


### Randomization  

--shuffle-y 
Randomly shuffle y-values across entire matrix. Often used for a NULL model to compare margin scores in the true model.

--shuffle-x1 
Randomly shuffle x1-values (motif scores) across entire matrix

--shuffle-x2 
Randomly shuffle x2-values (regulator scores) across entire matrix

### Saving model state  

--save-tree-only 
Save a pickled version of the learned ADT. Default: False.

--save-complete-data  
This will save every data structure including x1, x2, y, holdout, hierarchy, tree, and priors and generate a Python script to load these saved data structures. Default: False

--save-for-post-processing  
Saves x1, x2, tree, prior, configs and hierarchy as needed for any post-processing functionalities as well as a script to load these data structures. Default: True

### Additional Options  
 
--perf-metrics 
Can specify particular performance metrics to calculate. Default will print imbalanced error and store imbalanced and balanced error in the tree. Can specify any combination of [imbalanced_error,  balanced_error, auPRC, auROC] to store those metrics. When the labels have three values [-1,0,1], the metrics are calculated only on -1/1 labels. Default: None.

--compress-regulators  
Condense regulators that have the exact same pattern. Concatenates the labels and condenses to one column.

--compress-motifs (COMING)
Condense motifs that have the exact same pattern. Concatenates the labels and condenses to one column.

## Outputs:  

LOG_FILE.txt 
The first section is the exact command run. The second section shows the timing of the run.

global_rules__DATE_OUTPUT-PREFIX_MODEL-PARAMS_NUM-ITER.txt 
List of the rules learned at each iteration. Columns are x1_feat (motif), x2_feat (regulator), score (positive indicates predictive of increased accessibility/expression, negative indicates predictive of decreased accessibility/expression), above_rule (which node this rule was added on to), tree-depth (0 for root otherwise length of path added on to)

load_pickle_data_script.py 
Script to load pickled data (y, x1, x2, prior, tree object) to perform analysis in environment of model run. Can load in a python session with “execfile(‘/path/to/load_pickle_data_script.py’)

saved_complete_model__DATE_OUTPUT-PREFIX_MODEL-PARAMS_NUM-ITER.gz 
Pickled, gzipped dictionary of all data objects used in model: x1, x2, y, tree, prior

saved_tree_state__DATE_OUTPUT-PREFIX_MODEL-PARAMS_NUM-ITER.gz 
Pickled, gzipped tree state only. May be deprecated.

plots 
Folder for all plots. Standard 3 to produce with --plot are: balanced training/testing error, imbalanced training/testing error, margin of training/testing 


## Hierarchical Boosting

If you wish to impose a hierarchy (i.e. cell differentiation tree, time course, etc.) where rules can be applied to various points in the hierarchy, you should switch to the “hierarchical_boosting” branch that has this capability. 

Every rule added to the ADT has an additional feature of “hierarchy node” that indicates a pointer to a node in the hierarchy. The rule then applies to all children and sub-children of that hierarchy node including the node itself. When adding a new node to the ADT, the new node can either be added to the current hierarchy node of the ADT node being added to or to any of the hierarchy children and all of those nodes are checked for minimum loss. For example, if the first ADT node “ADT node 1” is added at the root of the hierarchy (i.e. applies to all nodes in the hierarchy) and that hierarchy node has two children, then the next ADT node in the ADT added on to “ADT node 1” can be applied to either the hierarchy root or its two children. Each path in the ADT can then follow a path down a hierarchy. 

To use this feature, you must first encode your hierarchy in hierarchy.py. Give your hierarchy a name. Then give every node in your hierarchy a number. Define the total number of nodes in your hierarchy and define two dictionaries: subtree_nodes and direct_children. For each key (i.e. hierarchy node number), subtree_nodes indicates all the other nodes including the node itself that are below that node in the hierarchy. Direct_children indicates just the direct children including the node itself. Direct_children is thus a subset of subtree_nodes. Include the rood as pointing to the first node in the hierarchy.

Take an example. If we have a 3 level hierarchy with a root (node 0) that has two children (nodes 1 and 2) and each of those two children has two children (nodes 3 and 4 coming from node 1 and nodes 5 and 6 coming from node 2), we could encode the hierarchy as follows:

    elif name == 'sample_7node_hierarchy':

        NUM_NODES = 7

        ### Define the direct children of each internal node

        direct_children = {}
        direct_children['root'] = [0]
        direct_children[0] = [0, 1, 2]
        direct_children[1] = [1, 3, 4]
        direct_children[2] = [2, 5, 6]
        direct_children[3] = [3]
        direct_children[4] = [4]
        direct_children[5] = [5]
        direct_children[6] = [6]

        ### Define the participant nodes in each subtree

        subtree_nodes = {}
        subtree_nodes['root'] = [0, 1, 2, 3, 4, 5, 6]
        subtree_nodes[0] = [0, 1, 2, 3, 4, 5, 6]
        subtree_nodes[1] = [1, 3, 4]
        subtree_nodes[2] = [2, 5, 6]
        subtree_nodes[3] = [3]
        subtree_nodes[4] = [4]
        subtree_nodes[5] = [5]
        subtree_nodes[6] = [6]

After adding your encoded hierarchy to hierarchy.py, use the following flag to impose this hierarchy on your rules:

--hierarchy_name
Name of hierarchy corresponding to your implemented hierarchy in hierarchy.py


# Post-processing 

Overview: Post-processing is performed separately with the saved and compressed main model. The post-processing module reloads the 3 matrices as well as the learned model to explore the conditions, genes or regions of interest you want to learn about based on (0, 1 or ) 2 index files you provide. One index file specifies conditions (columns of the target matrix) to look at, and the other index files specific genes/peaks (rows of the target matrix). You can use them separately or together. If you supply none then the margin will be calculated over all the data. Using a margin-score based approach, you can learn which motifs (x1 option) and regulators (x2 option) control each part of your target matrix. You can also look at motifs and regulators together (node option) as well as groups of motifs and regulators that are dependent in the model (path option). 

How to run vanilla post processing:

python run_post_processing.py --model-path /result/path/output_prefix/load_pickle_data_script.py --analysis-label
margin_score_output_name --run-margin-score --num-perm 100 --margin-score-methods x1,x2 --region-feat-file desired_peaks.txt --condition-feat-file desired_conditions.txt

## Margin Score

### Arguments  

--model-path 
path to load_pickle_data_script.py (e.g. /path/to/results/load_pickle_data_script.py)

--analysis-label 
label for particular margin score calculation. If calculating margin score over all examples, leave blank and default is “full_model” If providing --condition-feat-file or --region-feat-file be sure to provide a prefix

--run-margin-score  
Add flag to calculate margin score (in the future there will be more options so this will make more sense)

--condition-feat-file  
Path to a text file listing the conditions - exactly matching the labels for the target and x2 matrices - you are interested in learning the regulatory program for.

--region-feat-file  
Path to a text file listing the peaks (for predicting chromatin accessibility) or genes (for predicting expression) - exactly matching the labels for the target and x1 matrices -  you are interested in learning the regulatory program for.

--margin-score-methods 
options are: [x1, x2, node, path]. Include multiple options separated by commas. x1 calculates margin score by individual motif, x2 calculates margin score by individual regulator, node calculates margin score by individual node (joint motif+regulator) and path operates by removing an entire path (motif+regulator at every node leading to leaf node). People usually enter ‘x1,x2’

--num-perm 
Number of permutations of shuffling y matrix and re-calculating y matrix to calculate empirical p-values for each margin score in the real target matrix. Default 100, more recommended.

--null-tree-model 
Instead of shuffling the target matrix to generate empirical p-values you can provide an alternate model in the form of “/path/to/model/model_prefix/saved_tree_state__model_prefix.gz”. For example a model generated with the --shuffle-y flag. 

### Outputs:  

margin_scores
Folder for all post processing margin score. Results generally take the form: LABEL_{X1/X2/NODE/PATH}_{ALL/ENH/PROM}_{UP/DOWN}_margin_score.txt. Columns are:
margin_score: Raw difference in margin (-y*f(x)) between original model and model with feature removed. For x1, remove any node with specified motif or bundled with specified motif (--stable option). For x2, remove any node with specified regulator or bundled with specified regulator (--stable option). For node, remove specific node and all nodes added below the node. For path, remove effect from all nodes in path over only examples that follow to end of path (e.g. if path of length 2 from root -> A -> B, remove score from A * index of node B and score of B * index of node B). 
margin_score_norm:  margin score normalized by the the size of the index matrix (only elements in index matrix that are UP/DOWN
rule_index_fraction:  percent of index matrix the rule applies to (aka all examples where motif is present out of all examples in index matrix)
{x1/x2/node/path}_feat:  feature)
{x1/x2/node/path}_feat_bundles:  (features bundled with primary feature in --stable option)
pvalue - based on empirical distribution from shuffled y matrix and re-computed margin scores for the same index matrix

## Get example-by-feature matrix

This returns a matrix that contains all the normalized margin score for every motif and regulator for each peak and cell type. The rows are conditions concatenated with peaks (i.e. cell|peak for all peak and cell combinations provided in the file)  by motf|reg|node|path or features selected to contain. Writes matrix to folder “ex_by_feat_matrix” in the OUTPUT_PATH.

Example usage:
python run_post_processing.py --model-path /result/path/output_prefix/load_pickle_data_script.py --return-ex-by-feat-matrix  --features-to-use motif  --region-feat-file peak_file.txt --condition-feat-file cell_file.txt --analysis-label my_label

--model-path 
path to load_pickle_data_script.py (e.g. /path/to/results/load_pickle_data_script.py)

--return-ex-by-feat-matrix 
Write the example by feature matrix to a file

--condition-feat-file 
Path to a text file listing the conditions - exactly matching the labels for the target and x2 matrices - you are interested in learning the regulatory program for.

--region-feat-file 
Path to a text file listing the peaks (for predicting chromatin accessibility) or genes (for predicting expression) - exactly matching the labels for the target and x1 matrices -  you are interested in learning the regulatory program for.

--features-to-use 
Which features to form the example-by-feature matrix from. Enter multiple features separated by a comma. Options: [‘motif’, ‘reg’, ‘node’, ‘path’]

--analysis-label  
Use this for labelling the output of a subset example feature matrix

Suggestions for generating data matrices :

Motif Matrix:
User Homer (homer2 find) or Nathan Boley’s curated motifs

Regulator Matrix:
Subset genes to transcriptional regulators either by selecting genes with a DNA binding domain or genes with an associated GO term that suggests acting as a transcriptional regulator


## Additional Notes

If you have installed a multi-threaded version of numpy (e.g using a BLAS library for numpy or MKL), a larger number of threads may be used. You can limit this behavior by specifying “export OPENBLAS_NUM_THREADS=1”, “export MKL_NUM_THREADS=1”, etc.
When multiple rules have the exact same loss and stabilization is NOT used, then the algorithm will randomly sample one of the features for the rule


## Methods Description

We learn regulatory programs governing transitions across conditions using a confidence-rated boosting framework. We formulate this learning of regulatory programs as a binary prediction task to predict differences in accessibility across conditions from motifs present in ATAC-seq peaks and condition-specific expression of transcriptional regulators. The boosting model uses the Adaboost learning algorithm with a margin-based generalization of decision trees called alternating decision trees (ADTs). After boosting for many iterations we learn an alternating decision tree where each node contains the transcriptional regulator and motif that minimize loss for the current set of weights as well as the specific ATAC-seq examples for which that regulator-motif pair apply. This alternating decision tree is then dissected to understand the regulatory programs governing the changes in accessibility across conditions.

We used a margin-based score to learn the motifs and regulators governing specific sets of peaks for a given set of conditions. For each motif and regulator, we remove its individual contribution to the margin of prediction and calculate the overall change in margin for a specified subset of peaks and conditions. We rank all motifs and regulators by the change in margin, normalized for the total number of differential examples in the subset. We separately calculate scores for examples that increase in accessibility and examples that decrease in accessibility. 

To illustrate the differing regulatory programs across conditions, we calculate normalized margin-based scores for each motif and regulator on a per example basis for ATAC-seq peaks in a set of conditions. We concatenated these scores into a matrix of examples by motifs and regulators and then hierarchically clustered the scores to observe clusters of motifs and regulators that constitute regulatory programs governing co-regulated regions of the genome.
















			
		

