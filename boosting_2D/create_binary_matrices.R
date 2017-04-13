#!//usr/bin/Rscript
### Peyton Greenside
### Script to take matrices and output differential expression or accessibility
### 10/13/15
################################################################################################
################################################################################################

### Load libraries
################################################################################################
library(optparse)
library(preprocessCore)
library(doParallel)
library(foreach)

### Usage: 
################################################################################################
# DATA_PATH=/mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/data/
# SCRIPT_PATH=/users/pgreens/git/boosting_2D/boosting_2D/
# $SCRIPT_PATH"create_differential_matrices.R" -a $DATA_PATH"RNA_AML_Samples.txt" -f $DATA_PATH"cell_comparisons_w_leuk_all_hier.txt" \
# -c cell_type \
# -r $DATA_PATH"/rna_seq/merged_matrix/gene_level_counts.txt" -g /srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/regulator_names_bindingTFsonly.txt \
# -o $DATA_PATH"boosting_input/regulator_expression_deseq.txt" -m deseq

### Get arguments
################################################################################################

### Arguments with optparse
option_list <- list(
    make_option(c("-a", "--annot_file"), help="Annotation file that contains sample names with annotation to split groups on."),
    make_option(c("-f", "--comparison_file"), help="list of conditions to compare - groups separated by v, combine groups with '|'. ex: grp1|grp2vgrp3 ", default='none'),
    make_option(c("-c", "--comparison_column"), help="list of files ", default='cell_type'),
    make_option(c("-r", "--data_matrix_file"), help="matrix of counts (RNA/ATAC) or TPM (RNA)", default='none'),
    make_option(c("-g", "--regulator_file"), help="Name for analysis output directory and file names. E.g. list of regulators for an RNA matrix", default="none"),
    make_option(c("-o", "--output_file"), help="Name of PATH+FILE for output differential matrix"),
    make_option(c("-l", "--label_output_file"), help="Name of PATH+FILE for labels. If not provided, will not write label", default='none'),
    make_option(c("-m", "--binary_thresh"), help="threshold for being considered present", default=5),
    make_option(c("-t", "--out_format"), help="Specify ['dense', 'sparse'] to request specific output format", default='sparse'),
    make_option(c("-s", "--serial"), help="Specify ['dense', 'sparse'] to request specific output format", action="store_true", default=FALSE))

opt <- parse_args(OptionParser(option_list=option_list))

### Get Arguments
annot_file = opt$annot_file
comparison_file = opt$comparison_file
comparison_column = opt$comparison_column
data_matrix_file = opt$data_matrix_file
regulator_file = opt$regulator_file
output_file = opt$output_file
label_output_file = opt$label_output_file
binary_thresh = opt$binary_thresh
pval = opt$pval
out_format = opt$out_format
foldchange = opt$foldchange
serial = opt$serial

### Manual Inputs (Nadine)
# DATA_PATH = '/mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/data/'
# annot_file=paste(c(DATA_PATH,'RNA_AML_Samples.txt'), collapse="")
# comparison_file=paste(c(DATA_PATH,'cell_comparisons_w_leuk_all_hier_nadine_wrt_HSC.txt'), collapse="")
# comparison_column='cell_type'
# data_matrix_file=paste(c(DATA_PATH,'rna_seq/merged_matrices/gene_level_counts.txt'), collapse="")
# regulator_file='/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/regulator_names_bindingTFsonly.txt'
# output_file=paste(c(DATA_PATH,'boosting_input/regulator_expression_binary_nadine_dense.txt'), collapse="")
# binary_thresh=5

# Command Inputs (Nadine expression April 12 new regulator set)
# SCRIPT_PATH=/users/pgreens/git/boosting_2D/boosting_2D/
# $SCRIPT_PATH"create_binary_matrices.R" \
# -a /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/data/RNA_AML_Samples.txt \
# -r /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/data/rna_seq/merged_matrices/gene_level_counts.txt \
# -f /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/data/cell_comparisons_w_leuk_all_hier_nadine_binary.txt \
# -c cell_type \
# -g /mnt/lab_data/kundaje/users/pgreens/projects/modisco/data/combined_regulators_CISBP_and_GO:0003677_DNA_binding_unique_gene_names_n2790_4_12_17.txt \
# -o /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/data/boosting_input/expression_binary_nadine_dense_april12_new_regulators.txt \
# -m 5 -t dense


sprintf('output file: %s', output_file)
sprintf('regulator file: %s', regulator_file)
sprintf('out_format file: %s', out_format)
sprintf('annots file: %s', annot_file)
sprintf('pval: %s', pval)
sprintf('regulator_file: %s', regulator_file)

### Read in files
################################################################################################

annots = read.table(annot_file, sep="\t", header=TRUE, stringsAsFactors=FALSE, fill=TRUE)
comparisons = read.table(comparison_file, sep="\t", header=FALSE, stringsAsFactors=FALSE)
data_matrix = read.table(data_matrix_file, sep="\t", stringsAsFactors=FALSE, header=TRUE,check.names=FALSE,row.names=1)

### Get differential matrix
################################################################################################

# Initialize the whole matrix
unique_entries = unlist(unique(annots[comparison_column]))
binary_mat0 = matrix(0,ncol=length(unique_entries), nrow=nrow(data_matrix))
colnames(binary_mat0) = unique_entries
rownames(binary_mat0) = rownames(data_matrix)

###  PARALLELIZED
################################################################################################
################################################################################################

compute_binary_expression<-function(cell, data_matrix, binary_thresh) {
    # Print entry
    print(cell)

    # Get matrix with samples for comparison
    grp_labels = strsplit(strsplit(cell, 'v')[[1]][1], "\\|")[[1]]
    grp_samples = annots$sample_name[annots[,comparison_column] %in% grp_labels]

    # Get mean across sample
    sample_matrix = data_matrix[,grp_samples]
    sample_means = apply(sample_matrix, 1, mean)

    binary_labels = ifelse(sample_means > binary_thresh, 1, 0)
    return(binary_labels)
}

### PARALLEL VERSION
if (!serial){
  registerDoParallel(cores=1)
  x <- foreach(i=unique_entries, .combine='cbind') %dopar% compute_binary_expression(i, data_matrix, binary_thresh)
  x[is.na(x)]=0
  colnames(x)=colnames(binary_mat0)
  rownames(x)=rownames(binary_mat0)
  binary_mat0=x
} 

### SERIAL VERSION
if (serial){
    for (cell in unique_entries){
        binary_mat0[,cell]=compute_binary_expression(cell, data_matrix, binary_thresh)
    }
}

### Subset to allowable regulator list
if (regulator_file!='none'){
    regulator_list = read.table(regulator_file, sep="\t", stringsAsFactors=FALSE)[,1]
    binding_ind = which(rownames(binary_mat0) %in% regulator_list)
    binary_mat = binary_mat0[binding_ind,]
} else {
    binary_mat = binary_mat0
}

# Write out matrix
################################################################################################

dense_output_file = paste(c(strsplit(output_file, '.txt')[[1]][1], '_dense.txt'), collapse="")
sparse_output_file = output_file
write.table(binary_mat, dense_output_file, quote=FALSE, sep="\t", col.names=TRUE, row.names=TRUE)

if (out_format=='dense'){
    # Move to correct file name
    system(sprintf("mv %s %s", dense_output_file, output_file))
}
if (out_format=='sparse'){
    # Convert to sparse format
    system(sprintf('python /users/pgreens/git/boosting_2D/boosting_2D/convert_dense_to_sparse.py --input-file %s --output-file %s --with-labels', dense_output_file, sparse_output_file))

    # Remove dense version
    system(sprintf('rm %s', dense_output_file))
}

# Write out row labels
################################################################################################
if (label_output_file!='none'){
    write.table(data.frame(rownames(binary_mat)), label_output_file, sep="\t", quote=FALSE, col.names=FALSE, row.names=FALSE)
    sprintf('Wrote labels to: %s', label_output_file)
}

# Print when finished
sprintf('DONE: find the output matrix in: %s', sparse_output_file)



