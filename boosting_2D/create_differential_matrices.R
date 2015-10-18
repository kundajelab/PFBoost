### Peyton Greenside
### Script to take matrices and output differential expression or accessibility
### 10/13/15
################################################################################################
################################################################################################

### Load libraries
################################################################################################
library(optparse)
library(sva)
library(preprocessCore)
library(limma)

### Usage: 
################################################################################################
# DATA_PATH = '/mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/data/'
# RScript create_differential_matrices.R -a $DATA_PATH"RNA_AML_Samples.txt" -c $DATA_PATH"cell_comparisons_w_leuk_all_hier.txt" \
# -r $DATA_PATH"data/rna_seq/merged_matrix/gene_level_tpm.txt" -g /srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/regulator_names_bindingTFsonly.txt \
# -o $DATA_PATH"boosting_input/regulator_expression.txt"

### Get arguments
################################################################################################

### Arguments with optparse
option_list <- list(
	make_option(c("-a", "--rna_annot_file"). help="Annotation file that contains sample names with annotation to split groups on."),
	make_option(c("-c", "--comparison_file"), help="list of conditions to compare - groups separated by v, combine groups with '|'. ex: grp1|grp2vgrp3 ", default='none'),
	make_option(c("-c", "--comparison_column"), help="list of files ", default='cell_type'),
	make_option(c("-r", "--rna_matrix_file"), help=".bed file with regions to test for GWAS enrichment (works best with DHS only regions otherwise coverage discrepancies)", default='none'),
	make_option(c("-g", "--regulator_file"), help="Name for analysis output directory and file names"),
	make_option(c("-o", "--output_file"), help="Print plots of enrichment.", default=TRUE)
)

opt <- parse_args(OptionParser(option_list=option_list))

### Get Arguments
rna_annot_file = opt$rna_annot_file
comparison_file = opt$comparison_file
comparison_column = opt$comparison_column
rna_matrix_file = opt$rna_matrix_file
regulator_file = opt$regulator_file
output_file = opt$output_file

### Manual Inputs
DATA_PATH = '/mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/data/'
rna_annot_file=paste(c(DATA_PATH,'RNA_AML_Samples.txt'), collapse="")
comparison_file=paste(c(DATA_PATH,'cell_comparisons_w_leuk_all_hier.txt'), collapse="")
comparison_column='cell_type'
rna_matrix_file=paste(c(DATA_PATH,'rna_seq/merged_matrix/gene_level_tpm.txt'), collapse="")
regulator_file='/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/regulator_names_bindingTFsonly.txt'
output_file=paste(c(DATA_PATH,'boosting_input/regulator_expression.txt'), collapse="")

### Read in files
################################################################################################
rna_annots = read.table(rna_annot_file, sep="\t", header=TRUE, stringsAsFactors=FALSE)
comparisons = read.table(comparison_file, sep="\t", header=FALSE, stringsAsFactors=FALSE)
rna_matrix = read.table(rna_matrix_file, sep="\t", stringsAsFactors=FALSE, header=TRUE,check.names=FALSE)
regulator_list = read.table(regulator_file, sep="\t", stringsAsFactors=FALSE)[,1]

### Get differential matrix
################################################################################################

# Initialize the whole matrix
rna_diff_mat0 = matrix(0,ncol=length(comparisons[,1]), nrow=nrow(rna_matrix))
colnames(rna_diff_mat0) = comparisons[,1]
rownames(rna_diff_mat0) = rownames(rna_matrix)

# Iterate through comparisons and input differential 
for (comp in comparisons[,1]){

	# Print entry
	print(comp)

	# Get matrix with samples for comparison
	grp1_labels = strsplit(strsplit(comp, 'v')[[1]][1], "\\|")[[1]]
	grp2_labels = strsplit(strsplit(comp, 'v')[[1]][2], "\\|")[[1]]
	grp1_samples = rna_annots$sample_name[rna_annots[,comparison_column] %in% grp1_labels]
	grp2_samples = rna_annots$sample_name[rna_annots[,comparison_column] %in% grp2_labels]
	comp_matrix = rna_matrix[,c(grp1_samples, grp2_samples)]
	comp_matrix = comp_matrix[which(apply(comp_matrix, 1, max)>0),]

	# Normalize samples
	comp_matrix_asinh = asinh(comp_matrix)
	comp_matrix_asinh_qn = normalize.quantiles(as.matrix(comp_matrix_asinh), copy=TRUE)
	rownames(comp_matrix_asinh_qn) = rownames(comp_matrix_asinh)
	colnames(comp_matrix_asinh_qn) = colnames(comp_matrix_asinh)

	# Calculate differential expression between the samples 
	comp_matrix_asinh_qn_log = log2(as.matrix(comp_matrix_asinh_qn)+1)
	comp_annots = rna_annots[match(colnames(comp_matrix_asinh_qn_log),rna_annots$sample_name),]
	comp_annots['comp']=sapply(comp_annots[,comparison_column], function(x) ifelse(x %in% grp1_labels, "grp1", "grp2"))
	full.model <- model.matrix(~ as.factor(comp) , data = comp_annots)
	# null.model <- model.matrix(~  1, data = comp_annots)
	# svobj <- sva(dat=comp_matrix_asinh_qn_log, mod = full.model, mod0 = null.model)
	# full.model.sv <- cbind(full.model, svobj$sv)
	# fit <- lmFit(comp_matrix_asinh_qn_log, full.model.sv)
	fit <- lmFit(comp_matrix_asinh_qn_log, full.model)
	ebfit <- eBayes(fit)
	tophits <- topTable(ebfit, coef = "as.factor(comp)grp2", number = Inf)

	# Allocate results into complete matrix
	diff_genes_up = rownames(tophits)[intersect(which(tophits$adj.P.Val<=0.05), which(tophits$logFC>0))]
	diff_genes_down = rownames(tophits)[intersect(which(tophits$adj.P.Val<=0.05), which(tophits$logFC<0))]
	rna_diff_mat0[diff_genes_up,comp]=1
	rna_diff_mat0[diff_genes_down,comp]=-1

}

# Check the number of genes that are differentially significant between conditions
irw_counts = apply(rna_diff_mat0, 2, function(x) sum(x!=0))
twostep_counts = apply(rna_diff_mat0, 2, function(x) sum(x!=0))
nosva_counts = apply(rna_diff_mat0, 2, function(x) sum(x!=0))

### Subset to allowable regulator list
if (regulator_file!='none'){
	binding_ind = which(rownames(rna_diff_mat0) %in% regulator_list)
	rna_diff_mat = rna_diff_mat0[binding_ind,]
} else {
	rna_diff_mat = rna_diff_mat0
}

# Write out matrix
################################################################################################

dense_output_file = cpaste(c(strsplit(output_file, '.txt')[[1]][1], 'dense.txt'), collapse="")
sparse_output_file = output_file
write.table(rna_diff_mat, dense_output_file, quote=FALSE, sep="\t", col.names=TRUE, row.names=TRUE)

# Convert to sparse format
system(sprintf('/srv/git/boosting_2D/boosting_2D/convert_dense_to_sparse.py --input-file %s --output-file %s --with-labels', dense_output_file, sparse_output_file)

# Remove dense version
system(sprintf('rm %s', dense_output_file))

# Print when finished
sprintf('DONE: find the output matrix in: %s', sparse_output_file)



