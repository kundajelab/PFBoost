#!//usr/bin/Rscript
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
library(DESeq2)
library(doParallel)
library(foreach)
# library(BiocParallel)
# register(MulticoreParam(4))

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
	make_option(c("-m", "--method"), help="either [deseq_svaseq, sva_limma, deseq]", default="deseq"),
	make_option(c("-p", "--pval"), help="Instead of generating binary matrix [-1/0/+1] output p-value for each comp for each region", action="store_true", default=FALSE),
	make_option(c("-t", "--out_format"), help="Specify ['dense', 'sparse'] to request specific output format", default='sparse'),
	make_option(c("-s", "--serial"), help="Specify ['dense', 'sparse'] to request specific output format", action="store_true"))

opt <- parse_args(OptionParser(option_list=option_list))

### Get Arguments
annot_file = opt$annot_file
comparison_file = opt$comparison_file
comparison_column = opt$comparison_column
data_matrix_file = opt$data_matrix_file
regulator_file = opt$regulator_file
output_file = opt$output_file
label_output_file = opt$label_output_file
method = opt$method
pval = opt$pval
out_format = opt$out_format
serial = opt$serial

### Manual Inputs
# DATA_PATH = '/mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/data/'
# annot_file=paste(c(DATA_PATH,'RNA_AML_Samples.txt'), collapse="")
# comparison_file=paste(c(DATA_PATH,'cell_comparisons_w_leuk_all_hier.txt'), collapse="")
# comparison_column='cell_type'
# data_matrix_file=paste(c(DATA_PATH,'rna_seq/merged_matrix/gene_level_tpm.txt'), collapse="")
# data_matrix_file=paste(c(DATA_PATH,'rna_seq/merged_matrix/ .txt'), collapse="")
# regulator_file='/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/regulator_names_bindingTFsonly.txt'
# # regulator_file='/mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/data/regulator_names_GOterms_transcript_reg.txt'
# method='deseq_svaseq'
# output_file=paste(c(DATA_PATH,sprintf('boosting_input/regulator_expression_%s.txt', method)), collapse="")

### Manual Inputs 2
# annot_file="/mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/data/atac_seq/merged_matrices/Leuk_35M_annots_for_pseudorep_counts.txt"
# comparison_file="/mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/data/cell_comparisons_w_leuk_all_hier.txt"
# comparison_column='cell_type'
# data_matrix_file="/mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/data/atac_seq/merged_matrices/Leuk_35M_counts_per_peak_merged_macsqval5_pseudoreps_labelled.txt"
# regulator_file='none'
# # regulator_file='/mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/data/regulator_names_GOterms_transcript_reg.txt'
# method='deseq'
# output_file="/mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/data/boosting_input/regulator_expression_deseq_pval.txt"
# out_format = 'dense'
# pval=TRUE

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
data_diff_mat0 = matrix(0,ncol=length(comparisons[,1]), nrow=nrow(data_matrix))
colnames(data_diff_mat0) = comparisons[,1]
rownames(data_diff_mat0) = rownames(data_matrix)


###  PARALLELIZED
################################################################################################
################################################################################################

compute_pvals<-function(comp, method, data_diff_mat0) {
	# Print entry
	print(comp)

	# Get matrix with samples for comparison
	grp1_labels = strsplit(strsplit(comp, 'v')[[1]][1], "\\|")[[1]]
	grp2_labels = strsplit(strsplit(comp, 'v')[[1]][2], "\\|")[[1]]
	grp1_samples = annots$sample_name[annots[,comparison_column] %in% grp1_labels]
	grp2_samples = annots$sample_name[annots[,comparison_column] %in% grp2_labels]
	comp_matrix = data_matrix[,c(grp1_samples, grp2_samples)]
	comp_matrix = comp_matrix[which(apply(comp_matrix, 1, max)>0),]

	# Proceed with TPM data
	if (method=='sva_limma'){
		# Normalize samples
		comp_matrix_asinh = asinh(comp_matrix)
		comp_matrix_asinh_qn = normalize.quantiles(as.matrix(comp_matrix_asinh), copy=TRUE)
		rownames(comp_matrix_asinh_qn) = rownames(comp_matrix_asinh)
		colnames(comp_matrix_asinh_qn) = colnames(comp_matrix_asinh)

		# Calculate differential expression between the samples 
		comp_matrix_asinh_qn_log = log2(as.matrix(comp_matrix_asinh_qn)+1)
		comp_annots = annots[match(colnames(comp_matrix_asinh_qn_log),annots$sample_name),]
		comp_annots['comp']=sapply(comp_annots[,comparison_column], function(x) ifelse(x %in% grp1_labels, "grp1", "grp2"))
		full.model <- model.matrix(~ as.factor(comp) , data = comp_annots)
		null.model <- model.matrix(~  1, data = comp_annots)
		svobj <- sva(dat=comp_matrix_asinh_qn_log, mod = full.model, mod0 = null.model)
		full.model.sv <- cbind(full.model, svobj$sv)
		fit <- lmFit(comp_matrix_asinh_qn_log, full.model.sv)
		ebfit <- eBayes(fit)
		tophits <- topTable(ebfit, coef = "as.factor(comp)grp2", number = Inf)

		# data_diff_mat0[match(rownames(tophits), rownames(data_diff_mat0)),comp]=tophits$adj.P.Val
		if (pval==TRUE){
			ordered_pvals=tophits$adj.P.Val[rownames(data_diff_mat0),match(rownames(tophits))]
			return(ordered_pvals)
		} else {
			# Allocate results into complete matrix
			result_vec = data_diff_mat0[,comp]
			diff_genes_up = rownames(tophits)[intersect(which(tophits$adj.P.Val<=0.05), which(tophits$logFC>0))]
			diff_genes_down = rownames(tophits)[intersect(which(tophits$adj.P.Val<=0.05), which(tophits$logFC<0))]
			result_vec[diff_genes_up]=1
			result_vec[diff_genes_down]=-1
			return(unname(result_vec))
		}
	# Proceed with count data DESeq
	} else if (method=='deseq'){
		# Calculate differential expression with DESeq
		comp_matrix_rounded = round(comp_matrix)
		comp_annots = annots[match(colnames(comp_matrix_rounded),annots$sample_name),]
		comp_annots['comp']=sapply(comp_annots[,comparison_column], function(x) ifelse(x %in% grp1_labels, "grp1", "grp2"))
		conditions = factor(comp_annots[,'comp'])
		dds = DESeqDataSetFromMatrix(countData=comp_matrix_rounded, colData=comp_annots, design = ~comp)
		# dds <- DESeq(dds, parallel=TRUE)
		# res <- results(dds, independentFiltering=FALSE, parallel=TRUE)
		dds <- DESeq(dds, parallel=FALSE)
		res <- results(dds, independentFiltering=FALSE, parallel=FALSE)
		res = res[order(res$pval), ]

		# data_diff_mat0[match(rownames(res), rownames(data_diff_mat0)),comp]=res$padj
		if (pval==TRUE){
			ordered_pvals=res$padj[match(rownames(data_diff_mat0), rownames(res))]
			return(ordered_pvals)
		} else {
			# Allocate results into complete matrix
			result_vec = data_diff_mat0[,comp]
			diff_genes_up = rownames(res)[intersect(which(res$pval<=0.05), which(res$log2FoldChange>0))]
			diff_genes_down = rownames(res)[intersect(which(res$pval<=0.05), which(res$log2FoldChange<0))]
			result_vec[diff_genes_up]=1
			result_vec[diff_genes_down]=-1
			return(unname(result_vec))
		}
	# Proceed with count data DESeq with SVASeq
	} else if (method=="deseq_svaseq"){
		# Calculate differential expression with DESeq + SVA BATCH VARIABLES (http://genomicsclass.github.io/book/pages/rnaseq_gene_level.html)
		comp_matrix_rounded = round(comp_matrix)
		comp_matrix_rounded = as.matrix(comp_matrix_rounded[rowMeans(comp_matrix_rounded) > 1,])
		comp_annots = annots[match(colnames(comp_matrix),annots$sample_name),]
		comp_annots['comp']=sapply(comp_annots[,comparison_column], function(x) ifelse(x %in% grp1_labels, "grp1", "grp2"))
		conditions = factor(comp_annots[,'comp'])
		dds = DESeqDataSetFromMatrix(countData=comp_matrix_rounded, colData=comp_annots, design = ~comp)
		mod <- model.matrix(~ comp, colData(dds))
		mod0 <- model.matrix(~ 1, colData(dds))
		svaseq <- svaseq(comp_matrix_rounded, mod, mod0)
		dds.sva <- dds
		for (sv in seq(1,svaseq$n.sv)){
			eval(parse(text=sprintf('dds.sva$SV%s <- matrix(svaseq$sv, ncol=svaseq$n.sv)[,%s]', sv, sv)))
		}
		eval(parse(text=sprintf('design(dds.sva) <- ~ %s + comp', paste(sapply(seq(1,svaseq$n.sv), function(x) sprintf("SV%s", x)), collapse=" + "))))
		dds.sva <- DESeq(dds.sva)
		res <- results(dds.sva, independentFiltering=FALSE)
		res = res[order(res$pval),]

		# data_diff_mat0[match(rownames(res), rownames(data_diff_mat0)),comp]=res$padj
		if (pval==TRUE) {
			ordered_pvals=res$padj[match(rownames(data_diff_mat0), rownames(res))]
			return(ordered_pvals)
		} else {
			# Allocate results into complete matrix
			result_vec = data_diff_mat0[,comp]
			diff_genes_up = rownames(res)[intersect(which(res$pval<=0.05), which(res$log2FoldChange>0))]
			diff_genes_down = rownames(res)[intersect(which(res$pval<=0.05), which(res$log2FoldChange<0))]
			result_vec[diff_genes_up]=1
			result_vec[diff_genes_down]=-1
			return(unname(result_vec))
		}
	}
}

### PARALLEL VERSION
if (!serial){
	registerDoParallel(cores=4)
	x <- foreach(i=comparisons[,1], .combine='cbind') %dopar% compute_pvals(i, method, data_diff_mat0)
	x[is.na(x)]=1
	colnames(x)=colnames(data_diff_mat0)
	rownames(x)=rownames(data_diff_mat0)
	data_diff_mat0=x
} 

### SERIAL VERSION
if (serial){
	for (comp in comparisons[,1]){
		data_diff_mat0[,comp]=compute_pvals(comp, method, data_diff_mat0)
	}
}



# Check the number of genes that are differentially significant between conditions
# apply(data_diff_mat0, 2, function(x) sum(x!=0))
# apply(data_diff_mat, 2, function(x) sum(x!=0))

### Subset to allowable regulator list
if (regulator_file!='none'){
	regulator_list = read.table(regulator_file, sep="\t", stringsAsFactors=FALSE)[,1]
	binding_ind = which(rownames(data_diff_mat0) %in% regulator_list)
	data_diff_mat = data_diff_mat0[binding_ind,]
} else {
	data_diff_mat = data_diff_mat0
}

# Write out matrix
################################################################################################

dense_output_file = paste(c(strsplit(output_file, '.txt')[[1]][1], '_dense.txt'), collapse="")
sparse_output_file = output_file
write.table(data_diff_mat, dense_output_file, quote=FALSE, sep="\t", col.names=TRUE, row.names=TRUE)

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
	write.table(data.frame(rownames(data_diff_mat)), label_output_file, sep="\t", quote=FALSE, col.names=FALSE, row.names=FALSE)
	sprintf('Wrote labels to: %s', label_output_file)
}


# Print when finished
sprintf('DONE: find the output matrix in: %s', sparse_output_file)

### SERIAL VERSION
################################################################################################

# Iterate through comparisons and input differential 
# for (comp in comparisons[,1]){

# 	# Print entry
# 	print(comp)

# 	# Get matrix with samples for comparison
# 	grp1_labels = strsplit(strsplit(comp, 'v')[[1]][1], "\\|")[[1]]
# 	grp2_labels = strsplit(strsplit(comp, 'v')[[1]][2], "\\|")[[1]]
# 	grp1_samples = annots$sample_name[annots[,comparison_column] %in% grp1_labels]
# 	grp2_samples = annots$sample_name[annots[,comparison_column] %in% grp2_labels]
# 	comp_matrix = data_matrix[,c(grp1_samples, grp2_samples)]
# 	comp_matrix = comp_matrix[which(apply(comp_matrix, 1, max)>0),]

# 	# Proceed with TPM data
# 	if (method=='sva_limma'){
# 		# Normalize samples
# 		comp_matrix_asinh = asinh(comp_matrix)
# 		comp_matrix_asinh_qn = normalize.quantiles(as.matrix(comp_matrix_asinh), copy=TRUE)
# 		rownames(comp_matrix_asinh_qn) = rownames(comp_matrix_asinh)
# 		colnames(comp_matrix_asinh_qn) = colnames(comp_matrix_asinh)

# 		# Calculate differential expression between the samples 
# 		comp_matrix_asinh_qn_log = log2(as.matrix(comp_matrix_asinh_qn)+1)
# 		comp_annots = annots[match(colnames(comp_matrix_asinh_qn_log),annots$sample_name),]
# 		comp_annots['comp']=sapply(comp_annots[,comparison_column], function(x) ifelse(x %in% grp1_labels, "grp1", "grp2"))
# 		full.model <- model.matrix(~ as.factor(comp) , data = comp_annots)
# 		null.model <- model.matrix(~  1, data = comp_annots)
# 		svobj <- sva(dat=comp_matrix_asinh_qn_log, mod = full.model, mod0 = null.model)
# 		full.model.sv <- cbind(full.model, svobj$sv)
# 		fit <- lmFit(comp_matrix_asinh_qn_log, full.model.sv)
# 		ebfit <- eBayes(fit)
# 		tophits <- topTable(ebfit, coef = "as.factor(comp)grp2", number = Inf)

# 		if (pval==TRUE){
# 			data_diff_mat0[match(rownames(tophits), rownames(data_diff_mat0)),comp]=tophits$adj.P.Val
# 		} else {
# 			# Allocate results into complete matrix
# 			diff_genes_up = rownames(tophits)[intersect(which(tophits$adj.P.Val<=0.05), which(tophits$logFC>0))]
# 			diff_genes_down = rownames(tophits)[intersect(which(tophits$adj.P.Val<=0.05), which(tophits$logFC<0))]
# 			data_diff_mat0[diff_genes_up,comp]=1
# 			data_diff_mat0[diff_genes_down,comp]=-1
# 		}
# 	# Proceed with count data DESeq
# 	} else if (method=='deseq'){
# 		# Calculate differential expression with DESeq
# 		comp_matrix_rounded = round(comp_matrix)
# 		comp_annots = annots[match(colnames(comp_matrix_rounded),annots$sample_name),]
# 		comp_annots['comp']=sapply(comp_annots[,comparison_column], function(x) ifelse(x %in% grp1_labels, "grp1", "grp2"))
# 		conditions = factor(comp_annots[,'comp'])
# 		dds = DESeqDataSetFromMatrix(countData=comp_matrix_rounded, colData=comp_annots, design = ~comp)
# 		dds <- DESeq(dds)
# 		res <- results(dds, independentFiltering=FALSE)
# 		res = res[order(res$pval), ]

# 		if (pval==TRUE){
# 			data_diff_mat0[match(rownames(res), rownames(data_diff_mat0)),comp]=res$padj
# 		} else {
# 			# Allocate results into complete matrix
# 			diff_genes_up = rownames(res)[intersect(which(res$padj<=0.05), which(res$log2FoldChange>0))]
# 			diff_genes_down = rownames(res)[intersect(which(res$padj<=0.05), which(res$log2FoldChange<0))]
# 			data_diff_mat0[diff_genes_up,comp]=1
# 			data_diff_mat0[diff_genes_down,comp]=-1
# 		}
# 	# Proceed with count data DESeq with SVASeq
# 	} else if (method=="deseq_svaseq"){
# 		# Calculate differential expression with DESeq + SVA BATCH VARIABLES (http://genomicsclass.github.io/book/pages/rnaseq_gene_level.html)
# 		comp_matrix_rounded = round(comp_matrix)
# 		comp_matrix_rounded = as.matrix(comp_matrix_rounded[rowMeans(comp_matrix_rounded) > 1,])
# 		comp_annots = annots[match(colnames(comp_matrix),annots$sample_name),]
# 		comp_annots['comp']=sapply(comp_annots[,comparison_column], function(x) ifelse(x %in% grp1_labels, "grp1", "grp2"))
# 		conditions = factor(comp_annots[,'comp'])
# 		dds = DESeqDataSetFromMatrix(countData=comp_matrix_rounded, colData=comp_annots, design = ~comp)
# 		mod <- model.matrix(~ comp, colData(dds))
# 		mod0 <- model.matrix(~ 1, colData(dds))
# 		svaseq <- svaseq(comp_matrix_rounded, mod, mod0)
# 		dds.sva <- dds
# 		for (sv in seq(1,svaseq$n.sv)){
# 			eval(parse(text=sprintf('dds.sva$SV%s <- matrix(svaseq$sv, ncol=svaseq$n.sv)[,%s]', sv, sv)))
# 		}
# 		eval(parse(text=sprintf('design(dds.sva) <- ~ %s + comp', paste(sapply(seq(1,svaseq$n.sv), function(x) sprintf("SV%s", x)), collapse=" + "))))
# 		dds.sva <- DESeq(dds.sva)
# 		res <- results(dds.sva)
# 		res = res[order(res$pval),]

# 		if (pval==TRUE){
# 			data_diff_mat0[match(rownames(res), rownames(data_diff_mat0)),comp]=res$padj
# 		} else {
# 			# Allocate binary results into complete matrix
# 			diff_genes_up = rownames(res)[intersect(which(res$padj<=0.05), which(res$log2FoldChange>0))]
# 			diff_genes_down = rownames(res)[intersect(which(res$padj<=0.05), which(res$log2FoldChange<0))]
# 			data_diff_mat0[diff_genes_up,comp]=1
# 			data_diff_mat0[diff_genes_down,comp]=-1
# 		}
# 	}
# }


