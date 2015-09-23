### Peyton Greenside
### Script to take output from hema_gwas_enrich.sh non-rank enrichment and do multiple hypothesis corrections
### 8/5/15

key = read.delim('/srv/persistent/pgreens/projects/hqtl_gwas/non_rank_enrich_tests/all_study_abbreviations_all.txt', sep="\t", stringsAsFactors=FALSE)
key = key[,1:3]


### Multiple Hypothesis Corrections
########################################################################
path="/srv/persistent/pgreens/projects/hema_gwas/results/nonrank_enrich_results/"
### Use all files
use_files = system(sprintf("ls %s", path), intern=TRUE)
### Use peaks UP files
use_files = grep("up", system(sprintf("ls %s", path), intern=TRUE), value=T)
### Use peaks DOWN files
use_files = grep("down", system(sprintf("ls %s", path), intern=TRUE), value=T)

### Iterate through file and do multiple hypothesis corrections
for (file in use_files[grep("adjusted", use_files, invert=T)]){
	pval_df = read.table(sprintf('%s%s',path, file), stringsAsFactors=FALSE)
	colnames(pval_df)=c('study', 'pval', 'effect_size', 'conf05', 'conf95', 'sig_overlap', 'sig_no_overlap', 'nonsig_overlap', 'nonsig_no_overlap')
	pval_df[,'study_abbrev']=sapply(pval_df$study, function(x) ifelse(length(key$abbrev[key$full_name==x])>0, key$abbrev[key$full_name==x],x))
	pval_df['adj_pval']=p.adjust(pval_df$pval, method="BH")
	pval_df = pval_df[order(pval_df$pval, decreasing=FALSE),]
	new_file = paste(strsplit(file, ".txt")[[1]][1], "_adjusted.txt", sep="")
	write.table(pval_df, sprintf('%s%s',path, new_file), sep="\t", quote=FALSE, col.names=TRUE, row.names=FALSE)
}


### Plot p-value by effect size
########################################################################

plot_path="/srv/persistent/pgreens/projects/hema_gwas/plots/nonrank_enrich_plots/"

thresh="10e-5"

### Use all files
use_files = grep("adjusted", system(sprintf("ls %s", path), intern=TRUE), value=T)
### Use peaks UP files
use_files = grep("adjusted", grep("up", system(sprintf("ls %s", path), intern=TRUE), value=T), value=T)
### Use peaks DOWN files
use_files = grep("adjusted", grep("down", system(sprintf("ls %s", path), intern=TRUE), value=T), value=T)

for (file in use_files){
	cell_comp=strsplit(basename(file), "_peak")[[1]][1]
	pval_df = read.delim(sprintf('%s%s', path, file), stringsAsFactors=FALSE, header=TRUE)
	pval_df[,'neg_log10_adj_pval']=-log10(pval_df$adj_pval)
	pval_df = pval_df[order(pval_df$neg_log10_adj_pval,decreasing = TRUE),]
	# pval_df=pval_df[pval_df$neg_log10_adj_pval>0.02,]
	pval_df[,'log2_fold_enrich']=log2(pval_df$effect_size)
	pval_df=pval_df[pval_df$log2_fold_enrich>-1,]
	pval_df=pval_df[!is.infinite(pval_df$effect_size),]
	# pval_df$neg_log10_adj_pval[pval_df$adj_pval==0]=310
	pval_df$neg_log10_adj_pval[pval_df$adj_pval<2.2*10^(-16)]=16
	# PLOT
	pdf(sprintf('%s%s_thresh%s_pval_by_fold_enrich.pdf', plot_path, cell_comp, thresh), width=10, height=10)
	log_pval_thresh=2
	label_ind = which(pval_df$neg_log10_adj_pval>log_pval_thresh)
	no_label_ind =which(pval_df$neg_log10_adj_pval<=log_pval_thresh)
	# log_fold_thresh=1
	# label_ind = which(pval_df$log2_fold_enrich>log_fold_thresh)
	# no_label_ind =which(pval_df$log2_fold_enrich<=log_fold_thresh)
	xmin = min(pval_df$log2_fold_enrich)-0.1
	xmax = max(pval_df$log2_fold_enrich)+0.3
	ymin = min(pval_df$neg_log10_adj_pval[!is.infinite(pval_df$neg_log10_adj_pval)])-1
	ymax = max(pval_df$neg_log10_adj_pval[!is.infinite(pval_df$neg_log10_adj_pval)])+1
	par(mar=c(6.1,6.1,6.1,4.1))
	# plot(pval_df$log2_fold_enrich[label_ind], pval_df$neg_log10_adj_pval[label_ind], pch=16, col='blue', xlim=c(xmin, xmax), ylim=c(ymin, ymax), xlab="log2(odds ratio)", ylab="-log10(adj. p-value)", main=sprintf("Enrichment of %s hQTLs in GWAS\nlog2(odds ratio) by -log10(adj. pval)", mark), cex=1.6)
	plot(pval_df$log2_fold_enrich[label_ind], pval_df$neg_log10_adj_pval[label_ind], pch=16, col='blue', xlim=c(xmin, xmax), ylim=c(ymin, ymax), xlab="log2(odds ratio)", ylab="-log10(adj. p-value)", main="", cex=1.6, cex.lab=2.2, cex.axis=1.6) ## Illustrator version
	par(new=T)
	plot(pval_df$log2_fold_enrich[no_label_ind], pval_df$neg_log10_adj_pval[no_label_ind], pch=16, col='orange', xlim=c(xmin, xmax), ylim=c(ymin, ymax),ylab="", xlab="", cex=1.6, cex.axis=1.6)
	par(new=T)
	if (length(label_ind)>0){
		text(pval_df$log2_fold_enrich[label_ind], pval_df$neg_log10_adj_pval[label_ind], labels=pval_df$study_abbrev[label_ind], pos=4, xlim=c(xmin, xmax), ylim=c(ymin, ymax), xlab="", ylab="",srt=15, cex=0.75, offset=0.2)
		par(new=T)	
	}
	# text(pval_df$log2_fold_enrich[label_ind], pval_df$neg_log10_adj_pval[label_ind], labels=pval_df$study_abbrev[label_ind], pos=4, xlim=c(xmin, xmax), ylim=c(ymin, ymax), xlab="", ylab="")
	dev.off()
}

### SPIDER PLOTS PER DISEASE
########################################################################

by_disease_path='/srv/persistent/pgreens/projects/hema_gwas/results/nonrank_enrich_results/by_disease/'
plot_path='/srv/persistent/pgreens/projects/hema_gwas/plots/nonrank_enrich_plots/star_plots/'

result_df1 = read.table(sprintf('%s%s', by_disease_path, list.files(by_disease_path)[1]), sep="\t", stringsAsFactors=FALSE)
full_result_df = data.frame(matrix(nrow=length(list.files(by_disease_path)), ncol=nrow(result_df1))) # disease by cell type
rownames(full_result_df) = sapply(list.files(by_disease_path), function(x) strsplit(x, "_cell_")[[1]][1])
colnames(full_result_df) = sapply(result_df1[,12], function(x) strsplit(x, '_peak_')[[1]][1])

for (file in list.files(by_disease_path)){
	disease = strsplit(file, "_cell_")[[1]][1]
	result_df = read.table(sprintf('%s%s', by_disease_path, file), sep="\t", stringsAsFactors=FALSE, quote="")
	colnames(result_df)=c('study', 'pval', 'effect_size', 'conf05', 'conf95', 'sig_overlap', 'sig_no_overlap', 'nonsig_overlap', 'nonsig_no_overlap', 'study_abbrev', 'adj_pval', 'cell_file')
	result_df['cell_abbrev']=sapply(result_df$cell_file, function(x) strsplit(x, '_peak_')[[1]][1])
	result_df['neg_log10_adj_pval']=-log10(result_df$adj_pval)
	full_result_df[disease,result_df[,'cell_abbrev']]=result_df[,'neg_log10_adj_pval']
}
full_result_df = full_result_df[,grep("up|down", colnames(full_result_df))]
full_result_df = full_result_df[rownames(full_result_df)!="single_T1D_gwas_sorted_n500000",]


### Heatmap
my_palette <- colorRampPalette(c("green", "black", "red"))(n = 1000)

library(gplots)
pdf(sprintf('%sall_disease_by_cell_type_heatmap.pdf', plot_path))
par(mar=c(10.1,10.1,10.1,10.1))
heatmap.2(as.matrix(full_result_df), rowsep=0, colsep=0, sepwidth=c(0,0), dendrogram='none', trace='none', Rowv=FALSE, Colv=FALSE, col=my_palette)
dev.off()
### Poor attempts at spider plots
# for (dis in rownames(full_result_df)){
# 	pdf(sprintf('%s%sspider_plots.pdf', plot_path, dis))
# 	stars(full_result_df[dis,], full=TRUE, scale=TRUE, labels=result_df[,'cell_abbrev'])
# 	dev.off()	
# }
# for (row in seq(1,nrow(full_result_df),2)){
# 	pdf(sprintf('%s%sspider_plots.pdf', plot_path, row))
# 	stars(full_result_df[1,], full=TRUE, scale=TRUE, locations = c(0, 0), radius = FALSE, key.loc = c(0, 0))
# 	dev.off()
# }

### Plot effect size by p-value per disease
for (file in list.files(by_disease_path)){
	disease = strsplit(file, "_cell_")[[1]][1]
	result_df = read.table(sprintf('%s%s', by_disease_path, file), sep="\t", stringsAsFactors=FALSE, quote="")
	### Keep only up and down peaks
	colnames(result_df)=c('study', 'pval', 'effect_size', 'conf05', 'conf95', 'sig_overlap', 'sig_no_overlap', 'nonsig_overlap', 'nonsig_no_overlap', 'study_abbrev', 'adj_pval', 'cell_file')
	result_df['cell_abbrev']=sapply(result_df$cell_file, function(x) strsplit(x, '_peak_')[[1]][1])
	result_df['neg_log10_adj_pval']=-log10(result_df$adj_pval)
	result_df[,'log2_fold_enrich']=log2(result_df$effect_size)
	result_df = result_df[grep("up|down", result_df$cell_abbrev),]
	result_df = result_df[!is.infinite(result_df$log2_fold_enrich),]
	log_pval_thresh = 2
	label_ind = which(result_df$neg_log10_adj_pval>log_pval_thresh)
	no_label_ind =which(result_df$neg_log10_adj_pval<=log_pval_thresh)
	xmin = min(result_df$log2_fold_enrich)-0.1
	xmax = max(result_df$log2_fold_enrich)+1
	ymin = min(result_df$neg_log10_adj_pval[!is.infinite(result_df$neg_log10_adj_pval)])-1
	ymax = max(result_df$neg_log10_adj_pval[!is.infinite(result_df$neg_log10_adj_pval)])+1
	pdf(sprintf('%s%s_pval_by_effect_size.pdf', plot_path, disease))
	par(mar=c(6.1,6.1,6.1,4.1))
	plot(result_df$log2_fold_enrich[label_ind], result_df$neg_log10_adj_pval[label_ind], pch=16, col='green', xlim=c(xmin, xmax), ylim=c(ymin, ymax), xlab="log2(odds ratio)", ylab="-log10(adj. p-value)", main=sprintf("%s", disease), cex=1.6, cex.lab=2.2, cex.axis=1.6) ## Illustrator version
	par(new=T)
	plot(result_df$log2_fold_enrich[no_label_ind], result_df$neg_log10_adj_pval[no_label_ind], pch=16, col='black', xlim=c(xmin, xmax), ylim=c(ymin, ymax),ylab="", xlab="", cex=1.6, cex.axis=1.6)
	par(new=T)
	if (length(label_ind)>0){
		text(result_df$log2_fold_enrich[label_ind], result_df$neg_log10_adj_pval[label_ind], labels=result_df$cell_abbrev[label_ind], pos=4, xlim=c(xmin, xmax), ylim=c(ymin, ymax), xlab="", ylab="",srt=15, cex=0.75, offset=0.2)
		par(new=T)	
	}
	dev.off()
}

### HEMA HEATMAPS
######################################################################################

### Read in tables for each gwas  - HEMA to give ranking of blood cell types and enrichment of GWAS
######################################################################################

result_path='/srv/persistent/pgreens/projects/hema_gwas/results/roadmap_enrich_results/*/'
result_files=system('ls /srv/persistent/pgreens/projects/hema_gwas/results/roadmap_enrich_results/*/*roadmap_tissue_enrichment_table.txt', intern=TRUE)

blood_cells=c('E062', 'E034', 'E045', 'E033', 'E044', 'E043', 'E039', 'E041', 'E042', 'E040', 'E037', 'E048', 'E038', 'E047', 
'E029', 'E031', 'E035', 'E051', 'E050', 'E036', 'E032', 'E046', 'E030')
rank_df = data.frame(matrix(nrow=length(result_files), ncol=3))
colnames(rank_df)=c('max', 'mean', 'mean_top5')
rownames(rank_df)=sapply(result_files, function(x) tail(strsplit(x, '/')[[1]], n=2)[1])
for (file in result_files){
	row_label=tail(strsplit(file, '/')[[1]], n=2)[1]
	table=read.table(file, header=TRUE)
	blood_table = table[,blood_cells]
	rank_df[row_label, 'max'] = max(blood_table[nrow(blood_table),])
	rank_df[row_label, 'mean'] = mean(unlist(blood_table[nrow(blood_table),]))
	rank_df[row_label, 'mean_top5'] = mean(sort(unlist(blood_table[nrow(blood_table),]), decreasing=TRUE)[1:5])
}

sort_df = rank_df[order(rank_df$mean_top5, decreasing=TRUE),c(1,3)]

### Make matrix of all GWAS by cell types
######################################################################################
library(gplots)
library(cba)

file1 = read.table(result_files[1], header=TRUE)
enrich_df = data.frame(matrix(nrow=length(result_files), ncol=127))
colnames(enrich_df)=colnames(file1)[2:ncol(file1)]
rownames(enrich_df)=sapply(result_files, function(x) tail(strsplit(x, '/')[[1]], n=2)[1])

# Fill in matrix with the mean of the cell type in the last two bins
for (file in result_files){
	row_label=tail(strsplit(file, '/')[[1]], n=2)[1]
	table=read.table(file, header=TRUE)
	row_vals=apply(table[c(nrow(table)-1, nrow(table)),2:ncol(table)], 2, mean)
	if (sum(is.na(row_vals))>0){
		print(file)
	} else{
		enrich_df[row_label,]=row_vals
	}
}

# Enrichment DF only GWAS without NAs
enrich_df_plot <- enrich_df[is.na(apply(enrich_df, 1, sum))==FALSE,]

### Plot
annots = read.table('/srv/persistent/pgreens/projects/hema_gwas/jul2013.roadmapData_annotations.tsv', stringsAsFactors=FALSE, fill=T, header=T, sep='\t', comment.char='')
annots= annots[3:nrow(annots),]

plotannots = annots[match(colnames(enrich_df_plot), annots$NEW.EID),]
# get palette
col_breaks = c(0,seq(1,2,0.2),seq(3,8), 10)
my_palette <- heat.colors(n=length(col_breaks)-1)
# my_palette <- c("blue", colorRampPalette(c("white", "red"))(length(col_breaks)-2))

# Plot in heatmap
pdf("/users/pgreens/gwas_enrichment_across_roadmap_cell_types.pdf",width=20,height=20)
par(mar=c(7,4,4,12)+0.1)
par(oma=c(2,2,2,12))
heatmap.2(as.matrix(enrich_df_plot), rowsep=0, colsep=0, sepwidth=c(0,0), trace='none', keysize=1, margins=c(8,14), na.color='black', main="GWAS Enrichment across Roadmap Cell Types", col=my_palette, breaks=col_breaks, ColSideColors=plotannots$COLOR, labCol=plotannots$Epigenome.Mnemonic)
dev.off()

# Write out table
write.table(enrich_df_plot, '/srv/persistent/pgreens/projects/hema_gwas/results/roadmap_results/gwas_by_roadmap_enrich_mean_last_two_bins.txt', quote=FALSE, sep="\t", col.names =TRUE)

