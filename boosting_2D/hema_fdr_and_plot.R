### Peyton Greenside
### Script to take output from hema_gwas_enrich.sh non-rank enrichment and do multiple hypothesis corrections
### 8/5/15

key = read.delim('/srv/persistent/pgreens/projects/hqtl_gwas/non_rank_enrich_tests/all_study_abbreviations_all.txt', sep="\t", stringsAsFactors=FALSE)
key = key[,1:3]


### Multiple Hypothesis Corrections
########################################################################
path="/srv/persistent/pgreens/projects/hema_gwas/results/nonrank_enrich_results/"
use_files = system(sprintf("ls %s", path), intern=TRUE)
for (file in use_files[grep("adjusted", use_files, invert=T)]){
	pval_df = read.table(sprintf('%s%s',path, file), stringsAsFactors=FALSE)
	colnames(pval_df)=c('study', 'pval', 'effect_size', 'conf05', 'conf95', 'sig_overlap', 'sig_no_overlap', 'nonsig_overlap', 'nonsig_no_overlap')
	pval_df[,'study_abbrev']=sapply(pval_df$study, function(x) ifelse(length(key$abbrev[key$full_name==x])>0, key$abbrev[key$full_name==x],x))
	pval_df['adj_pval']=p.adjust(pval_df$pval, method="BH")
	pval_df = pval_df[order(pval_df$pval, decreasing=FALSE),]
	new_file = paste(strsplit(file, ".txt")[[1]][1], "_adjusted.txt", sep="")
	write.table(pval_df, sprintf('%s%s',path, new_file), sep="\t", quote=FALSE, col.names=TRUE, row.names=FALSE)
}


### Plot
########################################################################

plot_path="/srv/persistent/pgreens/projects/hema_gwas/plots/nonrank_enrich_plots/"

thresh="10e-5"
use_files = system(sprintf("ls %s/*thresh%s_sorted_adjusted.txt", path, thresh), intern=TRUE)
for (file in use_files){
	cell_comp=strsplit(basename(file), "_peak")[[1]][1]
	pval_df = read.delim(sprintf('%s', file), stringsAsFactors=FALSE, header=TRUE)
	pval_df[,'neg_log10_adj_pval']=-log10(pval_df$adj_pval)
	pval_df = pval_df[order(pval_df$neg_log10_adj_pval,decreasing = TRUE),]
	# pval_df=pval_df[pval_df$neg_log10_adj_pval>0.02,]
	pval_df[,'log2_fold_enrich']=log2(pval_df$effect_size)
	pval_df=pval_df[pval_df$log2_fold_enrich>-1,]
	pval_df=pval_df[!is.infinite(pval_df$effect_size),]
	# pval_df$neg_log10_adj_pval[pval_df$adj_pval==0]=310
	pval_df$neg_log10_adj_pval[pval_df$adj_pval<2.2*10^(-16)]=16
	# if (with_duplicates==FALSE){
	# 	# for duplicate diseases keep the highest ranked
	# 	dup_studies = names(table(pval_df$study_abbrev)[table(pval_df$study_abbrev)>1])
	# 	print(dup_studies)
	# 	for (study in dup_studies){
	# 		remove_ind = which(pval_df$study_abbrev==study)[2:length(which(pval_df$study_abbrev==study))]
	#  		pval_df = pval_df[-remove_ind,]		
	# 	}
	# }
	# if (with_duplicates==FALSE){
	# 	# pdf(sprintf('/srv/persistent/pgreens/projects/hqtl_gwas/non_rank_enrich_tests/plots/%s_thresh%s_pval_by_fold_enrich_just_gwas_no_duplicates.pdf', mark, thresh), width=10, height=10)
	# 	pdf(sprintf('%s%s_thresh%s_pval_by_fold_enrich_no_duplicates.pdf', plot_path, cell_comp, thresh), width=10, height=10)
	# } else{
	pdf(sprintf('%s%s_thresh%s_pval_by_fold_enrich.pdf', plot_path, cell_comp, thresh), width=10, height=10)
		# pdf(sprintf('/srv/persistent/pgreens/projects/hqtl_gwas/non_rank_enrich_tests/plots/%s_thresh%s_pval_by_fold_enrich.pdf', mark, thresh), width=10, height=10)
	# }
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
	text(pval_df$log2_fold_enrich[label_ind], pval_df$neg_log10_adj_pval[label_ind], labels=pval_df$study_abbrev[label_ind], pos=4, xlim=c(xmin, xmax), ylim=c(ymin, ymax), xlab="", ylab="",srt=15, cex=0.75, offset=0.2)
	par(new=T)
	# text(pval_df$log2_fold_enrich[label_ind], pval_df$neg_log10_adj_pval[label_ind], labels=pval_df$study_abbrev[label_ind], pos=4, xlim=c(xmin, xmax), ylim=c(ymin, ymax), xlab="", ylab="")
	dev.off()
}

