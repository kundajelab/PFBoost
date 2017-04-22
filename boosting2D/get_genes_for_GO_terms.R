### Script to get the genes associated with GO terms
### Peyton Greenside
### 10/25/15
############################################################

library(biomaRt)
### TEMPLATE
# ensembl = useMart("ensembl",dataset="hsapiens_gene_ensembl") #uses human ensembl annotations
# gene.data <- getBM(attributes=c('hgnc_symbol', 'ensembl_transcript_id', 'go_id'),
#                    filters = 'go_id', values = 'GO:0007507', mart = ensembl)

### Go IDS for transcriptional regulation
# GO:0006355	regulation of transcription, DNA-templated
# GO:0044212	transcription regulatory region DNA binding
# GO:0008134	transcription factor binding
# GO:0003700	transcription factor activity, sequence-specific DNA binding

go_ids = c('GO:0006355', 'GO:0044212', 'GO:0008134', 'GO:0003700')
ensembl = useMart("ensembl",dataset="hsapiens_gene_ensembl") #uses human ensembl annotations
gene_data <- getBM(attributes=c('hgnc_symbol', 'ensembl_transcript_id', 'go_id'),
                   filters = 'go_id', values = go_ids, mart = ensembl)
gene_names=unique(gene_data[,1])
gene_names=gene_names[gene_names!=""]

# Write out the list
out_file='/mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/data/regulator_names_GOterms_transcript_reg.txt'
write.table(data.frame(gene_names), file=out_file, quote=FALSE, sep='\t', row.names=FALSE, col.names=FALSE)

### Overlap with Jason's DNA binding things
### only 200 proteins with binding domains not in Jason's list
# regulator_file='/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/regulator_names_bindingTFsonly.txt'
# regulator_list = read.table(regulator_file, sep="\t", stringsAsFactors=FALSE)[,1]

