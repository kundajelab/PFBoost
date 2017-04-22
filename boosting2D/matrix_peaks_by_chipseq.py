### Peyton Greenside
### Make a "motif" TF binding matrix where the events are ChipSeq binding
### 9/22/15
####################################################################################
####################################################################################
####################################################################################

# RESULT_PATH='/srv/persistent/pgreens/projects/boosting/results/'
# execfile('{0}2015_08_15_hematopoeisis_23K_bindingTFsonly_adt_stable_1000iter/load_pickle_data_script.py'.format(RESULT_PATH))

### Make Chip-Seq based matrix instead of motifs to try boosting with ChIP-Seq
####################################################################################
####################################################################################

### Initialize files and paths
metadata=pd.read_table('/users/pgreens/data/global/ENCODE.hg19.TFBS.QC.metadata' \
    '.jun2012_TFs_SPP_pooled.txt')

OUT_PATH='{0}{1}/chip_seq/'.format(
            config.OUTPUT_PATH, config.OUTPUT_PREFIX) 
if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)


### CHIP PATH FOR IDR
CHIP_PATH='/mnt/data/ENCODE/peaks_spp/mar2012/distinct/idrOptimalBlackListFilt/'
RELAX_CHIP_PATH='/mnt/data/ENCODE/peaks_spp/mar2012/distinct/combrep/regionPeak/'
# K562_chips = [el for el in os.listdir(CHIP_PATH) if 'K562' in el]
K562_chips=metadata.ix[metadata.CELLTYPE=='K562','FILENAME'].tolist()
LCL_chips=metadata.ix[metadata.CELLTYPE=='GM12878','FILENAME'].tolist()
# HL60_chips=metadata.ix[metadata.CELLTYPE=='HL-60','FILENAME'].tolist()
# NB4_chips=metadata.ix[metadata.CELLTYPE=='NB4','FILENAME'].tolist()
# DND41_chips=metadata.ix[metadata.CELLTYPE=='Dnd41','FILENAME'].tolist()
# GM12864_chips=metadata.ix[metadata.CELLTYPE=='GM12864','FILENAME'].tolist()

row_labels=('/srv/persistent/pgreens/projects/boosting/data/' \
    'hematopoeisis_data/peak_headers_full_subset_CD34.txt')
regs = pd.read_table('/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/regulator_names_bindingTFsonly.txt', header=None).ix[:,0].tolist()
peaks = pd.read_table('/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/peak_headers_full_subset_CD34.txt', header=None).ix[:,0].tolist()

### Function to return list of intersections - returns LIST OF COUNTS
def intersect_peaks_with_chip(row_labels, chip_file, chip_label, relax_chip_file=None):
    # Write labels to temporary bed
    temp_bed = OUT_PATH+os.path.basename(row_labels).split('.')[0]+'_temp.bed'
    # File to store intersections
    # intersect_file = OUT_PATH+os.path.basename(row_labels).split('.')[0]+ \
    #     '_{0}_intersect.txt'.format(chip_label)
    # Command to calculate turn labels into bed file
    command = "cat {0} | tr ';' '\t' | cut -f1 | tr ':' '\t' | tr '-' '\t' > {1}".format(
        row_labels, temp_bed)
    os.system(command)
    # Bedtools intersect  to get bed
    if relax_chip_file == None:
        command = "zcat {0} | cut -f1-3 | bedtools intersect -a {1} -b - -c | cut -f4".format(
            chip_file, temp_bed)
        intersect_counts = [int(el.rstrip()) for el in os.popen(command).readlines()]
    elif relax_chip_file != None:
        num_peaks0 = os.popen('zcat {0} | wc -l'.format(chip_file)).readlines()[0]
        num_peaks = int(num_peaks0.rstrip())*2
        command = "zcat {0} | cut -f1-3 | head -n {1} | bedtools intersect -a {2} -b - -c | cut -f4".format(
            relax_chip_file, num_peaks, temp_bed)
        intersect_counts = [int(el.rstrip()) for el in os.popen(command).readlines()]
    return intersect_counts


def collapse_matrix_by_factor(chip_df):
    new_chip_df = pd.DataFrame(index=chip_df.index, columns=[])
    unique_factors = np.unique([el.split('_')[1] for el in chip_df.columns]).tolist()
    for tf in unique_factors:
        columns=[el for el in chip_df.columns if el.split('_')[1]==tf]
        new_chip_df[tf]=chip_df.ix[:,columns].apply(np.sum, 1)
    return new_chip_df


### Iterate over every chip seq experiment and get 
relax=True
chip_df = pd.DataFrame(index=peaks, columns=[])
for chip in K562_chips+LCL_chips:
    # get metadata
    tf = metadata.ix[metadata.FILENAME==chip,'HGNC TARGET NAME'].tolist()[0]
    # # Only include chips where the regulator is present
    if tf not in regs:
        continue    
    cell = metadata.ix[metadata.FILENAME==chip,'CELLTYPE'].tolist()[0]
    lab = metadata.ix[metadata.FILENAME==chip,'LAB'].tolist()[0]
    chip_label = '{0}_{1}_{2}'.format(cell, tf, lab)
    # Some files in metadata are not in the file system
    if len([el for el in os.listdir(CHIP_PATH) if chip in el])==0:
        continue
    chip_file = CHIP_PATH+[el for el in os.listdir(CHIP_PATH) if chip in el][0]
    if relax==True:
        relax_chip_file = RELAX_CHIP_PATH+[el for el in os.listdir(RELAX_CHIP_PATH) if chip in el][0]
        # get the bed file of intersections
        intersect_counts = intersect_peaks_with_chip(row_labels, chip_file, chip_label, relax_chip_file=relax_chip_file)
    else:
        intersect_counts = intersect_peaks_with_chip(row_labels, chip_file, chip_label)
    chip_df[chip_label]=intersect_counts
collapsed_chip_df = collapse_matrix_by_factor(chip_df)

# Check number of non-zero entries
sum(collapsed_chip_df.apply(max, 1)!=0)
sum(norelax_df.apply(max, 1)!=0)
sum(relax_df.apply(max, 1)!=0)

# Get only nonzero rows
final_relax_df = relax_df[(relax_df.T != 0).any()]
final_relax_df[final_relax_df != 0]=1
### Write out the matrix
DATA_PATH="/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/"
# Write out data
final_relax_df.T.to_csv('{0}annotationMatrix_full_subset_CD34_CHIPDATA.txt'.format(DATA_PATH), sep="\t", header=False, index=False)
# Convert to a sparse matrix
## !! BASH
python /users/pgreens/git/boosting_2D/boosting_2D/convert_dense_to_sparse.py --input-file /srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/annotationMatrix_full_subset_CD34_CHIPDATA.txt --output-file /srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/annotationMatrix_full_subset_CD34_CHIPDATA.txtB
mv /srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/annotationMatrix_full_subset_CD34_CHIPDATA.txtB /srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/annotationMatrix_full_subset_CD34_CHIPDATA.txt
## XX BASH
# Subset accessibility matrix as well
access0 = np.genfromtxt('{0}accessibilityMatrix_full_subset_CD34.txt'.format(DATA_PATH))
access = pd.DataFrame(csr_matrix(
            (access0[:,2], (access0[:,0]-1, access0[:,1]-1)),
            shape=(max(access0[:,0]),max(access0[:,1]))).toarray())
access.index = relax_df.index
final_access = access[(relax_df.T != 0).any()]
final_access.to_csv('{0}accessibilityMatrix_full_subset_CD34_CHIPDATA.txt'.format(DATA_PATH), sep="\t", header=False, index=False)
## !! BASH
python /users/pgreens/git/boosting_2D/boosting_2D/convert_dense_to_sparse.py --input-file /srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/accessibilityMatrix_full_subset_CD34_CHIPDATA.txt --output-file /srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/accessibilityMatrix_full_subset_CD34_CHIPDATA.txtB
mv /srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/accessibilityMatrix_full_subset_CD34_CHIPDATA.txtB /srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/accessibilityMatrix_full_subset_CD34_CHIPDATA.txt
## XX BASH

# Write out columns
chip_headers = final_relax_df.columns.tolist()
chip_peak_names = final_relax_df.index.tolist()
# Write out headers
f = open('{0}annotationMatrix_headers_CHIPDATA.txt'.format(DATA_PATH), 'w')
for item in chip_headers:
  f.write("%s\n" % item)
f.close()
# Write out row labels that correspond to non-zero rows
f = open('{0}peak_headers_full_subset_CD34_CHIPDATA.txt'.format(DATA_PATH), 'w')
for item in chip_peak_names:
  f.write("%s\n" % item)
f.close()



### Run model with ChIP matrix only


### Intersect ChipSeq with Peaks and see what rank of margin score comes out - discriminative margin score
####################################################################################
####################################################################################

# Path for output of chip-related analyses
OUT_PATH='{0}{1}/chip_seq/'.format(
            config.OUTPUT_PATH, config.OUTPUT_PREFIX) 
if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)
CHIP_PATH='/mnt/data/ENCODE/peaks_spp/mar2012/distinct/idrOptimalBlackListFilt/'
MARGIN_PATH='{0}{1}/margin_scores/'.format(
            config.OUTPUT_PATH, config.OUTPUT_PREFIX) 

### Function that takes in row labels and chip file and then get out peak labels corresponding to intersection
def intersect_peaks_with_chip(row_labels, chip_file, chip_label):
    # Write labels to temporary bed
    temp_bed = OUT_PATH+os.path.basename(row_labels).split('.')[0]+'_temp.bed'
    intersect_file = OUT_PATH+os.path.basename(row_labels).split('.')[0]+ \
        '_{0}_intersect.txt'.format(chip_label)
    command = "cat {0} | tr ';' '\t' | cut -f1 | tr ':' '\t' | tr '-' '\t' > {1}".format(
        row_labels, temp_bed)
    os.system(command)
    # Bedtools intersect to get bed
    # command = "zcat {0} | cut -f1-3 | bedtools intersect -a {1} -b - -wb > {2}".format(
    #     chip_file, temp_bed, intersect_file)
    # os.system(command)
    # Bedtools intersect  to get peak labels
    command = "zcat %s | cut -f1-3 | bedtools intersect -a %s -b - -wa |  \
        awk '{print $1\":\"$2\"-\"$3}' | grep -w -F -f /dev/stdin %s > %s" % (
        chip_file, temp_bed, row_labels, intersect_file)
    os.system(command)
    return intersect_file


# Iterate through all available K562 chip seqs, calculate margin score over index of intersection in cell type comparison
K562_chips = [el for el in os.listdir(CHIP_PATH) if 'K562' in el]
metadata=pd.read_table('/users/pgreens/data/global/ENCODE.hg19.TFBS.QC.metadata' \
    '.jun2012_TFs_SPP_pooled.txt')
K562_chips=metadata.ix[metadata.CELLTYPE=='K562','FILENAME'].tolist()
K562_tfs=metadata.ix[metadata.CELLTYPE=='K562','HGNC TARGET NAME'].tolist()
row_labels=('/srv/persistent/pgreens/projects/boosting/data/' \
    'hematopoeisis_data/peak_headers_full_subset_CD34.txt')
regs = pd.read_table('/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/regulator_names_bindingTFsonly.txt').ix[:,0].tolist()
cell_comparison = 'LSC_Blast_v_normal'
# cell_comparison = 'MPP_HSC_v_pHSC'

# For loop over all chip seqs 
for chip_file in K562_chips:
    # get metadata
    tf = metadata.ix[metadata.FILENAME==chip_file,'HGNC TARGET NAME'].tolist()[0]
    if tf not in regs:
        continue    
    cell = metadata.ix[metadata.FILENAME==chip_file,'CELLTYPE'].tolist()[0]
    chip_label = '{0}_{1}'.format(cell, tf)
    # If margin score already exists
    if os.path.isfile('{0}hema_{1}_{2}_ChipSeq_peaks_x2_all_up_margin_score.txt'.format(MARGIN_PATH, cell_comparison, chip_label)):
        continue
    print tf
    chip_file_full = CHIP_PATH+[el for el in os.listdir(CHIP_PATH) if chip_file in el][0]
    # get the bed file of intersections
    intersect_file = intersect_peaks_with_chip(row_labels, chip_file_full, chip_label)
    # run margin score
    RESULT_PATH='/srv/persistent/pgreens/projects/boosting/results/'
    command = ('python /users/pgreens/git/boosting_2D/bin/run_post_processing.py' \
    ' --model-path {0}2015_08_15_hematopoeisis_23K_bindingTFsonly_' \
    'adt_stable_1000iter/load_pickle_data_script.py --margin-score-prefix ' \
    'hema_{1}_{2}_ChipSeq_peaks --run-margin-score --num-perm 10 ' \
    '--margin-score-methods x1,x2 --region-feat-file /srv/persistent/pgreens/'
    'projects/boosting/results/2015_08_15_hematopoeisis_23K_bindingTFsonly_adt_' \
    'stable_1000iter/chip_seq/peak_headers_full_subset_CD34_{2}_intersect.txt ' \
    '--condition-feat-file /srv/persistent/pgreens/projects/boosting/data/' \
    'hematopoeisis_data/index_files/hema_{1}_cells.txt').format(
    RESULT_PATH, cell_comparison, chip_label)
    os.system(command)


## Read in all the ChipSeq results and annotate if the TF comes in at the top
RESULT_PATH='/srv/persistent/pgreens/projects/boosting/results/2015_08_15_hematopoeisis_23K_bindingTFsonly_adt_stable_1000iter/margin_scores/'
result_files=os.popen("ls {0}hema_MPP_HSC_v_pHSC*ChipSeq_peaks*.txt".format(
    RESULT_PATH)).read().split('\n')
result_files.pop()
cell_comparison = 'LSC_Blast_v_normal'
cell_comparison = 'MPP_HSC_v_pHSC'

### result_df is the rank of the regulator in the margin score for that cell type
result_df=pd.DataFrame(index=[], columns=['tf', 'x1_up', 'x1_down', 'x2_up', 'x2_down'])

# This has to string split at - for motifs defined as "MOTIF-MA0.blahblah" (should be deprecated)
def get_best_ind_motif(df, tf):
    bundle_lists = [el.split('|') for el in  df.ix[:,'x1_feat_bundles'].tolist()]
    feat_lists = [[el] for el in  df.ix[:,'x1_feat'].tolist()]
    joint_lists = [[el.split('-')[0] for el in bundle_lists[i]+feat_lists[i]] for i in xrange(len(bundle_lists))]
    x1_up_ind = [i for i in range(len(joint_lists)) if tf in joint_lists[i] or tf.capitalize() in joint_lists[i]]
    if len(x1_up_ind)==0:
        best_ind = None
    else:
        best_ind = x1_up_ind[0]
    return best_ind

# Get index of best regulator given a data frame with ranking of margin scores (output from model)
def get_best_ind_reg(df, tf):
    bundle_lists = [el.split('|') for el in  df.ix[:,'x2_feat_bundles'].tolist()]
    feat_lists = [[el] for el in  df.ix[:,'x2_feat'].tolist()]
    joint_lists = [bundle_lists[i]+feat_lists[i] for i in xrange(len(bundle_lists))]
    x1_up_ind = [i for i in range(len(joint_lists)) if tf in joint_lists[i]]
    if len(x1_up_ind)==0:
        best_ind = None
    else:
        best_ind = x1_up_ind[0]
    return best_ind

# Iterate through all chips and get the best rank of the margin score with that regulator/motif
for chip_file in K562_chips:
    # get metadata
    tf = metadata.ix[metadata.FILENAME==chip_file,'HGNC TARGET NAME'].tolist()[0]
    if tf not in regs:
        continue    
    if tf=="ESR1":
        continue
    cell = metadata.ix[metadata.FILENAME==chip_file,'CELLTYPE'].tolist()[0]
    chip_label = '{0}_{1}'.format(cell, tf)
    # Read in results
    x1_up = pd.read_table("{0}hema_{1}_{2}_ChipSeq_peaks_x1_all_up_margin_score.txt".format(
        RESULT_PATH, cell_comparison, chip_label))
    x1_down = pd.read_table("{0}hema_{1}_{2}_ChipSeq_peaks_x1_all_down_margin_score.txt".format(
        RESULT_PATH, cell_comparison, chip_label))
    x2_up = pd.read_table("{0}hema_{1}_{2}_ChipSeq_peaks_x2_all_up_margin_score.txt".format(
        RESULT_PATH, cell_comparison, chip_label))
    x2_down = pd.read_table("{0}hema_{1}_{2}_ChipSeq_peaks_x2_all_down_margin_score.txt".format(
        RESULT_PATH, cell_comparison, chip_label))
    # Get top rank margin score
    ### Need exact matching
    result_df = result_df.append({'tf':tf, 'x1_up':get_best_ind_motif(x1_up, tf), 'x1_down':get_best_ind_motif(x1_down, tf),
        'x2_up':get_best_ind_reg(x2_up, tf), 'x2_down':get_best_ind_reg(x2_down, tf)}, ignore_index=True)
result_df = result_df.drop_duplicates()
print result_df

result_df.dropna().apply(min,axis=1)


### Calculate a TF by TF margin score matrix
tf_by_tf_margin_score_dict = {}
tf_by_tf_margin_score_dict['x1_all_up']=pd.DataFrame(index=result_df.ix[:,0].tolist(), columns=result_df.ix[:,0].tolist())
tf_by_tf_margin_score_dict['x1_all_down']=pd.DataFrame(index=result_df.ix[:,0].tolist(), columns=result_df.ix[:,0].tolist())
tf_by_tf_margin_score_dict['x2_all_up']=pd.DataFrame(index=result_df.ix[:,0].tolist(), columns=result_df.ix[:,0].tolist())
tf_by_tf_margin_score_dict['x2_all_down']=pd.DataFrame(index=result_df.ix[:,0].tolist(), columns=result_df.ix[:,0].tolist())
for chip_file in K562_chips:
    # get metadata
    tf = metadata.ix[metadata.FILENAME==chip_file,'HGNC TARGET NAME'].tolist()[0]
    if tf not in regs:
        continue    
    if tf=="ESR1":
        continue
    cell = metadata.ix[metadata.FILENAME==chip_file,'CELLTYPE'].tolist()[0]
    chip_label = '{0}_{1}'.format(cell, tf)
    # Read in results
    for key in tf_by_tf_margin_score_dict.keys():
        df = pd.read_table("{0}hema_{1}_{2}_ChipSeq_peaks_{3}_margin_score.txt".format(
        RESULT_PATH, cell_comparison, chip_label, key))
        margin_score_vec = []
        if 'x2' in key:
            for tf_vec in tf_by_tf_margin_score_dict[key].index:
                margin_score_vec.append(get_best_ind_reg(df, tf_vec))
        elif 'x1' in key:
            for tf_vec in tf_by_tf_margin_score_dict[key].index:
                margin_score_vec.append(get_best_ind_motif(df, tf_vec))
        tf_by_tf_margin_score_dict[key].ix[:,tf]=margin_score_vec




### Get intersecting chips to see if can get regulator that discriminates them
####################################################################################
####################################################################################

### Function to take the intersection between two CHIPs
OUT_PATH='/srv/persistent/pgreens/projects/boosting/results/2015_08_15_hematopoeisis_23K_bindingTFsonly_adt_stable_1000iter/chip_seq/'
chip_label='peak_headers_full_subset_CD34_K562_GATA1_SP1'
CHIP_PATH='/mnt/data/ENCODE/peaks_spp/mar2012/distinct/idrOptimalBlackListFilt/'
chip_file1 = '{0}wgEncodeSydhTfbsK562Gata1UcdAlnRep0.bam_VS_wgEncodeSydhTfbsK562InputUcdAlnRep1.bam.regionPeak.gz'.format(CHIP_PATH)
chip_file2 = '{0}wgEncodeHaibTfbsK562Sp1Pcr1xAlnRep0.bam_VS_wgEncodeHaibTfbsK562RxlchPcr1xAlnRep0.bam.regionPeak.gz'.format(CHIP_PATH)
row_labels=('/srv/persistent/pgreens/projects/boosting/data/' \
    'hematopoeisis_data/peak_headers_full_subset_CD34.txt')

(intersect_file, nointersect_file) = intersect_chip_with_chip(row_labels, chip_file1, chip_file2, chip_label, OUT_PATH)

# Function to take two chips and row_labels and produce a bedfile of chip1 with and without chip2 (and peaks)
def intersect_chip_with_chip(row_labels, chip_file1, chip_file2, chip_label, OUT_PATH):
    temp_bed = OUT_PATH+os.path.basename(row_labels).split('.')[0]+'_temp.bed'
    command = "cat {0} | tr ';' '\t' | cut -f1 | tr ':' '\t' | tr '-' '\t' > {1}".format(
        row_labels, temp_bed)
    os.system(command)
    # Write labels to temporary bed
    intersect_file = '{0}{1}_INTERSECT.txt'.format(OUT_PATH, chip_label)
    nointersect_file = '{0}{1}_NO_INTERSECT.txt'.format(OUT_PATH, chip_label)
    # Bedtools intersect  to get peak labels WITH chip2
    command = "bedtools intersect -a %s -b %s -wa | bedtools intersect -a %s -b - -wa | \
        awk '{print $1\":\"$2\"-\"$3}' | grep -w -F -f /dev/stdin %s > %s" % (
        chip_file1, chip_file2, temp_bed, row_labels, intersect_file)
    os.system(command)
    # Bedtools intersect to get peak labels WITHOUT chip2
    command = "bedtools intersect -a %s -b %s -wa -v| bedtools intersect -a %s -b - -wa | \
        awk '{print $1\":\"$2\"-\"$3}' | grep -w -F -f /dev/stdin %s > %s" % (
        chip_file1, chip_file2, temp_bed, row_labels, nointersect_file)
    os.system(command)
    return (intersect_file, nointersect_file)

### Run in bash
factor1=GATA1
factor2=SP1
CHIP_PATH=/srv/persistent/pgreens/projects/boosting/results/2015_08_15_hematopoeisis_23K_bindingTFsonly_adt_stable_1000iter/chip_seq/
INDEX_PATH=/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/index_files/
cell_file=$INDEX_PATH"hema_MPP_HSC_v_pHSC_cells.txt"
label="hema_"$factor1"_intersecting_"$factor2
peak_file=$CHIP_PATH"peak_headers_full_subset_CD34_K562_"$factor1"_"$factor2"_INTERSECT.txt" # get from intersect file
peak_file2=$CHIP_PATH"peak_headers_full_subset_CD34_K562_"$factor1"_"$factor2"_NO_INTERSECT.txt" # get from nointersect file
RESULT_PATH=/srv/persistent/pgreens/projects/boosting/results/
python /users/pgreens/git/boosting_2D/bin/run_post_processing.py --model-path $RESULT_PATH"2015_08_15_hematopoeisis_23K_bindingTFsonly_adt_stable_1000iter/load_pickle_data_script.py" --margin-score-prefix $label --run-disc-margin-score --margin-score-methods x1,x2 --region-feat-file $peak_file --condition-feat-file $cell_file --region-feat-file2 $peak_file2 --condition-feat-file2 $cell_file



### Intersect gene enhancer links to make a bed file for testing in KNN
####################################################################################
####################################################################################

# Set paths
gene_enh_links=/mnt/data/epigenomeRoadmap/enh_gene_links/moduleLDA/Union_link_sig_FDR05_Sep.txt.gz
cluster_file_path=/srv/persistent/pgreens/projects/boosting/results/clustering_files/
peak_bed=/srv/persistent/pgreens/projects/boosting/results/2015_08_15_hematopoeisis_23K_bindingTFsonly_adt_stable_1000iter/chip_seq/peak_headers_full_subset_CD34_temp.bed
row_labels=/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/peak_headers_full_subset_CD34.txt
result_path=/srv/persistent/pgreens/projects/boosting/results/2015_08_15_hematopoeisis_23K_bindingTFsonly_adt_stable_1000iter/knn/
cell_type_file=/srv/persistent/pgreens/projects/boosting/results/margin_score_files/knn_cell_file.txt

# For given factor and cell type find the k-nearest neighbors
factor=SPI1
cell_comp=GMPvMono
zcat $gene_enh_links| awk -v f=$factor '$11==f' | cut -f2-4 > $cluster_file_path$factor"_enh.bed"
# Print just one cell type
# bedtools intersect -a $peak_bed -b $cluster_file_path$factor"_enh.bed" | \
#  awk '{print $1":"$2"-"$3}' | grep -w -F -f /dev/stdin $row_labels | \
#  awk -v OFS="\t" -v c=$cell_comp '{print $0,c}' > $cluster_file_path$factor"_enh_w_peaks.txt"
# Print all cell types
bedtools intersect -a $peak_bed -b $cluster_file_path$factor"_enh.bed" | \
awk '{print $1":"$2"-"$3}' | grep -w -F -f /dev/stdin $row_labels | \
awk -v OFS="\t" 'NR==FNR { a[$0]; next } { for (i in a) print i, $0 }' /dev/stdin $cell_type_file \
> $cluster_file_path$factor"_enh_w_peaks.txt"
echo $(cat $cluster_file_path$factor"_enh_w_peaks.txt" | wc -l)

# Look whether other enhancers also get picked up
numcol=$(cat $result_path$factor"_enh_w_peaks_with_knneighbors.txt" | head -n 1 | tr '\t' '\n' | wc -l)
for col in $(seq 1 $numcol);
do
    cat $result_path$factor"_enh_w_peaks_with_knneighbors.txt" | cut -f$col | \
    tr ';' '\t' | cut -f1 | tr ':' '\t' | tr '-' '\t' | bedtools intersect -a - -b $cluster_file_path$factor"_enh.bed" | wc -l
done



