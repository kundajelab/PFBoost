### Process tree structure into index and cell files
### Peyton Greenside
### 7/6/15
################################################################

comp_file=/users/pgreens/git/boosting_2D/hema_data/index_files/hema_tree_cell_comparisons.txt
all_cells=/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/cell_types_pairwise.txt
y_matrix=/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/accessibilityMatrix_full_subset_CD34.txt
INDEX_PATH=/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/index_files/

### Process normal trees
### For each cell comparison include every cell type that includes those cell types
cat $comp_file | while read comp
do
	echo $comp
	# cell1 is upstream cell type, cell2 is downstream cell type
	cell1=$(echo $comp | tr 'v' '\t' | cut -f1)
	cell2=$(echo $comp | tr 'v' '\t' | cut -f2)
	cell_search_string=$(echo $comp | tr 'v' '|' )
	cell_file=$INDEX_PATH"hema_"$cell1"_v_"$cell2"_cells.txt"
	# Write all cell type comparisons that include these two cell types (NON-LEUKEMIA)
	cat $all_cells |  grep -E  "(^${cell1}v|v${cell1}|^${cell2}v|v${cell2})" | grep -v "SU" > $cell_file
	peak_file=$INDEX_PATH"hema_"$cell1"_v_"$cell2"_peaks.txt"
	# Get the index of the y matrix that matches the cell type comparison
	comp_ind=$(grep -n $comp $all_cells | tr ':' '\t' | cut -f1)
	# Write all indices into one file, SPARSE matrix
	cat $y_matrix | awk -v c=$comp_ind '$2==c' | awk -v OFS="\t" '{print $1}' > $peak_file
done

### For each cell comparison include ONLY THE ONE COLUMN that directly compares them
cat $comp_file | while read comp
do
	echo $comp
	cell1=$(echo $comp | tr 'v' '\t' | cut -f1)
	cell2=$(echo $comp | tr 'v' '\t' | cut -f2)
	cell_file=$INDEX_PATH"hema_"$cell1"_v_"$cell2"_cells_direct_comp.txt"
	# Write all cell type comparisons that include these two cell types (NON-LEUKEMIA)
	echo $comp > $cell_file
done

### Hema GWAS enrichment
################################################################
################################################################

comp_file=/users/pgreens/git/boosting_2D/hema_data/index_files/hema_tree_cell_comparisons.txt
all_cells=/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/cell_types_pairwise.txt
peak_labels=/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/peak_headers_full.txt
INDEX_PATH=/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/index_files/
PEAK_PATH=/srv/persistent/pgreens/projects/hema_gwas/data/peak_files/
y_matrix=/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/accessibilityMatrix_full.txt

### Process peaks - UP AND DOWN TOGETHER
cat $comp_file | while read comp
do
    echo $comp
    # cell1 is upstream cell type, cell2 is downstream cell type
    cell1=$(echo $comp | tr 'v' '\t' | cut -f1)
    cell2=$(echo $comp | tr 'v' '\t' | cut -f2)
    peak_file=$PEAK_PATH"hema_"$cell1"_v_"$cell2"_peak_numbers.txt"
    peak_bed=$PEAK_PATH"hema_"$cell1"_v_"$cell2"_peaks.bed"
    # Get the index of the y matrix that matches the cell type comparison
    comp_ind=$(grep -n $comp $all_cells | tr ':' '\t' | cut -f1)
    # Write all indices into one file, SPARSE matrix
    cat $y_matrix | awk -v c=$comp_ind '$2==c' | awk -v OFS="\t" '{print $1}' > $peak_file
    awk 'FNR == NR { h[$1]; next } (FNR in h)' $peak_file $peak_labels | tr ';' '\t' | cut -f1 | tr ':' '\t' | tr '-' '\t' > $peak_bed
    ### Get 
    echo $comp
done

### Process peaks - SEPARATE UP AND DOWN
cat $comp_file | while read comp
do
    echo $comp
    # cell1 is upstream cell type, cell2 is downstream cell type
    cell1=$(echo $comp | tr 'v' '\t' | cut -f1)
    cell2=$(echo $comp | tr 'v' '\t' | cut -f2)
    peak_file_up=$PEAK_PATH"hema_"$cell1"_v_"$cell2"_peak_numbers_up.txt"
    peak_file_down=$PEAK_PATH"hema_"$cell1"_v_"$cell2"_peak_numbers_down.txt"
    peak_bed_up=$PEAK_PATH"hema_"$cell1"_v_"$cell2"_peaks_up.bed"
    peak_bed_down=$PEAK_PATH"hema_"$cell1"_v_"$cell2"_peaks_down.bed"
    # Get the index of the y matrix that matches the cell type comparison
    comp_ind=$(grep -n $comp $all_cells | tr ':' '\t' | cut -f1)
    # Write all indices into one file, SPARSE matrix
    cat $y_matrix | awk -v c=$comp_ind '$2==c' | awk '$3==1' | awk -v OFS="\t" '{print $1}' > $peak_file_up
    cat $y_matrix | awk -v c=$comp_ind '$2==c' | awk '$3==-1' | awk -v OFS="\t" '{print $1}' > $peak_file_down
    awk 'FNR == NR { h[$1]; next } (FNR in h)' $peak_file_up $peak_labels | tr ';' '\t' | cut -f1 | tr ':' '\t' | tr '-' '\t' > $peak_bed_up
    awk 'FNR == NR { h[$1]; next } (FNR in h)' $peak_file_down $peak_labels | tr ';' '\t' | cut -f1 | tr ':' '\t' | tr '-' '\t' > $peak_bed_down
    ### Get 
    echo $comp
done

### Process leukemic peaks (leukemia-specific.bed from J. Buenrostro Aug 6, 2015)
leuk_file=/srv/persistent/pgreens/projects/hema_gwas/data/leukemia-specific.bed

cat $leuk_file | awk -v OFS="\t" '{if ($4==1) print $1,$2,$3}' > /srv/persistent/pgreens/projects/hema_gwas/data/peak_files/hema_pHSC_v_normal_peaks_up.bed
cat $leuk_file | awk -v OFS="\t" '{if ($4==-1) print $1,$2,$3}' > /srv/persistent/pgreens/projects/hema_gwas/data/peak_files/hema_pHSC_v_normal_peaks_down.bed
cat $leuk_file | awk -v OFS="\t" '{if ($4!=0) print $1,$2,$3}' > /srv/persistent/pgreens/projects/hema_gwas/data/peak_files/hema_pHSC_v_normal_peaks.bed
cat $leuk_file | awk -v OFS="\t" '{if ($5==1) print $1,$2,$3}' > /srv/persistent/pgreens/projects/hema_gwas/data/peak_files/hema_LSC_Blast_v_normal_peaks_up.bed
cat $leuk_file | awk -v OFS="\t" '{if ($5==-1) print $1,$2,$3}' > /srv/persistent/pgreens/projects/hema_gwas/data/peak_files/hema_LSC_Blast_v_normal_peaks_down.bed
cat $leuk_file | awk -v OFS="\t" '{if ($5!=0) print $1,$2,$3}' > /srv/persistent/pgreens/projects/hema_gwas/data/peak_files/hema_LSC_Blast_v_normal_peaks.bed



