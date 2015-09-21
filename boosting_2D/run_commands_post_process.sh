### MARGIN SCORES
###############################################################

### Calculate margin score on whole matrix (5iter test)
RESULT_PATH=/srv/persistent/pgreens/projects/boosting/results/
python /users/pgreens/git/boosting_2D/run_post_processing.py --model-path $RESULT_PATH"2015_08_14_hematopoeisis_23K_bindingTFsonly_adt_stable_5iter/load_pickle_data_script.py" --run-margin-score --num-perm 100 --split-prom-enh-dist 0 --margin-score-methods node,path

### Calculate margin score on whole matrix (1000iter model)
RESULT_PATH=/srv/persistent/pgreens/projects/boosting/results/
python /users/pgreens/git/boosting_2D/run_post_processing.py --model-path $RESULT_PATH"2015_08_15_hematopoeisis_23K_bindingTFsonly_adt_stable_1000iter/load_pickle_data_script.py" --run-margin-score --num-perm 2 --margin-score-methods node

### Calculate margin score with subset of data
RESULT_PATH=/srv/persistent/pgreens/projects/boosting/results/
python /users/pgreens/git/boosting_2D/run_post_processing.py --model-path $RESULT_PATH"2015_08_14_hematopoeisis_23K_bindingTFsonly_adt_stable_5iter/load_pickle_data_script.py" --margin-score-prefix hema_CMP_v_Mono --run-margin-score --num-perm 100 --split-prom-enh-dist 0 --margin-score-methods x1,x2 --region-feat-file /srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/index_files/hema_CMP_v_Mono_peaks.txt --condition-feat-file /srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/index_files/hema_CMP_v_Mono_cells.txt

### Calculate margin score with subset of data with NULL model
RESULT_PATH=/srv/persistent/pgreens/projects/boosting/results/
python /users/pgreens/git/boosting_2D/run_post_processing.py --model-path $RESULT_PATH"2015_08_15_hematopoeisis_23K_bindingTFsonly_adt_stable_1000iter/load_pickle_data_script.py" --margin-score-prefix hema_CMP_v_Mono_NULL_model --run-margin-score --num-perm 1000 --split-prom-enh-dist 0 --margin-score-methods x1,x2,path,node --region-feat-file /srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/index_files/hema_CMP_v_Mono_peaks.txt --condition-feat-file /srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/index_files/hema_CMP_v_Mono_cells.txt --null-tree-model  $RESULT_PATH"2015_08_17_hematopoeisis_23K_bindingTFsonly_NULL_adt_stable_1000iter/saved_tree_state__2015_08_17_hematopoeisis_23K_bindingTFsonly_NULL_adt_stable_1000iter.gz"

### TADPOLE: Calculate margin score with subset of data with NULL model
RESULT_PATH=/srv/persistent/pgreens/projects/boosting/results/
python /users/pgreens/git/boosting_2D/run_post_processing.py --model-path $RESULT_PATH"2015_08_31_tadpole_tail_3K_adt_stable_1000iter/load_pickle_data_script.py" --run-margin-score --num-perm 100 --margin-score-methods x1,x2,path,node --margin-score-prefix tadpole_full_model  --region-feat-file /srv/persistent/pgreens/projects/boosting/data/tadpole_data/index_files/temp_regions.txt --condition-feat-file /srv/persistent/pgreens/projects/boosting/data/tadpole_data/index_files/temp_conditions.txt

### COmpute margin score with ChipSeq peaks  
RESULT_PATH=/srv/persistent/pgreens/projects/boosting/results/
python /users/pgreens/git/boosting_2D/run_post_processing.py --model-path $RESULT_PATH"2015_08_15_hematopoeisis_23K_bindingTFsonly_adt_stable_1000iter/load_pickle_data_script.py" --margin-score-prefix hema_MPP_HSC_v_pHSC_K562_GATA1_ChipSeq_peaks --run-margin-score --num-perm 5 --margin-score-methods x1,x2 --region-feat-file /srv/persistent/pgreens/projects/boosting/results/2015_08_15_hematopoeisis_23K_bindingTFsonly_adt_stable_1000iter/chip_seq/peak_headers_full_subset_CD34_K562_Gata1_intersect.txt --condition-feat-file /srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/index_files/hema_MPP_HSC_v_pHSC_cells.txt 

### Compute all hema comparisons between cell types
comp_file=/users/pgreens/git/boosting_2D/hema_data/index_files/hema_tree_cell_comparisons.txt
INDEX_PATH=/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/index_files/
RESULT_PATH=/srv/persistent/pgreens/projects/boosting/results/

# For each cell comparison include every cell type that includes those cell types (comp=LMPPvGMP)
cat $comp_file | while read comp
do
    echo $comp
    cell1=$(echo $comp | tr 'v' '\t' | cut -f1) # upstream cell type
    cell2=$(echo $comp | tr 'v' '\t' | cut -f2) # downstream cell type
    cell_search_string=$(echo $comp | tr 'v' '|' )
    cell_file=$INDEX_PATH"hema_"$cell1"_v_"$cell2"_cells_direct_comp.txt"
    # peak_bed_up=$PEAK_PATH"hema_"$cell1"_v_"$cell2"_peaks_up.bed"
    # peak_bed_down=$PEAK_PATH"hema_"$cell1"_v_"$cell2"_peaks_down.bed"
    peak_file=$INDEX_PATH"hema_"$cell1"_v_"$cell2"_peaks.txt"
    ### Process up peaks
    python /users/pgreens/git/boosting_2D/run_post_processing.py --model-path $RESULT_PATH"2015_08_15_hematopoeisis_23K_bindingTFsonly_adt_stable_1000iter/load_pickle_data_script.py" --margin-score-prefix "hema_"$comp --run-margin-score --num-perm 100 --margin-score-methods x2,node --region-feat-file $peak_file --condition-feat-file $cell_file
done

# Run pHSC v normal
cell_file=$INDEX_PATH"hema_pHSC_v_normal_cells.txt"
peak_file=$INDEX_PATH"hema_pHSC_v_normal_peaks.txt"
python /users/pgreens/git/boosting_2D/run_post_processing.py --model-path $RESULT_PATH"2015_08_15_hematopoeisis_23K_bindingTFsonly_adt_stable_1000iter/load_pickle_data_script.py" --margin-score-prefix hema_pHSC_v_normal --run-margin-score --num-perm 100 --margin-score-methods x2,node --region-feat-file $peak_file --condition-feat-file $cell_file

# Run Blast v normal
cell_file=$INDEX_PATH"hema_LSC_Blast_v_normal_cells.txt"
peak_file=$INDEX_PATH"hema_LSC_Blast_v_normal_peaks.txt"
python /users/pgreens/git/boosting_2D/run_post_processing.py --model-path $RESULT_PATH"2015_08_15_hematopoeisis_23K_bindingTFsonly_adt_stable_1000iter/load_pickle_data_script.py" --margin-score-prefix hema_LSC_Blast_v_normal --run-margin-score --num-perm 100 --margin-score-methods x2,node --region-feat-file $peak_file --condition-feat-file $cell_file

### KNN
###############################################################
RESULT_PATH=/srv/persistent/pgreens/projects/boosting/results/
python /users/pgreens/git/boosting_2D/run_post_processing.py --model-path $RESULT_PATH"2015_08_15_hematopoeisis_23K_bindingTFsonly_adt_stable_1000iter/load_pickle_data_script.py" --margin-score-prefix hema_CMP_v_Mono_NULL_model --run-knn-with-examples --examples-to-track /srv/persistent/pgreens/projects/boosting/results/clustering_files/hema_examples_to_track.txt --number-knneighbors 100

