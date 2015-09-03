### Calculate margin score on whole matrix (5iter test)
RESULT_PATH=/srv/persistent/pgreens/projects/boosting/results/
python /users/pgreens/git/boosting_2D/run_post_processing.py --model-path $RESULT_PATH"2015_08_14_hematopoeisis_23K_bindingTFsonly_adt_stable_5iter/load_pickle_data_script.py" --run-margin-score --num-perm 100 --split-prom-enh-dist 0 --margin-score-methods node,path

### Calculate margin score on whole matrix (1000iter model)
RESULT_PATH=/srv/persistent/pgreens/projects/boosting/results/
python /users/pgreens/git/boosting_2D/run_post_processing.py --model-path $RESULT_PATH"2015_08_15_hematopoeisis_23K_bindingTFsonly_adt_stable_1000iter/load_pickle_data_script.py" --run-margin-score --num-perm 20 --split-prom-enh-dist 0 --margin-score-methods x2

### Calculate margin score with subset of data
RESULT_PATH=/srv/persistent/pgreens/projects/boosting/results/
python /users/pgreens/git/boosting_2D/run_post_processing.py --model-path $RESULT_PATH"2015_08_14_hematopoeisis_23K_bindingTFsonly_adt_stable_5iter/load_pickle_data_script.py" --margin-score-prefix hema_CMP_v_Mono --run-margin-score --num-perm 100 --split-prom-enh-dist 0 --margin-score-methods x1,x2 --region-feat-file /srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/index_files/hema_CMP_v_Mono_peaks.txt --condition-feat-file /srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/index_files/hema_CMP_v_Mono_cells.txt

### Calculate margin score with subset of data with NULL model
RESULT_PATH=/srv/persistent/pgreens/projects/boosting/results/
python /users/pgreens/git/boosting_2D/run_post_processing.py --model-path $RESULT_PATH"2015_08_15_hematopoeisis_23K_bindingTFsonly_adt_stable_1000iter/load_pickle_data_script.py" --margin-score-prefix hema_CMP_v_Mono_NULL_model --run-margin-score --num-perm 1000 --split-prom-enh-dist 0 --margin-score-methods x1,x2,path,node --region-feat-file /srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/index_files/hema_CMP_v_Mono_peaks.txt --condition-feat-file /srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/index_files/hema_CMP_v_Mono_cells.txt --null-tree-model  $RESULT_PATH"2015_08_17_hematopoeisis_23K_bindingTFsonly_NULL_adt_stable_1000iter/saved_tree_state__2015_08_17_hematopoeisis_23K_bindingTFsonly_NULL_adt_stable_1000iter.gz"

### TADPOLE: Calculate margin score with subset of data with NULL model
RESULT_PATH=/srv/persistent/pgreens/projects/boosting/results/
python /users/pgreens/git/boosting_2D/run_post_processing.py --model-path $RESULT_PATH"2015_08_31_tadpole_tail_3K_adt_stable_1000iter/load_pickle_data_script.py" --run-margin-score --num-perm 100 --margin-score-methods x1,x2,path,node --margin-score-prefix tadpole_full_model  --region-feat-file /srv/persistent/pgreens/projects/boosting/data/tadpole_data/index_files/temp_regions.txt --condition-feat-file /srv/persistent/pgreens/projects/boosting/data/tadpole_data/index_files/temp_conditions.txt



