### BINARY - hematopoiesis

python /users/pgreens/git/boosting_2D/bin/run_boosting_2D.py \
--num-iter 1000 --output-prefix hema_nadine_hierarchy_march17 \
--input-format matrix --mult-format sparse  \
-x /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_binary_march17/motif_matrix.txt \
-z /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_binary_march17/expression_matrix_ordered.txt \
-y /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_binary_march17/accessibility_matrix_ordered_0_to_neg.txt \
-g /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_binary_march17/peak_names.txt \
-e /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_binary_march17/cell_types_ordered.txt \
-m /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_binary_march17/motif_names.txt \
-r /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_binary_march17/reg_names.txt \
--eta1 0.05 --eta2 0.01 --ncpu 1 --output-path /srv/persistent/pgreens/projects/boosting/results/ \
--hierarchy_name hema_16cell 
--stable 

### DIFFERENTIAL - hematopoiesis

python /users/pgreens/git/boosting_2D/bin/run_boosting_2D.py \
--num-iter 1000 --output-prefix hema_nadine_diff_wrt_HSC_hierarchy_march24 \
--input-format matrix --mult-format sparse  \
-x /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_diff_wrt_HSC_march24/motif_matrix.txt \
-z /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_diff_wrt_HSC_march24/expression_matrix_ordered.txt \
-y /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_diff_wrt_HSC_march24/accessibility_matrix_ordered.txt \
-g /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_diff_wrt_HSC_march24/peak_names.txt \
-e /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_diff_wrt_HSC_march24/cell_types_ordered.txt \
-m /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_diff_wrt_HSC_march24/motif_names.txt \
-r /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_diff_wrt_HSC_march24/reg_names.txt \
--eta1 0.05 --eta2 0.01 --ncpu 1 --output-path /srv/persistent/pgreens/projects/boosting/results/ \
--hierarchy_name hema_diff_wrt_HSC 
--stable 


### BINARY - hematopoiesis nadine TEST set

python /users/pgreens/git/boosting_2D/bin/run_boosting_2D.py \
--num-iter 1000 --output-prefix hema_nadine_test_set_apr11 \
--input-format matrix --mult-format dense  \
-x /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_subset_test/motif_matrix.txt \
-z /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_subset_test/expression_matrix_ordered.txt \
-y /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_subset_test/accessibility_matrix.txt \
-g /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_subset_test/peak_names.txt \
-e /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_subset_test/cell_types_ordered.txt \
-m /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_subset_test/motif_names.txt \
-r /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_subset_test/reg_names.txt \
--holdout-file /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_subset_test/holdout.txt \
--eta1 0.05 --eta2 0.01 --ncpu 5 --output-path /srv/persistent/pgreens/projects/boosting/results/ \
--hierarchy_name hema_16cell \
--stable


### BINARY - hematopoiesis nadine TEST set with CISBP MOTIFS

python /users/pgreens/git/boosting_2D/bin/run_boosting_2D.py \
--num-iter 1000 --output-prefix hema_nadine_test_set \
--input-format matrix --mult-format dense  \
-x /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_subset_test/motif_matrix_cisbp_filtered.txt \
-z /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_subset_test/expression_matrix_ordered.txt \
-y /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_subset_test/accessibility_matrix.txt \
-g /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_subset_test/peak_names.txt \
-e /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_subset_test/cell_types_ordered.txt \
-m /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_subset_test/motif_names_cisbp_filtered.txt \
-r /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_subset_test/reg_names.txt \
--holdout-file /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_subset_test/holdout.txt \
--eta1 0.05 --eta2 0.01 --ncpu 1 --output-path /srv/persistent/pgreens/projects/boosting/results/ \
--hierarchy_name hema_16cell 
--stable 

### BINARY - hematopoiesis nadine march30 set with CISBP MOTIFS

python /users/pgreens/git/boosting_2D/bin/run_boosting_2D.py \
--num-iter 1000 --output-prefix hema_nadine_test_set \
--input-format matrix --mult-format dense  \
-x /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_binary_march30/motif_matrix.txt \
-z /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_binary_march30/expression_matrix.txt \
-y /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_binary_march30/accessibility_matrix_0_to_neg.txt \
-g /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_binary_march30/peak_names.txt \
-e /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_binary_march30/cell_types_ordered.txt \
-m /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_binary_march30/motif_names.txt \
-r /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_binary_march30/reg_names.txt \
--eta1 0.05 --eta2 0.01 --ncpu 1 --output-path /srv/persistent/pgreens/projects/boosting/results/ \
--hierarchy_name hema_16cell 
--stable 

### BINARY - hematopoiesis nadine april4 set with CISBP MOTIFS

python /users/pgreens/git/boosting_2D/bin/run_boosting_2D.py \
--num-iter 1000 --output-prefix hema_nadine_test_set \
--input-format matrix --mult-format dense  \
-x /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_binary_april4/motif_matrix.txt \
-z /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_binary_april4/expression_matrix.txt \
-y /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_binary_april4/accessibility_matrix.txt \
-g /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_binary_april4/peak_names.txt \
-e /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_binary_april4/cell_types_ordered.txt \
-m /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_binary_april4/motif_names.txt \
-r /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_binary_april4/reg_names.txt \
--eta1 0.05 --eta2 0.01 --ncpu 1 --output-path /srv/persistent/pgreens/projects/boosting/results/ \
--hierarchy_name hema_16cell 
--stable 

### DIFFERENTIAL - hematopoiesis nadine april4 set with CISBP MOTIFS 100K

python /users/pgreens/git/boosting_2D/bin/run_boosting_2D.py \
--num-iter 1000 --output-prefix hema_nadine_test_set \
--input-format matrix --mult-format dense  \
-x /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_diff_wrt_HSC_april4/motif_matrix.txt \
-z /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_diff_wrt_HSC_april4/expression_matrix.txt \
-y /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_diff_wrt_HSC_april4/accessibility_matrix.txt \
-g /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_diff_wrt_HSC_april4/peak_names.txt \
-e /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_diff_wrt_HSC_april4/cell_types_ordered.txt \
-m /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_diff_wrt_HSC_april4/motif_names.txt \
-r /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_diff_wrt_HSC_april4/reg_names.txt \
--eta1 0.05 --eta2 0.01 --ncpu 1 --output-path /srv/persistent/pgreens/projects/boosting/results/ \
--hierarchy_name hema_diff_wrt_HSC 
--stable 

### DIFFERENTIAL - hematopoiesis nadine april4 set with CISBP MOTIFS 240K

python /users/pgreens/git/boosting_2D/bin/run_boosting_2D.py \
--num-iter 200 --output-prefix hema_nadine_test_set \
--input-format matrix --mult-format dense  \
-x /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_diff_wrt_HSC_april4_240K/motif_matrix.txt \
-z /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_diff_wrt_HSC_april4_240K/expression_matrix.txt \
-y /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_diff_wrt_HSC_april4_240K/accessibility_matrix.txt \
-g /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_diff_wrt_HSC_april4_240K/peak_names.txt \
-e /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_diff_wrt_HSC_april4_240K/cell_types_ordered.txt \
-m /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_diff_wrt_HSC_april4_240K/motif_names.txt \
-r /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_diff_wrt_HSC_april4_240K/reg_names.txt \
--eta1 0.05 --eta2 0.01 --ncpu 1 --output-path /srv/persistent/pgreens/projects/boosting/results/ \
--hierarchy_name hema_diff_wrt_HSC 
--stable 

### BINARY - hematopoiesis nadine april4 set NEW REGULATORS + CISBP motifs
python /users/pgreens/git/boosting_2D/bin/run_boosting_2D.py \
--num-iter 50 --output-prefix hema_nadine_test_set \
--input-format matrix --mult-format dense  \
-x /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_binary_april12/motif_matrix.txt \
-z /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_binary_april12/expression_matrix.txt \
-y /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_binary_april12/accessibility_matrix.txt \
-g /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_binary_april12/peak_names.txt \
-e /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_binary_april12/cell_types_ordered.txt \
-m /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_binary_april12/motif_names.txt \
-r /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_binary_april12/reg_names.txt \
--eta1 0.05 --eta2 0.01 --ncpu 1 --output-path /srv/persistent/pgreens/projects/boosting/results/ \
--hierarchy_name hema_16cell 
--stable 

### BINARY - hematopoiesis nadine TEST set (91K peaks)
python /users/pgreens/git/boosting_2D/bin/run_boosting_2D.py \
--num-iter 1000 --output-prefix hema_nadine_test_set_apr11 \
--input-format matrix --mult-format dense  \
-x /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_binary_test_april12/motif_matrix.txt \
-z /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_binary_test_april12/expression_matrix.txt \
-y /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_binary_test_april12/accessibility_matrix.txt \
-g /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_binary_test_april12/peak_names.txt \
-e /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_binary_test_april12/cell_types_ordered.txt \
-m /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_binary_test_april12/motif_names.txt \
-r /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_binary_test_april12/reg_names.txt \
--holdout-file /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_binary_test_april12/holdout.txt \
--eta1 0.05 --eta2 0.01 --ncpu 5 --output-path /srv/persistent/pgreens/projects/boosting/results/ \
--hierarchy-name hema_16cell \
--stable

### BINARY - 240K peaks, new set of regulators new motifs using empirical NULL
python /users/pgreens/git/boosting2D/bin/run_boosting2D.py \
--num-iter 100 --output-prefix hema_nadine_240K_set_apr14 \
--input-format matrix --mult-format dense  \
-x /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_binary_april14/motif_matrix.txt.gz \
-z /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_binary_april14/expression_matrix.txt.gz \
-y /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_binary_april14/accessibility_matrix.txt.gz \
-g /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_binary_april14/peak_names.txt \
-e /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_binary_april14/cell_types_ordered.txt \
-m /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_binary_april14/motif_names.txt \
-r /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_binary_april14/reg_names.txt \
--eta1 0.05 --eta2 0.01 --ncpu 5 --output-path /srv/persistent/pgreens/projects/boosting/results/ \
--hierarchy-name hema_16cell \
--stable


### DIFFERENTIAL - 240K peaks, new set of regulators new motifs using empirical NULL
python /users/pgreens/git/boosting_2D/bin/run_boosting_2D.py \
--num-iter 100 --output-prefix hema_nadine_240K_set_apr16_diff_wrtHSC_binary_reg \
--input-format matrix --mult-format sparse  \
-x /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_diff_wrt_HSC_april16_binary_reg/motif_matrix.txt.gz \
-z /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_diff_wrt_HSC_april16_binary_reg/regulator_matrix.txt.gz \
-y /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_diff_wrt_HSC_april16_binary_reg/output_matrix.txt \
-g /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_diff_wrt_HSC_april16_binary_reg/region_names.txt \
-e /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_diff_wrt_HSC_april16_binary_reg/conditions.txt \
-m /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_diff_wrt_HSC_april16_binary_reg/motif_names.txt \
-r /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_diff_wrt_HSC_april16_binary_reg/regulator_names.txt \
--eta1 0.05 --eta2 0.01 --ncpu 5 --output-path /srv/persistent/pgreens/projects/boosting/results/ \
--hierarchy-name hema_diff_wrt_HSC \


### DIFFERENTIAL previous cell type- 240K peaks, new set of regulators new motifs using empirical NULL
python /users/pgreens/git/boosting_2D/bin/run_boosting_2D.py \
--num-iter 100 --output-prefix hema_nadine_240K_set_apr16_diff_wrt_prev_cell \
--input-format matrix --mult-format sparse  \
-x /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_diff_wrt_previous_cell_type_april16/motif_matrix.txt.gz \
-z /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_diff_wrt_previous_cell_type_april16/regulator_matrix.txt.gz \
-y /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_diff_wrt_previous_cell_type_april16/output_matrix.txt \
-g /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_diff_wrt_previous_cell_type_april16/region_names.txt \
-e /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_diff_wrt_previous_cell_type_april16/conditions.txt \
-m /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_diff_wrt_previous_cell_type_april16/motif_names.txt \
-r /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_diff_wrt_previous_cell_type_april16/regulator_names.txt \
--eta1 0.05 --eta2 0.01 --ncpu 5 --output-path /srv/persistent/pgreens/projects/boosting/results/ \
--hierarchy-name hema_diff_wrt_HSC \


### DIFFERENTIAL - 50K peaks, new set of regulators new motifs using empirical NULL
python /users/pgreens/git/boosting_2D/bin/run_boosting_2D.py \
--num-iter 100 --output-prefix hema_nadine_50K_set_apr16_diff_wrtHSC \
--input-format matrix --mult-format sparse  \
-x /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_diff_wrt_HSC_april16_50K_subset/motif_matrix.txt \
-z /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_diff_wrt_HSC_april16_50K_subset/regulator_matrix.txt \
-y /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_diff_wrt_HSC_april16_50K_subset/output_matrix.txt \
-g /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_diff_wrt_HSC_april16_50K_subset/region_names.txt \
-e /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_diff_wrt_HSC_april16_50K_subset/conditions.txt \
-m /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_diff_wrt_HSC_april16_50K_subset/motif_names.txt \
-r /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_diff_wrt_HSC_april16_50K_subset/regulator_names.txt \
--eta1 0.05 --eta2 0.01 --ncpu 5 --output-path /srv/persistent/pgreens/projects/boosting/results/ \
--hierarchy-name hema_diff_wrt_HSC \

#### PRIOR MATRIX
############################

python /users/pgreens/git/boosting_2D/bin/run_boosting_2D.py \
--num-iter 1000 --output-prefix hema_nadine_test_set_apr11 \
--input-format matrix --mult-format dense  \
-x /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_subset_test/motif_matrix.txt \
-z /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_subset_test/expression_matrix_ordered.txt \
-y /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_subset_test/accessibility_matrix.txt \
-g /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_subset_test/peak_names.txt \
-e /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_subset_test/cell_types_ordered.txt \
-m /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_subset_test/motif_names.txt \
-r /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_subset_test/reg_names.txt \
--holdout-file /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_subset_test/holdout.txt \
--eta1 0.05 --eta2 0.01 --ncpu 5 --output-path /srv/persistent/pgreens/projects/boosting/results/ \
--use-prior \
--reg-reg-file /mnt/lab_data/kundaje/users/pgreens/projects/boosting_data/prior_matrices/prior_reg_reg_random.txt \
--reg-reg-row-labels /mnt/lab_data/kundaje/users/pgreens/projects/modisco/data/combined_regulators_CISBP_and_GO:0003677_DNA_binding_unique_gene_names_n2790_4_12_17.txt \
--reg-reg-col-labels /mnt/lab_data/kundaje/users/pgreens/projects/modisco/data/combined_regulators_CISBP_and_GO:0003677_DNA_binding_unique_gene_names_n2790_4_12_17.txt \
# --motif-reg-file /mnt/lab_data/kundaje/users/pgreens/projects/boosting_data/prior_matrices/prior_motif_reg_random.txt \
# --motif-reg-row-labels /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_subset_test/motif_names.txt \
# --motif-reg-col-labels /mnt/lab_data/kundaje/users/pgreens/projects/modisco/data/combined_regulators_CISBP_and_GO:0003677_DNA_binding_unique_gene_names_n2790_4_12_17.txt \

### BINARY - hematopoiesis nadine further SUBSET set (50K peaks)
python /users/pgreens/git/boosting_2D/bin/run_boosting_2D.py \
--num-iter 100 --output-prefix hema_nadine_test_set_apr21 \
--input-format matrix --mult-format dense  \
-x /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_subset_test_50K/motif_matrix.txt \
-z /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_subset_test_50K/expression_matrix.txt \
-y /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_subset_test_50K/accessibility_matrix.txt \
-g /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_subset_test_50K/peak_names.txt \
-e /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_subset_test_50K/cell_types_ordered.txt \
-m /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_subset_test_50K/motif_names.txt \
-r /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_subset_test_50K/reg_names.txt \
--holdout-file /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_subset_test_50K/holdout.txt \
--eta1 0.05 --eta2 0.01 --ncpu 5 --output-path /srv/persistent/pgreens/projects/boosting/results/ \
--hierarchy-name hema_16cell \
--stable


### ANNAs data 
python /users/pgreens/git/boosting_2D/bin/run_boosting_2D.py --output-prefix heterokaryon_differential_MEDUSA_filtered_run2\
--input-format triplet\
--mult-format sparse\
-x /srv/scratch/annashch/medusa_het/motifs/heterokaryon_moods_hits.differential.filtered.txt\
-z /srv/scratch/annashch/medusa_het/rnaseq/differential/heterokaryon_differential.tsv.regulators.filtered.sparse\
-y /srv/scratch/annashch/medusa_het/peaks/differential/new.labels.differential.binarized.differential.filtered.sparse\
-g /srv/scratch/annashch/medusa_het/peaks/differential/peaks.filtered\
-e /srv/scratch/annashch/medusa_het/peaks/differential/timeseries.differential.filtered\
-m /srv/scratch/annashch/medusa_het/motifs/motif_names.txt\
-r /srv/scratch/annashch/medusa_het/rnaseq/differential/regulators.differential.filtered\
--ncpu 1\
--output-path /srv/scratch/pgreens/boosting_output



### ANNA's non-hierarchy command

python /users/pgreens/git/boosting_2D/bin/run_boosting_2D.py --output-prefix heterokaryon_MEDUSA_run1\
--input-format triplet\
--mult-format sparse\
-x /srv/scratch/annashch/medusa_het/motifs/heterokaryon_moods_hits.txt\
-z /srv/scratch/annashch/medusa_het/rnaseq/presence/all_data.tpm.annotated.averaged.binarized.regulators.sparse\
-y /srv/scratch/annashch/medusa_het/peaks/presence/new.labels.presence.binarized.sparse\
-g /srv/scratch/annashch/medusa_het/peaks/presence/peaks\
-e /srv/scratch/annashch/medusa_het/peaks/presence/timeseries.presence\
-m /srv/scratch/annashch/medusa_het/motifs/motif_names.txt\
-r /srv/scratch/annashch/medusa_het/rnaseq/presence/regulators.presence\
--ncpu 10\
--output-path /srv/scratch/pgreens/boosting_output/ \
--stable\
--num-iter 1000


### Post-processing commands
##################################################################

# Nadine test model, 100 iterations
python /users/pgreens/git/boosting_2D/bin/run_post_processing.py \
--model-path /srv/persistent/pgreens/projects/boosting/results/2017_04_11_hema_nadine_test_set_apr11_adt_non_stable_100iter/load_pickle_data_script.py \
--run-margin-score \
--margin-score-methods x1,x2,node



