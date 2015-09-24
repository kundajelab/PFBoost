# Yeast, nandi
python /users/pgreens/git/boosting_2D/bin/run_boosting_2D.py  --num-iter 50  --output-prefix yeast_benchmark_50_sparse --input-format triplet --mult-format sparse -x /srv/persistent/pgreens/projects/boosting/data/yeast_data/motif_hits_MxG.tab.gz -z /srv/persistent/pgreens/projects/boosting/data/yeast_data/reg_exp_ExR.tab.gz -y /srv/persistent/pgreens/projects/boosting/data/yeast_data/target_exp_GxE.tab.gz -g /srv/persistent/pgreens/projects/boosting/data/yeast_data/target_gene_names_G.txt.gz -r /srv/persistent/pgreens/projects/boosting/data/yeast_data/reg_names_R.txt.gz -m /srv/persistent/pgreens/projects/boosting/data/yeast_data/motif_names_M.txt.gz -e /srv/persistent/pgreens/projects/boosting/data/yeast_data/expt_names_E.txt.gz --eta1 0.01 --eta2 0.01 --ncpu 1 --output-path /srv/persistent/pgreens/projects/boosting/results/ --holdout-file /srv/persistent/pgreens/projects/boosting/data/yeast_data/holdout_new.tab --holdout-format triplet --stable

# Hema CD34+, nandi
python /users/pgreens/git/boosting_2D/run_boosting_2D.py --num-iter 1000 --output-prefix hematopoeisis_23K_stable --input-format triplet --mult-format sparse  -x /srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/annotationMatrix_full_subset_CD34.txt -z /srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/regulatorExpression_full.txt -y /srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/accessibilityMatrix_full_subset_CD34.txt -g /srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/peak_headers_full_subset_CD34.txt -e /srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/cell_types_pairwise.txt -m /srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/annotationMatrix_headers_full.txt -r /srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/regulator_names_full.txt --eta1 0.05 --eta2 0.01 --ncpu 1 --output-path /srv/persistent/pgreens/projects/boosting/results/ --stable

# Hema CD34+, nandi, with PRIOR
DATA_PATH=/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/
python /users/pgreens/git/boosting_2D/run_boosting_2D.py --num-iter 50 --output-prefix hematopoeisis_23K_bindingTFsonly --input-format triplet --mult-format dense  -x $DATA_PATH"annotationMatrix_full_subset_CD34.txt" -z $DATA_PATH"regulatorExpression_bindingTFsonly.txt" -y $DATA_PATH"accessibilityMatrix_full_subset_CD34.txt" -g $DATA_PATH"peak_headers_full_subset_CD34.txt" -e $DATA_PATH"cell_types_pairwise.txt" -m $DATA_PATH"annotationMatrix_headers_full.txt" -r $DATA_PATH"regulator_names_bindingTFsonly.txt" --eta1 0.05 --eta2 0.01 --ncpu 1 --output-path /srv/persistent/pgreens/projects/boosting/results/ --stable --use-prior --prior-input-format matrix --motif-reg-file $DATA_PATH"prior_data/motifTFpriors.txt" --motif-reg-row-labels $DATA_PATH"prior_data/motifTFpriors.rows.txt" --motif-reg-col-labels $DATA_PATH"prior_data/motifTFpriors.columns_gene_only.txt"

# Hema CD34+, nandi
DATA_PATH=/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/
python /users/pgreens/git/boosting_2D/bin/run_boosting_2D.py --num-iter 20 --output-prefix hematopoeisis_23K_bindingTFsonly_NULL --input-format triplet --mult-format sparse -x $DATA_PATH"annotationMatrix_full_subset_CD34.txt" -z $DATA_PATH"regulatorExpression_bindingTFsonly.txt" -y $DATA_PATH"accessibilityMatrix_full_subset_CD34.txt" -g $DATA_PATH"peak_headers_full_subset_CD34.txt" -e $DATA_PATH"cell_types_pairwise.txt" -m $DATA_PATH"annotationMatrix_headers_full.txt" -r $DATA_PATH"regulator_names_bindingTFsonly.txt" --eta1 0.05 --eta2 0.01 --ncpu 1 --output-path /srv/persistent/pgreens/projects/boosting/results/ --stable --use-prior --prior-input-format matrix --motif-reg-file $DATA_PATH"prior_data/motifTFpriors.txt" --motif-reg-row-labels $DATA_PATH"prior_data/motifTFpriors.rows.txt" --motif-reg-col-labels $DATA_PATH"prior_data/motifTFpriors.columns_gene_only.txt" --plot --shuffle-y

### Tadpole tail data (Baker lab)
DATA_PATH=/srv/persistent/pgreens/projects/boosting/data/tadpole_tail_data/
python /users/pgreens/git/boosting_2D/bin/run_boosting_2D.py --num-iter 1000 --output-prefix tadpole_tail_3K --input-format matrix --mult-format sparse -x $DATA_PATH"Motif_matrix.txt" -z $DATA_PATH"RNA_matrix.txt" -y $DATA_PATH"ATAC_matrix.txt" -g $DATA_PATH"LABEL_peaks.txt" -e $DATA_PATH"LABEL_conditions.txt" -m $DATA_PATH"LABEL_motifs.txt" -r $DATA_PATH"LABEL_regulators.txt" --eta1 0.05 --eta2 0.01 --ncpu 1 --output-path /srv/persistent/pgreens/projects/boosting/results/ --stable --plot --compress-regulators

### Chip Data
DATA_PATH=/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/
python /users/pgreens/git/boosting_2D/bin/run_boosting_2D.py --num-iter 1000 --output-prefix hematopoeisis_23K_bindingTFsonly --input-format triplet --mult-format sparse  -x $DATA_PATH"annotationMatrix_full_subset_CD34_CHIPDATA.txt" -z $DATA_PATH"regulatorExpression_bindingTFsonly.txt" -y $DATA_PATH"accessibilityMatrix_full_subset_CD34.txt" -g $DATA_PATH"peak_headers_full_subset_CD34.txt" -e $DATA_PATH"cell_types_pairwise.txt" -m $DATA_PATH"annotationMatrix_headers_CHIPDATA.txt" -r $DATA_PATH"regulator_names_bindingTFsonly.txt" --eta1 0.05 --eta2 0.01 --ncpu 4 --output-path /srv/persistent/pgreens/projects/boosting/results/ --stable --plot --use-prior --prior-input-format matrix --motif-reg-file $DATA_PATH"prior_data/motifTFpriors.txt" --motif-reg-row-labels $DATA_PATH"prior_data/motifTFpriors.rows.txt" --motif-reg-col-labels $DATA_PATH"prior_data/motifTFpriors.columns_gene_only.txt"


### SCG3
######################################################

### Tadpole tail data (Baker lab) SCG3
DATA_PATH=/srv/gsfs0/projects/baker/jessica/data/regeneration/version_9.0/learning_model/matrices/
/srv/gsfs0/projects/kundaje/users/pgreens/downloads/anaconda/bin/python /srv/gsfs0/projects/kundaje/users/pgreens/git/boosting_2D/bin/run_boosting_2D.py --num-iter 1000 --output-prefix tadpole_tail_3K --input-format matrix --mult-format sparse -x $DATA_PATH"Motif_matrix.txt" -z $DATA_PATH"RNA_matrix_reduced.txt" -y $DATA_PATH"ATAC_matrix.txt" -g $DATA_PATH"LABEL_peaks.txt" -e $DATA_PATH"LABEL_conditions.txt" -m $DATA_PATH"LABEL_motifs.txt" -r $DATA_PATH"LABEL_regulators_reduced.txt" --eta1 0.05 --eta2 0.01 --ncpu 1 --output-path /srv/gsfs0/projects/kundaje/users/pgreens/git/ --stable --plot --compress-regulators

# /srv/gsfs0/projects/baker/jessica/data/regeneration/version_9.0/learning_model/output/ 

### YEAST SCG3th
DATA_PATH=/srv/gsfs0/projects/kundaje/users/pgreens/projects/boosting/data/yeast_data/
python /srv/gsfs0/projects/kundaje/users/pgreens/git/boosting_2D/bin/run_boosting_2D.py --num-iter 1000 --output-prefix yeast --input-format matrix --mult-format sparse -x $DATA_PATH"motif_hits_MxG.tab.gz" -z $DATA_PATH"reg_exp_RxE.tab.gz" -y $DATA_PATH"target_exp_GxE.tab.gz" -g $DATA_PATH"target_gene_names_G.txt.gz" -e $DATA_PATH"expt_names_E.txt.gz" -m $DATA_PATH"motif_names_M.txt.gz" -r $DATA_PATH"reg_names_R.txt.gz" --eta1 0.05 --eta2 0.01 --ncpu 1 --output-path /srv/gsfs0/projects/kundaje/users/pgreens/temp_output/ --stable --plot 


# UNCLEAR
# DATA_PATH=/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/
# python kernprof.py -l /users/pgreens/git/boosting_2D/run_boosting_2D.py --num-iter 10 --output-prefix hematopoeisis_23K_bindingTFsonly --input-format triplet --mult-format sparse  -x $DATA_PATH"annotationMatrix_full_subset_CD34.txt" -z $DATA_PATH"regulatorExpression_bindingTFsonly.txt" -y $DATA_PATH"accessibilityMatrix_full_subset_CD34.txt" -g $DATA_PATH"peak_headers_full_subset_CD34.txt" -e $DATA_PATH"cell_types_pairwise.txt" -m $DATA_PATH"annotationMatrix_headers_full.txt" -r $DATA_PATH"regulator_names_bindingTFsonly.txt" --eta1 0.05 --eta2 0.01 --ncpu 1 --output-path /srv/persistent/pgreens/projects/boosting/results/ --stable --use-prior --prior-input-format matrix --motif-reg-file $DATA_PATH"prior_data/motifTFpriors.txt" --motif-reg-row-labels $DATA_PATH"prior_data/motifTFpriors.rows.txt" --motif-reg-col-labels $DATA_PATH"prior_data/motifTFpriors.columns_gene_only.txt"

