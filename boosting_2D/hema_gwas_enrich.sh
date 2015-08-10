### Script to test peaks for GWAS enrichment
### Peyton Greenside
### 8/5/15

GWAS_PATH=/srv/scratch/pgreens/data/gwas/
SCRIPT_PATH=/users/pgreens/scripts/
PEAK_PATH=/srv/persistent/pgreens/projects/hema_gwas/data/peak_files/
RESULT_PATH=/srv/persistent/pgreens/projects/hema_gwas/results/rank_enrich_results/

### Run Maurano enrichment for every GWAS
##########################################################################################

for database in single grasp roadmap;
do
  for gwas_pruned in $(ls $GWAS_PATH"pruned_LD_geno/rsq_0.8/"$database"/"*_pruned_rsq_0.8.bed);
  do
    echo $gwas_pruned
    out_dir=$GWAS_PATH"expanded_LD_geno/rsq_0.8/"$database"/"
    gwas_expanded=$out_dir$(basename $gwas_pruned .bed)"_expanded_rsq_0.8.bed"
    ## Enrichment for all peaks together
    # for file in $(ls $PEAK_PATH*_peaks.bed);
    # do
    #   gwas_name0=$(basename $gwas_pruned)
    #   gwas_name=${gwas_name0/_pruned_rsq_0.8.bed/}
    #   cell_type0=$(basename $file)
    #   cell_type=${cell_type0/_peaks.bed/} # If running variable regions
    #   analysis_name=$database"_"$gwas_name"_"$cell_type
    #   ANALYSIS_PATH=$RESULT_PATH$database"_"$gwas_name"/"
    #   if [ ! -d $ANALYSIS_PATH ]; then
    #     mkdir $ANALYSIS_PATH
    #   fi
    #   echo Rscript $SCRIPT_PATH"gwas_enrichment_maurano_test_region_only.R" -t $gwas_pruned -l $gwas_expanded -n $analysis_name  -o $ANALYSIS_PATH -r $file -u 200 | qsub -N prune_hap -V -e $SCRIPT_PATH"script.err" -o $SCRIPT_PATH"script.out" -wd /srv/scratch # nandi 
    #   echo $cell_type
    # done
    ## Enrichment for all peaks increasing in accessibility
    # for file in $(ls $PEAK_PATH*_peaks_up.bed);
    # do
    #   gwas_name0=$(basename $gwas_pruned)
    #   gwas_name=${gwas_name0/_pruned_rsq_0.8.bed/}
    #   cell_type0=$(basename $file)
    #   cell_type=${cell_type0/_peaks_up.bed/} # If running variable regions
    #   analysis_name=$database"_"$gwas_name"_"$cell_type"_peaks_up"
    #   ANALYSIS_PATH=$RESULT_PATH$database"_"$gwas_name"/"
    #   if [ ! -d $ANALYSIS_PATH ]; then
    #     mkdir $ANALYSIS_PATH
    #   fi
    #   echo Rscript $SCRIPT_PATH"gwas_enrichment_maurano_test_region_only.R" -t $gwas_pruned -l $gwas_expanded -n $analysis_name  -o $ANALYSIS_PATH -r $file -u 200 | qsub -N prune_hap -V -e $SCRIPT_PATH"script.err" -o $SCRIPT_PATH"script.out" -wd /srv/scratch # nandi 
    #   echo $cell_type
    # done
    ### Enrichment for all peaks decreasing in accessibility
    for file in $(ls $PEAK_PATH*_peaks_down.bed);
    do
      gwas_name0=$(basename $gwas_pruned)
      gwas_name=${gwas_name0/_pruned_rsq_0.8.bed/}
      cell_type0=$(basename $file)
      cell_type=${cell_type0/_peaks_down.bed/} # If running variable regions
      analysis_name=$database"_"$gwas_name"_"$cell_type"_peaks_down"
      ANALYSIS_PATH=$RESULT_PATH$database"_"$gwas_name"/"
      if [ ! -d $ANALYSIS_PATH ]; then
        mkdir $ANALYSIS_PATH
      fi
      echo Rscript $SCRIPT_PATH"gwas_enrichment_maurano_test_region_only.R" -t $gwas_pruned -l $gwas_expanded -n $analysis_name  -o $ANALYSIS_PATH -r $file -u 200 | qsub -N prune_hap -V -e $SCRIPT_PATH"script.err" -o $SCRIPT_PATH"script.out" -wd /srv/scratch # nandi 
      echo $cell_type
    done
  done
done


### Non-Rank Fisher Test
##########################################################################################

RANK_PATH=/srv/persistent/pgreens/projects/hema_gwas/results/rank_enrich_results/
NONRANK_PATH=/srv/persistent/pgreens/projects/hema_gwas/results/nonrank_enrich_results/
PRUNE_PATH=/mnt/lab_data/kundaje/users/pgreens/gwas/pruned_LD_hap/rsq_0.8/
# cell_compare=/srv/persistent/pgreens/projects/hema_gwas/data/cell_comparisons_no_leukemia.txt
cell_compare=/srv/persistent/pgreens/projects/hema_gwas/data/cell_comparisons_w_leukemia.txt
thresh=10e-5

# Unload python
module unload python_anaconda/default

for comp in $(cat $cell_compare);
do
  echo $comp
  ### Process UP Peaks
  # pval_file=$NONRANK_PATH$comp"_up_peak_gwas_overlap_fisher_test_pvals_thresh"$thresh".txt"
  # pval_file_sorted=$NONRANK_PATH$comp"_up_peak_gwas_overlap_fisher_test_pvals_thresh"$thresh"_sorted.txt"
  # if [ -f $pval_file ]; then
  #   rm $pval_file
  # fi
  # touch $pval_file
  # for result in $(ls $RANK_PATH*"/"*"hema_"$comp"_peaks_up_test_region_overlaps.txt");
  # do
  #   gwas0=$(basename $result)
  #   var="_hema_"$comp"_peaks_up_test_region_overlaps.txt"
  #   gwas=${gwas0/$var/}
  #   num_rsid=$(cat $result | awk -v OFS="\t" -v t=$thresh '$5<t' | wc -l)
  #   if [ $num_rsid -ge 400 ]; then
  #     echo $gwas
  #     overlap_file=$result
  #     ### WITH half of gwas as back up
  #     Rscript /users/pgreens/scripts/gwas_enrichment_fisher_test.R -l $overlap_file -f $gwas -t $thresh -o $pval_file
  #   fi
  # done
  # cat $pval_file | sort -g -k2,2 > $pval_file_sorted
  # rm $pval_file
  ## Process DOWN Peaks
  pval_file=$NONRANK_PATH$comp"_down_peak_gwas_overlap_fisher_test_pvals_thresh"$thresh".txt"
  pval_file_sorted=$NONRANK_PATH$comp"_down_peak_gwas_overlap_fisher_test_pvals_thresh"$thresh"_sorted.txt"
  if [ -f $pval_file ]; then
    rm $pval_file
  fi
  touch $pval_file
  for result in $(ls $RANK_PATH*"/"*"hema_"$comp"_peaks_down_test_region_overlaps.txt");
  do
    gwas0=$(basename $result)
    var="_hema_"$comp"_peaks_down_test_region_overlaps.txt"
    gwas=${gwas0/$var/}
    num_rsid=$(cat $result | awk -v OFS="\t" -v t=$thresh '$5<t' | wc -l)
    if [ $num_rsid -ge 400 ]; then
      echo $gwas
      overlap_file=$result
      ### WITH half of gwas as back up
      Rscript /users/pgreens/scripts/gwas_enrichment_fisher_test.R -l $overlap_file -f $gwas -t $thresh -o $pval_file
    fi
  done
  cat $pval_file | sort -g -k2,2 > $pval_file_sorted
  rm $pval_file
done

### Multiple hypothesis corrections
