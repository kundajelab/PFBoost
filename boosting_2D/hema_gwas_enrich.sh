### Script to test peaks for GWAS enrichment
### Peyton Greenside
### 8/5/15

GWAS_PATH=/srv/scratch/pgreens/data/gwas/
# SCRIPT_PATH=/users/pgreens/scripts/
SCRIPT_PATH=/users/pgreens/git/gwas_util/
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
    for file in $(ls $PEAK_PATH*_peaks_up.bed);
    do
      gwas_name0=$(basename $gwas_pruned)
      gwas_name=${gwas_name0/_pruned_rsq_0.8.bed/}
      cell_type0=$(basename $file)
      cell_type=${cell_type0/_peaks_up.bed/} # If running variable regions
      analysis_name=$database"_"$gwas_name"_"$cell_type"_peaks_up"
      ANALYSIS_PATH=$RESULT_PATH$database"_"$gwas_name"/"
      if [ ! -d $ANALYSIS_PATH ]; then
        mkdir $ANALYSIS_PATH
      fi
    #   echo Rscript $SCRIPT_PATH"gwas_enrichment_maurano_test_region_only.R" -t $gwas_pruned -l $gwas_expanded -n $analysis_name  -o $ANALYSIS_PATH -r $file -u 200 | qsub -N prune_hap -V -e $SCRIPT_PATH"script.err" -o $SCRIPT_PATH"script.out" -wd /srv/scratch # nandi 
      echo Rscript $SCRIPT_PATH"get_regions_overlapping_gwas.R" -g $gwas_pruned -l $gwas_expanded -n $analysis_name  -o $ANALYSIS_PATH  -r $file -t 1e-5 | qsub -N prune_hap -V -e $SCRIPT_PATH"script.err" -o $SCRIPT_PATH"script.out" -wd /srv/scratch # nandi 
      echo $cell_type
     done
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
      # echo Rscript $SCRIPT_PATH"gwas_enrichment_maurano_test_region_only.R" -t $gwas_pruned -l $gwas_expanded -n $analysis_name  -o $ANALYSIS_PATH -r $file -u 200 | qsub -N prune_hap -V -e $SCRIPT_PATH"script.err" -o $SCRIPT_PATH"script.out" -wd /srv/scratch # nandi 
      echo Rscript $SCRIPT_PATH"get_regions_overlapping_gwas.R" -g $gwas_pruned -l $gwas_expanded -n $analysis_name  -o $ANALYSIS_PATH  -r $file -t 1e-5 | qsub -N prune_hap -V -e $SCRIPT_PATH"script.err" -o $SCRIPT_PATH"script.out" -wd /srv/scratch # nandi 
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
  pval_file=$NONRANK_PATH$comp"_up_peak_gwas_overlap_fisher_test_pvals_thresh"$thresh".txt"
  pval_file_sorted=$NONRANK_PATH$comp"_up_peak_gwas_overlap_fisher_test_pvals_thresh"$thresh"_sorted.txt"
  if [ -f $pval_file ]; then
    rm $pval_file
  fi
  touch $pval_file
  for result in $(ls $RANK_PATH*"/"*"hema_"$comp"_peaks_up_test_region_overlaps.txt");
  do
    gwas0=$(basename $result)
    var="_hema_"$comp"_peaks_up_test_region_overlaps.txt"
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

### !! Multiple hypothesis corrections + PLOT in hema_fdr_and_plot.R

# 
NONRANK_PATH=/srv/persistent/pgreens/projects/hema_gwas/results/nonrank_enrich_results/
for study in $(ls $NONRANK_PATH | grep up_peak | grep adjusted);
do
    echo $study
    cat $NONRANK_PATH$study | awk -v FS="\t" '$11<0.05' | cut -f1,11
done

for study in $(ls $NONRANK_PATH | grep down_peak | grep adjusted);
do
    echo $study
    cat $NONRANK_PATH$study | awk -v FS="\t" '$11<0.05' | cut -f1,11
done



### Re-write the enrichments by disease
##########################################################################################

thresh="10e-5"
NONRANK_PATH=/srv/persistent/pgreens/projects/hema_gwas/results/nonrank_enrich_results/
template_file=$NONRANK_PATH"CLP_v_Bcell_down_peak_gwas_overlap_fisher_test_pvals_thresh10e-5_sorted_adjusted.txt"
result_files=$(ls $NONRANK_PATH | grep adjusted | grep $thresh)

for dis in $(cat $template_file  | sed '1d' | cut -f1);
do
  echo $dis
  dis_file=$NONRANK_PATH"by_disease/"$dis"_cell_type_results.txt"
  if [ -f $dis_file ]; then
    rm $dis_file
  fi
  touch $dis_file
  for file in $result_files;
  do
    cat $NONRANK_PATH$file | awk -v d=$dis -v f=$file -v OFS="\t" '{if ($1==d) print $0, f}' >> $dis_file
  done
done



### Giant matrix of peak by GWAS
##########################################################################################


# Make file
intersect_file=$NONRANK_PATH"by_disease/"$dis"_cell_type_results.txt"
if [ -f $intersect_file ]; then
  rm $intersect_file
fi

# Get headers

# 
for gwas in all_gwas;
do

done








