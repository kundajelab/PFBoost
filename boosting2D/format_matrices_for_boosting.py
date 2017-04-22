### Script to format data frames of matrices to boosting input
### Peyton Greenside
### 4/16/17

import os
import sys
import argparse
import numpy as np
import pandas as pd

### Run arguments

# SCRIPT_PATH=/users/pgreens/git/boosting_2D/boosting_2D/
# python ${SCRIPT_PATH}format_matrices_for_boosting.py \
# --regulator-dataframe /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/data/boosting_input/expression_diff_wrt_HSC_nadine_dense_april16_new_regulators.txt \
# --output-dataframe /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/data/boosting_input/accessibility_diff_wrt_HSC_nadine_dense_april16.txt \
# --output-folder /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_diff_wrt_HSC_april16/ \
# --condition-order /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/data/cell_comparisons_w_leuk_all_hier_nadine_wrt_HSC.txt

# SCRIPT_PATH=/users/pgreens/git/boosting_2D/boosting_2D/
# python ${SCRIPT_PATH}format_matrices_for_boosting.py \
# --regulator-dataframe /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/data/boosting_input/expression_diff_wrt_previous_cell_type_nadine_dense_april16_new_regulators.txt \
# --output-dataframe /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/data/boosting_input/accessibility_diff_wrt_previous_cell_type_nadine_dense_april16.txt \
# --output-folder /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_diff_wrt_previous_cell_type_april16/ \
# --condition-order /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/data/cell_comparisons_w_leuk_all_hier_nadine_wrt_previous_cell_type_hierarchy_order.txt

# Parse arugments 
parser = argparse.ArgumentParser(description='Format Boosting Matrices')

parser.add_argument('--motif-dataframe', 
                    help='motif matrix, with header/index', default=None)
parser.add_argument('--regulator-dataframe', 
                    help='expression matrix, with header/index', default=None)
parser.add_argument('--output-dataframe', 
                    help='expression matrix, with header/index', default=None)
parser.add_argument('--output-folder', 
                    help='output folder for all matrices', default=None)
parser.add_argument('--condition-order', 
                    help='order for cell types', default=None)
args = parser.parse_args()

motif_dataframe = args.motif_dataframe
regulator_dataframe = args.regulator_dataframe
output_dataframe = args.output_dataframe
output_folder = args.output_folder
condition_order = args.condition_order

### Make output directory

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
motif_outfile = '{0}/motit_matrix.txt'.format(output_folder)
regulator_outfile = '{0}/regulator_matrix.txt'.format(output_folder)
output_outfile = '{0}/output_matrix.txt'.format(output_folder)
motif_names_file = '{0}/motit_names.txt'.format(output_folder)
regulator_names_file = '{0}/regulator_names.txt'.format(output_folder)
region_names_file = '{0}/region_names.txt'.format(output_folder)
condition_names_file = '{0}/condition_names.txt'.format(output_folder)

condition_order_list = pd.read_table(condition_order, header=None).ix[:, 0].tolist()

### Motif Data Frame

if motif_dataframe is not None:
    motif_df = pd.read_table(motif_dataframe)
    motif_df.T.to_csv(motif_outfile, sep="\t", index=False, header=False)
    pd.DataFrame(motif_df.index).to_csv(motif_names_file, header=False, index=False)

### Regulator Data Frame

if regulator_dataframe is not None:
    regulator_df = pd.read_table(regulator_dataframe)
    if condition_order is not None:
        assert len(condition_order_list) == regulator_df.shape[1]
        regulator_df = regulator_df.ix[:, condition_order_list]
    regulator_df.T.to_csv(regulator_outfile, sep="\t", index=False, header=False)
    pd.DataFrame(regulator_df.index).to_csv(regulator_names_file, header=False, index=False)

### Regulator Data Frame

if output_dataframe is not None:
    output_df = pd.read_table(output_dataframe)
    if condition_order is not None:
        assert len(condition_order_list) == output_df.shape[1]
        output_df = output_df.ix[:, condition_order_list]
    if np.unique(output_df).tolist() == [0,1]:
        print('Converting binary [0,1] to [-1,1]')
        output_df[output_df == 0] = -1
    output_df.to_csv(output_outfile, sep="\t", index=False, header=False)
    pd.DataFrame(output_df.index).to_csv(region_names_file, header=False, index=False)
    pd.DataFrame(output_df.columns).to_csv(condition_names_file, header=False, index=False)


# ### Generate command
# command="""
# python /users/pgreens/git/boosting_2D/bin/run_boosting_2D.py \
# --num-iter 100 --output-prefix hema_nadine_240K_set_apr14 \
# --input-format matrix --mult-format dense  \
# -x /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_binary_april14/motif_matrix.txt.gz \
# -z /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_binary_april14/expression_matrix.txt.gz \
# -y /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_binary_april14/accessibility_matrix.txt.gz \
# -g /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_binary_april14/peak_names.txt \
# -e /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_binary_april14/cell_types_ordered.txt \
# -m /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_binary_april14/motif_names.txt \
# -r /mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/boosting_data_sets/nadine_binary_april14/reg_names.txt \
# --eta1 0.05 --eta2 0.01 --ncpu 5 --output-path /srv/persistent/pgreens/projects/boosting/results/ \
# --hierarchy-name hema_15cell \
# """


