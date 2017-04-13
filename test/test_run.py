### Run test data set

import os
import sys
import argparse
import gzip
import cPickle as pickle
import numpy as np

parser = argparse.ArgumentParser(description='Extract Chromatin States')

parser.add_argument('--mult-format', help='options are: dense, sparse', default='dense')
parser.add_argument('--hierarchy',  help='Reference for hierarchy encoding in hierarchy.py', 
                    default=False, action="store_true")
args = parser.parse_args()

### TODO
# Remove results from previous test


def load_hema_results():
    with gzip.open('test/test_results/2017_04_13_hema_test_adt_non_stable_10iter/saved_complete_model__2017_04_13_hema_test_adt_non_stable_10iter.gz','rb') as f:
        model_dict = pickle.load(f)

    # Assign data structures
    x1 = model_dict['x1']
    x2 = model_dict['x2']
    y = model_dict['y']
    tree = model_dict['tree']
    return (x1, x2, y, tree)

def test_hema_results():
    # Load data
    (x1, x2, y, tree) = load_hema_results()

    # Test outputs
    assert x1.data.shape == (247, 30732)
    assert x2.data.shape == (36, 2661)
    assert y.data.shape == (30732, 36) 
    assert tree.nsplit == 11
    assert tree.split_x1 == [np.array(['root'],
       dtype='|S4'), 12, 183, 83, 62, 220, 218, 81, 79, 164, 11]
    assert tree.split_x2 == [np.array(['root'],
       dtype='|S4'), 1216, 1975, 2083, 676, 1689, 291, 883, 22, 203, 57]
    assert tree.split_node == ['root', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert tree.split_depth == [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    assert tree.scores == \
    [-0.18337401768094361,
     -0.56522607671148439,
     0.68935095856813644,
     -0.45593015304166179,
     0.39902812508944191,
     -0.79555001236674772,
     0.68933691408971587,
     -0.16938136988716937,
     0.32862638442980757,
     0.31511028764724786,
     -0.3238624490268987]
    assert tree.imbal_test_err == \
    [0.40906054279749476,
     0.40906054279749476,
     0.37615866388308977,
     0.37615866388308977,
     0.35981210855949897,
     0.3593110647181628,
     0.3482672233820459,
     0.34755741127348644,
     0.3402087682672234,
     0.32780793319415447,
     0.32647181628392485]
    assert tree.bal_test_err == \
    [0.5,
     0.5,
     0.45362613689854875,
     0.45362613689854875,
     0.42390667445923247,
     0.42389899363206063,
     0.40794168753513754,
     0.40756101780725135,
     0.3920363640504661,
     0.37087827070986656,
     0.3705017443014267]
    assert tree.train_margins == \
     [6401.2202092063781,
     16255.936856671109,
     20353.43895440011,
     25091.921034962099,
     27014.438541643034,
     30903.087002091699,
     32833.919698456994,
     37271.711589500825,
     37706.155669717031,
     38100.673749851383,
     42432.657868035196]
    assert tree.test_margins == \
     [1597.5544420363806,
     4132.5933960873881,
     5130.7735840940504,
     6279.2616396059948,
     6780.0419365932448,
     7719.5865011983742,
     8189.7142766075594,
     9288.9993671752891,
     9380.3575020467761,
     9515.2247051597988,
     10643.88534001854]
    return 0


def load_hema_hierarchy_results():
    with gzip.open('test/test_results/2017_04_13_hema_test_hierarchy_adt_non_stable_10iter_hierarchy_hema_16cell/saved_complete_model__2017_04_13_hema_test_hierarchy_adt_non_stable_10iter_hierarchy_hema_16cell.gz','rb') as f:
        model_dict = pickle.load(f)

    # Assign data structures
    x1 = model_dict['x1']
    x2 = model_dict['x2']
    y = model_dict['y']
    tree = model_dict['tree']
    return (x1, x2, y, tree)

def test_hema_hierarchy_results():

    # Load data
    (x1, x2, y, tree) = load_hema_hierarchy_results()

    # Tests
    assert x1.data.shape == (640, 91530)
    assert x2.data.shape == (16, 1319)
    assert y.data.shape == (91530, 16)
    assert tree.nsplit == 11
    assert tree.split_x1 == [np.array(['root'],
       dtype='|S4'), 557, 557, 483, 483, 483, 475, 483, 606, 483, 606]
    assert tree.split_x2 == [np.array(['root'],
       dtype='|S4'), 886, 141, 597, 138, 1242, 2, 1216, 138, 38, 1242]
    assert tree.split_node == ['root', 0, 0, 0, 0, 0, 0, 1, 0, 2, 0]
    assert tree.split_depth == [0, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1]
    assert tree.hierarchy_node == [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1]
    assert tree.scores == \
    [-0.037610956694015436,
     0.56510793155854178,
     -0.55821531663174551,
     0.31130045538989493,
     -1.2177871010080772,
     -0.33794997509899677,
     0.16457914560909584,
     -0.30498322429041247,
     -0.59032073962235454,
     0.28029568459886617,
     -0.1932279293421042]
    assert tree.imbal_test_err == \
    [0.48179978655282818,
     0.32407940234791888,
     0.35411568836712914,
     0.2958787620064034,
     0.2958787620064034,
     0.2958787620064034,
     0.26244951974386338,
     0.26244951974386338,
     0.26244951974386338,
     0.26244951974386338,
     0.26244951974386338]
    assert tree.bal_test_err == \
    [0.5,
     0.33011368616653763,
     0.36318936971991017,
     0.29904394234602732,
     0.29904394234602732,
     0.29904394234602732,
     0.26265426346244836,
     0.26265426346244836,
     0.26265426346244836,
     0.26265426346244836,
     0.26265426346244836]
    assert tree.train_margins == \
    [1242.6283982135753,
     80053.145641299328,
     154013.32580310575,
     185911.0382650868,
     218333.40204232582,
     248400.81132688368,
     267307.99273274787,
     253773.44720518784,
     269418.12744665949,
     252827.98646662178,
     269682.09937555745] 
    assert tree.test_margins == \
    [801.75276384632707,
     52997.946862251047,
     102263.23963158576,
     123539.38055566351,
     144973.6513205067,
     164929.93530007746,
     177433.01299200038,
     168522.01314468327,
     178877.41955913868,
     167866.84447672591,
     179100.34337695857]
    return 0


if args.hierarchy == False:

    command="""python bin/run_boosting_2D.py \
--num-iter 10 --output-prefix hema_test \
--input-format matrix --mult-format %s  \
-x test/data/hema_data/motif_matrix.txt \
-z test/data/hema_data/expression_matrix.txt \
-y test/data/hema_data/accessibility_matrix.txt \
-g test/data/hema_data/peak_names.txt \
-e test/data/hema_data/cell_types.txt \
-m test/data/hema_data/motif_names.txt \
-r test/data/hema_data/reg_names.txt \
--ncpu 1 --output-path test/test_results/ \
    """%(args.mult_format).rstrip('\n')

    os.system(command)

    result = test_hema_results()

    if result:
        print("TEST PASSED: hema data, without hierarchy")

else:

    command="""
python bin/run_boosting_2D.py \
--num-iter 10 --output-prefix hema_test_hierarchy \
--input-format matrix --mult-format %s  \
-x test/data/hema_data_hierarchy/motif_matrix.txt \
-z test/data/hema_data_hierarchy/expression_matrix.txt \
-y test/data/hema_data_hierarchy/accessibility_matrix.txt \
-g test/data/hema_data_hierarchy/peak_names.txt \
-e test/data/hema_data_hierarchy/cell_types.txt \
-m test/data/hema_data_hierarchy/motif_names.txt \
-r test/data/hema_data_hierarchy/reg_names.txt \
--holdout-file test/data/hema_data_hierarchy/holdout.txt \
--ncpu 1 --output-path test/test_results/ \
--hierarchy_name hema_16cell
    """%(args.mult_format).strip('\n')

    os.system(command)

    from IPython import embed; embed()

    result = test_hema_hierarchy_results()

    if result:
        print("TEST PASSED: hema data, with hierarchy")




