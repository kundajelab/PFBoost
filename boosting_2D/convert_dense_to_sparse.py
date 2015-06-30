### Convert sparse matrix to dense
### Peyton Greenside
### 6/30/15

import sys
import os
import random
import argparse

import numpy as np 
from scipy.sparse import *
import scipy.io 

### Usage: python /users/pgreens/git/boosting_2D/boosting_2D/convert_dense_to_sparse.py --input-file /srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/regulatorExpression_dense_bindingTFsonly.txt --output-file /srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/regulatorExpression_bindingTFsonly.txt --with-labels
# input_file="/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/regulatorExpression_dense_bindingTFsonly.txt"
# output_file="/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/regulatorExpression_bindingTFsonly.txt"
# with_labels=True

### Parse Arguments
####################################

parser = argparse.ArgumentParser(description='Extract Chromatin States')

parser.add_argument('--input-file', 
                    help='Input dense matrix')
parser.add_argument('--output-file', 
                    help='path to write the results to')
parser.add_argument('--with-labels', 
                    help='path to write the results to', action='store_true')

args = parser.parse_args()
input_file=args.input_file
output_file=args.output_file
with_labels=args.with_labels


### Read in data
####################################

if with_labels:
	dense_mat = np.genfromtxt(input_file, skip_header=1)
	dense_mat = csr_matrix(dense_mat[:,1::])

else:
	dense_mat = csr_matrix(np.genfromtxt(input_file))

### Write out in sparse form
####################################

scipy.io.mmwrite(output_file, dense_mat)

mtx_file=''.join([output_file, ".mtx"])

os.system("cat {0} | sed '1,3d' > {1} ".format(mtx_file, output_file))
os.system("rm {0}".format(mtx_file))

