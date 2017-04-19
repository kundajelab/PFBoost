### Implements a prior
### Peyton Greenside
### 6/18/15
##################################################

import sys
import os
import random

import numpy as np 
from scipy.sparse import *
import pandas as pd

from collections import namedtuple
import pdb

from boosting_2D import util


### Keeps track of constants
PriorParams = namedtuple('PriorParams', [
    'prior_constant', 'prior_decay_rate',
    'motif_reg_file', 'motif_reg_row_labels', 'motif_reg_col_labels',
    'reg_reg_file', 'reg_reg_labels'
])

### Define class of motif-regulator prior
### Define PPI prior (different )
### This will only be dense since everything has a value
class Prior(object):
    def __init__(self, data_file, row_labels, col_labels, x1, x2):
        self.data_file = data_file
        self.prior_row_labels = np.genfromtxt(row_labels, delimiter="\n", dtype="str")
        self.prior_col_labels = np.genfromtxt(col_labels, delimiter="\n", dtype="str")
        # self._fill_in_missing_motifs_regs(x1, x2)

class Prior_MotifReg(Prior):
    def __init__(self, data_file, row_labels, col_labels, x1, x2):
        Prior.__init__(self, data_file, row_labels, col_labels, x1, x2)
        self._load_data(x1, x2)
        self.num_row = self.data.shape[0]
        self.num_col = self.data.shape[1]
    def _load_data(self, x1, x2):
        final_data = np.zeros((x1.num_row, x2.num_col))
        final_row_labels = x1.row_labels
        final_col_labels = x2.col_labels

        prior_data = np.genfromtxt(self.data_file)
        prior_row_labels = self.prior_row_labels.tolist()
        assert prior_data.shape[0] == len(prior_row_labels)
        prior_col_labels = self.prior_col_labels.tolist()
        assert prior_data.shape[1] == len(prior_col_labels)

        prior_row_match_ind = pd.match(prior_row_labels, final_row_labels)
        prior_rows_to_transfer = [el for el in range(len(prior_row_labels))
                                  if prior_row_match_ind[el] != -1]
        final_rows_to_fill = prior_row_match_ind[prior_row_match_ind != -1]
        prior_col_match_ind = pd.match(prior_col_labels, final_col_labels)
        prior_cols_to_transfer = [el for el in range(len(prior_col_labels))
                                  if prior_col_match_ind[el] != -1]
        final_cols_to_fill = prior_col_match_ind[prior_col_match_ind != -1]
        final_data[np.ix_(final_rows_to_fill, final_cols_to_fill)] = \
            prior_data[prior_rows_to_transfer,:][:,prior_cols_to_transfer]

        self.row_labels = final_row_labels
        self.col_labels = final_col_labels

        self.data = final_data

class Prior_RegReg(Prior):
    def __init__(self, data_file, row_labels, col_labels, x1, x2):
        Prior.__init__(self, data_file, row_labels, col_labels, x1, x2)
        self._load_data(x1, x2)
        self.num_row = self.data.shape[0]
        self.num_col = self.data.shape[1]
    def _load_data(self, x1, x2):
        assert self.prior_row_labels.tolist() == self.prior_col_labels.tolist()
        final_data = np.zeros((x2.num_col, x2.num_col))
        final_labels = x2.col_labels

        prior_data = np.genfromtxt(self.data_file)
        prior_labels = self.prior_row_labels.tolist()
        assert prior_data.shape[0] == len(prior_labels)
        assert prior_data.shape[0] == prior_data.shape[1]

        prior_match_ind = pd.match(prior_labels, final_labels)
        prior_inds_to_transfer = [el for el in range(len(prior_labels))
                                  if prior_match_ind[el] != -1]
        final_inds_to_fill = prior_match_ind[prior_match_ind != -1]
        final_data[np.ix_(final_inds_to_fill, final_inds_to_fill)] = \
            prior_data[prior_inds_to_transfer,:][:,prior_inds_to_transfer]

        self.row_labels = final_labels
        self.col_labels = final_labels

        self.data = final_data

### Load the prior matrices
def parse_prior(prior_params, x1, x2):
    if prior_params.motif_reg_file is not None:
        prior_motifreg=Prior_MotifReg(prior_params.motif_reg_file,  
                                      prior_params.motif_reg_row_labels, 
                                      prior_params.motif_reg_col_labels, x1, x2)
    else:
        prior_motifreg = None
    if prior_params.reg_reg_file is not None:
        prior_regreg=Prior_RegReg(prior_params.reg_reg_file, 
                                  prior_params.reg_reg_labels, 
                                  prior_params.reg_reg_labels, x1, x2)
    else:
        prior_regreg = None
    if prior_params.reg_reg_file is None and prior_params.motif_reg_file is None:
        raise ValueError('--use-prior must be specified with a valid prior matrix and labels')
    print 'Loaded prior'
    return (prior_motifreg, prior_regreg)

### Update the loss based on priors provided
def update_loss_with_prior(loss_matrix, prior_params, prior_motifreg, 
                           prior_regreg, iteration, best_split_regulator=None):
    new_loss = loss_matrix
    # Apply motif-regulator prior
    if prior_motifreg is not None:
        ones_mat = np.ones(prior_motifreg.data.shape)
        loss_decrease = ones_mat - loss_matrix
        motifreg_multiplier = prior_motifreg.data * \
                              prior_params.prior_constant * \
                              np.power(prior_params.prior_decay_rate, iteration) + 1
        loss_decrease_weighted = util.element_mult(loss_decrease, motifreg_multiplier)
        new_loss = ones_mat - loss_decrease_weighted
    # Apply regulator-regulator prior
    if prior_regreg is not None:
        assert best_split_regulator is not None
        # Get prior for adding on to that regulator another regulator that interacts with that
        if best_split_regulator == 'root':
            return new_loss
        reg_row = prior_regreg.data[prior_regreg.row_labels == best_split_regulator,]
        reg_mat = np.vstack([reg_row for el in xrange(new_loss.shape[0])])
        regreg_multiplier = reg_mat * prior_params.prior_constant * \
                            np.power(prior_params.prior_decay_rate, iteration) + 1
        new_loss = util.element_mult(new_loss, regreg_multiplier)
    return new_loss



# prior = pd.read_table('/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/prior_data/motifTFpriors.txt',header=None)
# prior.index = np.genfromtxt('/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/prior_data/motifTFpriors.rows.txt', delimiter="\n", dtype="str")
# prior.columns= np.genfromtxt('/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/prior_data/motifTFpriors.columns.txt', delimiter="\n", dtype="str")