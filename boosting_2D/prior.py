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
    'prior_input_format',
    'motif_reg_file', 'motif_reg_row_labels', 'motif_reg_col_labels',
    'reg_reg_file', 'reg_reg_row_labels', 'reg_reg_col_labels'
])

### Define class of motif-regulator prior
### Define PPI prior (different )
### This will only be dense since everything has a value
class Prior(object):
    def __init__(self, data_file, row_labels, col_labels, input_format, x1, x2):
        self.data_file = data_file
        self.row_labels = np.genfromtxt(row_labels, delimiter="\n", dtype="str")
        self.col_labels = np.genfromtxt(col_labels, delimiter="\n", dtype="str")
        ## XXX ADD SOME ASSERTS
        self.input_format = input_format
        if self.input_format == 'triplet':
            self._load_sparse_data(x1, x2)
        elif self.input_format == 'matrix':
            self._load_dense_data(x1,x2)
        else:
            assert False, "Unrecognized data format '%s'" % self.input_format
        self.num_row = self.data.shape[0]
        self.num_col = self.data.shape[1]
        # self._fill_in_missing_motifs_regs(x1, x2)

    def _load_sparse_data(self, x1, x2):
        print "FUNCTION NOT IMPLEMENTED YET, PLEASE PROVIDE DENSE MATRIX"
        return 0

    def _load_dense_data(self, x1, x2):
        final_data=np.zeros((x1.num_row,x2.num_col))
        # final_row_labels=x1.col_labels # WHEN PROCESS LABELS EARLIER
        final_row_labels=[el.replace('-','_').replace('.','_') for el in x1.col_labels]
        final_col_labels=x2.row_labels
        prior_data=np.genfromtxt(self.data_file)
        prior_row_labels=self.row_labels.tolist()
        prior_col_labels=self.col_labels.tolist()
        # prior_row_match_ind=pd.match(prior_row_labels, [el.replace('-','_').replace('.','_') for el in final_row_labels.tolist()])
        prior_row_match_ind=pd.match(prior_row_labels, final_row_labels)
        prior_rows_to_transfer = [el for el in range(len(prior_row_labels)) if prior_row_match_ind[el]!=-1]
        final_rows_to_fill = prior_row_match_ind[prior_row_match_ind!=-1]
        prior_col_match_ind=pd.match(prior_col_labels, final_col_labels)
        prior_cols_to_transfer = [el for el in range(len(prior_col_labels)) if prior_col_match_ind[el]!=-1]
        final_cols_to_fill = prior_col_match_ind[prior_col_match_ind!=-1]
        final_data[np.ix_(final_rows_to_fill, final_cols_to_fill)]=prior_data[prior_rows_to_transfer,:][:,prior_cols_to_transfer]
        print 'sum of prior weights in final {0}'.format(final_data.sum())
        print 'sum of prior weights in full prior {0}'.format(prior_data.sum())
        self.data=final_data
        ### Show motifs/regulators not matching
        # mismatch_regs = [prior_col_labels[el] for el in xrange(len(prior_col_labels)) if prior_col_match_ind[el]==-1]
        # mismatch_motifs = [prior_row_labels[el] for el in xrange(len(prior_row_labels)) if prior_row_match_ind[el]==-1]

### Load the prior matrices
def parse_prior(prior_params, x1, x2):
    if prior_params.motif_reg_file!=None:
        prior_motifreg=Prior(prior_params.motif_reg_file,  
            prior_params.motif_reg_row_labels, prior_params.motif_reg_col_labels, 
            prior_params.prior_input_format, x1, x2)
    else:
        prior_motifreg=None
    if prior_params.reg_reg_file!=None:
        prior_regreg=Prior(prior_params.reg_reg_file, 
            prior_params.motif_reg_row_labels, prior_params.reg_reg_col_labels, 
            prior_params.prior_input_format, x1, x2)
        ### TEMPORARILY USE UNIFORM PPI WEIGHTS
        prior_regreg=np.ones((x2.num_col,x2.num_col))
    else:
        prior_regreg=None
    print 'loaded prior'
    return (prior_motifreg, prior_regreg)

### Update the loss based on priors provided
### motif-regulator and regulator-parent regulator
def update_loss_with_prior(loss_matrix, prior_params, prior_motifreg, prior_regreg):
    new_loss = loss_matrix
    if prior_motifreg!=None:
        motifreg_multiplier = prior_motifreg.data*prior_params.prior_constant*prior_params.prior_decay_rate + 1
        new_loss = util.element_mult(new_loss, motifreg_multiplier)
    if prior_regreg!=None:
        reg_row = reg_reg_prior[reg_reg_prior.row_labels==reg,]
        reg_mat = np.vstack([reg_row for el in xrange(reg_reg_prior.num_row)])
        regreg_multiplier = reg_mat*prior_params.prior_constant*prior_params.prior_decay_rate + 1
        new_loss = util.element_mult(new_loss, regreg_multiplier)
    # Update decay parameter (need to pass GLOBAL parameters so can update)
    return(new_loss)



