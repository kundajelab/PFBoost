### Define classes
import sys
import os
import random

import numpy as np 
from scipy.sparse import *
import pandas as pd

from boosting_2D.util import *

class Data(object):
    def _load_sparse_data(self):
        data = np.genfromtxt(self.data_file)
        # Load triplet data data as sparse matrices (
        # subtract 1 from index for 0 indices)
        if self.sparse:
            self.data = csr_matrix(
                (data[:,2], (data[:,0]-1, data[:,1]-1)),
                shape=(max(data[:,0]),max(data[:,1])))
        else:
            ### XXX FIX
            self.data = csr_matrix(
                (data[:,2], (data[:,0]-1, data[:,1]-1)),
                shape=(max(data[:,0]),max(data[:,1]))).toarray() # better way to do this?
           
    def _load_dense_data(self):
        if self.sparse:
            self.data = csr_matrix(np.genfromtxt(self.data_file))
        else:
            self.data = csr_matrix(np.genfromtxt(self.data_file)).toarray()

    def element_mult(self, other):
        if self.sparse:
            return self.data.multiply(other)
        else:
            return np.multiply(self.data, other)

    def matrix_mult(self, other):
        if self.sparse:
            return self.data.dot(other)
        else:
            return np.dot(self.data, other)

    def __init__( self, 
                  data_file, row_labels, col_labels, 
                  input_format, mult_format):
        self.data_file = data_file
        self.row_label_file=row_labels
        self.col_label_file=col_labels
        self.row_labels = np.genfromtxt(self.row_label_file, delimiter="\n",dtype="str")
        self.col_labels = np.genfromtxt(self.col_label_file, delimiter="\n",dtype="str")
        self.input_format = input_format
        self.mult_format = mult_format
        if self.mult_format == 'sparse':
            self.sparse = True
        elif self.mult_format == 'dense':
            self.sparse = False
        else:
            assert False, "Unrecognized mult format '%s'" % self.mult_format
        if self.input_format == 'triplet':
            self._load_sparse_data()
        elif self.input_format == 'matrix':
            self._load_dense_data()
        else:
            assert False, "Unrecognized data format '%s'" % self.input_format
        self.num_row = self.data.shape[0]
        self.num_col = self.data.shape[1]

class TargetMatrix(Data):
    pass

class Motifs(Data):
    pass

class Regulators(Data):
    pass

class Holdout(object):
    """
    Indices for training/testing examples split into target up and down

    """
    def __init__(self, y, mult_format, holdout_file=None, holdout_format=None):
        log('start')
        self.mult_format = mult_format
        if self.mult_format == 'sparse':
            self.sparse = True
        elif self.mult_format == 'dense':
            self.sparse = False
        else:
            assert False, "Unrecognized mult format '%s'" % self.mult_format
        if holdout_file!=None:
            if holdout_format=='triplet':
                data = np.genfromtxt(holdout_file)
                if self.sparse:
                    self.holdout = csr_matrix(
                        (data[:,2], (data[:,0]-1, data[:,1]-1)),
                        shape=(y.num_row,y.num_col))
                else:
                    self.holdout = csr_matrix(
                        (data[:,2], (data[:,0]-1, data[:,1]-1)),
                        shape=(y.num_row,y.num_col)).toarray()
            elif holdout_format=='matrix':
                if self.sparse:
                    self.holdout = csr_matrix(np.genfromtxt(holdout_file))
                else:
                    self.holdout = csr_matrix(np.genfromtxt(holdout_file)).toarray()
        if holdout_file==None:
            np.random.seed(1)
            if self.sparse:
                self.holdout = coo_matrix(
                np.reshape(
                    np.random.choice(a=[0,1], size=y.num_row*y.num_col,
                     replace=True, p=[0.8,0.2]),
                    (y.num_row, y.num_col))).tocsr()
            else:
                self.holdout = np.reshape(
                    np.random.choice(a=[0,1], size=y.num_row*y.num_col,
                     replace=True, p=[0.8,0.2]), (y.num_row, y.num_col))
        log('allocate holdout')
        self.ind_test_up =  element_mult(self.holdout, y.data==1)
        self.ind_test_down = element_mult(self.holdout, y.data==-1)
        self.ind_train_up =  element_mult(self.holdout!=1, y.data==1) 
        self.ind_train_down = element_mult(self.holdout!=1, y.data==-1)
        log('get holdout indices')

        self.ind_train_all = np.add(self.ind_train_up, self.ind_train_down)
        self.ind_test_all = np.add(self.ind_test_up, self.ind_test_down)
        self.n_train =  self.ind_train_all.sum()
        self.n_test = self.ind_test_all.sum()
        log('end holdout')

class DecisionTree(object):
    def __init__(self, holdout, y, x1, x2):
        ## Base model parameters
        # the number of nodes in the tree
        self.nsplit = 0

        # store the features used at each decision node
        self.split_x1 = [] # store motif split features
        self.split_x2 = [] # store regulator split features

        # node that it was added onto
        self.split_node = [] # For tree, specify node to split off of
        self.split_depth = [] # Store the depth of the new node (0 is root, 1 is the first layer)

        ### Stabilization Parameters
        self.bundle_x1 = [] # store motifs bundled with x1 min loss split
        self.bundle_x2 = [] # store motifs bundled with x2 min loss split
        self.bundle_set = 1

        ### Margin Score Parameters
        self.above_motifs = [] # motifs each split depends on
        self.above_regs = [] # regulators each split depends on
        self.above_nodes = []
        # self.above_pred = []
        self.train_margins = []
        self.test_margins = []

        ### Error parameters
        self.bal_train_err = []
        self.bal_test_err = []
        self.imbal_train_err = []
        self.imbal_test_err = []

        ### Prediction Parameters
        self.ind_pred_train = []
        self.ind_pred_test = []

        self.ind_pred_train.append(holdout.ind_train_all)
        self.ind_pred_test.append(holdout.ind_test_all)
        self.scores = []
        if y.sparse:
            self.pred_train = csr_matrix((y.num_row, y.num_col),dtype='float64')
            self.pred_test = csr_matrix((y.num_row, y.num_col),dtype='float64')
        else:
            self.pred_train = np.zeros((y.num_row, y.num_col),dtype='float64')
            self.pred_test = np.zeros((y.num_row, y.num_col),dtype='float64')

        ### Weights 
        self.weights = holdout.ind_train_all*1./holdout.n_train

        # Pre-allocate matrices
        if y.sparse:
            self.w_up_regup = csr_matrix((x1.num_row, x2.num_col),dtype='float64')
            self.w_down_regup = csr_matrix((x1.num_row, x2.num_col),dtype='float64')
            self.w_up_regdown = csr_matrix((x1.num_row, x2.num_col),dtype='float64') 
            self.w_down_regdown = csr_matrix((x1.num_row, x2.num_col),dtype='float64') 
            self.ones_mat = csr_matrix(np.ones((x1.num_row, x2.num_col)),dtype='float64')
        else:
            self.w_up_regup = np.zeros((x1.num_row, x2.num_col),dtype='float64')
            self.w_down_regup = np.zeros((x1.num_row, x2.num_col),dtype='float64')
            self.w_up_regdown = np.zeros((x1.num_row, x2.num_col),dtype='float64') 
            self.w_down_regdown = np.zeros((x1.num_row, x2.num_col),dtype='float64') 
            self.ones_mat = np.ones((x1.num_row, x2.num_col),dtype='float64')

        if y.mult_format=='sparse':
            self.sparse=True
        else:
            self.sparse=False
        # put in the root node
        self.init_root_node(holdout, y)
    
    def init_root_node(self, holdout, y):
        ## initialize the root node
        score_root = 0.5*np.log(
            element_mult(self.weights, holdout.ind_train_up).sum()/element_mult(
                self.weights, holdout.ind_train_down).sum())
        self.scores.append(score_root)
        # Add root node to first split
        self.split_x1.append(np.array(['root']))
        self.split_x2.append(np.array(['root']))
        self.nsplit += 1
        # Add empty lists for store
        self.above_motifs.append([])
        self.above_regs.append([])
        self.above_nodes.append([])
        # If stabilized add bundle
        self.bundle_x1.append([])
        self.bundle_x2.append([])
        self.split_node.append('root')
        self.split_depth.append(0)
        self._update_prediction(
            score_root, holdout.ind_train_all, holdout.ind_test_all)
        self._update_weights(holdout, y)
        # Initialize training error
        self._update_error(holdout, y)
        self._update_margin(y)

    # Store new rule
    def add_rule(self, motif, regulator, best_split, motif_bundle,
     regulator_bundle, rule_train_index, rule_test_index, 
     rule_score, above_motifs, above_regs, holdout, y):
        self.split_x1.append(motif)
        self.split_x2.append(regulator)
        self.split_node.append(best_split)

        self.nsplit += 1
        self.ind_pred_train.append(rule_train_index)
        self.ind_pred_test.append(rule_test_index)
        self.bundle_x1.append(motif_bundle)
        self.bundle_x2.append(regulator_bundle)
        self.scores.append(rule_score)

        if best_split==0:
            self.split_depth.append(1)
        else:
            self.split_depth.append(self.split_depth[best_split]+1)
        self.above_motifs.append(above_motifs)
        self.above_regs.append(above_regs)
        self.above_nodes.append([best_split]+self.above_nodes[best_split])

        self._update_prediction(rule_score, rule_train_index, rule_test_index)
        self._update_weights(holdout,y)
        self._update_error(holdout, y)
        self._update_margin(y)

    def _update_prediction(self, score, train_index, test_index):        
        # Update predictions
        self.pred_train = self.pred_train + score*train_index
        self.pred_test = self.pred_test + score*test_index

    def _update_weights(self, holdout, y):
        # Update weights
        if self.sparse:
            exp_term = np.negative(y.element_mult(self.pred_train))
            exp_term.data = np.exp(exp_term.data)
        else:
            exp_term = np.negative(y.element_mult(self.pred_train))
            exp_term[exp_term.nonzero()]=np.exp(exp_term[exp_term.nonzero()])
        new_weights = element_mult(exp_term, holdout.ind_train_all)
        # print (new_weights/new_weights.sum())[new_weights.nonzero()]
        self.weights = new_weights/new_weights.sum()

    def _update_error(self, holdout, y):
        # Identify incorrect processesedictions
        incorr_train = (y.element_mult(self.pred_train)<0)
        incorr_test = (y.element_mult(self.pred_test)<0)

        # Balanced error
        bal_train_err_i = (float(element_mult(incorr_train,
             holdout.ind_train_up).sum())/holdout.ind_train_up.sum()
            +float(element_mult(incorr_train, holdout.ind_train_down
                ).sum())/holdout.ind_train_down.sum())/2
        bal_test_err_i = (float(element_mult(incorr_test, 
            holdout.ind_test_up).sum())/holdout.ind_test_up.sum()
            +float(element_mult(incorr_test, holdout.ind_test_down
                ).sum())/holdout.ind_test_down.sum())/2

        ## Imbalanced error
        imbal_train_err_i=(float(element_mult(incorr_train,
             np.add(holdout.ind_train_up, holdout.ind_train_down)
             ).sum())/holdout.ind_train_all.sum())
        imbal_test_err_i=(float(element_mult(incorr_test, 
             np.add(holdout.ind_test_up, holdout.ind_test_down)
             ).sum())/holdout.ind_test_all.sum())

        # Store error 
        self.bal_train_err.append(bal_train_err_i)
        self.bal_test_err.append(bal_test_err_i)
        self.imbal_train_err.append(imbal_train_err_i)
        self.imbal_test_err.append(imbal_test_err_i)

    def _update_margin(self, y):
        train_margin=calc_margin(y.data, self.pred_train)
        test_margin=calc_margin(y.data, self.pred_test)
        self.train_margins.append(train_margin)
        self.test_margins.append(test_margin)

    def write_out_rules(self, tree, x1, x2, tuning_params,
         method_label, out_file=None):
        # Allocate matrix of rules
        rule_score_mat = pd.DataFrame(index=range(len(tree.split_x1)-1),
         columns=['x1_feat', 'x2_feat', 'score', 'above_rule', 'tree_depth'])
        for i in xrange(1,len(tree.split_x1)):
            x1_ind = [tree.split_x1[i]]+tree.bundle_x1[i]
            x2_ind = [tree.split_x2[i]]+tree.bundle_x2[i]
            above_node = tree.split_node[i]
            rule_score_mat.ix[i-1,'x1_feat'] = '|'.join(
                np.unique(x1.row_labels[x1_ind]).tolist())
            rule_score_mat.ix[i-1,'x2_feat'] = '|'.join(
                np.unique(x2.col_labels[x2_ind]).tolist())
            rule_score_mat.ix[i-1,'score'] = tree.scores[i]
            if tree.split_x1[above_node]=='root':
                rule_score_mat.ix[i-1,'above_rule'] = 'root'
            else:
                rule_score_mat.ix[i-1,'above_rule'] = '{0};{1}'.format(
                     '|'.join(np.unique(x1.row_labels[
                        [tree.split_x1[above_node]]+
                     tree.bundle_x1[above_node]]).tolist()),
                     '|'.join(np.unique(x2.col_labels[
                        [tree.split_x2[above_node]]+
                     tree.bundle_x2[above_node]]).tolist()))       
            rule_score_mat.ix[i-1,'tree_depth'] = tree.split_depth[i]
        if out_file!=None:
            print 'wrote rules to {0}'.format(out_file)
            rule_score_mat.to_csv(out_file, sep="\t", header=True, index=False)
            return 1
        else:
            return rule_score_mat
