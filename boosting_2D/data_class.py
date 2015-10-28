### Define classes
import sys
import os
import random

import numpy as np 
from scipy.sparse import *
import pandas as pd

from boosting_2D.util import *

import cPickle as pickle
import gzip
import config

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

    # This method assumes data is in a gzipped pickle file
    def _load_3D_data(self):
        self.data = pickle.load( gzip.open(self.data_file, 'rb') )
        self.data_hstacked = csc_matrix(hstack(self.data), dtype='float64')
        return None
        

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
                  input_format, mult_format, matrix_is_3d=False, make_3d_compatible=False):
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

        # Load the data very differently if 3D
        if matrix_is_3d:
            self._load_3D_data()
            self.matrix_is_3d = True
            self.num_row = self.data[0].shape[0]
            self.num_col = self.data[0].shape[1]
        else:
            self.matrix_is_3d = False
            if self.input_format == 'triplet':
                self._load_sparse_data()
            elif self.input_format == 'matrix':
                self._load_dense_data()
            else:
                assert False, "Unrecognized data format '%s'" % self.input_format
            self.num_row = self.data.shape[0]
            self.num_col = self.data.shape[1]

        # If target matrix,
        if make_3d_compatible:
            self._separateDataColumns()
            self.data = block_diag(self.dataColList, 'csr', dtype='float64')
            self.num_row = self.data.shape[0]
            self.num_col = self.data.shape[1]

    def _separateDataColumns(self):
        '''
        This function is used to make the 3D multiplication easier (x1 * y)
        '''
        [nrows, ncols] = self.data.shape
        self.dataColList = []
        for colNum in xrange(ncols):
            self.dataColList.append(self.data[:, colNum])
        return None

class TargetMatrix(Data):
    # 10/5 Don't convert to block diagonal until after holdout is generated??
    pass

class Motifs(Data):
    # Redefine matrix_mult here, if data object is of this class it will always use this method

    def is_3D(self):
        return self.matrix_is_3d

    def matrix_mult(self, other):
        if self.matrix_is_3d ==False:
            import time
            start = time.time()
            result = super(Motifs, self).matrix_mult(other)
            end = time.time()
            print 'time:', str(end - start)
            return result
        else:
            if True:

                # This stuff needs to NOT happen here!!
#                import time
#
#                start = time.time()
#                [nrows, ncols] = other.shape
#                otherColList = []
#                for colNum in xrange(ncols):
#                    otherColList.append(other[:,colNum])
#                end = time.time()
#                print "splitting into list:", str(end - start)

#                start = time.time()
#                other_blockdiag = block_diag(otherColList, 'csc', dtype='float64')
#                end = time.time()
#                print 'blocking:', str(end - start)
                # IF at all possible

                # Don't forget to convert to CSC

#                import time
#                start = time.time()
                result = self.data_hstacked.dot(csc_matrix(other))
#                end = time.time()
#                print 'time:', str(end - start)
                return csr_matrix(result)

            elif False:

                # for now assume that matrix 'other' is not column split, consider changing for speed up
                import time

                start = time.time()
                [nrows, ncols] = other.shape
                otherColList = []
                for colNum in xrange(ncols):
                    otherColList.append(other[:,colNum])

                # Now multiply
                multiplied_columns = [self.data[i].dot(otherColList[i]) for i in range(ncols)]
                multiplied_array = csr_matrix(hstack(multiplied_columns))

                end = time.time()
                print "current:", str(end - start)
                return multiplied_array

            else:

                # Try pre-allocate dense matrix and fill in?
#                results = np.empty([self.data[0].shape[0], ncols])
#                print results.shape
#                for i in range(ncols):
#                    tmp = self.data[i].dot(otherColList[i]).todense()
#                    import ipdb
#                    ipdb.set_trace()
#                    print tmp.shape
#                    results[:,i] = tmp[:,0]
#                results = csr_matrix(results)
                

                # Allocate memory first for output
                # np.dot use out option to send to pre-allocated output matrix

#                import ipdb
#                ipdb.set_trace()

#                result = np.empty([self.num_row, ncols])
#                for i in range(ncols):
#                    np.dot(self.data[i], otherColList[i], result[:,i])
#                    self.data[i].dot(otherColList[i], result[:,i])

#                multiplied_columns = [self.data[i].dot(otherColList[i], result[:,i]) for i in range(ncols)]
#                print result.shape

                # Notes w CS
                # Kronecker product of X2 with sparse identity matrix gives you block diagonal matrix scipy.sparse.kron
                # Hstack the x1 matrix
                # 1 matrix multiply for output
                other_blockdiag = block_diag(otherColList, 'csc', dtype='b')
                print other_blockdiag.shape

                self_hstacked = csc_matrix(hstack(self.data), dtype='b')
                print self_hstacked.shape
                
                
                start = time.time()
#                other_blockdiag = block_diag(otherColList, 'csc', dtype='b')
                result = self_hstacked.dot(other_blockdiag)
                end = time.time()

                print 'CS soln:', str(end - start)

                # Try a reshape method
                self_kron = kron(self.data[0], identity(ncols))
                print self_kron.shape
                other_col = csr_matrix(other.todense().reshape([ncols*nrows, 1]))
                print other_col.shape

                start = time.time()
                result = self_kron.dot(other_col)
                end = time.time()

                print "reshape method:", str(end - start)

                # The actual benchmark
                self_data = self.data[0]
                start = time.time()
                result = self_data.dot(other)
                end = time.time()
                print "normal:", str(end - start)

                import ipdb
                ipdb.set_trace()

                import time
                start = time.time()
                [nrows, ncols] = other.shape
                multiplied_columns = [self.data[i].dot(other[:,i]) for i in range(ncols)] # this is the long time step
                end1 = time.time()
                multiplied_array = csr_matrix(hstack(multiplied_columns))
                end2 = time.time()
                print '>> Matrix multiply end1:', str(end1 - start)
                print '>> Matrix multiply:', str(end2 -end1)

                # Test speed of pre-split other matrix
                start = time.time()
                self.data[0].dot(other[:,0])
                end = time.time()
                print '2D version:', str(end - start)

                # Test speed of tensordot
                print "testing other methods"
                
                D,M,N,R = 1,2,3,4
                A = np.random.rand(M,N,R)
                B = np.random.rand(N,R)
                
                print type(A)
                print type(B)
                print np.einsum('mnr,nr->mr', A, B).shape


                dense_columns = [np.array(self.data[i].todense()) for i in range(ncols)]
                array_3d = np.dstack(dense_columns)
                print array_3d.shape
                print other.shape
                print type(array_3d)
                print type(other)
                
                start = time.time()
                multiplied_array = np.einsum('mnr,nr->mr', array_3d, np.array(other.todense()))
                end = time.time()
                print "einsum:", str(end - start)
                print multiplied_array.shape

                return multiplied_array
        return None

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
            np.random.seed(1) # Deterministic randomness
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

        # Once holdout is made, make into the block diagonal format and use for rest of the run
        # Actually do before


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
        self.weights = holdout.ind_train_all*1./holdout.n_train # CHANGE FOR 3D SEQUENCE MATRIX; convert to block diag

        # Pre-allocate matrices
        if y.sparse:
            self.w_up_regup = csr_matrix((x1.num_row, x2.num_col),dtype='float64') # CHANGE FOR 3D SEQUENCE MATRIX
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
        if config.BOOST_MODE == 'ADABOOST':
            score_root = 0.5*np.log(
                element_mult(self.weights, holdout.ind_train_up).sum()/element_mult(
                    self.weights, holdout.ind_train_down).sum())
        elif config.BOOST_MODE == 'LOGITBOOST': # Score represents the summed working response Z
            score_root = 0.00000001
        else:
            print "YOU HAVE A PROBLEM WHAT IS BOOST MODE"
            score_root = 0
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

        self._update_prediction(rule_score, rule_train_index, rule_test_index) # Here's where it matters which one feature space you use!
        self._update_weights(holdout,y) # May want to use weights
        self._update_error(holdout, y)
        self._update_margin(y)

    def _update_prediction(self, score, train_index, test_index):        
        # Update predictions
        self.pred_train = self.pred_train + score*train_index
        self.pred_test = self.pred_test + score*test_index

    def _update_weights(self, holdout, y): 
        # Update example weights
        if config.BOOST_MODE == 'ADABOOST':
            if self.sparse:
                exp_term = np.negative(y.element_mult(self.pred_train))
                exp_term.data = np.exp(exp_term.data)
            else:
                exp_term = np.negative(y.element_mult(self.pred_train))
                exp_term[exp_term.nonzero()]=np.exp(exp_term[exp_term.nonzero()])
            new_weights = element_mult(exp_term, holdout.ind_train_all)
            # print (new_weights/new_weights.sum())[new_weights.nonzero()]
            self.weights = new_weights/new_weights.sum() # Change this for LOGITBOOST
        elif config.BOOST_MODE == 'LOGITBOOST':
            # Calculate p(x_i)
            example_probs = csr_matrix((1/(1+np.exp(-2*self.pred_train.data)), # Prob                                        
                                        self.pred_train.indices,
                                        self.pred_train.indptr),
                                       shape=self.pred_train.shape, dtype='float64')
            # Calculate the example weights, w_i = p(x_i) * ( 1 - p(x_i) )
#            self.weights = element_mult(example_probs, csr_matrix(np.ones(example_probs.shape)) - example_probs) # W
            self.weights = element_mult(example_probs, csr_matrix(example_probs>0) - example_probs)
        else:
            print "YOU HAVE A PROBLEM WHAT IS BOOST MODE"
            self.weights = None
            
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

    def write_out_rules(self, tree, x1, x2, tuning_params, out_file=None):
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
