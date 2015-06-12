### Define classes

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
        self.row_labels = np.genfromtxt(row_labels, delimiter="\n",dtype="str")
        self.col_labels = np.genfromtxt(col_labels, delimiter="\n",dtype="str")
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
               # holdout = csr_matrix((data[:,2], (data[:,0]-1, data[:,1]-1)),shape=(y.num_row,y.num_col))
            elif holdout_format=='matrix':
                if self.sparse:
                    self.holdout = csr_matrix(np.genfromtxt(holdout_file))
                else:
                    self.holdout = csr_matrix(np.genfromtxt(holdout_file)).toarray()
            # if self.holdout.shape[0]!=y.data.shape[0] or self.holdout.shape[1]!=y.data.shape[1]:
            #     assert False, "Holdout dimensions do not match target matrix '%s'" % holdout_file
        if holdout_file==None:
            np.random.seed(1)
            if self.sparse:
                self.holdout = coo_matrix(
                np.reshape(
                    np.random.choice(a=[0,1], size=y.num_row*y.num_col, replace=True, p=[0.8,0.2]),
                    (y.num_row, y.num_col)
                )).tocsr()
            else:
                self.holdout = np.reshape(
                    np.random.choice(a=[0,1], size=y.num_row*y.num_col, replace=True, p=[0.8,0.2]),
                    (y.num_row, y.num_col))
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
    def __init__(self):
        # Base model parameters
        self.nsplit = 0
        self.nsearch = 1
        self.npred = 1
        self.epsilon = 1./holdout.n_train
        self.split_x1 = [] # store motif split features
        self.split_x2 = [] # store regulator split features
        self.split_node = [] # For tree, specify node to split off of
        self.split_depth = [] # Store the depth of the new node (0 is root, 1 is the first layer)

        ### Stabilization Parameters
        self.bundle_x1 = [] # store motifs bundled with x1 min loss split
        self.bundle_x2 = [] # store motifs bundled with x2 min loss split
        self.bundle_set = 1

        ### Margin Score Parameters
        self.above_motifs = [] # motifs each split depends on
        self.above_regs = [] # regulators each split depends on
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

        # put in the root node
        self.init_root_node()
    
    def init_root_node(self):
        ## initialize the root node
        score_root = 0.5*np.log(
            element_mult(self.weights, holdout.ind_train_up).sum()/element_mult(self.weights, holdout.ind_train_down).sum())
        self.scores.append(score_root)
        # Add root node to first split
        self.split_x1.append(np.array(['root']))
        self.split_x2.append(np.array(['root']))
        # Add empty lists for store
        self.above_motifs.append([])
        self.above_regs.append([])
        # If stabilized add bundle
        self.bundle_x1.append([])
        self.bundle_x2.append([])
        self.split_node.append('root')
        self.split_depth.append(0)
        self.update_prediction(score_root, holdout.ind_train_all, holdout.ind_test_all)
        self.update_weights()
        # Initialize training error
        self.update_error()
        self.update_margin()

    # Store new rule
    def add_rule(self, m, r, best_split, m_bundle, r_bundle, rule_train_index, rule_test_index, rule_score, above_motifs, above_regs):
        self.split_x1.append(m)
        self.split_x2.append(r)
        self.split_node.append(best_split)

        self.nsplit += 1
        self.ind_pred_train.append(rule_train_index)
        self.ind_pred_test.append(rule_test_index)
        self.bundle_x1.append(m_bundle)
        self.bundle_x2.append(r_bundle)
        self.scores.append(rule_score)

        if best_split==0:
            self.split_depth.append(1)
        else:
            self.split_depth.append(self.split_depth[best_split]+1)
        self.above_motifs.append(above_motifs)
        self.above_regs.append(above_regs)
        self.npred += 1

        self.update_prediction(rule_score, rule_train_index, rule_test_index)
        self.update_weights()
        self.update_error()
        self.update_margin()

    def _update_prediction(self, score, train_index, test_index):        
        # Update predictions
        self.pred_train = self.pred_train + score*train_index
        self.pred_test = self.pred_test + score*test_index

    def _update_weights(self):
        # Update weights
        if y.sparse:
            exp_term = np.negative(y.element_mult(self.pred_train))
            exp_term.data = np.exp(exp_term.data)
        else:
            exp_term = np.negative(y.element_mult(self.pred_train))
            exp_term[exp_term.nonzero()]=np.exp(exp_term[exp_term.nonzero()])
        new_weights = element_mult(exp_term, holdout.ind_train_all)
        # print (new_weights/new_weights.sum())[new_weights.nonzero()]
        self.weights = new_weights/new_weights.sum()

    def _update_error(self):
        # Identify incorrect processesedictions
        incorr_train = (y.element_mult(self.pred_train)<0)
        incorr_test = (y.element_mult(self.pred_test)<0)

        # Balanced error
        bal_train_err_i = (float(element_mult(incorr_train, holdout.ind_train_up).sum())/holdout.ind_train_up.sum()
            +float(element_mult(incorr_train, holdout.ind_train_down).sum())/holdout.ind_train_down.sum())/2
        bal_test_err_i = (float(element_mult(incorr_test, holdout.ind_test_up).sum())/holdout.ind_test_up.sum()
            +float(element_mult(incorr_test, holdout.ind_test_down).sum())/holdout.ind_test_down.sum())/2

        ## Imbalanced error
        imbal_train_err_i=(float(element_mult(incorr_train, np.add(holdout.ind_train_up, holdout.ind_train_down)).sum())/holdout.ind_train_all.sum())
        imbal_test_err_i=(float(element_mult(incorr_test, np.add(holdout.ind_test_up, holdout.ind_test_down)).sum())/
            holdout.ind_test_all.sum())

        # Store error 
        self.bal_train_err.append(bal_train_err_i)
        self.bal_test_err.append(bal_test_err_i)
        self.imbal_train_err.append(imbal_train_err_i)
        self.imbal_test_err.append(imbal_test_err_i)

    def _update_margin(self):
        train_margin=calc_margin(y.data, self.pred_train)
        test_margin=calc_margin(y.data, self.pred_test)
        self.train_margins.append(train_margin)
        self.test_margins.append(test_margin)


