### Find best rule - NON-STABILIZED

# Wrapper - calc min loss with leaf training examples and current weights 
def find_leaf_and_min_loss(tree, leaf_index):
    print_time('start find_leaf_and_min_loss')
    example_weights=tree.weights
    ones_mat=tree.ones_mat
    leaf_training_examples = tree.ind_pred_train[leaf_index]
    print_time('start calc_min_leaf_loss')
    (best_loss, reg) = calc_min_leaf_loss(leaf_training_examples, example_weights, ones_mat)
    print_time('end calc_min_leaf_loss')
    return leaf_index, best_loss, reg

def find_rule_process_worker(tree, leaf_index_cntr, (
        lock, best_loss, best_leaf, shared_best_loss_mat, best_loss_reg)):
    # until we have processed all of the leafs
    while True:
        # get the leaf node to work on
        with leaf_index_cntr.get_lock():
            leaf_index = leaf_index_cntr.value
            leaf_index_cntr.value += 1
            
        # if this isn't a valid leaf, then we are done
        if leaf_index >= tree.nsearch: 
            return
        
        # calculate the loss for this leaf  
        leaf, loss_mat, loss_reg = find_leaf_and_min_loss(tree, leaf_index)

        # if the loss does not beat the current best loss, then
        # we are done
        loss = loss_mat.min()
        with lock:
            if loss > best_loss:
                continue

            # otherwise we know this rule currently produces the smallest loss, 
            # so save it
            best_leaf.value = leaf
            best_loss.value = loss
            # we update the array being stored in shared memory 
            if y.sparse:
                shared_best_loss_mat[:] = loss_mat.toarray().ravel()
            else:
                shared_best_loss_mat[:] = loss_mat.ravel()
            best_loss_reg.value = loss_reg
            print "="*80
            print leaf, loss_mat.shape, loss_reg
    
    return

def find_rule_processes(tree):
    rule_processes = []

    # this shoudl be an attribute of tree. Also, during the tree init,
    # accessing the global variables x1, x2, and y is really bad form. Since
    # you only need the dimensions you sohuld pass those as arguments into the 
    # init function. 
    nrow = x1.num_row
    ncol = x2.num_col
    
    # initialize a lock to control access to the best rule objects. We use
    # rawvalues because all access is governed through the lock
    lock = multiprocessing.Lock()
    # initialize this to a large number, so that the first loss
    # will be chosen
    best_loss = multiprocessing.RawValue(ctypes.c_double, 1e100)
    best_leaf = multiprocessing.RawValue('i', -1)
    shared_best_loss_mat = multiprocessing.RawArray(
        ctypes.c_double, nrow*ncol)
    best_loss_reg = multiprocessing.RawValue('i', 0)

    # store the value of the next leaf index that needs to be processed, so that
    # the workers know what leaf to work on
    leaf_index_cntr = multiprocessing.Value('i', 0)

    # pack arguments for the worker processes
    args = [tree, leaf_index_cntr, (
            lock, best_loss, best_leaf, shared_best_loss_mat, best_loss_reg)]
    
    # fork worker processes, and wait for them to return
    from grit.lib.multiprocessing_utils import fork_and_wait
    fork_and_wait(NCPU, find_rule_process_worker, args)
    
    # covert all of the shared types into standard python values
    best_leaf = int(best_leaf.value)
    best_loss_reg = int(best_loss_reg.value)
    # we convert the raw array into a numpy array
    best_loss_mat = np.reshape(np.array(shared_best_loss_mat), (nrow, ncol))
    
    # return rule_processes
    return (best_leaf, best_loss_reg, best_loss_mat)

# Function - calc min loss with leaf training examples and current weights  
def calc_min_leaf_loss(leaf_training_examples, example_weights, ones_mat):
    print_time('start find_rule_weights')
    rule_weights = find_rule_weights(leaf_training_examples, example_weights, ones_mat)
    print_time('end find_rule_weights')
    
    ## Calculate Loss
    if tuning_params.use_corrected_loss==True:
        loss_regup = corrected_loss(rule_weights.w_up_regup, rule_weights.w_down_regup, rule_weights.w_zero_regup)
        loss_regdown = corrected_loss(rule_weights.w_up_regdown, rule_weights.w_down_regdown, rule_weights.w_zero_regdown)
    else:
        loss_regup = calc_loss(rule_weights.w_up_regup, rule_weights.w_down_regup, rule_weights.w_zero_regup)
        loss_regdown = calc_loss(rule_weights.w_up_regdown, rule_weights.w_down_regdown, rule_weights.w_zero_regdown)

    ## Get loss matrix and regulator status
    loss_best_s = np.min([loss_regup.min(), loss_regdown.min()])
    loss_arg_min = np.argmin([loss_regup.min(), loss_regdown.min()])
    if loss_arg_min==0:
        reg_s=1
    else:
        reg_s=-1

    loss = [loss_regup, loss_regdown][loss_arg_min]
    return (loss, reg_s)

# Get rule weights of positive and negative examples
def find_rule_weights(leaf_training_examples, example_weights, ones_mat):
    """
    Find rule weights, and return an object store containing them. 

    """
    # print_time('find_rule_weights start')
    w_temp = element_mult(example_weights, leaf_training_examples)
    # print_time('weights element-wise')
    w_pos = element_mult(w_temp, holdout.ind_train_up)
    # print_time('weights element-wise')
    w_neg = element_mult(w_temp, holdout.ind_train_down) 
    # print_time('weights element-wise')
    x2_pos = x2.element_mult(x2.data>0)
    # print_time('x2 element-wise')
    x2_neg = abs(x2.element_mult(x2.data<0))
    # print_time('x2 element-wise')
    x1wpos = x1.matrix_mult(w_pos)
    # print_time('x1 weights dot')
    x1wneg = x1.matrix_mult(w_neg)
    # print_time('x1 weights dot')
    w_up_regup = matrix_mult(x1wpos, x2_pos)
    # print_time('x1w x2 dot')
    w_up_regdown = matrix_mult(x1wpos, x2_neg)
    # print_time('x1w x2 dot')
    w_down_regup = matrix_mult(x1wneg, x2_pos)
    # print_time('x1w x2 dot')
    w_down_regdown = matrix_mult(x1wneg, x2_neg)
    # print_time('x1w x2 dot')
    w_zero_regup = ones_mat - w_up_regup - w_down_regup
    # print_time('weights subtraction')
    w_zero_regdown = ones_mat - w_up_regdown - w_down_regdown
    # print_time('weights subtraction')
    return ObjectStore(w_up_regup, w_up_regdown, w_down_regup, w_down_regdown, w_zero_regup, w_zero_regdown, w_pos, w_neg) 

# Get best leaf, loss and regulator
def find_best_split_from_losses(rule_processes):
    ind_of_leaf = [el[0] for el in rule_processes]
    best_loss_by_leaf = [el[1] for el in rule_processes]
    reg_by_leaf = [el[2] for el in rule_processes]
    best_split = ind_of_leaf[np.argmin([el.min() for el in best_loss_by_leaf])]
    return best_split, reg_by_leaf[best_split], best_loss_by_leaf[best_split]

def get_current_rule(tree, best_split, reg, loss_best):
    # if y.sparse:
    #     m,r=np.where(loss_best.toarray() == loss_best.min())
    # else:
    m,r=np.where(loss_best == loss_best.min())
    # If multiple rules have the same loss, randomly select one
    if len(m)>1:
        choice = random.sample(range(len(m)), 1)
        m = np.array(m[choice])
        r = np.array(r[choice])
       
    ## Get examples where rule applies
    valid_m = np.nonzero(x1.data[m,:])[1]
    if x2.sparse:
        valid_r = np.where(x2.data.toarray()[:,r]==reg)[0]
    else:
        valid_r = np.where(x2.data[:,r]==reg)[0]
    
    ### Get rule index - training and testing
    if y.sparse:
        valid_mat = csr_matrix((y.num_row,y.num_col), dtype=np.bool)
    else:
        valid_mat = np.zeros((y.num_row,y.num_col), dtype=np.bool)
    valid_mat[np.ix_(valid_m, valid_r)]=1 # XX not efficient
    rule_train_index = element_mult(valid_mat, tree.ind_pred_train[best_split])
    rule_test_index = element_mult(valid_mat, tree.ind_pred_test[best_split])

    return m,r,reg,rule_train_index,rule_test_index

