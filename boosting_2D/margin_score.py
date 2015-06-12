### Margin Score functions
#######################

def calc_margin_score_x1_wrapper(args):
    return calc_margin_score_x1(*args)

# calc_margin_score_x1(tree, y, x1, x2, csr_matrix(np.ones((y.num_row, y.num_col))), 'YML081W_YPD')
def calc_margin_score_x1(tree, y, x1, x2, index_mat, x1_feat_index):
    x1_feat_name = x1.col_labels[x1_feat_index]
    allowed_x1_rules = [el for el in range(tree.npred)
         if x1_feat_index not in tree.above_motifs[el]
         +tree.split_x1[el].tolist()
         +tree.bundle_x1[el]]
    x1_feat_nodes = [el for el in range(tree.npred) 
         if x1_feat_index in tree.split_x1[el].tolist()
         +tree.bundle_x1[el]]
    x1_bundles = ['|'.join(x1.col_labels[[el for el in tree.split_x1[node].tolist()
         +tree.bundle_x1[node] if el != x1_feat_index]]) for node in x1_feat_nodes]
    x1_bundle_string = '--'.join([el if len(el)>0 else "none" for el in x1_bundles])
    # All rules where the motif is not above it, used in the split, or bundled in the split
    allowed_rules = [el for el in range(tree.npred)
         if x1_feat_index not in tree.above_motifs[el]
         +tree.split_x1[el].tolist()
         +tree.bundle_x1[el]]
    # New predictions without rule
    if y.sparse:
        pred_adj = csr_matrix((y.num_row,y.num_col), dtype='float64')
    else:
        pred_adj = np.zeros((y.num_row,y.num_col), dtype='float64')
    for rule in allowed_rules:
        pred_adj = pred_adj + tree.scores[rule]*tree.ind_pred_train[rule]
    margin_score = element_mult(y.element_mult(tree.pred_train-pred_adj), index_mat).sum()
    # Get all rules where x1 feat is above or in rule or bundle
    rules_w_x1_feat = [el for el in range(tree.npred)
         if x1_feat_index in tree.above_motifs[el]+
         tree.split_x1[el].tolist()
         +tree.bundle_x1[el]]
    rule_index_mat = tree.ind_pred_train[rules_w_x1_feat[0]]
    for r in rules_w_x1_feat:
        rule_index_mat = rule_index_mat + tree.ind_pred_train[r]
    # Index where x1 feat is used 
    rule_index_mat = (rule_index_mat>0)
    # Index where x1 feat is used and examples of interest
    rule_index_joint = element_mult(index_mat, rule_index_mat)
    # Fraction of examples of interest where x1 feat used
    rule_index_fraction = float(rule_index_joint.sum())/index_mat.sum()
    print rule_index_fraction
    return [x1_feat_name, x1_bundle_string, margin_score, rule_index_fraction]

def calc_margin_score_x2_wrapper(args):
    return calc_margin_score_x2(*args)

# calc_margin_score_x2(tree, y, x1, x2, csr_matrix(np.ones((y.num_row, y.num_col))), 'YDR085C')
def calc_margin_score_x2(tree, y, x1, x2, index_mat, x2_feat_index):
    x2_feat_name = x2.row_labels[x2_feat_index]
    allowed_x2_rules = [el for el in range(tree.npred)
         if x2_feat_index not in tree.above_regs[el]
         +tree.split_x2[el].tolist()
         +tree.bundle_x2[el]]
    x2_feat_nodes = [el for el in range(tree.npred)
         if x2_feat_index in tree.split_x2[el].tolist()
         +tree.bundle_x2[el]]
    x2_bundles = ['|'.join(x2.row_labels[[el for el in tree.split_x2[node].tolist()
         +tree.bundle_x2[node] if el != x2_feat_index]]) for node in x2_feat_nodes]
    x2_bundle_string = '--'.join([el  if len(el)>0 else "none" for el in x2_bundles])
    # New Prediction Matrix 
    if y.sparse:
        pred_adj = csr_matrix((y.num_row,y.num_col), dtype='float64')
    else:
        pred_adj = np.zeros((y.num_row,y.num_col), dtype='float64')
    for rule in allowed_x2_rules:
        pred_adj = pred_adj + tree.scores[rule]*tree.ind_pred_train[rule]
    margin_score = element_mult(y.element_mult(tree.pred_train-pred_adj), index_mat).sum()
    # Get all rules where x2 feat is above or in rule or bundle
    rules_w_x2_feat = [el for el in range(tree.npred)
         if x2_feat_index in tree.above_regs[el]+
         tree.split_x2[el].tolist()
         +tree.bundle_x2[el]]
    rule_index_mat = tree.ind_pred_train[rules_w_x2_feat[0]]
    for r in rules_w_x2_feat:
        rule_index_mat = rule_index_mat + tree.ind_pred_train[r]
    # Index where rule is used
    rule_index_mat = (rule_index_mat>0)
    # Index where rule is used and examples of interest
    rule_index_joint = element_mult(index_mat, rule_index_mat)
    # Fraction of examples of interest where rule applies
    rule_index_fraction = float(rule_index_joint.sum())/index_mat.sum()
    print rule_index_fraction
    return [x2_feat_name, x2_bundle_string, margin_score, rule_index_fraction]

def calc_margin_score_rule_wrapper(args):
    return calc_margin_score_rule(*args)

# calc_margin_score_rule(tree, y, x1, x2, csr_matrix(np.ones((y.num_row, y.num_col))), 'RIM101_YPD', 'YBR026C')
def calc_margin_score_rule(tree, y, x1, x2, index_mat, x1_feat_index, x2_feat_index):
    # Feature names
    x1_feat_name = x1.col_labels[x1_feat_index]
    x2_feat_name = x2.row_labels[x2_feat_index]

    # All rules where the x1/x2 feat is not above it, used in the split, or bundled in the split
    allowed_x1_rules = [el for el in range(tree.npred)
         if x1_feat_index not in tree.above_motifs[el]
         +tree.split_x1[el].tolist()
         +tree.bundle_x1[el]]
    allowed_x2_rules = [el for el in range(tree.npred)
         if x2_feat_index not in tree.above_regs[el]
         +tree.split_x2[el].tolist()
         +tree.bundle_x2[el]]
    # All the nodes that contain x1 and x2 in bundle
    pair_nodes = [el for el in range(tree.npred) 
         if x1_feat_index in tree.split_x1[el].tolist()
         +tree.bundle_x1[el] and x2_feat_index in 
         tree.split_x2[el].tolist()+tree.bundle_x2[el]]

    x1_bundles = ['|'.join(x1.col_labels[[el for el in tree.split_x1[node].tolist()
         +tree.bundle_x1[node] if el != x1_feat_index]]) for node in pair_nodes]
    x1_bundle_string = '--'.join([el if len(el)>0 else "none" for el in x1_bundles])

    x2_bundles = ['|'.join(x2.row_labels[[el for el in tree.split_x2[node].tolist()
         +tree.bundle_x2[node] if el != x2_feat_index]]) for node in pair_nodes]
    x2_bundle_string = '--'.join([el if len(el)>0 else "none" for el in x2_bundles ])

    allowed_rules = np.unique(allowed_x1_rules+allowed_x2_rules).tolist()
    # Allocate Prediction Matrix
    if y.sparse:
        pred_adj = csr_matrix((y.num_row,y.num_col), dtype='float64')
    else:
        pred_adj = np.zeros((y.num_row,y.num_col), dtype='float64')
    for rule in allowed_rules:
        pred_adj = pred_adj + tree.scores[rule]*tree.ind_pred_train[rule]
    margin_score = element_mult(y.element_mult(tree.pred_train-pred_adj), index_mat).sum()
    ### ! If considering specific rule only (rule with both motif and reg)
    rules_w_x1_feat_and_x2_feat = list(
        set([el for el in range(tree.npred)
         if x2_feat_index in tree.above_regs[el]+
         tree.split_x2[el].tolist()
         +tree.bundle_x2[el]])
        &
        set([el for el in range(tree.npred)
         if x1_feat_index in tree.above_motifs[el]+
         tree.split_x1[el].tolist()
         +tree.bundle_x1[el]]))
    ### ! If considering joint power of motif + reg (any rule with either)
    # rules_w_x1_feat_or_x2_feat = np.unique(
    #     [el for el in range(npred)
    #      if reg_index in above_regs[el]
    #      +tree.split_x2[el].tolist()
    #      +tree.bundle_x2[el]]+gut oy
    #      [el for el in range(npred)
    #      if motif_index in above_motifs[el]+
    #      tree.split_x1[el].tolist()+ 
    #      tree.bundle_x1[el]]).tolist()
    # Add up where every rule is used
    rule_index_mat = tree.ind_pred_train[rules_w_x1_feat_and_x2_feat[0]]
    for r in rules_w_x1_feat_and_x2_feat:
        rule_index_mat = rule_index_mat + tree.ind_pred_train[r]
    # Index where rule  is used 
    rule_index_mat = (rule_index_mat>0)
    # Index where rules is used and examples of interest
    rule_index_joint = element_mult(index_mat, rule_index_mat)
    # Fraction of examples of interest where rule used
    rule_index_fraction = float(rule_index_joint.sum())/index_mat.sum()
    print rule_index_fraction
    return [x1_feat_name, x1_bundle_string, x2_feat_name, x2_bundle_string, margin_score, rule_index_fraction]


def rank_by_margin_score(tree, y, x1, x2, index_mat, by_x1=False, by_x2=False, by_rules=True):
    # Rank x1 features only
    if by_x1:
        # All x1 features used in a treem, not equal to root
        used_x1_feats = np.unique([el.tolist()[0] for el in tree.split_x1 if el != 'root']+ \
                [el for listy in tree.bundle_x1 for el in listy if el != 'root']).tolist()
        # Get margin score for each x1 features
        rule_processes = pool.map(calc_margin_score_x1_wrapper, iterable=[ \
            (tree, y, x1, x2, index_mat, m) \
            for m in used_x1_feats])
        # Report data frame with feature 
        ranked_score_df = pd.DataFrame({'x1_feat':[el[0] for el in rule_processes], \
            'x1_feat_bundles':[el[1] for el in rule_processes], \
            'margin_score':[el[2] for el in rule_processes], \
            'rule_index_fraction':[el[3] for el in rule_processes]}).sort(columns=['margin_score'], ascending=False)
    # Rank x2 features only
    if by_x2:
        # All x2 features used in a treem, not equal to root
        used_x2_feats = np.unique([el.tolist()[0] for el in tree.split_x2 if el != 'root']+ \
                [el for listy in tree.bundle_x2 for el in listy if el != 'root']).tolist()
        # Get margin score for each x2 feature
        rule_processes = pool.map(calc_margin_score_x2_wrapper, iterable=[ \
            (tree, y, x1, x2, index_mat, r) \
            for r in used_x2_feats])
        # Report data frame with feature 
        ranked_score_df = pd.DataFrame({'x2_feat':[el[0] for el in rule_processes], \
            'x2_feat_bundles':[el[1] for el in rule_processes], \
            'margin_score':[el[2] for el in rule_processes], \
            'rule_index_fraction':[el[3] for el in rule_processes]}).sort(columns=['margin_score'], ascending=False)
    # Rank by rules 
    if by_rules:
        # unlike previous take non-unique sets
        used_x1_feats = [el.tolist()[0] for el in tree.split_x1 if el != 'root']+ \
                [el for listy in tree.bundle_x1 for el in listy if el != 'root']
        used_x2_feats = [el.tolist()[0] for el in tree.split_x2 if el != 'root']+ \
                [el for listy in tree.bundle_x2 for el in listy if el != 'root']
        # Get margin score for each rule
        rule_processes = pool.map(calc_margin_score_rule_wrapper, iterable=[ \
            (tree, y, x1, x2, index_mat, used_x1_feats[i], used_x2_feats[i])  \
            for i in range(len(used_x2_feats))])

        # Report data frame with feature 
        ranked_score_df = pd.DataFrame({'x1_feat':[el[0] for el in rule_processes], \
            'x1_feat_bundles':[el[1] for el in rule_processes], \
            'x2_feat':[el[2] for el in rule_processes], \
            'x2_feat_bundles':[el[3] for el in rule_processes], \
            'margin_score':[el[4] for el in rule_processes], \
            'rule_index_fraction':[el[5] for el in rule_processes]}).sort(columns=['margin_score'], ascending=False)
    ranked_score_df.drop_duplicates()
    # Return matrix
    return ranked_score_df


### Get the index with a text file or document of things of interest
# index_mat = get_index(y, x1, x2, tree, x2_feat_file='/scratch/shared/bmi212boost_doNotDelete/index_files/SJ_ethnicity_is_NotHispanic_row16_x2_index.txt')
def get_index(y, x1, x2, tree, x1_feat_file=None, x2_feat_file=None):
    if x1_feat_file!=None:
        x1_file = pd.read_table(x1_feat_file, header=None)
        # if providing index numbers
        if x1_file.applymap(lambda x: isinstance(x, (int, float))).sum().tolist()[0]==x1_file.shape[0]:
            x1_index = x1_file.ix[:,0].tolist()
        # if providing labels
        else:
            x1_index = [el for el in range(y.data.shape[0]) if y.row_labels[el] in x1_file.ix[:,0].tolist()]
    # else allow all
    else:
        x1_index = range(x1.num_col)
    if x2_feat_file!=None:
        x2_file = pd.read_table(x2_feat_file, header=None)
        # if providing index numbers
        if x2_file.applymap(lambda x: isinstance(x, (int, float))).sum().tolist()[0]==x2_file.shape[0]: # this is terrible fix
            x2_index = x2_file.ix[:,0].tolist()
        # if providing labels
        else:
            x2_index = [el for el in range(y.data.shape[1]) if y.col_labels[el] in x2_file.ix[:,0].tolist()]
    else:
        x2_index = range(x2.num_row)
    index_mat = np.zeros((y.num_row, y.num_col), dtype=bool)
    index_mat[np.ix_(x1_index, x2_index)]=1
    if y.sparse:
        index_mat = csr_matrix(index_mat)
    return index_mat


