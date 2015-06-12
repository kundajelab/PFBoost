
def parse_args():
    # Get arguments
    parser = argparse.ArgumentParser(description='Extract Chromatin States')

    parser.add_argument('--output-prefix', 
                        help='Analysis name for output plots')
    parser.add_argument('--output-path', 
                        help='path to write the results to', 
                        default='/users/pgreens/projects/boosting/results/')

    parser.add_argument('-p', '--data-path', help='Path for all data files')
    parser.add_argument('-f', '--format', help='options are: matrix, triplet')

    parser.add_argument('-y', '--target-file', 
                        help='target matrix - dimensionality GxE')
    parser.add_argument('-g', '--target_row_labels', 
                        help='row labels for y matrix (dimension G)')
    parser.add_argument('-e', '--target_col_labels', 
                        help='column labels for y matrix (dimension E)')

    parser.add_argument('-x', '--motifs-file', 
                        help='x1 features - dimensionality MxG')
    parser.add_argument('-m', '--m-col-labels', 
                        help='column labels for x1 matrix (dimension M)')

    parser.add_argument('-z', '--regulators-file', 
                        help='x2 features - dimensionality ExR')
    parser.add_argument('-r', '--r-row-labels', 
                        help='row labels for x2 matrix (dimension R)')

    parser.add_argument('-n', '--num_iter', 
                        help='Number of iterations', default=500, type=int)

    parser.add_argument('--eta1', help='stabilization threshold 1', type=float)
    parser.add_argument('--eta2', help='stabilization threshold 2', type=float)

    parser.add_argument('-s', '--stumps', 
                        help='specify to do stumps instead of adt', 
                        action='store_true')
    parser.add_argument('-d', '--stable', 
                        help='bundle rules/implement stabilized boosting', 
                        action='store_true')
    parser.add_argument('-c', '--corrected-loss', 
                        action='store_true', help='For corrected Loss')

    parser.add_argument('-u', '--ncpu', 
                        help='number of cores to run on', type=int)

    parser.add_argument('--holdout-file', 
                        help='Specify holdout matrix, same as y dimensions', default=None)
    parser.add_argument('--holdout-format', 
                        help='format for holdout matrix', 
                        default=None)

    # Parse arguments
    args = parser.parse_args()
    
    global OUTPUT_PATH
    OUTPUT_PATH = args.output_path

    global OUTPUT_PREFIX
    OUTPUT_PATH = args.output_prefix
    
    global tuning_params
    tuning_params = TuningParams(
        args.num_iter, 
        args.stumps, args.stable, args.corrected_loss,
        args.eta1, args.eta2 )

    global NCPU
    NCPU = args.ncpu

    print_time('load y start ')
    y = TargetMatrix(os.path.join(args.data_path, args.target_file), 
                     os.path.join(args.data_path, args.target_row_labels), 
                     os.path.join(args.data_path, args.target_col_labels),
                     args.format)
    print_time('load y stop')

    print_time('load x1 start')
    x1 = Motifs(os.path.join(args.data_path, args.motifs_file), 
                os.path.join(args.data_path, args.target_row_labels), 
                os.path.join(args.data_path, args.m_col_labels),
                args.format)
    print_time('load x1 stop')
    
    print_time('load x2 start')
    x2 = Regulators(os.path.join(args.data_path, args.regulators_file), 
                    os.path.join(args.data_path, args.r_row_labels), 
                    os.path.join(args.data_path, args.target_col_labels),
                    args.format)
    print_time('load x2 stop')
   
    # model_state = ModelState()
    print_time('load holdout start')
    global holdout
    holdout_file = args.holdout_file
    holdout_format = args.holdout_format
    holdout = Holdout(y, holdout_file, holdout_format)
    print_time('load holdout stop')
    
    return (x1, x2, y)

def main():
    print 'starting main loop'

    ### Parse arguments
    print_time('parse args start')
    (x1, x2, y) = parse_args()
    print_time('parse args end')

    ### Time implementation
    t = time.time()

    ### Create tree object
    print_time('make tree start')
    tree = DecisionTree()
    print_time('make tree stop')

    ### Keeps track of if there are any terms to bundle
    bundle_set=1

    ### Main Loop
    for i in range(1,tuning_params.num_iter):

        print 'iteration {0}'.format(i)

        # State number of parameters to search
        if tuning_params.use_stumps:
            tree.nsearch = 1
        else:
            tree.nsearch = tree.npred

        ## Calculate loss at all search nodes
        print_time('start rule_processes')
        # rule_processes = pool.map(find_leaf_and_min_loss_wrapper, iterable=[ \
        #     (tree, x) \
        #     for x in range(tree.nsearch)]) # find rules with class call
        rule_processes = rule_processes_wrapper(tree)
        print_time('end rule_processes')

        # Find the best loss and split leaf
        print_time('start find_best_split_from_losses')
        best_split, reg, loss_best = find_best_split_from_losses(rule_processes)
        print_time('end find_best_split_from_losses')

        # Get rule weights for the best split
        print_time('start find_rule_weights')
        rule_weights = find_rule_weights(tree.ind_pred_train[best_split], tree.weights, tree.ones_mat)
        print_time('end find_rule_weights')

        # Get current rule, no stabilization
        print_time('start get_current_rule')
        m,r,reg,rule_train_index,rule_test_index = get_current_rule(tree, best_split, reg, loss_best)
        print_time('end get_current_rule')


        ## Update score without stabilization,  if stabilization results in one rule or if stabilization criterion not met
        rule_score = calc_score(tree, rule_weights, rule_train_index)
        m_bundle = []
        r_bundle = []

        ### Store motifs/regulators above this node (for margin score)
        above_motifs = tree.above_motifs[best_split]+tree.split_x1[best_split].tolist()
        above_regs = tree.above_regs[best_split]+tree.split_x2[best_split].tolist()

        ### Add the rule with best loss
        tree.add_rule(m, r, best_split, m_bundle, r_bundle, rule_train_index, rule_test_index, rule_score, above_motifs, above_regs)

        ### Update training/testing errors
        print_time('start update tree')
        tree.update_prediction(rule_score, rule_train_index, rule_test_index)
        tree.update_weights()
        tree.update_error()
        tree.update_margin()
        print_time('end update tree')

        ### Return default to bundle
        bundle_set = 1

        ### Print progress
        print_progress(tree, i)

    ## Write out rules
    if tuning_params.use_stable:
        list_rules(split_x1, split_x2, 
                   bundle_x1, bundle_x2, 
                   scores, split_node, split_depth, ind_pred_train, ind_pred_test, out_file='{0}rule_score_matrix_{1}_{2}_{3}_iter_stabilized_eta1_{4}_eta2_{5}.txt'.format(out_path, analysis_name, method, niter, eta_1, eta_2))
    else:
        list_rules(split_x1, split_x2, bundle_x1, bundle_x2, scores, split_node, split_depth, ind_pred_train, ind_pred_test, out_file='{0}rule_score_matrix_{1}_{2}_{3}_iter.txt'.format(out_path, analysis_name, method, niter))

    ### Make plots
    plot_margin(train_margins, test_margins, method, niter)
    plot_balanced_error(loss_train, loss_test, method, niter)
    plot_imbalanced_error(imbal_train, imbal_test, method, niter)
