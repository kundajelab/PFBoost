from boosting_2D import config
from boosting_2D import util
from boosting_2D import plot
from boosting_2D import margin_score
from boosting_2D import stabilize

log = util.log

def parse_args():
    # Get arguments
    parser = argparse.ArgumentParser(description='Extract Chromatin States')

    parser.add_argument('--output-prefix', 
                        help='Analysis name for output plots')
    parser.add_argument('--output-path', 
                        help='path to write the results to', 
                        default='/users/pgreens/projects/boosting/results/')

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
    
    config.OUTPUT_PATH = args.output_path
    config.OUTPUT_PREFIX = args.output_prefix
    config.TUNING_PARAMS = TuningParams(
        args.num_iter, 
        args.stumps, args.stable, args.corrected_loss,
        args.eta1, args.eta2 )
    config.NCPU = args.ncpu

    log('load y start ')
    y = TargetMatrix(args.target_file, 
                     args.target_row_labels, 
                     args.target_col_labels,
                     args.format)
    log('load y stop')

    log('load x1 start')
    x1 = Motifs(args.motifs_file, 
                args.target_row_labels, 
                args.m_col_labels,
                args.format)
    log('load x1 stop')
    
    log('load x2 start')
    x2 = Regulators(args.regulators_file, 
                    args.r_row_labels, 
                    args.target_col_labels,
                    args.format)
    log('load x2 stop')
   
    # model_state = ModelState()
    log('load holdout start')
    holdout = Holdout(y, args.holdout_file, args.holdout_format)
    log('load holdout stop')
    
    return (x1, x2, y, holdout)

def find_next_decision_node(tree, x1, x2):
    # State number of parameters to search
    if tuning_params.use_stumps:
        tree.nsearch = 1
    else:
        tree.nsearch = tree.npred

    ## Calculate loss at all search nodes
    log('start rule_processes')
    rule_processes = rule_processes_wrapper(tree)
    log('end rule_processes')

    # Find the best loss and split leaf
    log('start find_best_split_from_losses')
    best_split, reg, loss_best = find_best_split_from_losses(rule_processes)
    log('end find_best_split_from_losses')

    # Get rule weights for the best split
    log('start find_rule_weights')
    rule_weights = find_rule_weights(
        tree.ind_pred_train[best_split], tree.weights, tree.ones_mat)
    log('end find_rule_weights')

    ### get_bundled_rules (returns the current rule if no bundling)  
    # Get current rule, no stabilization
    log('start get_current_rule')
    motif,regulator,reg_sign,rule_train_index,rule_test_index = get_current_rule(
        tree, best_split, reg, loss_best)
    log('end get_current_rule')

    ## Update score without stabilization,  if stabilization results 
    ## in one rule or if stabilization criterion not met
    rule_score = calc_score(tree, rule_weights, rule_train_index)
    motif_bundle = []
    regulator_bundle = []

    ### Store motifs/regulators above this node (for margin score)
    above_motifs = tree.above_motifs[best_split]+tree.split_x1[best_split].tolist()
    above_regs = tree.above_regs[best_split]+tree.split_x2[best_split].tolist()

    return (motif, regulator, best_split, 
            motif_bundle, regulator_bundle, 
            rule_train_index, rule_test_index, rule_score, 
            above_motifs, above_regs)

def main():
    print 'starting main loop'

    ### Parse arguments
    log('parse args start')
    (x1, x2, y) = parse_args()
    log('parse args end')

    ### Create tree object
    log('make tree start')
    tree = DecisionTree()
    log('make tree stop')

    ### Keeps track of if there are any terms to bundle
    bundle_set=1

    ### Main Loop
    for i in range(1,tuning_params.num_iter):
        log('iteration {0}'.format(i), level='VERBOSE')
        
        (motif, regulator, best_split, 
         motif_bundle, regulator_bundle, 
         rule_train_index, rule_test_index, rule_score, 
         above_motifs, above_regs) = find_next_rule(tree)
        
        ### Add the rule with best loss
        tree.add_rule(m, r, best_split, 
                      m_bundle, r_bundle, 
                      rule_train_index, rule_test_index, rule_score, 
                      above_motifs, above_regs)

        ### Update training/testing errors
        log('start update tree')
        tree.update_prediction(rule_score, rule_train_index, rule_test_index)
        tree.update_weights()
        tree.update_error()
        tree.update_margin()
        log('end update tree')

        ### Return default to bundle
        bundle_set = 1

        ### Print progress
        util.log_progress(tree, i)


    ### Get rid of this, add a method to the tree:
    ## Write out rules
    list_rules(split_x1, split_x2, 
               bundle_x1, bundle_x2, 
               scores, 
               split_node, split_depth, 
               ind_pred_train, ind_pred_test, 
               out_file='{0}rule_score_matrix_{1}_{2}_{3}_iter.txt'.format(
                   out_path, analysis_name, method, niter))

    ### Make plots
    plot_margin(train_margins, test_margins, method, niter)
    plot_balanced_error(loss_train, loss_test, method, niter)
    plot_imbalanced_error(imbal_train, imbal_test, method, niter)
