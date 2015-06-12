### Plotting functions
#######################

# Plot balanced training and testing error
def plot_balanced_error(tree, tuning_params, method_label, iteration):
    plt.figure(figsize=(12, 9))
    plt.plot(range(iteration), tree.bal_train_err[0:iteration], color="red", label='Training')
    plt.plot(range(iteration), tree.bal_test_err[0:iteration], color="blue", label='Testing')
    plt.xlabel('Iteration')
    plt.ylabel('Balanced Error')
    plt.title('Balanced Error - 2D Boosting \n {0} method, {1} iterations \n Training error {2:.4f}, Testing error: {3:.4f}'.format(method_label, iteration, tree.bal_train_err[iteration-1], tree.bal_test_err[iteration-1]))
    plt.legend(loc=1)
    plt.savefig('{0}plots/{1}_balanced_train_test_error_{2}_{3}iter.png'.format(OUTPUT_PATH, OUTPUT_PREFIX, method_label, tuning_params.num_iter))

# Plot imbalanced training and testing error
def plot_imbalanced_error(tree, tuning_params, method_label, iteration):
    plt.figure(figsize=(12, 9))
    plt.plot(range(iteration), tree.imbal_train_err[0:iteration], color="red", label='Training')
    plt.plot(range(iteration), tree.imbal_test_err[0:iteration], color="blue", label='Testing')
    plt.xlabel('Iteration')
    plt.ylabel('Imbalanced Error')
    plt.title('Imbalanced Error - 2D Boosting \n {0} method, {1} iterations \n Training error {2:.4f}, Testing error: {3:.4f}'.format(method_label, iteration, tree.imbal_train_err[iteration-1], tree.imbal_test_err[iteration-1]))
    plt.legend(loc=1)
    plt.savefig('{0}plots/{1}_imbalanced_train_test_error_{2}_{3}iter.png'.format(OUTPUT_PATH, OUTPUT_PREFIX, method_label, tuning_params.num_iter))

# Plot margin
def plot_margin(tree, tuning_params, method_label, iteration):
    plt.figure(figsize=(12, 9))
    plt.plot(range(iteration), tree.train_margins[0:iteration], color="red", label='Training')
    plt.plot(range(iteration), tree.test_margins[0:iteration], color="blue", label='Testing')
    plt.ylim((0,np.max(tree.train_margins[0:iteration])+1))
    plt.xlabel('Iteration')
    plt.ylabel('Margin')
    plt.title('Margin - 2D Boosting \n {0} method, {1} iterations \n Training error {2:.4f}, Testing error: {3:.4f}'.format(method_label, iteration, tree.imbal_train_err[iteration-1], tree.imbal_test_err[iteration-1]))
    plt.legend(loc=4)
    plt.savefig('{0}plots/{1}_margin_train_test_{2}_{3}iter.png'.format(OUTPUT_PATH, OUTPUT_PREFIX, method_label, tuning_params.num_iter))
