import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import numpy as np

from boosting_2D import config

### Plotting functions
#######################

# Create plot directory in output folder
def configure_plot_dir(tree, iteration):
    if not os.path.exists('{0}{1}/plots'.format(config.OUTPUT_PATH, config.OUTPUT_PREFIX)):
        os.makedirs('{0}{1}/plots'.format(config.OUTPUT_PATH, config.OUTPUT_PREFIX))

# Plot balanced training and testing error
def plot_balanced_error(tree, iteration):
    plt.figure(figsize=(12, 9))
    plt.plot(range(iteration), tree.bal_train_err[0:iteration], color="red", label='Training')
    plt.plot(range(iteration), tree.bal_test_err[0:iteration], color="blue", label='Testing')
    plt.xlabel('Iteration')
    plt.ylabel('Balanced Error')
    plt.title('Balanced Error - 2D Boosting \n {0} - {1} iterations \n Training error {2:.4f}, Testing error: {3:.4f}'.format(
        config.OUTPUT_PREFIX, iteration, tree.bal_train_err[iteration-1], tree.bal_test_err[iteration-1]))
    plt.legend(loc=1)
    plt.savefig('{0}{1}/plots/balanced_train_test_error__{1}.png'.format(config.OUTPUT_PATH, config.OUTPUT_PREFIX))

# Plot imbalanced training and testing error
def plot_imbalanced_error(tree, iteration):
    plt.figure(figsize=(12, 9))
    plt.plot(range(iteration), tree.imbal_train_err[0:iteration], color="red", label='Training')
    plt.plot(range(iteration), tree.imbal_test_err[0:iteration], color="blue", label='Testing')
    plt.xlabel('Iteration')
    plt.ylabel('Imbalanced Error')
    plt.title('Imbalanced Error - 2D Boosting \n {0} - {1} iterations \n Training error {2:.4f}, Testing error: {3:.4f}'.format(
        config.OUTPUT_PREFIX, iteration, tree.imbal_train_err[iteration-1], tree.imbal_test_err[iteration-1]))
    plt.legend(loc=1)
    plt.savefig('{0}{1}/plots/imbalanced_train_test_error__{1}.png'.format(config.OUTPUT_PATH, config.OUTPUT_PREFIX))

# Plot margin
def plot_margin(tree, iteration):
    plt.figure(figsize=(12, 9))
    plt.plot(range(iteration), tree.train_margins[0:iteration], color="red", label='Training')
    plt.plot(range(iteration), tree.test_margins[0:iteration], color="blue", label='Testing')
    plt.ylim((0,np.max(tree.train_margins[0:iteration])+1))
    plt.xlabel('Iteration')
    plt.ylabel('Margin')
    plt.title('Margin - 2D Boosting \n {0} - {1} iterations \n Training error {2:.4f}, Testing error: {3:.4f}'.format(
        config.OUTPUT_PREFIX, iteration, tree.imbal_train_err[iteration-1], tree.imbal_test_err[iteration-1]))
    plt.legend(loc=4)
    plt.savefig('{0}{1}/plots/margin_train_test__{1}.png'.format(config.OUTPUT_PATH, config.OUTPUT_PREFIX))
