### Encode hierarchy for use with boosting
### Peyton Greenside
### 3/20/17

import os
import sys
import numpy as np
import pandas as pd
from scipy.sparse import *

from boosting2D import util

### Return specific hierarchy encoding

class Hierarchy:
    def __init__(self, num_nodes, direct_children, subtree_nodes, name):
        self.num_nodes = num_nodes
        self.direct_children = direct_children
        self.subtree_nodes = subtree_nodes
        self.name = name

def get_hierarchy(name='hema_16cell'):

    if name == None:

        hierarchy = None

    elif name == 'hema_16cell':

        NUM_NODES = 16

        ### Define the direct children of each internal node

        direct_children = {}
        direct_children['root'] = [0]
        direct_children[0] = [0, 1, 7]
        direct_children[1] = [1, 2, 3]
        direct_children[2] = [2, 4, 5]
        direct_children[3] = [3, 5, 6]
        direct_children[4] = [4, 9, 10, 11, 12]
        direct_children[5] = [5, 13]
        direct_children[6] = [6, 14]
        direct_children[7] = [7, 8]
        direct_children[8] = [8, 15]
        direct_children[9] = [9]
        direct_children[10] = [10]
        direct_children[11] = [11]
        direct_children[12] = [12]
        direct_children[13] = [13]
        direct_children[14] = [14]
        direct_children[15] = [15]

        ### Define the participant nodes in each subtree

        subtree_nodes = {}
        subtree_nodes['root'] = range(NUM_NODES)
        subtree_nodes[0] = range(NUM_NODES)
        subtree_nodes[1] = [1, 2, 4, 9, 10, 11, 12, 3, 5, 6, 13, 14]
        subtree_nodes[2] = [2, 4, 9, 10, 11, 12, 5, 13]
        subtree_nodes[3] = [3, 5, 6, 13, 14]
        subtree_nodes[4] = [4, 9, 10, 11, 12]
        subtree_nodes[5] = [5, 13]
        subtree_nodes[6] = [6, 14]
        subtree_nodes[7] = [7, 8, 15]
        subtree_nodes[8] = [8, 15]
        subtree_nodes[9] = [9]
        subtree_nodes[10] = [10]
        subtree_nodes[11] = [11]
        subtree_nodes[12] = [12]
        subtree_nodes[13] = [13]
        subtree_nodes[14] = [14]
        subtree_nodes[15] = [15]

        hierarchy = Hierarchy(NUM_NODES, direct_children, subtree_nodes, name)


    elif name == 'hema_diff_wrt_HSC':

        NUM_NODES = 15

        ### Define the direct children of each internal node

        direct_children = {}
        direct_children['root'] = [0]
        direct_children[0] = [0, 1, 2]
        direct_children[1] = [1, 3, 4]
        direct_children[2] = [2, 4, 5]
        direct_children[3] = [3, 8, 9, 10, 11]
        direct_children[4] = [4, 12]
        direct_children[5] = [5, 13]
        direct_children[6] = [6, 7]
        direct_children[7] = [7, 14]
        direct_children[8] = [8]
        direct_children[9] = [9]
        direct_children[10] = [10]
        direct_children[11] = [11]
        direct_children[12] = [12]
        direct_children[13] = [13]
        direct_children[14] = [14]

        ### Define the participant nodes in each subtree

        subtree_nodes = {}
        subtree_nodes['root'] = range(NUM_NODES)
        subtree_nodes[0] = range(NUM_NODES)
        subtree_nodes[1] = [1, 3, 8, 9, 10, 11, 4, 12]
        subtree_nodes[2] = [2, 4, 12, 5, 13]
        subtree_nodes[3] = [3, 5, 6, 13, 14]
        subtree_nodes[4] = [4, 8, 9, 10, 11]
        subtree_nodes[5] = [5, 13]
        subtree_nodes[6] = [6, 7, 14]
        subtree_nodes[7] = [7, 14]
        subtree_nodes[8] = [8]
        subtree_nodes[9] = [9]
        subtree_nodes[10] = [10]
        subtree_nodes[11] = [11]
        subtree_nodes[12] = [12]
        subtree_nodes[13] = [13]
        subtree_nodes[14] = [14]

        hierarchy = Hierarchy(NUM_NODES, direct_children, subtree_nodes, name)

    else:
        raise ValueError('No existing hierarchy for provided name: %s'%name)


    return hierarchy


def get_hierarchy_index(hier_node, hierarchy, training_index, tree):
    """Mask training index with hierarchy node if there is a hierarchy
    """
    if hierarchy is None:
        return training_index

    cells = hierarchy.subtree_nodes[hier_node]
    cell_matrix = np.zeros(training_index.shape, dtype=bool)
    cell_matrix[:, cells] = True
    if tree.sparse:
        leaf_training_examples = util.element_mult(training_index, 
                                                   csr_matrix(cell_matrix))
    else:
        leaf_training_examples = util.element_mult(training_index,
                                                   cell_matrix)
    return leaf_training_examples





