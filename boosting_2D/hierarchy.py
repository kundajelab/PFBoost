### Encode hierarchy for use with boosting
### Peyton Greenside
### 3/20/17

import os
import sys
import numpy as np
import pandas as pd
from scipy.sparse import *

from boosting_2D import util

### Return specific hierarchy encoding

class Hierarchy:
    def __init__(self, num_nodes, max_children, num_internal_nodes,
                 direct_children, subtree_nodes):
        self.num_nodes = num_nodes
        self.max_children = max_children
        self.num_internal_nodes = num_internal_nodes
        self.direct_children = direct_children
        self.subtree_nodes = subtree_nodes

def get_hierarchy(name='hema_16cell'):

    if name == None:

        hierarchy = None

    elif name == 'hema_16cell':

        NUM_NODES = 16
        MAX_CHILDREN = 6
        NUM_INT_NODES = 9

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
        subtree_nodes[2] = [2, 4, 9, 10, 11, 12]
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

    elif name == 'hema_diff_wrt_HSC':

        NUM_NODES = 15
        MAX_CHILDREN = 6
        NUM_INT_NODES = 9

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
        subtree_nodes[4] = [3, 8, 9, 10, 11]
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

    else:
        raise ValueError('No existing hierarchy for provided name: %s'%name)

    hierarchy = Hierarchy(NUM_NODES, MAX_CHILDREN, NUM_INT_NODES,
                          direct_children, subtree_nodes)

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



# def get_hierarchy(name='hema_16cell'):

#     if name == 'hema_16cell':

#         NUM_NODES = 16
#         MAX_CHILDREN = 6
#         NUM_INT_NODES = 9

#         ### Define the direct children of each internal node

#         direct_children = pd.DataFrame(index=range(NUM_INT_NODES), 
#                                        columns=range(MAX_CHILDREN))
#         direct_children.ix[0, 0:1] = [1, 7]
#         direct_children.ix[1, 0:1] = [2, 3]
#         direct_children.ix[2, 0:1] = [4, 5]
#         direct_children.ix[3, 0:1] = [5, 6]
#         direct_children.ix[4, 0:3] = [9, 10, 11, 12]
#         direct_children.ix[5, 0] = 13
#         direct_children.ix[6, 0] = 14
#         direct_children.ix[7, 0] = 8
#         direct_children.ix[8, 0] = 15

#         ### Define the participant nodes in each subtree

#         subtree_nodes = pd.DataFrame(index=range(NUM_NODES),
#                                      columns=range(NUM_NODES))
#         subtree_nodes.ix[0, :] = range(NUM_NODES)
#         subtree_nodes.ix[1, 0:11] = [1, 2, 4, 9, 10, 11, 12, 3, 5, 6, 13, 14]
#         subtree_nodes.ix[2, 0:5] = [2, 4, 9, 10, 11, 12]
#         subtree_nodes.ix[3, 0:4] = [3, 5, 6, 13, 14]
#         subtree_nodes.ix[4, 0:4] = [4, 9, 10, 11, 12]
#         subtree_nodes.ix[5, 0:1] = [5, 13]
#         subtree_nodes.ix[6, 0:1] = [6, 14]
#         subtree_nodes.ix[7, 0:2] = [7, 8, 15]
#         subtree_nodes.ix[8, 0:1] = [8, 15]
#         subtree_nodes.ix[9, 0] = 9
#         subtree_nodes.ix[10, 0] = 10
#         subtree_nodes.ix[11, 0] = 11
#         subtree_nodes.ix[12, 0] = 12
#         subtree_nodes.ix[13, 0] = 13
#         subtree_nodes.ix[14, 0] = 14
#         subtree_nodes.ix[15, 0] = 15

#     else:
#         raise ValueError('No existing hierarchy for provided name: %s'%name)

#     hierarchy = Hierarchy(NUM_NODES, MAX_CHILDREN, NUM_INT_NODES,
#                           direct_children, subtree_nodes)

#     return hierarchy






