### Encode hierarchy for use with boosting
### Peyton Greenside
### 3/20/17

import os
import sys
import numpy as numpy
import pandas as pd

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

    if name == 'hema_16cell':

        NUM_NODES = 16
        MAX_CHILDREN = 6
        NUM_INT_NODES = 9

        ### Define the direct children of each internal node

        direct_children = pd.DataFrame(index=range(NUM_INT_NODES), 
                                       columns=range(MAX_CHILDREN))
        direct_children.ix[0, 0:1] = [1, 7]
        direct_children.ix[1, 0:1] = [2, 3]
        direct_children.ix[2, 0:1] = [4, 5]
        direct_children.ix[3, 0:1] = [5, 6]
        direct_children.ix[4, 0:3] = [9, 10, 11, 12]
        direct_children.ix[5, 0] = 13
        direct_children.ix[6, 0] = 14
        direct_children.ix[7, 0] = 8
        direct_children.ix[8, 0] = 15

        ### Define the participant nodes in each subtree

        subtree_nodes = pd.DataFrame(index=range(NUM_NODES),
                                     columns=range(NUM_NODES))
        subtree_nodes.ix[0, :] = range(NUM_NODES)
        subtree_nodes.ix[1, 0:11] = [1, 2, 4, 9, 10, 11, 12, 3, 5, 6, 13, 14]
        subtree_nodes.ix[2, 0:5] = [2, 4, 9, 10, 11, 12]
        subtree_nodes.ix[3, 0:4] = [3, 5, 6, 13, 14]
        subtree_nodes.ix[4, 0:4] = [4, 9, 10, 11, 12]
        subtree_nodes.ix[5, 0:1] = [5, 13]
        subtree_nodes.ix[6, 0:1] = [6, 14]
        subtree_nodes.ix[7, 0:2] = [7, 8, 15]
        subtree_nodes.ix[8, 0:1] = [8, 15]
        subtree_nodes.ix[9, 0] = 9
        subtree_nodes.ix[10, 0] = 10
        subtree_nodes.ix[11, 0] = 11
        subtree_nodes.ix[12, 0] = 12
        subtree_nodes.ix[13, 0] = 13
        subtree_nodes.ix[14, 0] = 14
        subtree_nodes.ix[15, 0] = 15

    else:
        raise ValueError('No existing hierarchy for provided name: %s'%name)

    hierarchy = Hierarchy(NUM_NODES, MAX_CHILDREN, NUM_INT_NODES,
                          direct_children, subtree_nodes)

    return hierarchy






