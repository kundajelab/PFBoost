### TEMPORARY Function to load data only 
### Peyton Greenside
### 6/18/15

import sys
import os
import random

import numpy as np 
from scipy.sparse import *

import argparse
import pandas as pd
import multiprocessing
import ctypes

from functools import partial
import time
from collections import namedtuple
import pdb
import pickle

from boosting_2D import config
from boosting_2D import util
from boosting_2D import plot
from boosting_2D import margin_score
from boosting_2D.data_class import *
from boosting_2D.find_rule import *
from boosting_2D import stabilize

y = TargetMatrix('/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/accessibilityMatrix_full_subset_CD34.txt', 
                 '/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/peak_headers_full_subset_CD34.txt', 
                 '/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/cell_types_pairwise.txt',
                 'triplet',
                 'sparse')

x1 = Motifs('/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/annotationMatrix_full_subset_CD34.txt', 
            '/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/peak_headers_full_subset_CD34.txt', 
            '/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/annotationMatrix_headers_full.txt',
            'triplet',
            'sparse')

x2 = Regulators('/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/regulatorExpression_full.txt', 
                '/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/regulator_names_full.txt', 
                '/srv/persistent/pgreens/projects/boosting/data/hematopoeisis_data/cell_types_pairwise.txt',
                'triplet',
                'sparse')

holdout = Holdout(y, 'sparse')
