'''This module contains decision trees models'''

import numpy as np
import pandas as pd

from .utils import *
from ..common.utils import compute_metric


from ..optimizers.adam import AdamOptimizer


class DecisionTreeClassifier ():
    '''equivalent to sklearn with limited options:
    binary classification 0/1 only
    handles numeric data only
    criterion : gini only
    splitter  : best only
    max_depth : None only (nodes are expanded until
         all leaves are pure or until all leaves
         contain less than min_samples_split samples
    min_samples_split : integers only
         (fraction of n_samples not supported)
    min_samples_leaf : 1 only
         (only the min_samples_splits criterions
         stops the growth of the tree)
    min_weight_fraction_leaf : 0 only
    max_features :None only (all features)
    max_leaf_nodes : None only (unlimited)
    min_impurity_decrease : 0 only
    min_impurity_split : 0 only
    '''
    def __init__(self, min_samples_split):
        self.min_samples_split=min_samples_split
        self.tree={'level':0,
                   'name':'root',
                   'feature':None
                   'left':None,
                   'right':None,
                   }
