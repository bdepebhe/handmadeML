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
                   'type':'root',
                   'feature':None,
                   'threshold':None,
                   'left':None,
                   'right':None,
                   }

    def fit(self, X_train, y_train):
        def node_construction(subtree,subX,suby):
            'modify tree ? to be tested'
            splits={}
            for index,col in subX.iteritems():
                splits[index]=np.sort(col.sample(1000,replace=True).unique())
                splits[index]=list((splits[index][1:]+splits[index][:-1])/2)
            splits=[[(key,threshold) for threshold in value] for key,value in splits.items()]
            splits=[y for x in splits for y in x]

            for split in splits:
                mask=subX[split[0]]>split[1]
                y_left=suby[mask]
                y_right=suby[-mask]

        node_construction(self.tree, X_train, y_train, min_samples_split=self.min_samples_split)

    def predict(self, X_test):
        def node_navigation(subtree, X_sample):
            '''X_sample : df with a single line'''
            if tree['type']=='leaf':
                return subtree['category']
            if X_sample[subtree['feature']]>subtree['threshold']:
                return node_navigation(subtree['right'], X_sample)
            return node_navigation(subtree['left'], X_sample)

        return X.apply(lambda x: node_navigation(self.tree, x), axis=1)
