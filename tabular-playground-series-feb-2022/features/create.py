import pandas as pd
import numpy as np
import re as re

from base import Feature, get_arguments, generate_features

Feature.dir = 'features'


# class Pclass(Feature):
#     def create_features(self):
#         self.train['Pclass'] = train['Pclass']
#         self.test['Pclass'] = test['Pclass']

class Native(Feature):
    def create_features(self):
        self.train = train.copy()
        self.test = test.copy()
        self.train.drop(['row_id', 'target'], axis=1, inplace=True)
        self.test.drop('row_id', axis=1, inplace=True)


if __name__ == '__main__':
    args = get_arguments()

    train = pd.read_feather('./data/input/train.feather')
    test = pd.read_feather('./data/input/test.feather')

    generate_features(globals(), args.force)
