import pandas as pd
from sklearn.preprocessing import LabelEncoder
import datetime
import time
from config import const


def load_datasets(feats):
    dfs = [pd.read_feather(f'features/{f}_train.ftr') for f in feats]
    X_train = pd.concat(dfs, axis=1, sort=False)
    dfs = [pd.read_feather(f'features/{f}_test.ftr') for f in feats]
    X_test = pd.concat(dfs, axis=1, sort=False)
    return X_train, X_test


def load_target(target_name=None):
    if target_name is None:
        target_name = const.target_name
    le = LabelEncoder()
    train = pd.read_csv('./data/input/train.csv')
    train.drop(['row_id'], axis=1, inplace=True)
    train.drop_duplicates(keep='first', inplace=True)
    train.reset_index(drop=True, inplace=True)
    train[target_name] = le.fit_transform(train[target_name])

    return train[target_name]


def save_submission(y_pred, cross_val_acc, target_name, prefix=''):
    sample_submission = pd.read_csv('./data/input/sample_submission.csv')
    sample_submission[target_name] = y_pred

    le = LabelEncoder()
    train = pd.read_csv('./data/input/train.csv')
    le.fit(train[target_name])

    sample_submission[target_name] = le.inverse_transform(
        sample_submission[target_name])
    sample_submission.to_csv(
        './data/output/{}-{}-{}.csv'.format(int(time.time()), prefix, cross_val_acc), index=False)
