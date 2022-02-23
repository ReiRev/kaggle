from cv2 import reduce
import pandas as pd
import numpy as np
import re as re
from sklearn.preprocessing import StandardScaler
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


class Standardized(Feature):
    def create_features(self):
        scaler = StandardScaler()
        self.train = train.copy()
        self.test = test.copy()
        scaler.fit(self.train)
        self.train.iloc[:, :] = scaler.transform(self.train)
        # ここではfitをしてはいけない．
        self.test.iloc[:, :] = scaler.transform(self.test)


class Robust(Feature):
    def create_features(self):
        self.train = train.copy()
        self.test = test.copy()

        from sklearn.preprocessing import RobustScaler
        from scipy.stats import norm

        scaler = RobustScaler()
        scaler.fit(self.train)
        coefficient = norm.ppf(0.75)-norm.ppf(0.25)

        self.train.iloc[:, :] = scaler.transform(self.train)*coefficient
        self.test.iloc[:, :] = scaler.transform(self.test)*coefficient


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(
        100 * (start_mem - end_mem) / start_mem))

    return df


if __name__ == '__main__':
    args = get_arguments()
    train = pd.read_csv('./data/input/train.csv')
    test = pd.read_csv('./data/input/test.csv')
    train.drop(['row_id', 'target'], axis=1, inplace=True)
    test.drop('row_id', axis=1, inplace=True)

    print("{} columns are duplicated".format(train.duplicated().sum()))

    train.drop_duplicates(keep='first', inplace=True)
    train.reset_index(drop=True, inplace=True)

    train = reduce_mem_usage(train)
    test = reduce_mem_usage(test)

    generate_features(globals(), args.force)
