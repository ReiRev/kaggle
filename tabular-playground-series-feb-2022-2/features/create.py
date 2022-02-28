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


class MultipliedATGC(Feature):
    def create_features(self):
        import re
        atgc_columns = [[], [], [], []]
        columns = train.columns
        for i, c in enumerate(columns):
            a = re.findall(r'\d+', c)
            for j, atgc in enumerate(["A", "T", "G", "C"]):
                if a[j] == 0:
                    continue
                self.train['column-{}-{}'.format(c, atgc)] = train[c]*int(a[j])
                self.test['column-{}-{}'.format(c, atgc)] = test[c]*int(a[j])
                atgc_columns[j].append('column-{}-{}'.format(c, atgc))

        for j, atgc in enumerate(["A", "T", "G", "C"]):
            self.train['{}-sum'.format(atgc)
                       ] = self.train[atgc_columns[j]].sum(axis=1)
            self.test['{}-sum'.format(atgc)
                      ] = self.test[atgc_columns[j]].sum(axis=1)

            self.train['{}-mean'.format(atgc)
                       ] = self.train[atgc_columns[j]].mean(axis=1)
            self.test['{}-mean'.format(atgc)
                      ] = self.test[atgc_columns[j]].mean(axis=1)

            self.train['{}-std'.format(atgc)
                       ] = self.train[atgc_columns[j]].std(axis=1)
            self.test['{}-std'.format(atgc)
                      ] = self.test[atgc_columns[j]].std(axis=1)

            self.train['{}-median'.format(atgc)
                       ] = self.train[atgc_columns[j]].median(axis=1)
            self.test['{}-median'.format(atgc)
                      ] = self.test[atgc_columns[j]].median(axis=1)

            self.train['{}-max'.format(atgc)
                       ] = self.train[atgc_columns[j]].max(axis=1)
            self.test['{}-max'.format(atgc)
                      ] = self.test[atgc_columns[j]].max(axis=1)

            self.train['{}-min'.format(atgc)
                       ] = self.train[atgc_columns[j]].min(axis=1)
            self.test['{}-min'.format(atgc)
                      ] = self.test[atgc_columns[j]].min(axis=1)

            self.train['{}-skew'.format(atgc)
                       ] = self.train[atgc_columns[j]].skew(axis=1)
            self.test['{}-skew'.format(atgc)
                      ] = self.test[atgc_columns[j]].skew(axis=1)

            for i, rate in enumerate([0.01, 0.05, 0.1, 0.25, 0.75, 0.9, 0.95, 0.99, 0.4, 0.6]):
                self.train['{}-quantile-{}'.format(atgc, rate)
                           ] = self.train[atgc_columns[j]].quantile(q=rate, axis=1)
                self.test['{}-quantile-{}'.format(atgc, rate)
                          ] = self.test[atgc_columns[j]].quantile(q=rate, axis=1)


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

    print("{} columns are duplicated".format(
        train.duplicated(keep='first').sum()))

    train.drop_duplicates(keep='first', inplace=True)
    train.reset_index(drop=True, inplace=True)

    # train = reduce_mem_usage(train)
    # test = reduce_mem_usage(test)

    generate_features(globals(), args.force)
