# source: https://amalog.hateblo.jp/entry/kaggle-feature-management
# コメントはReiRevによる

import argparse
import inspect
import re
import time
from abc import ABCMeta, abstractmethod
from pathlib import Path
from contextlib import contextmanager
import pandas as pd

# contextmanagerはwith文と共に用いられるクラスに使用される．
# with文を使うときには，__enter__, __exit__を使うことで，始めと終わりの処理を記述できる．
# @contextlib.contextmanagerを用いると，これを簡潔に実装できるようになる．
# yieldの時点でwithブロックの中の実行が開始し，withを抜けるときにyiled以降が実行される．


@contextmanager
def timer(name):
    t0 = time.time()
    print(f'[{name}] start')
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')

# ABCMetaはよくわからず．調べる．


class Feature(metaclass=ABCMeta):
    prefix = ''
    suffix = ''
    dir = '.'

    def __init__(self):
        self.name = self.__class__.__name__
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        # pathlib.Pathは/でパスを繋げていけるというカッコイイテクがある．
        self.train_path = Path(self.dir) / f'{self.name}_train.ftr'
        self.test_path = Path(self.dir) / f'{self.name}_test.ftr'

    def run(self):
        with timer(self.name):
            self.create_features()
            prefix = self.prefix + '_' if self.prefix else ''
            suffix = '_' + self.suffix if self.suffix else ''
            self.train.columns = prefix + self.train.columns + suffix
            self.test.columns = prefix + self.test.columns + suffix
        return self

    # 抽象化したいメソッドには@abstractmethodをつける．
    # 継承したクラスで実行されていないとエラーが起きる．
    # 中身はおそらくpassでも問題ない．
    @abstractmethod
    def create_features(self):
        raise NotImplementedError

    def save(self):
        self.train.to_feather(str(self.train_path))
        self.test.to_feather(str(self.test_path))


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', '-f', action='store_true',
                        help='Overwrite existing files')
    return parser.parse_args()


def get_features(namespace):
    for k, v in namespace.items():
        if inspect.isclass(v) and issubclass(v, Feature) and not inspect.isabstract(v):
            yield v()


def generate_features(namespace, overwrite):
    for f in get_features(namespace):
        if f.train_path.exists() and f.test_path.exists() and not overwrite:
            print(f.name, 'was skipped')
        else:
            f.run().save()
