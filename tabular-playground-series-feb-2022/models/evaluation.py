import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score
from config import const


def cross_validation_score(model, X, y, metrics=accuracy_score, n_splits=5, random_state=0, stratified=True, shuffle=True):
    oof_train = cross_validation_predict(model, X, y)

    y_pred_oof = np.argmax(oof_train, axis=1)
    return accuracy_score(y, y_pred_oof)


def cross_validation_predict(model, X, y, X_test=None, metrics=accuracy_score, n_splits=5, random_state=0, stratified=True, shuffle=True):
    oof_train = np.zeros((len(y), const.n_class))
    y_preds = []
    X = X.copy()
    y = y.copy()
    if X_test is not None:
        X_test = X_test.copy()
    if stratified:
        kfold = StratifiedKFold(
            n_splits=n_splits, random_state=random_state, shuffle=shuffle)
        for fold_id, (train_index, valid_index) in enumerate(kfold.split(X, y)):
            X_train = X.iloc[train_index, :]
            X_valid = X.iloc[valid_index, :]
            y_train = y.iloc[train_index]
            y_valid = y.iloc[valid_index]

            model.fit(X_train, y_train)
            oof_train[valid_index] = model.predict(X_valid)

            if X_test is not None:
                y_preds.append(model.predict(X_test))

    else:
        kfold = KFold(n_splits, random_state=random_state, shuffle=shuffle)
        for fold_id, (train_index, valid_index) in enumerate(kfold.split(X)):
            X_train = X.iloc[train_index, :]
            X_valid = X.iloc[valid_index, :]
            y_train = y.iloc[train_index]
            y_valid = y.iloc[valid_index]

            model.fit(X_train, y_train)
            oof_train[valid_index] = model.predict(X_valid)

            if X_test is not None:
                y_preds.append(model.predict(X_test))

    if X_test is None:
        return oof_train
    else:
        return oof_train, sum(y_preds) / len(y_preds)


def stacking(models, X_train, y_train, X_test, n_round=10):
    X_train = X_train.copy()
    y_train = y_train.copy()
    X_test = X_test.copy()
    y_preds = []
    oofs = []
    for i in range(n_round):
        y_preds = []
        for j, model in enumerate(models):
            print("{}th {}".format(i, str(model)))
            columns = []
            for k in range(const.n_class):
                columns.append("{}-round-{}th-{}-{}th-class".format(i,
                                                                    j, str(model), k))
            oof_train, y_pred = cross_validation_predict(
                model, X_train, y_train, X_test)
            X_append_train = pd.DataFrame(oof_train, columns=columns)
            X_append_test = pd.DataFrame(y_pred, columns=columns)
            X_train = pd.concat([X_train, X_append_train], axis=1)
            X_test = pd.concat([X_test, X_append_test], axis=1)
            y_preds.append(y_pred)
            oofs.append(oof_train)

    return oofs, y_preds


def ensemble_predict(models, X_train, y_train, X_test=None):
    X_train = X_train.copy()
    y_train = y_train.copy()
    if X_test is not None:
        X_test = X_test.copy()
    pass
