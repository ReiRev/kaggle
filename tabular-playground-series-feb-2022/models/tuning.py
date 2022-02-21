import optuna
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
import numpy as np

# params = {'learning_rate': [0.001, 0.5]}などと指定する．


def beyesian_optimization(model, X, y, params, n_trials=100):
    X_train, X_valid, y_train, y_valid = \
        train_test_split(X, y, test_size=0.3,
                         random_state=0, stratify=y)

    def objective(trial):
        optuna_params = {}
        for k in params.keys():
            if type(params[k]) is list:
                if type(params[k][0]) is float:
                    optuna_params[k] = trial.suggest_float(
                        k, params[k][0], params[k][1])
                elif type(params[k][0]) is int:
                    optuna_params[k] = trial.suggest_int(
                        k, params[k][0], params[k][1])
                else:
                    raise TypeError("please specify float or int!")
            else:
                optuna_params[k] = params[k]

        #  元のmodelが変更されてしまうのでまずい？
        model.params = optuna_params
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)
        score = log_loss(y_valid, y_pred)
        return score

    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    study.optimize(objective, n_trials=n_trials)

    res = {}
    for k in params.keys():
        if k in study.best_params.keys():
            res[k] = study.best_params[k]
        else:
            res[k] = params[k]
    return res
