" *** Using hyperopt for hyperparam tuning *** "

import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection, decomposition, pipeline
from sklearn import ensemble
from sklearn import metrics
from skopt import space
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.base import scope
from functools import partial


def optimise(params, x, y):
    model = ensemble.RandomForestClassifier(**params)
    kf = model_selection.StratifiedKFold(n_splits=5)
    accuracies = []
    for idx in kf.split(X=x, y=y):
        train_idx = idx[0]
        test_idx = idx[1]

        X_train = x[train_idx]
        y_train = y[train_idx]

        X_test = x[test_idx]
        y_test = y[test_idx]

        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        fold_acc = metrics.accuracy_score(y_test, pred)

        accuracies.append(fold_acc)

    return -1.0 * np.mean(accuracies)


if __name__ == "__main__":
    df = pd.read_csv("../input/train.csv")
    X = df.drop("price_range", axis=1).values
    y = df['price_range'].values

    param_space = {
        "max_depth": scope.int(hp.quniform("max_dpeth", 3, 15, 1)),
        "n_estimators": scope.int(hp.quniform("n_estimators", 100, 600, 1)),
        "criterion": hp.choice("criterion", ["gini", "entropy"]),
        "max_features": hp.uniform("max_features", 0.1, 1)
    }

    optimisation_function = partial(optimise, x=X, y=y)

    trial = Trials()

    result = fmin(
        fn=optimisation_function,
        space=param_space,
        algo=tpe.suggest,
        max_evals=15,
        trials=trial,
    )

    print(result)


""" 
    *** best result got for hyperopt ***

    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [02:05<00:00,  8.35s/trial, best loss: -0.908]
{'criterion': 1, 'max_dpeth': 12.0, 'max_features': 0.726868604092706, 'n_estimators': 305.0}

"""