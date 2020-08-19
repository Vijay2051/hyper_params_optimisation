"*** bayesian opimisation ***"

import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection, decomposition, pipeline
from sklearn import ensemble
from sklearn import metrics
from functools import partial
from skopt import space
from skopt import gp_minimize

def optimise(params, param_names, x, y):
    params = dict(zip(param_names, params))
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
    df =pd.read_csv("../input/train.csv")
    X = df.drop("price_range", axis=1).values
    y = df['price_range'].values

    param_space = [
        space.Integer(3, 15, name="max_depth"),
        space.Integer(100, 600, name="n_estimators"),
        space.Categorical(["gini", "entropy"], name="criterion"),
        space.Real(0.01, 1, prior="uniform", name="max_features")
    ]

    param_names = [
        "max_depth",
        "n_estimators",
        "criterion",
        "max_features"
    ]

    optimisation_function = partial(
        optimise,
        param_names = param_names,
        x = X,
        y = y
    )

    result = gp_minimize(optimisation_function, dimensions=param_space, n_random_starts=10, n_calls=15, verbose=10)

    print(
        dict(zip(
            param_names, result.x
        ))
    )
    



"""
 *** some resullts ***

Time taken: 10.7025
Function value obtained: -0.9055
Current minimum: -0.9110
Iteration No: 13 started. Searching for the next optimal point.
Iteration No: 13 ended. Search finished for the next optimal point.
Time taken: 2.2517
Function value obtained: -0.7575
Current minimum: -0.9110
Iteration No: 14 started. Searching for the next optimal point.
Iteration No: 14 ended. Search finished for the next optimal point.
Time taken: 3.9038
Function value obtained: -0.9055
Current minimum: -0.9110
Iteration No: 15 started. Searching for the next optimal point.
Iteration No: 15 ended. Search finished for the next optimal point.
Time taken: 1.5560
Function value obtained: -0.7120
Current minimum: -0.9110
{'max_depth': 14, 'n_estimators': 122, 'criterion': 'entropy', 'max_features': 0.45311881546933286}

"""
