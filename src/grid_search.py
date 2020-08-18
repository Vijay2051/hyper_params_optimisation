import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection, decomposition, pipeline
from sklearn import ensemble
from sklearn import metrics


if __name__ == "__main__":
    df = pd.read_csv("../input/train.csv")
    X = df.drop("price_range", axis=1).values
    y = df['price_range'].values

    scl = preprocessing.StandardScaler()
    pca = decomposition.PCA()
    rf_classifier = ensemble.RandomForestClassifier(n_jobs=-1)

    classifier = pipeline.Pipeline([
        ("scaling", scl),
        ("pca", pca),
        ("random_forest", rf_classifier)
    ])

    hyper_params_for_grid_search = {
        "n_estimators": [100, 200, 300, 400],
        "max_depth": [1, 3, 5, 7],
        "criterion": ["gini", "entropy"]
    }
    hyper_params_for_randomised_search = {
        "n_estimators": np.arange(100, 1500, 100),
        "max_depth": np.arange(1,20),
        "criterion": ["gini", "entropy"]
    }

    # model = model_selection.GridSearchCV(
    #     estimator=classifier, 
    #     param_grid=hyper_params_for_grid_search,
    #     n_jobs=1, 
    #     cv=5, 
    #     verbose=10, 
    #     scoring="accuracy"
    # )
    model = model_selection.RandomizedSearchCV(
        estimator = rf_classifier,
        param_distributions=hyper_params_for_randomised_search,
        n_iter=10,
        n_jobs=1,
        verbose=10,
        scoring="accuracy",
        cv=5
    )
    model.fit(X,y)
    print(model.best_score_)                    # gives the best score for the model
    print(model.best_estimator_.get_params())   # gives the best params    # the main params are criterion, n_esitamtors and max_depth

"""
    best params found out for the model.gridsearchcv 

{'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini', 'max_depth': 7, 'max_features': 'auto', 'max_leaf_nodes': None,
 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 2,
 'min_weight_fraction_leaf': 0.0, 'n_estimators': 200, 'n_jobs': -1, 'oob_score': False, 'random_state': None, 'verbose': 0, 'warm_start': False}

    best params for the randomised search cv

{'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'entropy', 'max_depth': 15, 'max_features': 'auto', 'max_leaf_nodes': None,
 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 2,
 'min_weight_fraction_leaf': 0.0, 'n_estimators': 900, 'n_jobs': -1, 'oob_score': False, 'random_state': None, 'verbose': 0, 'warm_start': False}

 """