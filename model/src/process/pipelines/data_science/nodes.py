from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import explained_variance_score, mean_squared_error
from sklearn.model_selection import train_test_split


def _split_data(feature_table: pd.DataFrame, response: str) -> Tuple:
    feature_names = [i for i in list(feature_table) if response != i]
    X = feature_table[feature_names]
    y = feature_table[response]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
    return X_train, X_test, y_train, y_test


def gradientBoostingRegressor(X_train, X_test, y_train, y_test):
    params = {
        "n_estimators": 500,
        "max_depth": 4,
        "min_samples_split": 5,
        "learning_rate": 0.01
    }

    model = GradientBoostingRegressor(**params)
    sfm = RFE(model, n_features_to_select=4)
    sfm.fit(X_train, np.ravel(y_train))

    feature_list = list(X_train.columns[sfm.get_support(indices=True)])
    X_important_train = X_train[feature_list]
    X_important_test = X_test[feature_list]

    model.fit(X_important_train, y_train)
    y_pred = model.predict(X_important_test[feature_list])

    mse = mean_squared_error(y_test, model.predict(X_test[feature_list]))
    print("MSE: {:.4f}".format(mse))
    print("Explained variance: {:.3f}".format(explained_variance_score(y_test, y_pred)))
    return model, metrics
