from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import explained_variance_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeRegressor, plot_tree


def _split_data(feature_table: pd.DataFrame, response: str, test_size: float) -> Tuple:
    """Split data into train and test set using scikit learn package.

    Args:
        feature_table (pd.DataFrame): standardized feature table
        response (str): 6 month rolling ltv
        test_size (float): Percentage of test size

    Returns:
        Tuple: X_train, X_test, y_train, y_test
    """    
    feature_names = [i for i in list(feature_table) if response != i]
    X = feature_table[feature_names]
    y = feature_table[response]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42
    )
    return X_train, X_test, y_train, y_test


def gradientBoostingRegressor(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame, n_feat: int) -> Any:
    """Gradient boosting regression model with recursive feature elimination.

    Model Parameters:
        n_estimators: 500
        max_depth: 4
        min_samples_split: 5
        learning_rate: 0.01

    Args:
        X_train (pd.DataFrame): Feature training set
        X_test (pd.DataFrame): Feature test set
        y_train (pd.DataFrame): Response training set
        y_test (pd.DataFrame): Response testing set
        n_feat (int): Top N features to select

    Returns:
        GradientBoostingRegressor: scikit learn object
        Dict: Model Evaluation Metrics
    """    
    params = {
        "n_estimators": 500,
        "max_depth": 4,
        "min_samples_split": 5,
        "learning_rate": 0.01
    }
    #TODO Add HistGradientBoostingRegressor
    model = GradientBoostingRegressor(**params)
    sfm = RFE(model, n_features_to_select=4)
    sfm.fit(X_train, np.ravel(y_train))

    feature_list = list(X_train.columns[sfm.get_support(indices=True)])
    X_important_train = X_train[feature_list]
    X_important_test = X_test[feature_list]

    model.fit(X_important_train, y_train)
    y_pred = model.predict(X_important_test[feature_list])

    mse = mean_squared_error(y_test, model.predict(X_test[feature_list]))
    var = explained_variance_score(y_test, y_pred)
    print("MSE: {:.4f}".format(mse))
    print("Explained variance: {:.3f}".format(var))

    metrics = {"model": "Gradient Boost", "metrics": {"mse": mse, "explained variance": var}}

    test_score = np.zeros((params["n_estimators"],), dtype=np.float64)
    for i, y_pred in enumerate(model.staged_predict(X_test[feature_list])):
        test_score[i] = model.loss_(y_test, y_pred)

    fig = plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1)
    plt.title("Deviance")
    plt.plot(
        np.arange(params["n_estimators"]) + 1,
        model.train_score_,
        "b-",
        label="Training Set Deviance",
    )
    plt.plot(
        np.arange(params["n_estimators"]) + 1, test_score, "r-", label="Test Set Deviance"
    )
    plt.legend(loc="upper right")
    plt.xlabel("Boosting Iterations")
    plt.ylabel("Deviance")
    fig.tight_layout()
    plt.show()


    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + 0.5
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.barh(pos, feature_importance[sorted_idx], align="center")
    plt.yticks(pos, np.array(feature_list)[sorted_idx])
    plt.title("Feature Importance (MDI)")

    result = permutation_importance(
        model, X_test[feature_list], y_test, n_repeats=10, random_state=42, n_jobs=2
    )
    sorted_idx = result.importances_mean.argsort()
    plt.subplot(1, 2, 2)
    plt.boxplot(
        result.importances[sorted_idx].T,
        vert=False,
        labels=np.array(feature_list)[sorted_idx],
    )
    plt.title("Permutation Importance (test set)")
    fig.tight_layout()
    plt.show()

    return model, metrics

def DecisionTree(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> Any:
    """Scikit Learn Decision Tree Regression Model.

    Args:
        X_train (pd.DataFrame): Feature training set
        X_test (pd.DataFrame): Feature test set
        y_train (pd.DataFrame): Response training set
        y_test (pd.DataFrame): Response testing set

    Returns:
        DecisionTreeRegressor: scikit learn object
        Dict: Model Evaluation Metrics
    """    
    params = {
        "max_depth": 4,
        "random_state": 42,
        "max_leaf_nodes": 4
    }
    model = DecisionTreeRegressor(**params)
    model.fit(X_train, y_train)

    print('Base model with all features ...')
    plot_tree(model,feature_names = list(X_train))

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, model.predict(X_test))
    var = explained_variance_score(y_test, y_pred)
    print("MSE: {:.4f}".format(mse))
    print("Explained variance: {:.3f}".format(var))
    metrics = {"model": "Gradient Boost", "metrics": {"mse": mse, "explained variance": var}}
    return model, metrics
