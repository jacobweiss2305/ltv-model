import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn import preprocessing
from statsmodels.stats.outliers_influence import variance_inflation_factor

logger = logging.getLogger(__name__)


def target_variable(transactions: pd.DataFrame) -> pd.DataFrame:
    """Calculate the rolling 6 month LTV

    Args:
        transactions (pd.DataFrame): customer transactions

    Returns:
        pd.DataFrame: customer transactions with rolling 6 month LTV
    """
    transactions['rolling_6_months'] = (
        (transactions['transaction_days_after_joining']-1) / 180).astype(int)
    transactions['rolling_6_month_ltv'] = transactions.groupby(
        ['customer_id', 'rolling_6_months'])['transaction_value'].cumsum()
    return transactions.groupby(['customer_id', 'rolling_6_months'])['rolling_6_month_ltv'].last().reset_index().set_index('customer_id')


def continuous_variables(df: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
    """Scale continuous variables using scikit learn Standard Scaler with mean imputation

    Args:
        df (pd.DataFrame): raw feature table
        parameters ([type], optional): dictionary of feature and type mapping.

    Returns:
        Tuple: scaled dense feature table and scalar mapping
    """
    values = [i for i, j in parameters["feature_engineering_map"].items() if j ==
              "continuous" and i in list(df)]
    x = df[values]
    scaler = preprocessing.StandardScaler()
    scaler.fit(x)
    # Standard Scalar sets mean to 0. We can apply 0 mean imputation for NaN values.
    return pd.DataFrame(scaler.transform(x), columns=values).fillna(0)


def categorical_variables(df: pd.DataFrame, parameters: Dict) -> Tuple:
    """Convert categorical variables to continuous variables using scikit learn label encoder. Also provide label encode mapping as seperate mapping.

    Args:
        df (pd.DataFrame): dense raw features
        parameters ([type], optional): dictionary of feature and type mapping. Defaults to None.

    Returns:
        Tuple: categorical features and label mapping
    """
    param = "label"
    values = [i for i, j in parameters["feature_engineering_map"].items() if j ==
              param and i in list(df)]
    le = preprocessing.LabelEncoder()
    collect = []
    for column in values:
        df[column] = df[column].replace('', "None").fillna("None").astype(str)
        le.fit(df[column])
        df[column + f'_{param}'] = le.transform(df[column])
        temp = pd.concat([pd.DataFrame(le.classes_, columns=['input']),
                          pd.DataFrame(le.transform(le.classes_), columns=['output'])], axis=1)
        temp['attribute'] = column
        collect.append(temp)
    final = df[[i for i in list(df) if f'_{param}' in i]]
    return final, pd.concat(collect)


def _calculate_vif(feature_table: pd.DataFrame, thresh=5.0) -> pd.DataFrame:
    """Remove multicollinearity in feature table using vif with a threshold of 5.0 (leverages parallel runs).

    Args:
        feature_table (pd.DataFrame): standardized feature table
        thresh (float, optional): vif threshold. Defaults to 5.0.

    Returns:
        pd.DataFrame: standardized feature table with low multicollinearity
    """
    X = feature_table.dropna(axis=1)
    variables = [X.columns[i] for i in range(X.shape[1])]
    dropped = True
    while dropped:
        dropped = False
        print(len(variables))
        vif = Parallel(n_jobs=-1, verbose=5)(delayed(variance_inflation_factor)
                                             (X[variables].values, ix) for ix in range(len(variables)))
        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print(time.ctime() + ' dropping \'' +
                  X[variables].columns[maxloc] + '\' at index: ' + str(maxloc))
            variables.pop(maxloc)
            dropped = True
    print('Remaining variables:')
    print([variables])
    return X[[i for i in variables]]


def standardized_feature_table(response: pd.DataFrame, cont_features: pd.DataFrame, cat_features: pd.DataFrame) -> pd.DataFrame:
    """Concatenates features and response variable into trainable dataset. Removes Multicollinearity between categorical and continuous features.

    Args:
        response (pd.DataFrame): Rolling 6 month LTV
        cont_features (pd.DataFrame): Standard Scaled continuous features
        cat_features (pd.DataFrame): Label encoded categorical features

    Returns:
        pd.DataFrame: standardized feature table
    """
    vif = _calculate_vif(pd.concat([cat_features, cont_features], axis=1))
    return pd.concat([response, vif], axis=1)
