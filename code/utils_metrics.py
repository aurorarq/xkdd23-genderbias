#####################################################################
# File: utils_metrics.py
# Goal: Functions to calculate performance and fairness metrics
# Author: Aurora Ramirez (Univ. of Cordoba, Spain)
#####################################################################

import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame
from fairlearn.metrics import (
    count,
    selection_rate,
    equalized_odds_difference,
    false_positive_rate,
    false_negative_rate,
)
from fairlearn.reductions import EqualizedOdds


def evaluate_fairness_metrics(y_test, y_pred, df_test, gender_feat):
    """
    Calculate fairness metrics
    :param y_test: Labels in the test partition
    :param y_pred: Predicted labels
    :param df_test: Feature values in the test partition
    :param gender_feat: Name of the gender attribute
    :return Fairness metrics computed by group, difference and overall, and equalized odds difference
    """
    fairness_metrics = {
        "count": count,
        "selection_rate": selection_rate,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
    }

    metricframe = MetricFrame(
    metrics=fairness_metrics,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=df_test[gender_feat]
    )

    eod = equalized_odds_difference(y_true=y_test, y_pred=y_pred, sensitive_features=df_test[gender_feat])

    return metricframe.by_group, metricframe.difference(),  metricframe.overall, eod


def get_df_eof_and_best_predictor(predictors, x_test, y_test, df_test, gender_feat):
    """
    Find the best predictor based on the equalized odds difference fairness metric
    :param predictors: a set of trained models
    :param x_test: feature values of the test partition
    :param y_test: true labels of the test partition
    :param df_test: test partition as data frame
    :param gender_feat: Name of the gender attribute
    :return Dataframe with the equalized odds difference of each model and the best predictor
    """
    eod_values = []
    predictor_idx = range(0, len(predictors))
    best_eod = np.inf
    best_predictor = None
    for p in predictors:
        y_pred = p.predict(x_test)
        eod = equalized_odds_difference(y_true=y_test, y_pred=y_pred, sensitive_features=df_test[gender_feat])
        eod_values.append(eod)
        if eod < best_eod:
            best_predictor = p
            best_eod = eod
    df_eof = pd.DataFrame(data={'predictor':predictor_idx, 'eod':eod_values})
    return df_eof, best_predictor