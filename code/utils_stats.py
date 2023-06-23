#############################################################
# File: utils_stats.py
# Goal: Functions to extract statistics of FP and FN
# Author: Aurora Ramirez (Univ. of Cordoba, Spain)
#############################################################

import numpy as np
import utils_ml as ml

def fp_stats_by_gender(df_test, y_test, y_pred, gender_feat):
    """
    This functions obtains the number and percentage of FP for each gender
    :param df_test: Data frame with the test partition (feature values)
    :param y_test: The true labels of the test partition
    :param y_pred: The predicted labels of the test partition
    :param gender_feat: The name of the gender attribute
    :return Number of FP in the female group, Percentage of FP in the female group, number of FP in the male group, percentage of FP in the male group
    """
    df_fp = ml.get_fp(df_test, y_test, y_pred)

    num_fp_fem = df_fp[df_fp[gender_feat]==0].shape[0]
    num_fem_test = df_test[df_test[gender_feat]==0].shape[0]
    perc_fp_fem = np.round(num_fp_fem / num_fem_test * 100, decimals=2)

    num_fp_male = df_fp[df_fp[gender_feat]==1].shape[0]
    num_male_test = df_test[df_test[gender_feat]==1].shape[0]
    perc_fp_male = np.round(num_fp_male / num_male_test * 100, decimals=2)

    return num_fp_fem, perc_fp_fem, num_fp_male, perc_fp_male

def fn_stats_by_gender(df_test, y_test, y_pred, gender_feat):
    """
    This functions obtains the number and percentage of FN for each gender
    :param df_test: Data frame with the test partition (feature values)
    :param y_test: The true labels of the test partition
    :param y_pred: The predicted labels of the test partition
    :param gender_feat: The name of the gender attribute
    :return Number of FN in the female group, percentage of FN in the female group, number of FN in the male group, percentage of FN in the male group
    """
    df_fn = ml.get_fn(df_test, y_test, y_pred)

    num_fn_fem = df_fn[df_fn[gender_feat]==0].shape[0]
    num_fem_test = df_test[df_test[gender_feat]==0].shape[0]
    perc_fn_fem = np.round(num_fn_fem / num_fem_test * 100, decimals=2)

    num_fn_male = df_fn[df_fn[gender_feat]==1].shape[0]
    num_male_test = df_test[df_test[gender_feat]==1].shape[0]
    perc_fn_male = np.round(num_fn_male / num_male_test * 100, decimals=2)

    return num_fn_fem, perc_fn_fem, num_fn_male, perc_fn_male