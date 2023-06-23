#######################################################
# File: utils_ml.py
# Goal: Functions to extract FP, FN, TP and TN
# Author: Aurora Ramirez (Univ. of Cordoba, Spain)
#######################################################

import pandas as pd
import numpy as np

def get_fp(df_test, y_test, y_pred):
    '''
    This function returns a data frame with the false positives.
    :param df_test: Feature values of samples in the test partition as data frame
    :param y_test: Target value of samples in the test partition as array
    :param y_pred: Predictions for samples in the test partition as array
    :return A data frame with the test samples that were wrongly classified as positive.
    '''
    fp_samples = []
    for i in range(0, len(y_test)):
        if y_test.values[i] == 0 and y_pred[i] == 1:
            fp_samples.append(df_test.iloc[i])
    df_fp = pd.DataFrame(columns=df_test.columns, data=fp_samples)
    return df_fp

def get_fn(df_test, y_test, y_pred):
    '''
    This function returns a data frame with the false negatives.
    :param df_test: Feature values of samples in the test partition as data frame
    :param y_test: Target value of samples in the test partition as array
    :param y_pred: Predictions for samples in the test partition as array
    :return A data frame with the test samples that were wrongly classified as negative.
    '''
    fn_samples = []
    for i in range(0, len(y_test)):
        if y_test.values[i] == 1 and y_pred[i] == 0:
            fn_samples.append(df_test.iloc[i])
    df_fn = pd.DataFrame(columns=df_test.columns, data=fn_samples)
    return df_fn

def get_tp(df_test, y_test, y_pred):
    '''
    This function returns a data frame with the true positives.
    :param df_test: Feature values of samples in the test partition as data frame
    :param y_test: Target value of samples in the test partition as array
    :param y_pred: Predictions for samples in the test partition as array
    :return A data frame with the test samples that were correctly classified as positive.
    '''
    tp_samples = []
    for i in range(0, len(y_test)):
        if y_test.values[i] == 1 and y_pred[i] == 1:
            tp_samples.append(df_test.iloc[i])
    df_tp = pd.DataFrame(columns=df_test.columns, data=tp_samples)
    return df_tp

def get_tn(df_test, y_test, y_pred):
    '''
    This function returns a data frame with the true negatives.
    :param df_test: Feature values of samples in the test partition as data frame
    :param y_test: Target value of samples in the test partition as array
    :param y_pred: Predictions for samples in the test partition as array
    :return A data frame with the test samples that were correctly classified as negative.
    '''
    tn_samples = []
    for i in range(0, len(y_test)):
        if y_test.values[i] == 0 and y_pred[i] == 0:
            tn_samples.append(df_test.iloc[i])
    df_tn = pd.DataFrame(columns=df_test.columns, data=tn_samples)
    return df_tn


def save_confusion_matrix_text(cf, filename):
    """
    Save the performance metrics obtained from the confusion matrix
    :param cf: Confusion matrix
    :param filename: Name of the file to save the metrics
    """
    accuracy  = np.trace(cf) / float(np.sum(cf))
    precision = cf[1,1] / sum(cf[:,1])
    recall    = cf[1,1] / sum(cf[1,:])
    f1_score  = 2*precision*recall / (precision + recall)
    stats_text = "TP={:0.3f}\nFP={:0.3f}\nFN={:0.3f}\nTN={:0.3f}\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                cf[1,1], cf[0,1], cf[1,0], cf[0,0], accuracy,precision,recall,f1_score)

    try:
        f = open(filename, "w")
        f.write(stats_text)
        f.close()
    except IOError:
            print("problems writting confusion stats text")
