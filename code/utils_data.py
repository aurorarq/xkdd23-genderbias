#######################################################
# File: utils_data.py
# Goal: Functions to split datasets
# Author: Aurora Ramirez (Univ. of Cordoba, Spain)
#######################################################

from sklearn.model_selection import train_test_split

def split_train_test_full(data, target_feat):
    """
    Stratified split in train/test of the full dataset
    :param data: The original data frame
    :param target_feat: The name of the target variable
    :return Data split in x_train, x_test, y_train, y_test
    """
    # Train and test partition for the full date frame, stratified by target attribute
    target_values = data[target_feat]
    df_train, df_test, y_train, y_test = train_test_split(data,
                                                                target_values,
                                                                test_size=0.3,
                                                                random_state=0,
                                                                stratify=target_values)
    x_train = df_train.drop(target_feat, axis=1)
    x_test = df_test.drop(target_feat, axis=1)
    return x_train, x_test, y_train, y_test


def split_train_test_nogender(data, target_feat, gender_feat):
    """
    Stratified split in train/test of the the dataset without the gender attribute
    :param data: The original data frame
    :param target_feat: The name of the target variable
    :param gender_feat: The name of the gender attribute
    :return Data split in x_train, x_test, y_train, y_test and training data as data frame (df_train, df_test)
    """
    # Train and test partition for the date frame without gender feature, stratified by target attribute
    target_values = data[target_feat]
    df_train, df_test, y_train, y_test = train_test_split(data,
                                                                target_values,
                                                                test_size=0.3,
                                                                random_state=0,
                                                                stratify=target_values)
    
    x_train = df_train.drop(columns=[target_feat, gender_feat], axis=1)
    x_test = df_test.drop(columns=[target_feat, gender_feat], axis=1)
    return x_train, x_test, y_train, y_test, df_train[gender_feat], df_test[gender_feat]


def split_train_test_gender(data, target_feat, gender_feat, gender_value):
    """
    Stratified split in train/test of the dataset for a gender group
    :param data: The original data frame
    :param target_feat: The name of the target variable
    :param gender_feat: The name of the gender attribute
    :param gender_value: The value of the gender attribute to identify the gender group
    :return Data split in x_train, x_test, y_train, y_test
    """
    # Train and test partition for the female data frame, stratified by target attribute
    df_gender = data[data[gender_feat]==gender_value]
    target_values = df_gender[target_feat]
    df_train, df_test, y_train, y_test = train_test_split(df_gender,
                                                                target_values,
                                                                test_size=0.30,
                                                                random_state=0,
                                                                stratify=target_values)
    x_train = df_train.drop([target_feat, gender_feat], axis=1)
    x_test = df_test.drop([target_feat, gender_feat], axis=1)
    return x_train, x_test, y_train, y_test
