##########################################################################################################################
# Script for the paper "Exploring gender bias in misclassification with clustering and explainable methods". 
# The script runs the experiments to address the two research questions of the paper:
# - RQ1: Analysis of misclassification
# - RQ2: Analysis of local explanations
# This script uses the adult dataset and the "mit_in" classification strategy described in the paper.
# Author: Aurora Ramírez (University of Córdoba)
##########################################################################################################################

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from fairlearn.reductions import ExponentiatedGradient
from fairlearn.reductions import EqualizedOdds
from sklearn.metrics import confusion_matrix
import os

import utils_data as data
import utils_metrics as metrics
import utils_ml as ml
import utils_stats as stats
import clustering_errors as clue
import clustering_stats as clus
import local_explanations as local

import warnings
warnings.simplefilter("ignore")

results_path = '../results/metrics'
dataset_name = 'adult'
folder = results_path + '/' + dataset_name
if not os.path.exists(folder):
    os.makedirs(folder)


# ## Preprocessing data
# ### Load and clean data
df = pd.read_csv('../data/adult.csv')

# Set features of interest
target_feat = 'income'
gender_feat = 'sex'

# ### Convert to numeric data
df['sex']=df['sex'].replace('Female', 0)
df['sex']=df['sex'].replace('Male', 1)

# For other features with more than 2 categories, use one-hot-encoding
col_names = ["workclass","marital-status","occupation","relationship","race","native-country"]
dummies = pd.get_dummies(df[col_names], dtype='int')
df.drop(columns=col_names, inplace=True)
df = df.join(dummies)

# Distribution by gender
num_females = df[df[gender_feat]==0].shape[0]
num_males = df[df[gender_feat]==1].shape[0]
total = df.shape[0]

print(f"Female: {num_females}, Perc: {num_females/total*100}%")
print(f"Male: {num_males}, Perc: {num_males/total*100}%")

# Split in train/test

# Train and test partition for the full date frame, stratified by target attribute
x_train_all, x_test_all, y_train_all, y_test_all = data.split_train_test_full(df, target_feat)

# Train and test partition for the full date frame without gender
x_train_nogender, x_test_nogender, y_train_nogender, y_test_nogender, x_train_gender, x_test_gender = data.split_train_test_nogender(df, target_feat, gender_feat)

# Train and test partition for the female data frame, stratified by target attribute
x_train_fem, x_test_fem, y_train_fem, y_test_fem = data.split_train_test_gender(df, target_feat, gender_feat, 0)

# Train and test partition for the male date frame, stratified by target attribute
x_train_male, x_test_male, y_train_male, y_test_male = data.split_train_test_gender(df, target_feat, gender_feat, 1)

# Building classifiers

# Create a copy as data frame
df_train_all = pd.DataFrame(x_train_all)
df_test_all = pd.DataFrame(x_test_all)

# Configure the method
# For epsilon, the 1/sqrt(samples) value is recommended
threshold = np.round(1/np.sqrt(df.shape[0]), 3)
exp_grad_est = ExponentiatedGradient(
        estimator=RandomForestClassifier(random_state=0),
        constraints=EqualizedOdds(difference_bound=threshold),
)

# Fit and get the inner predictors
exp_grad_est.fit(x_train_all, y_train_all, sensitive_features=df_train_all[gender_feat])
predictors = exp_grad_est.predictors_
print(f"Number of predictors: {len(predictors)}")

# Get a summary of equalized odds difference for each predictor and best one
df_eof, best_rf = metrics.get_df_eof_and_best_predictor(predictors, x_test_all, y_test_all, df_test_all, gender_feat)
df_eof.to_csv("../results/metrics/adult/rf_mit_in_eof.csv", index=False)

# Performance metrics
y_pred_rf_mit_in = best_rf.predict(x_test_all)
cf_matrix = confusion_matrix(y_test_all, y_pred_rf_mit_in)
ml.save_confusion_matrix_text(cf_matrix, filename="../results/metrics/adult/rf_mit_in_metrics.txt")

# Fairness metrics
fmetric_by_group, fmetric_dif, fmetric_overall, eod = metrics.evaluate_fairness_metrics(y_test_all, y_pred_rf_mit_in, df_test_all, gender_feat)

print('== Metrics by gender ==')
print(fmetric_by_group)

print('== Metric differences ==')
print(fmetric_dif)

print('== Overall metrics ==')
print(fmetric_overall)

print(f"Equalized odds difference: {eod}")

# FP stats by gender
num_fp_fem, perc_fp_fem, num_fp_male, perc_fp_male = stats.fp_stats_by_gender(df_test_all, y_test_all, y_pred_rf_mit_in, gender_feat)
print(f"FP (female): {num_fp_fem}, percentage (all female in test): {perc_fp_fem}")
print(f"FP (male): {num_fp_male}, percentage (all male in test): {perc_fp_male}")

# FN stats by gender
num_fn_fem, perc_fn_fem, num_fn_male, perc_fn_male = stats.fn_stats_by_gender(df_test_all, y_test_all, y_pred_rf_mit_in, gender_feat)
print(f"FN (female): {num_fn_fem}, percentage (all female in test): {perc_fn_fem}")
print(f"FN (male): {num_fn_male}, percentage (all male in test): {perc_fn_male}")

# Configure the method
# For epsilon, the 1/sqrt(samples) value is recommended
exp_grad_est = ExponentiatedGradient(
        estimator=GradientBoostingClassifier(random_state=0),
        constraints=EqualizedOdds(difference_bound=threshold),
)

# Fit and get the inner predictors
exp_grad_est.fit(x_train_all, y_train_all, sensitive_features=df_train_all[gender_feat])
predictors = exp_grad_est.predictors_
print(f"Number of predictors: {len(predictors)}")

# Get a summary of equalized odds difference for each predictor and best one
df_eof, best_gb = metrics.get_df_eof_and_best_predictor(predictors, x_test_all, y_test_all, df_test_all, gender_feat)
df_eof.to_csv("../results/metrics/adult/gb_mit_in_eof.csv", index=False)

# Performance metrics
y_pred_gb_mit_in = best_gb.predict(x_test_all)
cf_matrix = confusion_matrix(y_test_all, y_pred_gb_mit_in)
ml.save_confusion_matrix_text(cf_matrix, filename="../results/metrics/adult/gb_mit_in_metrics.txt")

# Fairness metrics
fmetric_by_group, fmetric_dif, fmetric_overall, eod = metrics.evaluate_fairness_metrics(y_test_all, y_pred_gb_mit_in, df_test_all, gender_feat)

print('== Metrics by gender ==')
print(fmetric_by_group)

print('== Metric differences ==')
print(fmetric_dif)

print('== Overall metrics ==')
print(fmetric_overall)

print(f"Equalized odds difference: {eod}")

# FP stats by gender
num_fp_fem, perc_fp_fem, num_fp_male, perc_fp_male = stats.fp_stats_by_gender(df_test_all, y_test_all, y_pred_gb_mit_in, gender_feat)
print(f"FP (female): {num_fp_fem}, percentage (all female in test): {perc_fp_fem}")
print(f"FP (male): {num_fp_male}, percentage (all male in test): {perc_fp_male}")

# FN stats by gender
num_fn_fem, perc_fn_fem, num_fn_male, perc_fn_male = stats.fn_stats_by_gender(df_test_all, y_test_all, y_pred_gb_mit_in, gender_feat)
print(f"FN (female): {num_fn_fem}, percentage (all female in test): {perc_fn_fem}")
print(f"FN (male): {num_fn_male}, percentage (all male in test): {perc_fn_male}")

# RQ1: Analysis of misclassifications

results_path = '../results/rq1/'
folder = results_path + '/' + dataset_name
if not os.path.exists(folder):
    os.makedirs(folder)

stats_fp = clus.ClusterStatistics()
stats_fn = clus.ClusterStatistics()

# Random Forest + FP + female
clf_name = 'rf_mit_in'
df_fp = ml.get_fp(df_test_all, y_test_all, y_pred_rf_mit_in)
df_fp_fem = df_fp[df_fp[gender_feat]==0]
filename = folder + '/' + dataset_name + '_' + clf_name + '_fp_samples.csv'
df_fp.to_csv(filename, index=False)

explainer = clue.ClusterExplainer(df_fp_fem.values)
clusters_rf_mit_in_fp_fem, exemplars_rf_mit_in_fp_fem = explainer.cluster_affinity()
filename = folder + '/' + dataset_name + '_' + clf_name + '_fp_female_clustering.txt'
stats_fp.save_clustering_results(filename, df_fp_fem, clf_name, 'female', clusters_rf_mit_in_fp_fem, exemplars_rf_mit_in_fp_fem)

# Random Forest + FP + male
df_fp_male = df_fp[df_fp[gender_feat]==1]
explainer = clue.ClusterExplainer(df_fp_male.values)
clusters_rf_mit_in_fp_male, exemplars_rf_mit_in_fp_male = explainer.cluster_affinity()
filename = folder + '/' + dataset_name + '_' + clf_name + '_fp_male_clustering.txt'
stats_fp.save_clustering_results(filename, df_fp_male, clf_name, 'male', clusters_rf_mit_in_fp_male, exemplars_rf_mit_in_fp_male)

# Random Forest + FN + Female
df_fn = ml.get_fn(df_test_all, y_test_all, y_pred_rf_mit_in)
df_fn_fem = df_fn[df_fn[gender_feat]==0]
filename = folder + '/' + dataset_name + '_' + clf_name + '_fn_samples.csv'
df_fn.to_csv(filename, index=False)

explainer = clue.ClusterExplainer(df_fn_fem.values)
clusters_rf_mit_in_fn_fem, exemplars_rf_mit_in_fn_fem = explainer.cluster_affinity()
filename = folder + '/' + dataset_name + '_' + clf_name + '_fn_female_clustering.txt'
stats_fn.save_clustering_results(filename, df_fn_fem, clf_name, 'female', clusters_rf_mit_in_fn_fem, exemplars_rf_mit_in_fn_fem)

# Random Forest + FN + Male
df_fn_male = df_fn[df_fn[gender_feat]==1]
explainer = clue.ClusterExplainer(df_fn_male.values)
clusters_rf_mit_in_fn_male, exemplars_rf_mit_in_fn_male = explainer.cluster_affinity()
filename = folder + '/' + dataset_name + '_' + clf_name + '_fn_male_clustering.txt'
stats_fn.save_clustering_results(filename, df_fn_male, clf_name, 'male', clusters_rf_mit_in_fn_male, exemplars_rf_mit_in_fn_male)

# Gradient Boosting + FP + Female
clf_name = 'gb_mit_in'
df_fp = ml.get_fp(df_test_all, y_test_all, y_pred_gb_mit_in)
df_fp_fem = df_fp[df_fp[gender_feat]==0]
filename = folder + '/' + dataset_name + '_' + clf_name + '_fp_samples.csv'
df_fp.to_csv(filename, index=False)

explainer = clue.ClusterExplainer(df_fp_fem.values)
clusters_gb_mit_in_fp_fem, exemplars_gb_mit_in_fp_fem = explainer.cluster_affinity()
filename = folder + '/' + dataset_name + '_' + clf_name + '_fp_female_clustering.txt'
stats_fp.save_clustering_results(filename, df_fp_fem, clf_name, 'female', clusters_gb_mit_in_fp_fem, exemplars_gb_mit_in_fp_fem)

# Gradient Boosting + FP + Male
df_fp_male = df_fp[df_fp[gender_feat]==1]
explainer = clue.ClusterExplainer(df_fp_male.values)
clusters_gb_mit_in_fp_male, exemplars_gb_mit_in_fp_male = explainer.cluster_affinity()
filename = folder + '/' + dataset_name + '_' + clf_name + '_fp_male_clustering.txt'
stats_fp.save_clustering_results(filename, df_fp_male, clf_name, 'male', clusters_gb_mit_in_fp_male, exemplars_gb_mit_in_fp_male)

# Gradient Boosting + FN + Female
df_fn = ml.get_fn(df_test_all, y_test_all, y_pred_gb_mit_in)
df_fn_fem = df_fn[df_fn[gender_feat]==0]
filename = folder + '/' + dataset_name + '_' + clf_name + '_fn_samples.csv'
df_fn.to_csv(filename, index=False)

explainer = clue.ClusterExplainer(df_fn_fem.values)
clusters_gb_mit_in_fn_fem, exemplars_gb_mit_in_fn_fem = explainer.cluster_affinity()
filename = folder + '/' + dataset_name + '_' + clf_name + '_fn_female_clustering.txt'
stats_fn.save_clustering_results(filename, df_fn_fem, clf_name, 'female', clusters_gb_mit_in_fn_fem, exemplars_gb_mit_in_fn_fem)

# Gradient Boosting + FN + Male
df_fn_male = df_fn[df_fn[gender_feat]==1]
explainer = clue.ClusterExplainer(df_fn_male.values)
clusters_gb_mit_in_fn_male, exemplars_gb_mit_in_fn_male = explainer.cluster_affinity()
filename = folder + '/' + dataset_name + '_' + clf_name + '_fn_male_clustering.txt'
stats_fn.save_clustering_results(filename, df_fn_male, clf_name, 'male', clusters_gb_mit_in_fn_male, exemplars_gb_mit_in_fn_male)

filename = results_path + '/' + dataset_name + '/' + dataset_name + '_clustering_stats_mit_in_fp.csv'
stats_fp.save_statistics(filename)
filename = results_path + '/' + dataset_name + '/' + dataset_name + '_clustering_stats_mit_in_fn.csv'
stats_fn.save_statistics(filename)

# RQ2: Analysis of local explanations

results_path = '../results/rq2/'
folder = results_path + '/' + dataset_name
if not os.path.exists(folder):
    os.makedirs(folder)

# Classifiers with full data

# Random Forest

# FP + Female
clf_name = 'rf_mit_in'
df_train=pd.DataFrame(x_train_all)
df_fp = ml.get_fp(df_test_all, y_test_all, y_pred_rf_mit_in)
df_fp_fem = df_fp[df_fp[gender_feat]==0]
prototypes_indexes = exemplars_rf_mit_in_fp_fem
prototypes_to_explain = df_fp_fem.iloc[prototypes_indexes]
df_fp_fem_contributions = local.summary_feature_contribution_bd(best_rf, df_train, y_train_all, prototypes_to_explain, 5)
filename = folder + '/' + dataset_name + '_' + clf_name + '_contributions_fp_fem.csv'
df_fp_fem_contributions.to_csv(filename, index=False)

# FP + Male
df_fp_male = df_fp[df_fp[gender_feat]==1]
prototypes_indexes = exemplars_rf_mit_in_fp_male
prototypes_to_explain = df_fp_male.iloc[prototypes_indexes]
df_fp_male_contributions = local.summary_feature_contribution_bd(best_rf, df_train, y_train_all, prototypes_to_explain, 5)
filename = folder + '/' + dataset_name + '_' + clf_name + '_contributions_fp_male.csv'
df_fp_male_contributions.to_csv(filename, index=False)

# FN + Female
df_fn = ml.get_fn(df_test_all, y_test_all, y_pred_rf_mit_in)
df_fn_fem = df_fn[df_fn[gender_feat]==0]
prototypes_indexes = exemplars_rf_mit_in_fn_fem
prototypes_to_explain = df_fn_fem.iloc[prototypes_indexes]
df_fn_fem_contributions = local.summary_feature_contribution_bd(best_rf, df_train, y_train_all, prototypes_to_explain, 5)
filename = folder + '/' + dataset_name + '_' + clf_name + '_contributions_fn_fem.csv'
df_fn_fem_contributions.to_csv(filename, index=False)

# FN + Male
df_fn_male = df_fn[df_fn[gender_feat]==1]
prototypes_indexes = exemplars_rf_mit_in_fn_male
prototypes_to_explain = df_fn_male.iloc[prototypes_indexes]
df_fn_male_contributions = local.summary_feature_contribution_bd(best_rf, df_train, y_train_all, prototypes_to_explain, 5)
filename = folder + '/' + dataset_name + '_' + clf_name + '_contributions_fn_male.csv'
df_fn_male_contributions.to_csv(filename, index=False)

# Gradient boosting

# FP + Female
clf_name = 'gb_mit_in'
df_fp = ml.get_fp(df_test_all, y_test_all, y_pred_gb_mit_in)
df_fp_fem = df_fp[df_fp[gender_feat]==0]
prototypes_indexes = exemplars_gb_mit_in_fp_fem
prototypes_to_explain = df_fp_fem.iloc[prototypes_indexes]
df_fp_fem_contributions = local.summary_feature_contribution_bd(best_gb, df_train, y_train_all, prototypes_to_explain, 5)
filename = folder + '/' + dataset_name + '_' + clf_name + '_contributions_fp_fem.csv'
df_fp_fem_contributions.to_csv(filename, index=False)

# FP + Male
df_fp_male = df_fp[df_fp[gender_feat]==1]
prototypes_indexes = exemplars_gb_mit_in_fp_male
prototypes_to_explain = df_fp_male.iloc[prototypes_indexes]
df_fp_male_contributions = local.summary_feature_contribution_bd(best_gb, df_train, y_train_all, prototypes_to_explain, 5)
filename = folder + '/' + dataset_name + '_' + clf_name + '_contributions_fp_male.csv'
df_fp_male_contributions.to_csv(filename, index=False)

# FN + Female
df_fn = ml.get_fn(df_test_all, y_test_all, y_pred_gb_mit_in)
df_fn_fem = df_fn[df_fn[gender_feat]==0]
prototypes_indexes = exemplars_gb_mit_in_fn_fem
prototypes_to_explain = df_fn_fem.iloc[prototypes_indexes]
df_fn_fem_contributions = local.summary_feature_contribution_bd(best_gb, df_train, y_train_all, prototypes_to_explain, 5)
filename = folder + '/' + dataset_name + '_' + clf_name + '_contributions_fn_fem.csv'
df_fn_fem_contributions.to_csv(filename, index=False)

# FN + Male
df_fn_male = df_fn[df_fn[gender_feat]==1]
prototypes_indexes = exemplars_gb_mit_in_fn_male
prototypes_to_explain = df_fn_male.iloc[prototypes_indexes]
df_fn_male_contributions = local.summary_feature_contribution_bd(best_gb, df_train, y_train_all, prototypes_to_explain, 5)
filename = folder + '/' + dataset_name + '_' + clf_name + '_contributions_fn_male.csv'
df_fn_male_contributions.to_csv(filename, index=False)