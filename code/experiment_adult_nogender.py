##########################################################################################################################
# Script for the paper "Exploring gender bias in misclassification with clustering and explainable methods". 
# The script runs the experiments to address the two research questions of the paper:
# - RQ1: Analysis of misclassification
# - RQ2: Analysis of local explanations
# This script uses the adult dataset and the "no_gender" classification strategy described in the paper.
# Author: Aurora Ramírez (University of Córdoba)
##########################################################################################################################

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
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

# Preprocessing data
# Load and clean data
df = pd.read_csv('../data/adult.csv')

# Set features of interest
target_feat = 'income'
gender_feat = 'sex'

# Convert to numeric data
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

# Train and test partition for the full date frame without gender
x_train_nogender, x_test_nogender, y_train_nogender, y_test_nogender, x_train_gender, x_test_gender = data.split_train_test_nogender(df, target_feat, gender_feat)

# Building classifiers

# Train and test the model with data excluding gender attribute
clf = RandomForestClassifier(random_state=0)
rf_nogender = clf.fit(x_train_nogender, y_train_nogender)
y_pred_rf_nogender = rf_nogender.predict(x_test_nogender)

# Performance metrics
cf_matrix = confusion_matrix(y_test_nogender, y_pred_rf_nogender)
ml.save_confusion_matrix_text(cf_matrix, filename="../results/metrics/adult/rf_nogender_metrics.txt")

# Add gender attribute temporarily to calculate metrics that require gender information
df_test_nogender = pd.DataFrame(x_test_nogender)
df_test_nogender.insert(0, gender_feat, x_test_gender)

# Fairness metrics
fmetric_by_group, fmetric_dif, fmetric_overall, eod = metrics.evaluate_fairness_metrics(y_test_nogender, y_pred_rf_nogender, df_test_nogender, gender_feat)

print('== Metrics by gender ==')
print(fmetric_by_group)

print('== Metric differences ==')
print(fmetric_dif)

print('== Overall metrics ==')
print(fmetric_overall)

print(f"Equalized odds difference: {eod}")

# FP stats by gender
num_fp_fem, perc_fp_fem, num_fp_male, perc_fp_male = stats.fp_stats_by_gender(df_test_nogender, y_test_nogender, y_pred_rf_nogender, gender_feat)
print(f"FP (female): {num_fp_fem}, percentage (all female in test): {perc_fp_fem}")
print(f"FP (male): {num_fp_male}, percentage (all male in test): {perc_fp_male}")

# FN stats by gender
num_fn_fem, perc_fn_fem, num_fn_male, perc_fn_male = stats.fn_stats_by_gender(df_test_nogender, y_test_nogender, y_pred_rf_nogender, gender_feat)
print(f"FN (female): {num_fn_fem}, percentage (all female in test): {perc_fn_fem}")
print(f"FN (male): {num_fn_male}, percentage (all male in test): {perc_fn_male}")

# Train and test the model with data excluding gender attribute
clf = GradientBoostingClassifier(random_state=0)
gb_nogender = clf.fit(x_train_nogender, y_train_nogender)
y_pred_gb_nogender = gb_nogender.predict(x_test_nogender)

# Performance metrics
cf_matrix = confusion_matrix(y_test_nogender, y_pred_gb_nogender)
ml.save_confusion_matrix_text(cf_matrix, filename="../results/metrics/adult/gb_nogender_metrics.txt")

# Fairness metrics
fmetric_by_group, fmetric_dif, fmetric_overall, eod = metrics.evaluate_fairness_metrics(y_test_nogender, y_pred_gb_nogender, df_test_nogender, gender_feat)

print('== Metrics by gender ==')
print(fmetric_by_group)

print('== Metric differences ==')
print(fmetric_dif)

print('== Overall metrics ==')
print(fmetric_overall)

print(f"Equalized odds difference: {eod}")

# FP stats by gender
num_fp_fem, perc_fp_fem, num_fp_male, perc_fp_male = stats.fp_stats_by_gender(df_test_nogender, y_test_nogender, y_pred_gb_nogender, gender_feat)
print(f"FP (female): {num_fp_fem}, percentage (all female in test): {perc_fp_fem}")
print(f"FP (male): {num_fp_male}, percentage (all male in test): {perc_fp_male}")

# FN stats by gender
num_fn_fem, perc_fn_fem, num_fn_male, perc_fn_male = stats.fn_stats_by_gender(df_test_nogender, y_test_nogender, y_pred_gb_nogender, gender_feat)
print(f"FN (female): {num_fn_fem}, percentage (all female in test): {perc_fn_fem}")
print(f"FN (male): {num_fn_male}, percentage (all male in test): {perc_fn_male}")

# RQ1: Analysis of misclassifications

results_path = '../results/rq1/'
folder = results_path + '/' + dataset_name
if not os.path.exists(folder):
    os.makedirs(folder)

stats_fp = clus.ClusterStatistics()
stats_fn = clus.ClusterStatistics()

# Random Forest + FP + Female
clf_name = 'rf_nogender'
df_fp = ml.get_fp(df_test_nogender, y_test_nogender, y_pred_rf_nogender)
df_fp_fem = df_fp[df_fp[gender_feat]==0]
filename = folder + '/' + dataset_name + '_' + clf_name + '_fp_samples.csv'
df_fp.to_csv(filename, index=False)

explainer = clue.ClusterExplainer(df_fp_fem.values)
clusters_rf_nogender_fp_fem, exemplars_rf_nogender_fp_fem = explainer.cluster_affinity()
filename = folder + '/' + dataset_name + '_' + clf_name + '_fp_female_clustering.txt'
stats_fp.save_clustering_results(filename, df_fp_fem, clf_name, 'female', clusters_rf_nogender_fp_fem, exemplars_rf_nogender_fp_fem)

# Random Forest + FP + Male
df_fp_male = df_fp[df_fp[gender_feat]==1]
explainer = clue.ClusterExplainer(df_fp_male.values)
clusters_rf_nogender_fp_male, exemplars_rf_nogender_fp_male = explainer.cluster_affinity()
filename = folder + '/' + dataset_name + '_' + clf_name + '_fp_male_clustering.txt'
stats_fp.save_clustering_results(filename, df_fp_male, clf_name, 'male', clusters_rf_nogender_fp_male, exemplars_rf_nogender_fp_male)

# Random Forest + FN + Female
df_fn = ml.get_fn(df_test_nogender, y_test_nogender, y_pred_rf_nogender)
df_fn_fem = df_fn[df_fn[gender_feat]==0]
filename = folder + '/' + dataset_name + '_' + clf_name + '_fn_samples.csv'
df_fn.to_csv(filename, index=False)

explainer = clue.ClusterExplainer(df_fn_fem.values)
clusters_rf_nogender_fn_fem, exemplars_rf_nogender_fn_fem = explainer.cluster_affinity()
filename = folder + '/' + dataset_name + '_' + clf_name + '_fn_female_clustering.txt'
stats_fn.save_clustering_results(filename, df_fn_fem, clf_name, 'female', clusters_rf_nogender_fn_fem, exemplars_rf_nogender_fn_fem)

# Random Forest + FN + Male
df_fn_male = df_fn[df_fn[gender_feat]==1]
explainer = clue.ClusterExplainer(df_fn_male.values)
clusters_rf_nogender_fn_male, exemplars_rf_nogender_fn_male = explainer.cluster_affinity()
filename = folder + '/' + dataset_name + '_' + clf_name + '_fn_male_clustering.txt'
stats_fn.save_clustering_results(filename, df_fn_male, clf_name, 'male', clusters_rf_nogender_fn_male, exemplars_rf_nogender_fn_male)

# Gradient Boosting + FP + Female
clf_name = 'gb_nogender'
df_fp = ml.get_fp(df_test_nogender, y_test_nogender, y_pred_gb_nogender)
df_fp_fem = df_fp[df_fp[gender_feat]==0]
filename = folder + '/' + dataset_name + '_' + clf_name + '_fp_samples.csv'
df_fp.to_csv(filename, index=False)

explainer = clue.ClusterExplainer(df_fp_fem.values)
clusters_gb_nogender_fp_fem, exemplars_gb_nogender_fp_fem = explainer.cluster_affinity()
filename = folder + '/' + dataset_name + '_' + clf_name + '_fp_female_clustering.txt'
stats_fp.save_clustering_results(filename, df_fp_fem, clf_name, 'female', clusters_gb_nogender_fp_fem, exemplars_gb_nogender_fp_fem)

# Gradient Boosting + FP + Male
df_fp_male = df_fp[df_fp[gender_feat]==1]
explainer = clue.ClusterExplainer(df_fp_male.values)
clusters_gb_nogender_fp_male, exemplars_gb_nogender_fp_male = explainer.cluster_affinity()
filename = folder + '/' + dataset_name + '_' + clf_name + '_fp_male_clustering.txt'
stats_fp.save_clustering_results(filename, df_fp_male, clf_name, 'male', clusters_gb_nogender_fp_male, exemplars_gb_nogender_fp_male)

# Gradient Boosting + FN + Female
df_fn = ml.get_fn(df_test_nogender, y_test_nogender, y_pred_gb_nogender)
df_fn_fem = df_fn[df_fn[gender_feat]==0]
filename = folder + '/' + dataset_name + '_' + clf_name + '_fn_samples.csv'
df_fn.to_csv(filename, index=False)

explainer = clue.ClusterExplainer(df_fn_fem.values)
clusters_gb_nogender_fn_fem, exemplars_gb_nogender_fn_fem = explainer.cluster_affinity()
filename = folder + '/' + dataset_name + '_' + clf_name + '_fn_female_clustering.txt'
stats_fn.save_clustering_results(filename, df_fn_fem, clf_name, 'female', clusters_gb_nogender_fn_fem, exemplars_gb_nogender_fn_fem)

# Gradient Boosting + FN + Male
df_fn_male = df_fn[df_fn[gender_feat]==1]
explainer = clue.ClusterExplainer(df_fn_male.values)
clusters_gb_nogender_fn_male, exemplars_gb_nogender_fn_male = explainer.cluster_affinity()
filename = folder + '/' + dataset_name + '_' + clf_name + '_fn_male_clustering.txt'
stats_fn.save_clustering_results(filename, df_fn_male, clf_name, 'male', clusters_gb_nogender_fn_male, exemplars_gb_nogender_fn_male)

filename = results_path + '/' + dataset_name + '/' + dataset_name + '_clustering_stats_nogender_fp.csv'
stats_fp.save_statistics(filename)
filename = results_path + '/' + dataset_name + '/' + dataset_name + '_clustering_stats_nogender_fn.csv'
stats_fn.save_statistics(filename)

# RQ2: Analysis of local explanations

results_path = '../results/rq2/'
dataset_name = 'adult'
folder = results_path + '/' + dataset_name
if not os.path.exists(folder):
    os.makedirs(folder)

# Random Forest

# FP + Female
clf_name = 'rf_nogender'
df_train=pd.DataFrame(x_train_nogender)
df_fp = ml.get_fp(df_test_nogender, y_test_nogender, y_pred_rf_nogender)
df_fp_fem = df_fp[df_fp[gender_feat]==0]
prototypes_indexes = exemplars_rf_nogender_fp_fem
prototypes_to_explain = df_fp_fem.iloc[prototypes_indexes]
prototypes_to_explain.drop(columns=[gender_feat], inplace=True)
df_fp_fem_contributions = local.summary_feature_contribution_bd(rf_nogender, df_train, y_train_nogender, prototypes_to_explain, 5)
filename = folder + '/' + dataset_name + '_' + clf_name + '_contributions_fp_fem.csv'
df_fp_fem_contributions.to_csv(filename, index=False)

# FP + Male
df_fp_male = df_fp[df_fp[gender_feat]==1]
prototypes_indexes = exemplars_rf_nogender_fp_male
prototypes_to_explain = df_fp_male.iloc[prototypes_indexes]
prototypes_to_explain.drop(columns=[gender_feat], inplace=True)
df_fp_male_contributions = local.summary_feature_contribution_bd(rf_nogender, df_train, y_train_nogender, prototypes_to_explain, 5)
filename = folder + '/' + dataset_name + '_' + clf_name + '_contributions_fp_male.csv'
df_fp_male_contributions.to_csv(filename, index=False)

# FN + Female
df_fn = ml.get_fn(df_test_nogender, y_test_nogender, y_pred_rf_nogender)
df_fn_fem = df_fn[df_fn[gender_feat]==0]
prototypes_indexes = exemplars_rf_nogender_fn_fem
prototypes_to_explain = df_fn_fem.iloc[prototypes_indexes]
prototypes_to_explain.drop(columns=[gender_feat], inplace=True)
df_fn_fem_contributions = local.summary_feature_contribution_bd(rf_nogender, df_train, y_train_nogender, prototypes_to_explain, 5)
filename = folder + '/' + dataset_name + '_' + clf_name + '_contributions_fn_fem.csv'
df_fn_fem_contributions.to_csv(filename, index=False)

# FN + Male
df_fn_male = df_fn[df_fn[gender_feat]==1]
prototypes_indexes = exemplars_rf_nogender_fn_male
prototypes_to_explain = df_fn_male.iloc[prototypes_indexes]
prototypes_to_explain.drop(columns=[gender_feat], inplace=True)
df_fn_male_contributions = local.summary_feature_contribution_bd(rf_nogender, df_train, y_train_nogender, prototypes_to_explain, 5)
filename = folder + '/' + dataset_name + '_' + clf_name + '_contributions_fn_male.csv'
df_fn_male_contributions.to_csv(filename, index=False)

# Gradient boosting

# FP + Female
clf_name = 'gb_nogender'
df_fp = ml.get_fp(df_test_nogender, y_test_nogender, y_pred_gb_nogender)
df_fp_fem = df_fp[df_fp[gender_feat]==0]
prototypes_indexes = exemplars_gb_nogender_fp_fem
prototypes_to_explain = df_fp_fem.iloc[prototypes_indexes]
prototypes_to_explain.drop(columns=[gender_feat], inplace=True)
df_fp_fem_contributions = local.summary_feature_contribution_bd(gb_nogender, df_train, y_train_nogender, prototypes_to_explain, 5)
filename = folder + '/' + dataset_name + '_' + clf_name + '_contributions_fp_fem.csv'
df_fp_fem_contributions.to_csv(filename, index=False)

# FP + Male
df_fp_male = df_fp[df_fp[gender_feat]==1]
prototypes_indexes = exemplars_gb_nogender_fp_male
prototypes_to_explain = df_fp_male.iloc[prototypes_indexes]
prototypes_to_explain.drop(columns=[gender_feat], inplace=True)
df_fp_male_contributions = local.summary_feature_contribution_bd(gb_nogender, df_train, y_train_nogender, prototypes_to_explain, 5)
filename = folder + '/' + dataset_name + '_' + clf_name + '_contributions_fp_male.csv'
df_fp_male_contributions.to_csv(filename, index=False)

# FN + Female
df_fn = ml.get_fn(df_test_nogender, y_test_nogender, y_pred_gb_nogender)
df_fn_fem = df_fn[df_fn[gender_feat]==0]
prototypes_indexes = exemplars_gb_nogender_fn_fem
prototypes_to_explain = df_fn_fem.iloc[prototypes_indexes]
prototypes_to_explain.drop(columns=[gender_feat], inplace=True)
df_fn_fem_contributions = local.summary_feature_contribution_bd(gb_nogender, df_train, y_train_nogender, prototypes_to_explain, 5)
filename = folder + '/' + dataset_name + '_' + clf_name + '_contributions_fn_fem.csv'
df_fn_fem_contributions.to_csv(filename, index=False)

# FN + Male
df_fn_male = df_fn[df_fn[gender_feat]==1]
prototypes_indexes = exemplars_gb_nogender_fn_male
prototypes_to_explain = df_fn_male.iloc[prototypes_indexes]
prototypes_to_explain.drop(columns=[gender_feat], inplace=True)
df_fn_male_contributions = local.summary_feature_contribution_bd(gb_nogender, df_train, y_train_nogender, prototypes_to_explain, 5)
filename = folder + '/' + dataset_name + '_' + clf_name + '_contributions_fn_male.csv'
df_fn_male_contributions.to_csv(filename, index=False)