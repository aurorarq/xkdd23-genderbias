##########################################################################################################################
# Script for the paper "Exploring gender bias in misclassification with clustering and explainable methods". 
# The script runs the experiments to address the two research questions of the paper:
# - RQ1: Analysis of misclassification
# - RQ2: Analysis of local explanations
# This script uses the dutch census dataset and the "mit_post" classification strategy described in the paper.
# Author: Aurora Ramírez (University of Córdoba)
##########################################################################################################################

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from fairlearn.postprocessing import ThresholdOptimizer
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
dataset_name = 'census'
folder = results_path + '/' + dataset_name
if not os.path.exists(folder):
    os.makedirs(folder)

# Preprocessing data

# Load and clean data
df = pd.read_csv('../data/dutch_census.csv', index_col=0)

# Set features of interest
target_feat = 'Occupation'
gender_feat = 'Sex'

# Convert to numeric data

# Force categorical features
categorical_features = ['EducationLevel', 'HouseholdPosition', 'HouseholdSize', 'Country', 'EconomicStatus', 'CurEcoActivity', 'MaritalStatus']
for c in categorical_features:
    df[c] = df[c].astype('category')

# Convert types for binary columns
df['Sex'].replace([2], [0], inplace=True) # 1: Male, 2: Female (change to 0)

# For other features with more than 2 categories, use one-hot-encoding
data_non_num = df.select_dtypes(exclude=['int', 'float', 'bool'])
col_names = data_non_num.columns.values
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

# Building classifiers

# Create a copy as data frame
df_train_all = pd.DataFrame(x_train_all)
df_test_all = pd.DataFrame(x_test_all)

# Train RF
clf = RandomForestClassifier(random_state=0)
rf_all = clf.fit(x_train_all, y_train_all)

# Configure the postprocessing method
model_rf_mit_post = ThresholdOptimizer(
    estimator=rf_all,
    constraints="equalized_odds",  # Optimize FPR and FNR simultaneously
    objective="balanced_accuracy_score",
    prefit=True,
    predict_method="predict_proba",
)

# Fit the new model and use it to get new predictions
model_rf_mit_post.fit(X=x_train_all, y=y_train_all, sensitive_features=df_train_all[gender_feat])
y_pred_rf_mit_post = model_rf_mit_post.predict(x_test_all, sensitive_features=df_test_all[gender_feat])

# Performance metrics
cf_matrix = confusion_matrix(y_test_all, y_pred_rf_mit_post)
ml.save_confusion_matrix_text(cf_matrix, filename="../results/metrics/census/rf_mit_post_metrics.txt")

# Fairness metrics
fmetric_by_group, fmetric_dif, fmetric_overall, eod = metrics.evaluate_fairness_metrics(y_test_all, y_pred_rf_mit_post, df_test_all, gender_feat)

print('== Metrics by gender ==')
print(fmetric_by_group)

print('== Metric differences ==')
print(fmetric_dif)

print('== Overall metrics ==')
print(fmetric_overall)

print(f"Equalized odds difference: {eod}")

# FP stats by gender
num_fp_fem, perc_fp_fem, num_fp_male, perc_fp_male = stats.fp_stats_by_gender(df_test_all, y_test_all, y_pred_rf_mit_post, gender_feat)
print(f"FP (female): {num_fp_fem}, percentage (all female in test): {perc_fp_fem}")
print(f"FP (male): {num_fp_male}, percentage (all male in test): {perc_fp_male}")

# FN stats by gender
num_fn_fem, perc_fn_fem, num_fn_male, perc_fn_male = stats.fn_stats_by_gender(df_test_all, y_test_all, y_pred_rf_mit_post, gender_feat)
print(f"FN (female): {num_fn_fem}, percentage (all female in test): {perc_fn_fem}")
print(f"FN (male): {num_fn_male}, percentage (all male in test): {perc_fn_male}")

# Train GB
clf = GradientBoostingClassifier(random_state=0)
gb_all = clf.fit(x_train_all, y_train_all)

# Configure the postprocessing method
model_gb_mit_post = ThresholdOptimizer(
    estimator=gb_all,
    constraints="equalized_odds",  # Optimize FPR and FNR simultaneously
    objective="balanced_accuracy_score",
    prefit=True,
    predict_method="predict_proba",
)

# Fit the new model and use it to get new predictions
model_gb_mit_post.fit(X=x_train_all, y=y_train_all, sensitive_features=df_train_all[gender_feat])
y_pred_gb_mit_post = model_gb_mit_post.predict(x_test_all, sensitive_features=df_test_all[gender_feat])

# Performance metrics
cf_matrix = confusion_matrix(y_test_all, y_pred_gb_mit_post)
ml.save_confusion_matrix_text(cf_matrix, filename="../results/metrics/census/gb_mit_post_metrics.txt")

# Fairness metrics
fmetric_by_group, fmetric_dif, fmetric_overall, eod = metrics.evaluate_fairness_metrics(y_test_all, y_pred_gb_mit_post, df_test_all, gender_feat)

print('== Metrics by gender ==')
print(fmetric_by_group)

print('== Metric differences ==')
print(fmetric_dif)

print('== Overall metrics ==')
print(fmetric_overall)

print(f"Equalized odds difference: {eod}")

# FP stats by gender
num_fp_fem, perc_fp_fem, num_fp_male, perc_fp_male = stats.fp_stats_by_gender(df_test_all, y_test_all, y_pred_gb_mit_post, gender_feat)
print(f"FP (female): {num_fp_fem}, percentage (all female in test): {perc_fp_fem}")
print(f"FP (male): {num_fp_male}, percentage (all male in test): {perc_fp_male}")

# FN stats by gender
num_fn_fem, perc_fn_fem, num_fn_male, perc_fn_male = stats.fn_stats_by_gender(df_test_all, y_test_all, y_pred_gb_mit_post, gender_feat)
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
clf_name = 'rf_mit_post'
df_fp = ml.get_fp(df_test_all, y_test_all, y_pred_rf_mit_post)
df_fp_fem = df_fp[df_fp[gender_feat]==0]
filename = folder + '/' + dataset_name + '_' + clf_name + '_fp_samples.csv'
df_fp.to_csv(filename, index=False)

explainer = clue.ClusterExplainer(df_fp_fem.values)
clusters_rf_mit_post_fp_fem, exemplars_rf_mit_post_fp_fem = explainer.cluster_affinity()
filename = folder + '/' + dataset_name + '_' + clf_name + '_fp_female_clustering.txt'
stats_fp.save_clustering_results(filename, df_fp_fem, clf_name, 'female', clusters_rf_mit_post_fp_fem, exemplars_rf_mit_post_fp_fem)

# Random Forest + FP + male
df_fp_male = df_fp[df_fp[gender_feat]==1]
explainer = clue.ClusterExplainer(df_fp_male.values)
clusters_rf_mit_post_fp_male, exemplars_rf_mit_post_fp_male = explainer.cluster_affinity()
filename = folder + '/' + dataset_name + '_' + clf_name + '_fp_male_clustering.txt'
stats_fp.save_clustering_results(filename, df_fp_male, clf_name, 'male', clusters_rf_mit_post_fp_male, exemplars_rf_mit_post_fp_male)

# Random Forest + FN + Female
df_fn = ml.get_fn(df_test_all, y_test_all, y_pred_rf_mit_post)
df_fn_fem = df_fn[df_fn[gender_feat]==0]
filename = folder + '/' + dataset_name + '_' + clf_name + '_fn_samples.csv'
df_fn.to_csv(filename, index=False)

explainer = clue.ClusterExplainer(df_fn_fem.values)
clusters_rf_mit_post_fn_fem, exemplars_rf_mit_post_fn_fem = explainer.cluster_affinity()
filename = folder + '/' + dataset_name + '_' + clf_name + '_fn_female_clustering.txt'
stats_fn.save_clustering_results(filename, df_fn_fem, clf_name, 'female', clusters_rf_mit_post_fn_fem, exemplars_rf_mit_post_fn_fem)

# Random Forest + FN + Male
df_fn_male = df_fn[df_fn[gender_feat]==1]
explainer = clue.ClusterExplainer(df_fn_male.values)
clusters_rf_mit_post_fn_male, exemplars_rf_mit_post_fn_male = explainer.cluster_affinity()
filename = folder + '/' + dataset_name + '_' + clf_name + '_fn_male_clustering.txt'
stats_fn.save_clustering_results(filename, df_fn_male, clf_name, 'male', clusters_rf_mit_post_fn_male, exemplars_rf_mit_post_fn_male)

# Gradient Boosting + FP + Female
clf_name = 'gb_mit_post'
df_fp = ml.get_fp(df_test_all, y_test_all, y_pred_gb_mit_post)
df_fp_fem = df_fp[df_fp[gender_feat]==0]
filename = folder + '/' + dataset_name + '_' + clf_name + '_fp_samples.csv'
df_fp.to_csv(filename, index=False)

explainer = clue.ClusterExplainer(df_fp_fem.values)
clusters_gb_mit_post_fp_fem, exemplars_gb_mit_post_fp_fem = explainer.cluster_affinity()
filename = folder + '/' + dataset_name + '_' + clf_name + '_fp_female_clustering.txt'
stats_fp.save_clustering_results(filename, df_fp_fem, clf_name, 'female', clusters_gb_mit_post_fp_fem, exemplars_gb_mit_post_fp_fem)

# Gradient Boosting + FP + Male
df_fp_male = df_fp[df_fp[gender_feat]==1]
explainer = clue.ClusterExplainer(df_fp_male.values)
clusters_gb_mit_post_fp_male, exemplars_gb_mit_post_fp_male = explainer.cluster_affinity()
filename = folder + '/' + dataset_name + '_' + clf_name + '_fp_male_clustering.txt'
stats_fp.save_clustering_results(filename, df_fp_male, clf_name, 'male', clusters_gb_mit_post_fp_male, exemplars_gb_mit_post_fp_male)

# Gradient Boosting + FN + Female
df_fn = ml.get_fn(df_test_all, y_test_all, y_pred_gb_mit_post)
df_fn_fem = df_fn[df_fn[gender_feat]==0]
filename = folder + '/' + dataset_name + '_' + clf_name + '_fn_samples.csv'
df_fn.to_csv(filename, index=False)

explainer = clue.ClusterExplainer(df_fn_fem.values)
clusters_gb_mit_post_fn_fem, exemplars_gb_mit_post_fn_fem = explainer.cluster_affinity()
filename = folder + '/' + dataset_name + '_' + clf_name + '_fn_female_clustering.txt'
stats_fn.save_clustering_results(filename, df_fn_fem, clf_name, 'female', clusters_gb_mit_post_fn_fem, exemplars_gb_mit_post_fn_fem)

# Gradient Boosting + FN + Male
df_fn_male = df_fn[df_fn[gender_feat]==1]
explainer = clue.ClusterExplainer(df_fn_male.values)
clusters_gb_mit_post_fn_male, exemplars_gb_mit_post_fn_male = explainer.cluster_affinity()
filename = folder + '/' + dataset_name + '_' + clf_name + '_fn_male_clustering.txt'
stats_fn.save_clustering_results(filename, df_fn_male, clf_name, 'male', clusters_gb_mit_post_fn_male, exemplars_gb_mit_post_fn_male)

filename = results_path + '/' + dataset_name + '/' + dataset_name + '_clustering_stats_mit_post_fp.csv'
stats_fp.save_statistics(filename)
filename = results_path + '/' + dataset_name + '/' + dataset_name + '_clustering_stats_mit_post_fn.csv'
stats_fn.save_statistics(filename)

# RQ2: Analysis of local explanations

results_path = '../results/rq2/'
folder = results_path + '/' + dataset_name
if not os.path.exists(folder):
    os.makedirs(folder)

# Classifiers with full data

# Random Forest

# FP + Female
clf_name = 'rf_mit_post'
df_train=pd.DataFrame(x_train_all)
df_fp = ml.get_fp(df_test_all, y_test_all, y_pred_rf_mit_post)
df_fp_fem = df_fp[df_fp[gender_feat]==0]
prototypes_indexes = exemplars_rf_mit_post_fp_fem
prototypes_to_explain = df_fp_fem.iloc[prototypes_indexes]
df_fp_fem_contributions = local.summary_feature_contribution_bd(rf_all, df_train, y_train_all, prototypes_to_explain, 5)
filename = folder + '/' + dataset_name + '_' + clf_name + '_contributions_fp_fem.csv'
df_fp_fem_contributions.to_csv(filename, index=False)

# FP + Male
df_fp_male = df_fp[df_fp[gender_feat]==1]
prototypes_indexes = exemplars_rf_mit_post_fp_male
prototypes_to_explain = df_fp_male.iloc[prototypes_indexes]
df_fp_male_contributions = local.summary_feature_contribution_bd(rf_all, df_train, y_train_all, prototypes_to_explain, 5)
filename = folder + '/' + dataset_name + '_' + clf_name + '_contributions_fp_male.csv'
df_fp_male_contributions.to_csv(filename, index=False)

# FN + Female
df_fn = ml.get_fn(df_test_all, y_test_all, y_pred_rf_mit_post)
df_fn_fem = df_fn[df_fn[gender_feat]==0]
prototypes_indexes = exemplars_rf_mit_post_fn_fem
prototypes_to_explain = df_fn_fem.iloc[prototypes_indexes]
df_fn_fem_contributions = local.summary_feature_contribution_bd(rf_all, df_train, y_train_all, prototypes_to_explain, 5)
filename = folder + '/' + dataset_name + '_' + clf_name + '_contributions_fn_fem.csv'
df_fn_fem_contributions.to_csv(filename, index=False)

# FN + Male
df_fn_male = df_fn[df_fn[gender_feat]==1]
prototypes_indexes = exemplars_rf_mit_post_fn_male
prototypes_to_explain = df_fn_male.iloc[prototypes_indexes]
df_fn_male_contributions = local.summary_feature_contribution_bd(rf_all, df_train, y_train_all, prototypes_to_explain, 5)
filename = folder + '/' + dataset_name + '_' + clf_name + '_contributions_fn_male.csv'
df_fn_male_contributions.to_csv(filename, index=False)

# Gradient boosting

# FP + Female
clf_name = 'gb_mit_post'
df_fp = ml.get_fp(df_test_all, y_test_all, y_pred_gb_mit_post)
df_fp_fem = df_fp[df_fp[gender_feat]==0]
prototypes_indexes = exemplars_gb_mit_post_fp_fem
prototypes_to_explain = df_fp_fem.iloc[prototypes_indexes]
df_fp_fem_contributions = local.summary_feature_contribution_bd(gb_all, df_train, y_train_all, prototypes_to_explain, 5)
filename = folder + '/' + dataset_name + '_' + clf_name + '_contributions_fp_fem.csv'
df_fp_fem_contributions.to_csv(filename, index=False)

# FP + Male
df_fp_male = df_fp[df_fp[gender_feat]==1]
prototypes_indexes = exemplars_gb_mit_post_fp_male
prototypes_to_explain = df_fp_male.iloc[prototypes_indexes]
df_fp_male_contributions = local.summary_feature_contribution_bd(gb_all, df_train, y_train_all, prototypes_to_explain, 5)
filename = folder + '/' + dataset_name + '_' + clf_name + '_contributions_fp_male.csv'
df_fp_male_contributions.to_csv(filename, index=False)

# FN + Female
df_fn = ml.get_fn(df_test_all, y_test_all, y_pred_gb_mit_post)
df_fn_fem = df_fn[df_fn[gender_feat]==0]
prototypes_indexes = exemplars_gb_mit_post_fn_fem
prototypes_to_explain = df_fn_fem.iloc[prototypes_indexes]
df_fn_fem_contributions = local.summary_feature_contribution_bd(gb_all, df_train, y_train_all, prototypes_to_explain, 5)
filename = folder + '/' + dataset_name + '_' + clf_name + '_contributions_fn_fem.csv'
df_fn_fem_contributions.to_csv(filename, index=False)

# FN + Male
df_fn_male = df_fn[df_fn[gender_feat]==1]
prototypes_indexes = exemplars_gb_mit_post_fn_male
prototypes_to_explain = df_fn_male.iloc[prototypes_indexes]
df_fn_male_contributions = local.summary_feature_contribution_bd(gb_all, df_train, y_train_all, prototypes_to_explain, 5)
filename = folder + '/' + dataset_name + '_' + clf_name + '_contributions_fn_male.csv'
df_fn_male_contributions.to_csv(filename, index=False)