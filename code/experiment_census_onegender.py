##########################################################################################################################
# Script for the paper "Exploring gender bias in misclassification with clustering and explainable methods". 
# The script runs the experiments to address the two research questions of the paper:
# - RQ1: Analysis of misclassification
# - RQ2: Analysis of local explanations
# This script uses the dutch census dataset and the "fem/male" classification strategy described in the paper.
# Author: Aurora Ramírez (University of Córdoba)
##########################################################################################################################

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import os

import utils_data as data
import utils_ml as ml
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

# Split in train/test

# Train and test partition for the female data frame, stratified by target attribute
x_train_fem, x_test_fem, y_train_fem, y_test_fem = data.split_train_test_gender(df, target_feat, gender_feat, 0)

# Train and test partition for the male date frame, stratified by target attribute
x_train_male, x_test_male, y_train_male, y_test_male = data.split_train_test_gender(df, target_feat, gender_feat, 1)

# Building classifiers
print("== Building classifiers ==")

# Train and test the model with female data
clf = RandomForestClassifier(random_state=0)
rf_fem = clf.fit(x_train_fem, y_train_fem)
y_pred_rf_fem = rf_fem.predict(x_test_fem)

# Performance metrics
cf_matrix = confusion_matrix(y_test_fem, y_pred_rf_fem)
ml.save_confusion_matrix_text(cf_matrix, filename="../results/metrics/census/rf_female_metrics.txt")

# Train and test the model with female data
clf = GradientBoostingClassifier(random_state=0)
gb_fem = clf.fit(x_train_fem, y_train_fem)
y_pred_gb_fem = gb_fem.predict(x_test_fem)

# Performance metrics
cf_matrix = confusion_matrix(y_test_fem, y_pred_gb_fem)
ml.save_confusion_matrix_text(cf_matrix, filename="../results/metrics/census/gb_female_metrics.txt")

# Train and test the model with male data
clf = RandomForestClassifier(random_state=0)
rf_male = clf.fit(x_train_male, y_train_male)
y_pred_rf_male = rf_male.predict(x_test_male)

# Performance metrics
cf_matrix = confusion_matrix(y_test_male, y_pred_rf_male)
ml.save_confusion_matrix_text(cf_matrix, filename="../results/metrics/census/rf_male_metrics.txt")

# Train and test the model with male data
clf = GradientBoostingClassifier(random_state=0)
gb_male = clf.fit(x_train_male, y_train_male)
y_pred_gb_male = gb_male.predict(x_test_male)

# Performance metrics
cf_matrix = confusion_matrix(y_test_male, y_pred_rf_male)
ml.save_confusion_matrix_text(cf_matrix, filename="../results/metrics/census/gb_male_metrics.txt")

# RQ1: Analysis of misclassifications

print("== Clustering ==")

results_path = '../results/rq1/'
folder = results_path + '/' + dataset_name
if not os.path.exists(folder):
    os.makedirs(folder)

stats_fp = clus.ClusterStatistics()
stats_fn = clus.ClusterStatistics()

# Random Forest + FP + Female
print("\t== RF+FP+Female ==")
clf_name = 'rf_fem'
df_test_fem = pd.DataFrame(x_test_fem)
df_fp = ml.get_fp(df_test_fem, y_test_fem, y_pred_rf_fem)
filename = folder + '/' + dataset_name + '_' + clf_name + '_fp_samples.csv'
df_fp.to_csv(filename, index=False)

explainer = clue.ClusterExplainer(df_fp.values)
clusters_rf_fem_fp, exemplars_rf_fem_fp = explainer.cluster_affinity()
filename = folder + '/' + dataset_name + '_' + clf_name + '_fp_female_clustering.txt'
stats_fp.save_clustering_results(filename, df_fp, clf_name, 'female', clusters_rf_fem_fp, exemplars_rf_fem_fp)

# Random Forest + FN + Female
print("\t== RF+FN+Female ==")
df_fn = ml.get_fn(df_test_fem, y_test_fem, y_pred_rf_fem)
filename = folder + '/' + dataset_name + '_' + clf_name + '_fn_samples.csv'
df_fn.to_csv(filename, index=False)

explainer = clue.ClusterExplainer(df_fn.values)
clusters_rf_fem_fn, exemplars_rf_fem_fn = explainer.cluster_affinity()
filename = folder + '/' + dataset_name + '_' + clf_name + '_fn_female_clustering.txt'
stats_fn.save_clustering_results(filename, df_fn, clf_name, 'female', clusters_rf_fem_fn, exemplars_rf_fem_fn)

# Gradient Boosting + FP + Female
print("\t== GB+FP+Female ==")
clf_name = 'gb_fem'
df_fp = ml.get_fp(df_test_fem, y_test_fem, y_pred_gb_fem)
filename = folder + '/' + dataset_name + '_' + clf_name + '_fp_samples.csv'
df_fp.to_csv(filename, index=False)

explainer = clue.ClusterExplainer(df_fp.values)
clusters_gb_fem_fp, exemplars_gb_fem_fp = explainer.cluster_affinity()
filename = folder + '/' + dataset_name + '_' + clf_name + '_fp_female_clustering.txt'
stats_fp.save_clustering_results(filename, df_fp, clf_name, 'female', clusters_gb_fem_fp, exemplars_gb_fem_fp)

# Gradient Boosting + FN + Female
print("\t== GB+NP+Female ==")
df_fn = ml.get_fn(df_test_fem, y_test_fem, y_pred_gb_fem)
filename = folder + '/' + dataset_name + '_' + clf_name + '_fn_samples.csv'
df_fn.to_csv(filename, index=False)

explainer = clue.ClusterExplainer(df_fn.values)
clusters_gb_fem_fn, exemplars_gb_fem_fn = explainer.cluster_affinity()
filename = folder + '/' + dataset_name + '_' + clf_name + '_fn_female_clustering.txt'
stats_fn.save_clustering_results(filename, df_fn, clf_name, 'female', clusters_gb_fem_fn, exemplars_gb_fem_fn)

# Random Forest + FP + Male
print("\t== RF+FP+Male ==")
clf_name = 'rf_male'
df_test_male = pd.DataFrame(x_test_male)
df_fp = ml.get_fp(df_test_male, y_test_male, y_pred_rf_male)
filename = folder + '/' + dataset_name + '_' + clf_name + '_fp_samples.csv'
df_fp.to_csv(filename, index=False)

explainer = clue.ClusterExplainer(df_fp.values)
clusters_rf_male_fp, exemplars_rf_male_fp = explainer.cluster_affinity()
filename = folder + '/' + dataset_name + '_' + clf_name + '_fp_male_clustering.txt'
stats_fp.save_clustering_results(filename, df_fp, clf_name, 'male', clusters_rf_male_fp, exemplars_rf_male_fp)

# Random Forest + FN + Male
print("\t== RF+FN+Male ==")
df_fn = ml.get_fn(df_test_male, y_test_male, y_pred_rf_male)
filename = folder + '/' + dataset_name + '_' + clf_name + '_fn_samples.csv'
df_fn.to_csv(filename, index=False)

explainer = clue.ClusterExplainer(df_fn.values)
clusters_rf_male_fn, exemplars_rf_male_fn = explainer.cluster_affinity()
filename = folder + '/' + dataset_name + '_' + clf_name + '_fn_male_clustering.txt'
stats_fn.save_clustering_results(filename, df_fn, clf_name, 'male', clusters_rf_male_fn, exemplars_rf_male_fn)

# Gradient Boosting + FP + Male
print("\t== GB+FP+Male ==")
clf_name = 'gb_male'
df_test_male = pd.DataFrame(x_test_male)
df_fp = ml.get_fp(df_test_male, y_test_male, y_pred_gb_male)
filename = folder + '/' + dataset_name + '_' + clf_name + '_fp_samples.csv'
df_fp.to_csv(filename, index=False)

explainer = clue.ClusterExplainer(df_fp.values)
clusters_gb_male_fp, exemplars_gb_male_fp = explainer.cluster_affinity()
filename = folder + '/' + dataset_name + '_' + clf_name + '_fp_male_clustering.txt'
stats_fp.save_clustering_results(filename, df_fp, clf_name, 'male', clusters_gb_male_fp, exemplars_gb_male_fp)

# Gradient Boosting + FN + Male
print("\t== RF+FN+Male ==")
df_fn = ml.get_fn(df_test_male, y_test_male, y_pred_gb_male)
filename = folder + '/' + dataset_name + '_' + clf_name + '_fn_samples.csv'
df_fn.to_csv(filename, index=False)

explainer = clue.ClusterExplainer(df_fn.values)
clusters_gb_male_fn, exemplars_gb_male_fn = explainer.cluster_affinity()
filename = folder + '/' + dataset_name + '_' + clf_name + '_fn_male_clustering.txt'
stats_fn.save_clustering_results(filename, df_fn, clf_name, 'male', clusters_gb_male_fn, exemplars_gb_male_fn)

print("\t== Save results ==")
filename = results_path + '/' + dataset_name + '/' + dataset_name + '_clustering_onegender_all_fp.csv'
stats_fp.save_statistics(filename)
filename = results_path + '/' + dataset_name + '/' + dataset_name + '_clustering_onegender_all_fn.csv'
stats_fn.save_statistics(filename)

# RQ2: Analysis of local explanations
print("== Local explanations ==")
results_path = '../results/rq2/'
dataset_name = 'census'
folder = results_path + '/' + dataset_name
if not os.path.exists(folder):
    os.makedirs(folder)

# Classifiers with full data

# Random Forest

# FP + Female
print("\t== RF+FP+Female ==")
clf_name = 'rf_fem'
df_train=pd.DataFrame(x_train_fem)
df_fp_fem = ml.get_fp(df_test_fem, y_test_fem, y_pred_rf_fem)
prototypes_indexes = exemplars_rf_fem_fp
prototypes_to_explain = df_fp_fem.iloc[prototypes_indexes]
df_fp_fem_contributions = local.summary_feature_contribution_bd(rf_fem, df_train, y_train_fem, prototypes_to_explain, 5)
filename = folder + '/' + dataset_name + '_' + clf_name + '_contributions_fp_fem.csv'
df_fp_fem_contributions.to_csv(filename, index=False)

# FP + Male
print("\t== RF+FP+Male ==")
clf_name = 'rf_male'
df_train=pd.DataFrame(x_train_male)
df_fp_male = ml.get_fp(df_test_male, y_test_male, y_pred_rf_male)
prototypes_indexes = exemplars_rf_male_fp
prototypes_to_explain = df_fp_male.iloc[prototypes_indexes]
df_fp_male_contributions = local.summary_feature_contribution_bd(rf_male, df_train, y_train_male, prototypes_to_explain, 5)
filename = folder + '/' + dataset_name + '_' + clf_name + '_contributions_fp_male.csv'
df_fp_male_contributions.to_csv(filename, index=False)

# FN + Female
print("\t== RF+FN+Female ==")
clf_name = 'rf_fem'
df_train =pd.DataFrame(x_train_fem) 
df_fn_fem = ml.get_fn(df_test_fem, y_test_fem, y_pred_rf_fem)
prototypes_indexes = exemplars_rf_fem_fn
prototypes_to_explain = df_fn_fem.iloc[prototypes_indexes]
df_fn_fem_contributions = local.summary_feature_contribution_bd(rf_fem, df_train, y_train_fem, prototypes_to_explain, 5)
filename = folder + '/' + dataset_name + '_' + clf_name + '_contributions_fn_fem.csv'
df_fn_fem_contributions.to_csv(filename, index=False)

# FN + Male
print("\t== RF+FN+Male ==")
clf_name = 'rf_male'
df_train=pd.DataFrame(x_train_male)
df_fn_male = ml.get_fn(df_test_male, y_test_male, y_pred_rf_male)
prototypes_indexes = exemplars_rf_male_fn
prototypes_to_explain = df_fn_male.iloc[prototypes_indexes]
df_fn_male_contributions = local.summary_feature_contribution_bd(rf_male, df_train, y_train_male, prototypes_to_explain, 5)
filename = folder + '/' + dataset_name + '_' + clf_name + '_contributions_fn_male.csv'
df_fn_male_contributions.to_csv(filename, index=False)

# Gradient boosting

# FP + Female
print("\t== GB+FP+Female ==")
clf_name = 'gb_fem'
df_train=pd.DataFrame(x_train_fem)
df_fp_fem = ml.get_fp(df_test_fem, y_test_fem, y_pred_gb_fem)
prototypes_indexes = exemplars_gb_fem_fp
prototypes_to_explain = df_fp_fem.iloc[prototypes_indexes]
df_fp_fem_contributions = local.summary_feature_contribution_bd(gb_fem, df_train, y_train_fem, prototypes_to_explain, 5)
filename = folder + '/' + dataset_name + '_' + clf_name + '_contributions_fp_fem.csv'
df_fp_fem_contributions.to_csv(filename, index=False)

# FP + Male
print("\t== GB+FP+Male ==")
clf_name = 'gb_male'
df_train=pd.DataFrame(x_train_male)
df_fp_male = ml.get_fp(df_test_male, y_test_male, y_pred_gb_male)
prototypes_indexes = exemplars_gb_male_fp
prototypes_to_explain = df_fp_male.iloc[prototypes_indexes]
df_fp_male_contributions = local.summary_feature_contribution_bd(gb_male, df_train, y_train_male, prototypes_to_explain, 5)
filename = folder + '/' + dataset_name + '_' + clf_name + '_contributions_fp_male.csv'
df_fp_male_contributions.to_csv(filename, index=False)

# FN + Female
print("\t== GB+FN+Female ==")
clf_name = 'gb_fem'
df_train =pd.DataFrame(x_train_fem) 
df_fn_fem = ml.get_fn(df_test_fem, y_test_fem, y_pred_gb_fem)
prototypes_indexes = exemplars_gb_fem_fn
prototypes_to_explain = df_fn_fem.iloc[prototypes_indexes]
df_fn_fem_contributions = local.summary_feature_contribution_bd(gb_fem, df_train, y_train_fem, prototypes_to_explain, 5)
filename = folder + '/' + dataset_name + '_' + clf_name + '_contributions_fn_fem.csv'
df_fn_fem_contributions.to_csv(filename, index=False)

# FN + Male
print("\t== GB+FN+Male ==")
clf_name = 'gb_male'
df_train=pd.DataFrame(x_train_male)
df_fn_male = ml.get_fn(df_test_male, y_test_male, y_pred_gb_male)
prototypes_indexes = exemplars_gb_male_fn
prototypes_to_explain = df_fn_male.iloc[prototypes_indexes]
df_fn_male_contributions = local.summary_feature_contribution_bd(gb_male, df_train, y_train_male, prototypes_to_explain, 5)
filename = folder + '/' + dataset_name + '_' + clf_name + '_contributions_fn_male.csv'
df_fn_male_contributions.to_csv(filename, index=False)