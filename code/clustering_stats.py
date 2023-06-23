######################################################################################################################
# File: clustering_stats.py
# Goal: Functions to save clustering statistics in data frame
# Author: Aurora Ramirez (Univ. of Cordoba, Spain)
#######################################################################################################################

import pandas as pd
import numpy as np
from clustering_result import ClusterResult

class ClusterStatistics:
    """
    A class to store statistics of several runs of clustering methods
    """
    def __init__(self):
        """
        Create the object
        """
        self.df = pd.DataFrame(columns=['classifier', 'gender', 'silhouette', 'calinski', 'davies', 'num_clusters'])

    def add_row_values(self, classifier=None, gender=None, silhouette=None, calinski=None, davies=None, num_clusters=None):
        """
        Add one value to each column
        :param classifier: Classifier name
        :param gender: Gender under analysis
        :param silhouette: Silhouette metric value
        :param calinski: Calinski metric value
        :param davies: Davies metric value
        :param num_clusters: Number of clusters
        """
        new_row = {
            'classifier': classifier,
            'gender': gender,
            'silhouette': silhouette,
            'calinski': calinski,
            'davies': davies,
            'num_clusters': num_clusters
        }
        self.df.loc[len(self.df)] = new_row

    def save_statistics(self, filename):
        """
        Save the data frame as a CSV file
        :param filename: Name of the file
        """
        self.df.to_csv(filename, index=False)

    def save_clustering_results(self, filename, df_fp, clf_name, gender, clusters, exemplars):
        if clusters is None or exemplars is None:
            self.add_row_values(clf_name, gender, np.nan, np.nan, np.nan, np.nan)
        else:
            fp_cluster_results = ClusterResult(df_fp.values, df_fp.columns, clusters, exemplars)
            fp_cluster_results.save_results(filename)
            num_clusters = fp_cluster_results.number_clusters()
            if num_clusters > 1:
                sil_coef = fp_cluster_results.compute_silhouette_coefficient()
                cal_sco = fp_cluster_results.compute_calinski_score()
                dav_ind = fp_cluster_results.compute_davies_index()
                self.add_row_values(clf_name, gender, sil_coef, cal_sco, dav_ind, num_clusters)
            else:
                self.add_row_values(clf_name, gender, np.nan, np.nan, np.nan, num_clusters)
