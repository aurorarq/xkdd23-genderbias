######################################################################################################################
# File: local_explanations.py
# Goal: Functions to analyse local explanations of the prototypes
# Author: Aurora Ramirez (Univ. of Cordoba, Spain)
#######################################################################################################################

import pandas as pd
import numpy as np
import dalex as dx

def summary_feature_contribution_bd(model, df_train, y_train, prototypes, num_top):
    """
    Create a data frame with the top positive and top negative feature contributions in the prototypes.
    :param model: the classification model
    :param df_train: data frame with training instances
    :param y_train: labels for training instances
    :param prototypes: the feature values of the prototype instances
    :param num_top: number of feature contributions to consider for the summary data frame
    :return A data frame with the name of the features and how many times they appear in the num_top positive contributions and num_top negative contributions
    """
    explainer = dx.Explainer(model=model, data=df_train, y=y_train, label="")
    
    pos_contributions = np.zeros(len(df_train.columns))
    neg_contributions = np.zeros(len(df_train.columns))
    col_names = df_train.columns.values

    for index, p in prototypes.iterrows():
       
        # Get bd values and associated variables (different column order each time)
        explanation = explainer.predict_parts(p, type='break_down')
        bd_values = explanation.result['contribution'].values
        bd_variables = explanation.result['variable_name'].values

        # Exclude the first (intercept) and the last (prediction)
        bd_values = bd_values[1:len(bd_values)-1]           
        bd_variables = bd_variables[1:len(bd_variables)-1]
        
        # Find the attributes contributing the most (positive and negative)
        pos_bd_values = np.flip(np.sort(bd_values[bd_values > 0]))
        neg_bd_values = np.sort(bd_values[bd_values < 0])
        if len(pos_bd_values) < num_top:
            n = len(pos_bd_values)
        else:
            n = num_top
        top_pos = pos_bd_values[0:n]
        
        if len(neg_bd_values) < num_top:
            n = len(neg_bd_values)
        else:
            n = num_top
        top_neg = neg_bd_values[0:n]

        # Update contribution count for the corresponding feature
        for i in range(0, len(top_pos)):
            top_pos_feat_index = np.where(bd_values == top_pos[i])
            feature_name_in_bd = bd_variables[top_pos_feat_index][0]
            index_in_df = np.where(col_names == feature_name_in_bd)
            pos_contributions[index_in_df] += 1 

        for i in range(0, len(top_neg)):
            top_neg_feat_index = np.where(bd_values == top_neg[i])
            feature_name_in_bd = bd_variables[top_neg_feat_index][0]
            index_in_df = np.where(col_names == feature_name_in_bd)
            neg_contributions[index_in_df] += 1  

    # Save as dataframe with the original column order
    df = pd.DataFrame({'feature': col_names, 'top_pos_contribution': pos_contributions, 'top_neg_contribution': neg_contributions})
    return df
    