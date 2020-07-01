# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 04:04:39 2020

@author: Aron
"""

import sqlite3
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.spatial import distance
from sklearn.metrics import roc_curve, roc_auc_score, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d


def readDatabase(path):
    conn = sqlite3.connect(path)
    df = pd.read_sql_query('select * from keystroke_datas', conn, parse_dates=['date'])
    df.drop(df[df['password'] != 'greyc laboratory'].index, inplace = True)
    conn.close()
    
    tempData = []
    n_data_rows = df.shape[0]
    columns = ["user_id"]
    
    for i in range(60):
        columns.append("ft_" + str(i+1))
        
    for i in range(n_data_rows):
        
        user_id = [df.iloc[i]["user_id"]]
    
        time_to_type = [df.iloc[i]["time_to_type"]]
        
        vector = df.iloc[i]["vector"].split()      
     
        if(len(vector) == 60 ):
          
            tempData.append(user_id  + list(map(int, vector)))
    
    df = pd.DataFrame(tempData, columns = columns)
    
    tempData.clear()
    
    subjects = df["user_id"].unique()
    
    train_users = []
    
    dev_users = []
    
    test_users = []
    
    for subject in subjects:
        current_user_data = df.loc[df.user_id == subject, :]
    
        if len(current_user_data) == 5:
            train, dev = train_test_split(current_user_data, train_size = 0.6, random_state=43, shuffle=True)
            dev , test = train_test_split(dev, train_size = 0.5, random_state=43, shuffle=True)
        
        else:
            train, dev = train_test_split(current_user_data, train_size = 0.80, random_state=43, shuffle=True)
            dev , test = train_test_split(dev, train_size = 0.5, random_state=43, shuffle=True)
            
        train_users.append(train)
        dev_users.append(dev)
        test_users.append(test)
    
    train_users = pd.concat(train_users)
    dev_users = pd.concat(dev_users)
    test_users = pd.concat(test_users)
    
    return train_users, dev_users, test_users

def euclideanDistance(a, b):
    return distance.euclidean(a.values, b.values)

def manhattanDistance(a, b):
    return distance.cityblock(a.values, b.values)

def cosDistance(a,b):
    return distance.cosine(a.values, b.values)

def euclideanStandardization(score, coef = 1):
    coef = 1 / 37767187.5 #Valor de alfa
    return 1 /( (score) *  coef + 1 )
    
def manhattanStandardization(score, coef = 1):
    coef = 1 / 179191692.5 #Valor de alfa
    return 1 /( (score) *  coef + 1 )
    
def cosineStandardization(score):
    return 1 - score

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def evaluate_EER_Thresh_Distance(genuine_scores, impostor_scores):
    #Se etiquetan los usuarios legítimos con 0 e impostores con 1
    labels = [0]*len(genuine_scores) + [1]*len(impostor_scores)
    
    #Se utiliza el metodo de roc_curve para hallar los fpr, tpr y umbrales
    fpr, tpr, thresholds = roc_curve(labels, genuine_scores + impostor_scores)
    
    #Se calcula el EER cuando el punto del fpr y del fpr se encuentran
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh
    
def evaluate_EER_Thresh_Prob(genuine_scores, impostor_scores):
    
    #Se etiquetan los usuarios legítimos con 1 e impostores con 0
    labels = [1]*len(genuine_scores) + [0]*len(impostor_scores)
    
    #Se utiliza el metodo de roc_curve para hallar los fpr, tpr y umbrales
    fpr, tpr, thresholds = roc_curve(labels, genuine_scores + impostor_scores, pos_label = 1)
    
    #Se calcula el EER cuando el punto del fpr y del fpr se encuentran
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh

def find_fpr_and_tpr_given_a_threshold_Distance(genuine_scores, impostor_scores, threshold):
    labels = [0] * len(genuine_scores) + [1] * len(impostor_scores)
    fprs, tprs, thresholds = roc_curve(labels, genuine_scores + impostor_scores)
    idx, value = find_nearest(thresholds, threshold)
    return fprs[idx], tprs[idx], value

def find_fpr_and_tpr_given_a_threshold_Prob(genuine_scores, impostor_scores, threshold):
    labels = [1] * len(genuine_scores) + [0] * len(impostor_scores)
    fprs, tprs, thresholds = roc_curve(labels, genuine_scores + impostor_scores, pos_label = 1)
    idx, value = find_nearest(thresholds, threshold)
    return fprs[idx], tprs[idx], value

def evaluate_AUC_Distance(genuine_scores, impostor_scores):
    #Se etiquetan los usuarios legítimos con 0 e impostores con 1
    labels = [0] * len(genuine_scores) + [1] * len(impostor_scores)
    auc_score = roc_auc_score(labels, genuine_scores + impostor_scores)
    return auc_score

def evaluate_AUC_Prob(genuine_scores, impostor_scores):
    #Se etiquetan los usuarios legítimos con 1 e impostores con 0
    labels = [1] * len(genuine_scores) + [0] * len(impostor_scores)
    auc_score = roc_auc_score(labels, genuine_scores + impostor_scores)
    return auc_score
