# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 04:16:27 2020

@author: Aron
"""

import pandas as pd
import numpy as np
from utils import euclideanDistance, manhattanDistance, cosDistance, euclideanStandardization, manhattanStandardization, cosineStandardization
from utils import evaluate_EER_Thresh_Prob, evaluate_EER_Thresh_Distance, find_fpr_and_tpr_given_a_threshold_Distance, find_fpr_and_tpr_given_a_threshold_Prob
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def EuclideanModel(train_users, dev_users, test_users):
    
    subjects = train_users["user_id"].unique()
    
    
    #Se calcula la media de cada usuario agrupando el dataframe de train
    groupby = train_users.groupby("user_id").mean().copy()
    
    #Se incluye la columna del usuario
    train_users = groupby.reset_index()
        
    users_evaluation_dev = []

    #Se hace el cálculo para cada usuario
    for subject in subjects:
        
        #Media del vector del usuario actual
        mean_vector= train_users.loc[train_users.user_id == subject, "ft_1":"ft_60"]
        
        #Para cada registro del subdataset de test
        for index, row in dev_users.iterrows():
    
            temp_obj = {}
    
            #userid del del registro actual del subdataset de test
            current_user_id = row[0]
    
            #Vector de tiempo del registro actual del subdataset de test
            current_data = row[1:]
    
            temp_obj["user_model"] = subject
    
            temp_obj["user_id"] = current_user_id
    
            temp_obj["score"] = euclideanDistance(mean_vector, current_data)
            
            temp_obj["std_score"] = euclideanStandardization(euclideanDistance(mean_vector, current_data))
            
            #temp_obj["std_score"] = standardizationMinMax(euclideanDistance(mean_vector, current_data))
            
            if subject == current_user_id:
                temp_obj["y_test"] = "genuine"
            else:
                temp_obj["y_test"] = "impostor"
    
            users_evaluation_dev.append(temp_obj)
    
    users_evaluation_dev = pd.DataFrame(users_evaluation_dev)
    
    #Obtenemos la listas de scores de los registros que deberian de catalogarse como genuinos por los modelos
    genuine_scores_dev = list(users_evaluation_dev.loc[users_evaluation_dev.y_test == "genuine", "std_score"])
    
    #Obtenemos la listas de scores de los registros que deberian de catalogarse como impostores por los modelos
    impostor_scores_dev = list(users_evaluation_dev.loc[users_evaluation_dev.y_test == "impostor", "std_score"])
    
    #Calculo del ERR y del umbral de decisión global
    err_dev, thresh_dev = evaluate_EER_Thresh_Distance(genuine_scores_dev, impostor_scores_dev)
    
    #-----------------------------------------------------------------------------------
    
    users_evaluation_test = []
    
    #Se hace el cálculo para cada usuario
    for subject in subjects:
        
        #Media del vector del usuario actual
        mean_vector= train_users.loc[train_users.user_id == subject, "ft_1":"ft_60"]
        
        #Para cada registro del subdataset de test
        for index, row in test_users.iterrows():
    
            temp_obj = {}
    
            #userid del del registro actual del subdataset de test
            current_user_id = row[0]
    
            #Vector de tiempo del registro actual del subdataset de test
            current_data = row[1:]
    
            temp_obj["user_model"] = subject
    
            temp_obj["user_id"] = current_user_id
    
            temp_obj["score"] = euclideanDistance(mean_vector, current_data)
            
            temp_obj["std_score"] = euclideanStandardization(euclideanDistance(mean_vector, current_data))
            
            #temp_obj["std_score"] = standardizationMinMax(euclideanDistance(mean_vector, current_data))
            
            if subject == current_user_id:
                temp_obj["y_test"] = "genuine"
            else:
                temp_obj["y_test"] = "impostor"
    
            users_evaluation_test.append(temp_obj)
    
    users_evaluation_test = pd.DataFrame(users_evaluation_test)
    
    # OJO, aca se esta usando el score como umbral, si usaramos el score estandarizado, deberiamos de cambiar el sentido de la comparación
    users_evaluation_test['y_pred'] = np.where(users_evaluation_test['std_score'] >= thresh_dev, 'genuine', 'impostor')
    
    #Obtenemos los y_test y y_pred de nuestros resultados
    y_test_test = users_evaluation_test.loc[:, "y_test"]
    y_pred_test = users_evaluation_test.loc[:, "y_pred"]
    
    #-----------------------------------------------------------------------------------
        
    #Obtenemos la listas de scores de los registros que deberian de catalogarse como genuinos por los modelos
    genuine_scores_test = list(users_evaluation_test.loc[users_evaluation_test.y_test == "genuine", "std_score"])
    
    #Obtenemos la listas de scores de los registros que deberian de catalogarse como impostores por los modelos
    impostor_scores_test = list(users_evaluation_test.loc[users_evaluation_test.y_test == "impostor", "std_score"])
    
    thresh_x, thresh_y, _ = find_fpr_and_tpr_given_a_threshold_Distance(genuine_scores_test, impostor_scores_test, thresh_dev)
    
    thresh_std = round(thresh_dev.tolist(), 3) 
    
    return y_test_test, y_pred_test ,genuine_scores_test, impostor_scores_test, thresh_std, thresh_x, thresh_y

def ManhattanModel(train_users, dev_users, test_users):
    
    subjects = train_users["user_id"].unique()
    
    #Se calcula la media de cada usuario agrupando el dataframe de train
    groupby = train_users.groupby("user_id").mean().copy()
    
    #Se incluye la columna del usuario
    train_users = groupby.reset_index()
        
    users_evaluation_dev = []

    #Se hace el cálculo para cada usuario
    for subject in subjects:
        
        #Media del vector del usuario actual
        mean_vector= train_users.loc[train_users.user_id == subject, "ft_1":"ft_60"]
        
        #Para cada registro del subdataset de test
        for index, row in dev_users.iterrows():
    
            temp_obj = {}
    
            #userid del del registro actual del subdataset de test
            current_user_id = row[0]
    
            #Vector de tiempo del registro actual del subdataset de test
            current_data = row[1:]
    
            temp_obj["user_model"] = subject
    
            temp_obj["user_id"] = current_user_id
    
            temp_obj["score"] = manhattanDistance(mean_vector, current_data)
            
            temp_obj["std_score"] = manhattanStandardization(manhattanDistance(mean_vector, current_data))
            
            #temp_obj["std_score"] = standardizationMinMax(euclideanDistance(mean_vector, current_data))
            
            if subject == current_user_id:
                temp_obj["y_test"] = "genuine"
            else:
                temp_obj["y_test"] = "impostor"
    
            users_evaluation_dev.append(temp_obj)
    
    users_evaluation_dev = pd.DataFrame(users_evaluation_dev)
    
    #Obtenemos la listas de scores de los registros que deberian de catalogarse como genuinos por los modelos
    genuine_scores_dev = list(users_evaluation_dev.loc[users_evaluation_dev.y_test == "genuine", "std_score"])
    
    #Obtenemos la listas de scores de los registros que deberian de catalogarse como impostores por los modelos
    impostor_scores_dev = list(users_evaluation_dev.loc[users_evaluation_dev.y_test == "impostor", "std_score"])
    
    #Calculo del ERR y del umbral de decisión global
    err_dev, thresh_dev = evaluate_EER_Thresh_Distance(genuine_scores_dev, impostor_scores_dev)
    
    #-----------------------------------------------------------------------------------
    
    users_evaluation_test = []
    
    #Se hace el cálculo para cada usuario
    for subject in subjects:
        
        #Media del vector del usuario actual
        mean_vector= train_users.loc[train_users.user_id == subject, "ft_1":"ft_60"]
        
        #Para cada registro del subdataset de test
        for index, row in test_users.iterrows():
    
            temp_obj = {}
    
            #userid del del registro actual del subdataset de test
            current_user_id = row[0]
    
            #Vector de tiempo del registro actual del subdataset de test
            current_data = row[1:]
    
            temp_obj["user_model"] = subject
    
            temp_obj["user_id"] = current_user_id
    
            temp_obj["score"] = manhattanDistance(mean_vector, current_data)
            
            temp_obj["std_score"] = manhattanStandardization(manhattanDistance(mean_vector, current_data))
            
            #temp_obj["std_score"] = standardizationMinMax(euclideanDistance(mean_vector, current_data))
            
            if subject == current_user_id:
                temp_obj["y_test"] = "genuine"
            else:
                temp_obj["y_test"] = "impostor"
    
            users_evaluation_test.append(temp_obj)
    
    users_evaluation_test = pd.DataFrame(users_evaluation_test)
    
    # OJO, aca se esta usando el score como umbral, si usaramos el score estandarizado, deberiamos de cambiar el sentido de la comparación
    users_evaluation_test['y_pred'] = np.where(users_evaluation_test['std_score'] >= thresh_dev, 'genuine', 'impostor')
    
    #Obtenemos los y_test y y_pred de nuestros resultados
    y_test_test = users_evaluation_test.loc[:, "y_test"]
    y_pred_test = users_evaluation_test.loc[:, "y_pred"]
    
    #-----------------------------------------------------------------------------------
        
    #Obtenemos la listas de scores de los registros que deberian de catalogarse como genuinos por los modelos
    genuine_scores_test = list(users_evaluation_test.loc[users_evaluation_test.y_test == "genuine", "std_score"])
    
    #Obtenemos la listas de scores de los registros que deberian de catalogarse como impostores por los modelos
    impostor_scores_test = list(users_evaluation_test.loc[users_evaluation_test.y_test == "impostor", "std_score"])
    
    thresh_x, thresh_y, _ = find_fpr_and_tpr_given_a_threshold_Distance(genuine_scores_test, impostor_scores_test, thresh_dev)
    
    thresh_std = round(thresh_dev.tolist(), 3) 
    

    return y_test_test, y_pred_test ,genuine_scores_test, impostor_scores_test, thresh_std, thresh_x, thresh_y
    
def CosineModel(train_users, dev_users, test_users):
    
    subjects = train_users["user_id"].unique()
    
    #Se calcula la media de cada usuario agrupando el dataframe de train
    groupby = train_users.groupby("user_id").mean().copy()
    
    #Se incluye la columna del usuario
    train_users = groupby.reset_index()
        
    users_evaluation_dev = []

    #Se hace el cálculo para cada usuario
    for subject in subjects:
        
        #Media del vector del usuario actual
        mean_vector= train_users.loc[train_users.user_id == subject, "ft_1":"ft_60"]
        
        #Para cada registro del subdataset de test
        for index, row in dev_users.iterrows():
    
            temp_obj = {}
    
            #userid del del registro actual del subdataset de test
            current_user_id = row[0]
    
            #Vector de tiempo del registro actual del subdataset de test
            current_data = row[1:]
    
            temp_obj["user_model"] = subject
    
            temp_obj["user_id"] = current_user_id
    
            temp_obj["score"] = cosDistance(mean_vector, current_data)
            
            temp_obj["std_score"] = cosineStandardization(cosDistance(mean_vector, current_data))
            
            #temp_obj["std_score"] = standardizationMinMax(euclideanDistance(mean_vector, current_data))
            
            if subject == current_user_id:
                temp_obj["y_test"] = "genuine"
            else:
                temp_obj["y_test"] = "impostor"
    
            users_evaluation_dev.append(temp_obj)
    
    users_evaluation_dev = pd.DataFrame(users_evaluation_dev)
    
    #Obtenemos la listas de scores de los registros que deberian de catalogarse como genuinos por los modelos
    genuine_scores_dev = list(users_evaluation_dev.loc[users_evaluation_dev.y_test == "genuine", "score"])
    
    #Obtenemos la listas de scores de los registros que deberian de catalogarse como impostores por los modelos
    impostor_scores_dev = list(users_evaluation_dev.loc[users_evaluation_dev.y_test == "impostor", "score"])
    
    #Calculo del ERR y del umbral de decisión global
    err_dev, thresh_dev = evaluate_EER_Thresh_Distance(genuine_scores_dev, impostor_scores_dev)
    
    #-----------------------------------------------------------------------------------
    
    users_evaluation_test = []
    
    #Se hace el cálculo para cada usuario
    for subject in subjects:
        
        #Media del vector del usuario actual
        mean_vector= train_users.loc[train_users.user_id == subject, "ft_1":"ft_60"]
        
        #Para cada registro del subdataset de test
        for index, row in test_users.iterrows():
    
            temp_obj = {}
    
            #userid del del registro actual del subdataset de test
            current_user_id = row[0]
    
            #Vector de tiempo del registro actual del subdataset de test
            current_data = row[1:]
    
            temp_obj["user_model"] = subject
    
            temp_obj["user_id"] = current_user_id
    
            temp_obj["score"] = cosDistance(mean_vector, current_data)
            
            temp_obj["std_score"] = cosineStandardization(cosDistance(mean_vector, current_data))
            
            #temp_obj["std_score"] = standardizationMinMax(euclideanDistance(mean_vector, current_data))
            
            if subject == current_user_id:
                temp_obj["y_test"] = "genuine"
            else:
                temp_obj["y_test"] = "impostor"
    
            users_evaluation_test.append(temp_obj)
    
    users_evaluation_test = pd.DataFrame(users_evaluation_test)
    
    # OJO, aca se esta usando el score como umbral, si usaramos el score estandarizado, deberiamos de cambiar el sentido de la comparación
    users_evaluation_test['y_pred'] = np.where(users_evaluation_test['score'] <= thresh_dev, 'genuine', 'impostor')
    
    #Obtenemos los y_test y y_pred de nuestros resultados
    y_test_test = users_evaluation_test.loc[:, "y_test"]
    y_pred_test = users_evaluation_test.loc[:, "y_pred"]
    
    #-----------------------------------------------------------------------------------
        
    #Obtenemos la listas de scores de los registros que deberian de catalogarse como genuinos por los modelos
    genuine_scores_test = list(users_evaluation_test.loc[users_evaluation_test.y_test == "genuine", "score"])
    
    #Obtenemos la listas de scores de los registros que deberian de catalogarse como impostores por los modelos
    impostor_scores_test = list(users_evaluation_test.loc[users_evaluation_test.y_test == "impostor", "score"])
    
    thresh_x, thresh_y, _ = find_fpr_and_tpr_given_a_threshold_Distance(genuine_scores_test, impostor_scores_test, thresh_dev)
    
    thresh_std = round(cosineStandardization(thresh_dev), 3)    
    
    return y_test_test, y_pred_test ,genuine_scores_test, impostor_scores_test, thresh_std, thresh_x, thresh_y
    
def SVMModel(train_users, dev_users, test_users):
    
    subjects = train_users["user_id"].unique()

    users_evaluation_dev = []
    
    #Se hace el cálculo para cada usuario
    for subject in subjects:
        
        #----------------------------------------------------------------
        #Generamos una copia temporal del dataset de entrenamiento
        temp1 = train_users.copy()
    
        #Reemplazamos todos los users_ids que son distintos al sujeto actual por 0
        temp1["user_id"] = temp1["user_id"].mask(temp1["user_id"] != subject, 0)
    
        #Obtenemos los registros considerados genuinos del entrenamiento
        genuine_data = temp1.loc[temp1.user_id == subject, :]
    
        #Obtenemos los registros considerados impostores del entrenamiento.
        #Este debe de ser del mismo tamaño que de los registros genuinos
        impostor_data = temp1.loc[temp1.user_id != subject, :].sample(n= genuine_data.shape[0], random_state=43)
    
        #Unimos los dos anteriores variables en un solo dataset de entrenamiento del modelo
        train = pd.concat([genuine_data, impostor_data])
    
        #Obtenemos el X_train
        X_train = train.loc[:, "ft_1":"ft_60" ]
    
        #Obtenemos el y_train
        y_train = train.loc[:, "user_id"]
        
        #----------------------------------------------------------------
        
        #Generamos una copia temporal del dataset de desarrollo
        temp2 = dev_users.copy()
    
        #Reemplazamos todos los users_ids que son distintos al usuario 1 por 0
        temp2["user_id"] = temp2["user_id"].mask(temp2["user_id"] != subject, 0)
    
        #df.sample(frac=0.5, replace=True, random_state=1)
        X_dev = temp2.loc[:, "ft_1":"ft_60"]
        y_dev = temp2.loc[:, "user_id"]
    
        #----------------------------------------------------------------
    
        #Generamos una copia temporal del dataset de test
        temp3 = test_users.copy()
    
        #Reemplazamos todos los users_ids que son distintos al usuario 1 por 0
        temp3["user_id"] = temp3["user_id"].mask(temp3["user_id"] != subject, 0)
    
        X_test = temp3.loc[:, "ft_1":"ft_60"]
        y_test = temp3.loc[:, "user_id"]
        
        #----------------------------------------------------------------
        
        #Entrenamos el modelo SVM
        
        clf = SVC(C = 10.0, gamma = 'scale', kernel = 'rbf', probability = True, random_state=43)
        #clf = SVC(probability = True)
        
        clf.fit(X_train,y_train)
        
        #Obtenemos probabilidades de cada registro del dataset de test
        y_prob = clf.predict_proba(X_dev)
        y_prob = pd.DataFrame(y_prob, columns = ["probImpos", "probLegi"])
    
        i = 0
    
        #Para cada registro del subdataset de test
        for index, row in dev_users.iterrows():
    
            temp_obj = {}
    
            #user id del registro actual del subdataset de test
            current_user_id = row[0]
    
            #Vector de tiempo del registro actual del subdataset de test
            current_data = row[1:]
    
            #Actual modelo del usuario a evaluar
            temp_obj["user_model"] = subject
    
            #user id del registro actual
            temp_obj["user_id"] = current_user_id
    
            #Puntaje o score del modelo
            temp_obj["score"] = y_prob.iloc[i]["probLegi"]
    
            #Normalizacion del score
            temp_obj["std_score"] = y_prob.iloc[i]["probLegi"]
    
            #Variable que indica si el registro deberia de ser clasificado como genuino o impostor
            if current_user_id == subject:
                temp_obj["y_test"] = "genuine"
            else:
                temp_obj["y_test"] = "impostor"
    
            users_evaluation_dev.append(temp_obj)
    
            i += 1
    
    users_evaluation_dev = pd.DataFrame(users_evaluation_dev)

    #Obtenemos la listas de scores de los registros que deberian de catalogarse como genuinos por los modelos
    genuine_scores_dev = list(users_evaluation_dev.loc[users_evaluation_dev.y_test == "genuine", "score"])
    
    #Obtenemos la listas de scores de los registros que deberian de catalogarse como impostores por los modelos
    impostor_scores_dev = list(users_evaluation_dev.loc[users_evaluation_dev.y_test == "impostor", "score"])
        
    #Calculo del ERR y del umbral de decisión global
    err_dev, thresh_dev = evaluate_EER_Thresh_Prob(genuine_scores_dev, impostor_scores_dev)
    
    users_evaluation_test = []
    
    #Se hace el cálculo para cada usuario
    for subject in subjects:
        
        #----------------------------------------------------------------
        #Generamos una copia temporal del dataset de entrenamiento
        temp1 = train_users.copy()
    
        #Reemplazamos todos los users_ids que son distintos al sujeto actual por 0
        temp1["user_id"] = temp1["user_id"].mask(temp1["user_id"] != subject, 0)
    
        #Obtenemos los registros considerados genuinos del entrenamiento
        genuine_data = temp1.loc[temp1.user_id == subject, :]
    
        #Obtenemos los registros considerados impostores del entrenamiento.
        #Este debe de ser del mismo tamaño que de los registros genuinos
        impostor_data = temp1.loc[temp1.user_id != subject, :].sample(n= genuine_data.shape[0], random_state=43)
    
        #Unimos los dos anteriores variables en un solo dataset de entrenamiento del modelo
        train = pd.concat([genuine_data, impostor_data])
    
        #Obtenemos el X_train
        X_train = train.loc[:, "ft_1":"ft_60" ]
    
        #Obtenemos el y_train
        y_train = train.loc[:, "user_id"]
        
        #----------------------------------------------------------------
        
        #Generamos una copia temporal del dataset de desarrollo
        temp2 = dev_users.copy()
    
        #Reemplazamos todos los users_ids que son distintos al usuario 1 por 0
        temp2["user_id"] = temp2["user_id"].mask(temp2["user_id"] != subject, 0)
    
        #df.sample(frac=0.5, replace=True, random_state=1)
        X_dev = temp2.loc[:, "ft_1":"ft_60"]
        y_dev = temp2.loc[:, "user_id"]
    
        #----------------------------------------------------------------
    
        #Generamos una copia temporal del dataset de test
        temp3 = test_users.copy()
    
        #Reemplazamos todos los users_ids que son distintos al usuario 1 por 0
        temp3["user_id"] = temp3["user_id"].mask(temp3["user_id"] != subject, 0)
    
        X_test = temp3.loc[:, "ft_1":"ft_60"]
        y_test = temp3.loc[:, "user_id"]
        
        #----------------------------------------------------------------
        
        #Entrenamos el modelo SVM
        
        clf = SVC(C = 10.0, gamma = 'scale', kernel = 'rbf', probability = True, random_state=43)
        #clf = SVC(probability = True)
        
        clf.fit(X_train,y_train)
        
        #Obtenemos probabilidades de cada registro del dataset de test
        y_prob = clf.predict_proba(X_test)
        y_prob = pd.DataFrame(y_prob, columns = ["probImpos", "probLegi"])
    
        i = 0
    
        #Para cada registro del subdataset de test
        for index, row in test_users.iterrows():
    
            temp_obj = {}
    
            #user id del registro actual del subdataset de test
            current_user_id = row[0]
    
            #Vector de tiempo del registro actual del subdataset de test
            current_data = row[1:]
    
            #Actual modelo del usuario a evaluar
            temp_obj["user_model"] = subject
    
            #user id del registro actual
            temp_obj["user_id"] = current_user_id
    
            #Puntaje o score del modelo
            temp_obj["score"] = y_prob.iloc[i]["probLegi"]
    
            #Normalizacion del score
            temp_obj["std_score"] = y_prob.iloc[i]["probLegi"]
    
            #Variable que indica si el registro deberia de ser clasificado como genuino o impostor
            if current_user_id == subject:
                temp_obj["y_test"] = "genuine"
            else:
                temp_obj["y_test"] = "impostor"
    
            users_evaluation_test.append(temp_obj)
    
            i += 1
    
    users_evaluation_test = pd.DataFrame(users_evaluation_test)
    
    users_evaluation_test['y_pred'] = np.where(users_evaluation_test['score'] >= thresh_dev, 'genuine', 'impostor')
    
    #Obtenemos los y_test y y_pred de nuestros resultados
    y_test_test = users_evaluation_test.loc[:, "y_test"]
    y_pred_test = users_evaluation_test.loc[:, "y_pred"]
    
    genuine_scores_test = list(users_evaluation_test.loc[users_evaluation_test.y_test == "genuine", "score"])

    #Obtenemos la listas de scores de los registros que deberian de catalogarse como impostores por los modelos
    impostor_scores_test = list(users_evaluation_test.loc[users_evaluation_test.y_test == "impostor", "score"])
    
    thresh_x, thresh_y, _ = find_fpr_and_tpr_given_a_threshold_Prob(genuine_scores_test, impostor_scores_test, thresh_dev)
    
    thresh_std = round(thresh_dev.tolist(), 3)
    
    return y_test_test, y_pred_test ,genuine_scores_test, impostor_scores_test, thresh_std, thresh_x, thresh_y



def RFModel(train_users, dev_users, test_users):
    
    subjects = train_users["user_id"].unique()

    users_evaluation_dev = []
    
    #Se hace el cálculo para cada usuario
    for subject in subjects:
        
        #----------------------------------------------------------------
        #Generamos una copia temporal del dataset de entrenamiento
        temp1 = train_users.copy()
    
        #Reemplazamos todos los users_ids que son distintos al sujeto actual por 0
        temp1["user_id"] = temp1["user_id"].mask(temp1["user_id"] != subject, 0)
    
        #Obtenemos los registros considerados genuinos del entrenamiento
        genuine_data = temp1.loc[temp1.user_id == subject, :]
    
        #Obtenemos los registros considerados impostores del entrenamiento.
        #Este debe de ser del mismo tamaño que de los registros genuinos
        impostor_data = temp1.loc[temp1.user_id != subject, :].sample(n= genuine_data.shape[0], random_state=43)
    
        #Unimos los dos anteriores variables en un solo dataset de entrenamiento del modelo
        train = pd.concat([genuine_data, impostor_data])
    
        #Obtenemos el X_train
        X_train = train.loc[:, "ft_1":"ft_60" ]
    
        #Obtenemos el y_train
        y_train = train.loc[:, "user_id"]
        
        #----------------------------------------------------------------
        
        #Generamos una copia temporal del dataset de desarrollo
        temp2 = dev_users.copy()
    
        #Reemplazamos todos los users_ids que son distintos al usuario 1 por 0
        temp2["user_id"] = temp2["user_id"].mask(temp2["user_id"] != subject, 0)
    
        #df.sample(frac=0.5, replace=True, random_state=1)
        X_dev = temp2.loc[:, "ft_1":"ft_60"]
        y_dev = temp2.loc[:, "user_id"]
    
        #----------------------------------------------------------------
    
        #Generamos una copia temporal del dataset de test
        temp3 = test_users.copy()
    
        #Reemplazamos todos los users_ids que son distintos al usuario 1 por 0
        temp3["user_id"] = temp3["user_id"].mask(temp3["user_id"] != subject, 0)
    
        X_test = temp3.loc[:, "ft_1":"ft_60"]
        y_test = temp3.loc[:, "user_id"]
        
        #----------------------------------------------------------------
        
        #Entrenamos el modelo SVM
        
        clf = RandomForestClassifier(random_state = 43, max_depth= 20, 
                                 max_features= 'auto',  min_samples_leaf= 1,
                                   min_samples_split= 3, n_estimators= 700)
        
        clf.fit(X_train,y_train)
        
        #Obtenemos probabilidades de cada registro del dataset de test
        y_prob = clf.predict_proba(X_dev)
        y_prob = pd.DataFrame(y_prob, columns = ["probImpos", "probLegi"])
    
        i = 0
    
        #Para cada registro del subdataset de test
        for index, row in dev_users.iterrows():
    
            temp_obj = {}
    
            #user id del registro actual del subdataset de test
            current_user_id = row[0]
    
            #Vector de tiempo del registro actual del subdataset de test
            current_data = row[1:]
    
            #Actual modelo del usuario a evaluar
            temp_obj["user_model"] = subject
    
            #user id del registro actual
            temp_obj["user_id"] = current_user_id
    
            #Puntaje o score del modelo
            temp_obj["score"] = y_prob.iloc[i]["probLegi"]
    
            #Normalizacion del score
            temp_obj["std_score"] = y_prob.iloc[i]["probLegi"]
    
            #Variable que indica si el registro deberia de ser clasificado como genuino o impostor
            if current_user_id == subject:
                temp_obj["y_test"] = "genuine"
            else:
                temp_obj["y_test"] = "impostor"
    
            users_evaluation_dev.append(temp_obj)
    
            i += 1
    
    users_evaluation_dev = pd.DataFrame(users_evaluation_dev)

    #Obtenemos la listas de scores de los registros que deberian de catalogarse como genuinos por los modelos
    genuine_scores_dev = list(users_evaluation_dev.loc[users_evaluation_dev.y_test == "genuine", "score"])
    
    #Obtenemos la listas de scores de los registros que deberian de catalogarse como impostores por los modelos
    impostor_scores_dev = list(users_evaluation_dev.loc[users_evaluation_dev.y_test == "impostor", "score"])
        
    #Calculo del ERR y del umbral de decisión global
    err_dev, thresh_dev = evaluate_EER_Thresh_Prob(genuine_scores_dev, impostor_scores_dev)
    
    users_evaluation_test = []
    
    #Se hace el cálculo para cada usuario
    for subject in subjects:
        
        #----------------------------------------------------------------
        #Generamos una copia temporal del dataset de entrenamiento
        temp1 = train_users.copy()
    
        #Reemplazamos todos los users_ids que son distintos al sujeto actual por 0
        temp1["user_id"] = temp1["user_id"].mask(temp1["user_id"] != subject, 0)
    
        #Obtenemos los registros considerados genuinos del entrenamiento
        genuine_data = temp1.loc[temp1.user_id == subject, :]
    
        #Obtenemos los registros considerados impostores del entrenamiento.
        #Este debe de ser del mismo tamaño que de los registros genuinos
        impostor_data = temp1.loc[temp1.user_id != subject, :].sample(n= genuine_data.shape[0], random_state=43)
    
        #Unimos los dos anteriores variables en un solo dataset de entrenamiento del modelo
        train = pd.concat([genuine_data, impostor_data])
    
        #Obtenemos el X_train
        X_train = train.loc[:, "ft_1":"ft_60" ]
    
        #Obtenemos el y_train
        y_train = train.loc[:, "user_id"]
        
        #----------------------------------------------------------------
        
        #Generamos una copia temporal del dataset de desarrollo
        temp2 = dev_users.copy()
    
        #Reemplazamos todos los users_ids que son distintos al usuario 1 por 0
        temp2["user_id"] = temp2["user_id"].mask(temp2["user_id"] != subject, 0)
    
        #df.sample(frac=0.5, replace=True, random_state=1)
        X_dev = temp2.loc[:, "ft_1":"ft_60"]
        y_dev = temp2.loc[:, "user_id"]
    
        #----------------------------------------------------------------
    
        #Generamos una copia temporal del dataset de test
        temp3 = test_users.copy()
    
        #Reemplazamos todos los users_ids que son distintos al usuario 1 por 0
        temp3["user_id"] = temp3["user_id"].mask(temp3["user_id"] != subject, 0)
    
        X_test = temp3.loc[:, "ft_1":"ft_60"]
        y_test = temp3.loc[:, "user_id"]
        
        #----------------------------------------------------------------
        
        #Entrenamos el modelo SVM
        
        clf = RandomForestClassifier(random_state = 43, max_depth= 20, 
                                 max_features= 'auto',  min_samples_leaf= 1,
                                   min_samples_split= 3, n_estimators= 700)
        
        clf.fit(X_train,y_train)
        
        #Obtenemos probabilidades de cada registro del dataset de test
        y_prob = clf.predict_proba(X_test)
        y_prob = pd.DataFrame(y_prob, columns = ["probImpos", "probLegi"])
    
        i = 0
    
        #Para cada registro del subdataset de test
        for index, row in test_users.iterrows():
    
            temp_obj = {}
    
            #user id del registro actual del subdataset de test
            current_user_id = row[0]
    
            #Vector de tiempo del registro actual del subdataset de test
            current_data = row[1:]
    
            #Actual modelo del usuario a evaluar
            temp_obj["user_model"] = subject
    
            #user id del registro actual
            temp_obj["user_id"] = current_user_id
    
            #Puntaje o score del modelo
            temp_obj["score"] = y_prob.iloc[i]["probLegi"]
    
            #Normalizacion del score
            temp_obj["std_score"] = y_prob.iloc[i]["probLegi"]
    
            #Variable que indica si el registro deberia de ser clasificado como genuino o impostor
            if current_user_id == subject:
                temp_obj["y_test"] = "genuine"
            else:
                temp_obj["y_test"] = "impostor"
    
            users_evaluation_test.append(temp_obj)
    
            i += 1
    
    users_evaluation_test = pd.DataFrame(users_evaluation_test)
    
    users_evaluation_test['y_pred'] = np.where(users_evaluation_test['score'] >= thresh_dev, 'genuine', 'impostor')
    
    #Obtenemos los y_test y y_pred de nuestros resultados
    y_test_test = users_evaluation_test.loc[:, "y_test"]
    y_pred_test = users_evaluation_test.loc[:, "y_pred"]
    
    genuine_scores_test = list(users_evaluation_test.loc[users_evaluation_test.y_test == "genuine", "score"])

    #Obtenemos la listas de scores de los registros que deberian de catalogarse como impostores por los modelos
    impostor_scores_test = list(users_evaluation_test.loc[users_evaluation_test.y_test == "impostor", "score"])
    
    thresh_x, thresh_y, _ = find_fpr_and_tpr_given_a_threshold_Prob(genuine_scores_test, impostor_scores_test, thresh_dev)
    
    thresh_std = round(thresh_dev.tolist(), 3)
    
    return y_test_test, y_pred_test ,genuine_scores_test, impostor_scores_test, thresh_std, thresh_x, thresh_y
