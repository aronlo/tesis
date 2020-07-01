# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 04:05:14 2020

@author: Aron
"""

from utils import readDatabase, evaluate_AUC_Distance, evaluate_AUC_Prob
from models import EuclideanModel, ManhattanModel, CosineModel, SVMModel, RFModel
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, recall_score ,f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

path = "./data/grey/keystroke.db"
train_users, dev_users, test_users = readDatabase(path)
    



#------------------------------------------------------------------
#RFModel 1

y_test_test, y_pred_test, genuine_scores_test, impostor_scores_test, thresh_std, thresh_x, thresh_y = RFModel(train_users, dev_users, test_users)

report = classification_report(y_test_test, y_pred_test, labels = [ "genuine", "impostor"])
cm = confusion_matrix(y_test_test, y_pred_test, labels = [ "genuine", "impostor"])
accuracy = accuracy_score(y_test_test, y_pred_test)
recall = recall_score(y_test_test, y_pred_test, pos_label  = "genuine")
f1 = f1_score(y_test_test, y_pred_test, pos_label  = "genuine")
print("\n-------------------------------------")
print("Distancia Random Forest")
print("Accuracy", accuracy)
print("Recall", recall)
print("F1", f1)
print(report)
print(cm)
  
labels = [1] * len(genuine_scores_test) + [0] * len(impostor_scores_test)
fpr, tpr, thresholds = roc_curve(labels, genuine_scores_test + impostor_scores_test, pos_label = 1)
roc_auc = evaluate_AUC_Prob(genuine_scores_test, impostor_scores_test)

plt.plot(fpr, tpr, 'purple', label = 'RF = %0.2f' % roc_auc)
plt.scatter(thresh_x ,thresh_y, color = "purple")
#plt.text(thresh_x + 0.025, thresh_y - 0.05 , thresh_std)

#------------------------------------------------------------------
#SVM 2

y_test_test, y_pred_test, genuine_scores_test, impostor_scores_test, thresh_std, thresh_x, thresh_y = SVMModel(train_users, dev_users, test_users)

report = classification_report(y_test_test, y_pred_test, labels = [ "genuine", "impostor"])
cm = confusion_matrix(y_test_test, y_pred_test, labels = [ "genuine", "impostor"])
accuracy = accuracy_score(y_test_test, y_pred_test)
recall = recall_score(y_test_test, y_pred_test, pos_label  = "genuine")
f1 = f1_score(y_test_test, y_pred_test, pos_label  = "genuine")
print("\n-------------------------------------")
print("Distancia SVM")
print("Accuracy", accuracy)
print("Recall", recall)
print("F1", f1)
print(report)
print(cm)
  
labels = [1] * len(genuine_scores_test) + [0] * len(impostor_scores_test)
fpr, tpr, thresholds = roc_curve(labels, genuine_scores_test + impostor_scores_test, pos_label = 1)
roc_auc = evaluate_AUC_Prob(genuine_scores_test, impostor_scores_test)

plt.plot(fpr, tpr, 'red', label = 'SVM = %0.2f' % roc_auc)
plt.scatter(thresh_x ,thresh_y, color = "red")
#plt.text(thresh_x + 0.025, thresh_y - 0.05 , thresh_std)



#------------------------------------------------------------------
#Distancia Manhattan 3 

y_test_test, y_pred_test, genuine_scores_test, impostor_scores_test, thresh_std, thresh_x, thresh_y = ManhattanModel(train_users, dev_users, test_users)

report = classification_report(y_test_test, y_pred_test, labels = [ "genuine", "impostor"])
cm = confusion_matrix(y_test_test, y_pred_test, labels = [ "genuine", "impostor"])
accuracy = accuracy_score(y_test_test, y_pred_test)
recall = recall_score(y_test_test, y_pred_test, pos_label  = "genuine")
f1 = f1_score(y_test_test, y_pred_test, pos_label  = "genuine")
print("\n-------------------------------------")
print("Distancia Manhattan")
print("Accuracy", accuracy)
print("Recall", recall)
print("F1", f1)
print(report)
print(cm)
  
labels = [0] * len(genuine_scores_test) + [1] * len(impostor_scores_test)
fpr, tpr, thresholds = roc_curve(labels, genuine_scores_test + impostor_scores_test)
roc_auc = evaluate_AUC_Distance(genuine_scores_test, impostor_scores_test)

plt.plot(fpr, tpr, 'orange', label = 'Manhattan = %0.2f' % roc_auc)
plt.scatter(thresh_x ,thresh_y, color = "orange")
#plt.text(thresh_x + 0.025, thresh_y - 0.05 , thresh_std)

#------------------------------------------------------------------
#Distancia Coseno 4

y_test_test, y_pred_test, genuine_scores_test, impostor_scores_test, thresh_std, thresh_x, thresh_y = CosineModel(train_users, dev_users, test_users)

report = classification_report(y_test_test, y_pred_test, labels = [ "genuine", "impostor"])
cm = confusion_matrix(y_test_test, y_pred_test, labels = [ "genuine", "impostor"])
accuracy = accuracy_score(y_test_test, y_pred_test)
recall = recall_score(y_test_test, y_pred_test, pos_label  = "genuine")
f1 = f1_score(y_test_test, y_pred_test, pos_label  = "genuine")
print("\n-------------------------------------")
print("Distancia Coseno")
print("Accuracy", accuracy)
print("Recall", recall)
print("F1", f1)
print(report)
print(cm)
  
labels = [0] * len(genuine_scores_test) + [1] * len(impostor_scores_test)
fpr, tpr, thresholds = roc_curve(labels, genuine_scores_test + impostor_scores_test)
roc_auc = evaluate_AUC_Distance(genuine_scores_test, impostor_scores_test)

plt.plot(fpr, tpr, 'green', label = 'Coseno = %0.2f' % roc_auc)
plt.scatter(thresh_x ,thresh_y, color = "green")
#plt.text(thresh_x + 0.025, thresh_y - 0.05 , thresh_std)


#------------------------------------------------------------------
#Distancia Euclidiana 5
y_test_test, y_pred_test, genuine_scores_test, impostor_scores_test, thresh_std, thresh_x, thresh_y = EuclideanModel(train_users, dev_users, test_users)

report = classification_report(y_test_test, y_pred_test, labels = [ "genuine", "impostor"])
cm = confusion_matrix(y_test_test, y_pred_test, labels = [ "genuine", "impostor"])
accuracy = accuracy_score(y_test_test, y_pred_test)
recall = recall_score(y_test_test, y_pred_test, pos_label  = "genuine")
f1 = f1_score(y_test_test, y_pred_test, pos_label  = "genuine")
print("\n-------------------------------------")
print("Distancia Euclidiana")
print("Accuracy", accuracy)
print("Recall", recall)
print("F1", f1)
print(report)
print(cm)
  
labels = [0] * len(genuine_scores_test) + [1] * len(impostor_scores_test)
fpr, tpr, thresholds = roc_curve(labels, genuine_scores_test + impostor_scores_test)
roc_auc = evaluate_AUC_Distance(genuine_scores_test, impostor_scores_test)

plt.plot(fpr, tpr, 'blue', label = 'Euclidiana = %0.2f' % roc_auc)
plt.scatter(thresh_x ,thresh_y, color = "blue")
#plt.text(thresh_x + 0.025, thresh_y - 0.05 , thresh_std)


#------------------------------------------------------------------
#Configuraciones generales

plt.title("Curva de ROC de los modelos")
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(title='AUC')
plt.show()



#-----------------------------------------------------------------
#Rango

size = 120

plt.title("Umbras de decisión de los modelos")
plt.plot([0, 1], [0,0], color = "black")

plt.scatter(0.54080952, 0, color = "purple", label = "RF = " + str(0.54),  s= size)
plt.scatter(0.52927178, 0, color = "red", label = "SVM = " + str(0.53),  s= size)
plt.scatter(0.79826325, 0, color = "orange", label = "Manhattan = " + str(0.8),  s= size)
plt.scatter(0.93226290, 0, color = "green", label = "Coseno = " + str(0.93),  s= size)
plt.scatter(0.82878230, 0, color = "blue", label = "Euclidiana = " + str(0.83),  s= size)

plt.legend(title='Umbrales')
plt.legend(loc = 'lower right', title = "Umbrales")
plt.show()




