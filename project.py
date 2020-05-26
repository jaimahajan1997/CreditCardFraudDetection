# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 12:37:07 2019

@author: Jai Mahajan, Aditya Aggarwal
"""

import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import *
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler,StandardScaler,label_binarize
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score,f1_score,roc_curve,auc,classification_report,cohen_kappa_score
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split,learning_curve
import time
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.preprocessing import *
from sklearn.linear_model import *
from sklearn.neural_network import *
from sklearn.linear_model import LogisticRegression
import scikitplot
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import axes3d, Axes3D
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from imblearn.under_sampling import RandomUnderSampler
########################################################################## FUNCTIONS ###################################################################

def Plot_PCA(X, y,title):   
    X = PCA(n_components=2).fit_transform(X)
    plt.scatter(X[y==0, 0],X[y==0, 1],c='r', s=5, alpha=0.6,label='Non-fraud')
    plt.scatter(X[y==1, 0],X[y==1, 1],c='g',s=5, alpha=0.6,label='fraud')    
    plt.title(title)
    plt.legend(loc='upper left')
    plt.show()

def plot_tsne(X, y,title):
    X = TSNE(n_components=2).fit_transform(X)
    print(X.shape)
    plt.scatter(X[y==0, 0],X[y==0, 1],c='r', s=5, alpha=0.6, label='Non-fraud')
    plt.scatter(X[y==1, 0],X[y==1, 1],c='g', s=5, alpha=0.6, label='fraud')
    plt.legend(loc='upper left')
    plt.title(title)
    plt.show()
    
def plot_learning_curve():
    # Ref : https://scikit-learn.org/stable/modules/learning_curve.html
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    
################################################################### VISUALIZING DATA ###################################################################    
    
df = pd.read_csv('data.csv')
data = df.iloc[:,:-1].to_numpy()
labels = df.iloc[:,-1].to_numpy()
scaler = MinMaxScaler(feature_range=(0,1))
data = scaler.fit_transform(data)
X_train,X_test,y_train,y_test=train_test_split(data,labels,test_size=0.2,random_state=50,stratify=labels)

smote_oversampling = SMOTE(random_state=132,n_jobs=-1)
X_equalized, y_equalized = smote_oversampling.fit_resample(X_train,y_train)

# Ref :https://imbalanced-learn.readthedocs.io/en/stable/under_sampling.html
undersampling=RandomUnderSampler(random_state=60)
X_under,y_under = undersampling.fit_resample(X_train,y_train)

# Visualizing PCA
Plot_PCA(X_under,y_under,'PCA undersampled')

# Visualizing tSNE
plot_tsne(X_under,y_under,'tSNE undersampled')

################################################################### PRE-MIDSEM MODELS #####################################################################
# Reference : https://www.kaggle.com/klaudiajankowska/binary-classification-methods-comparison

classifier = GaussianNB()
t1 = time.clock()
classifier.fit(X_train,y_train)
t2 = time.clock()
print(t2-t1)
accuracy_Gaussian = classifier.score(X_test,y_test)
F1_Gaussian = f1_score(y_test,classifier.predict(X_test))
kappa_Gaussian = cohen_kappa_score(y_test,classifier.predict(X_test))
print(classification_report(y_test, classifier.predict(X_test), target_names=['class 0','class 1']))
Roc_gaussian = scikitplot.metrics.plot_roc(y_test,classifier.predict_proba(X_test))
kappa_gaussian = cohen_kappa_score(y_test,classifier.predict(X_test))

classifier = LogisticRegression()
t1 = time.clock()
classifier.fit(X_train,y_train)
t2 = time.clock()
print(t2-t1)
accuracy_Logistic = classifier.score(X_test,y_test)
F1_Logistic = f1_score(y_test,classifier.predict(X_test))
print(classification_report(y_test, classifier.predict(X_test), target_names=['class 0','class 1']))
Roc_Logistic = scikitplot.metrics.plot_roc(y_test,classifier.predict_proba(X_test))
kappa_Logistic = cohen_kappa_score(y_test,classifier.predict(X_test))

#################################################################### POST-MIDSEM MODELS ###############################################################

# KNN without SMOTE
print("KNN without SMOTE")
classifier = KNeighborsClassifier()
t1 = time.clock()
classifier.fit(X_train,y_train)
t2 = time.clock()
print(t2-t1)
pred_knn = classifier.predict(X_test)
F1_KNN = f1_score(y_test,pred_knn,average='weighted')
accuracy_KNN = accuracy_score(pred_knn,y_test)
print(classification_report(y_test, pred_knn, target_names=['class 0','class 1']))
kappa_KNN = cohen_kappa_score(y_test,pred_knn)

# MLP without SMOTE
print("MLP without SMOTE")

NN_sklearn = MLPClassifier(activation='relu', alpha=0.01,hidden_layer_sizes=(100,80,50), random_state=6,verbose=1)
t1 = time.clock()
NN_sklearn.fit(X_train,y_train)
t2 = time.clock()
print(t2-t1)
accuracy_MLP = NN_sklearn.score(X_test,y_test)
F1_MLP = f1_score(y_test,NN_sklearn.predict(X_test),average='weighted')
nn_pred = NN_sklearn.predict(X_test)
print(classification_report(y_test, nn_pred))
kappa_MLP = cohen_kappa_score(y_test,nn_pred)

# RandomForest without SMOTE
print("RandomForest without SMOTE")

classifier = RandomForestClassifier()
t1 = time.clock()
classifier.fit(X_train,y_train)
t2 = time.clock()
print(t2-t1)
accuracy_RF = classifier.score(X_test,y_test)
rf_pred = classifier.predict(X_test)
F1_RF = f1_score(y_test,rf_pred,average='weighted')
print(classification_report(y_test, rf_pred))
kappa_RF = cohen_kappa_score(y_test,rf_pred)

# KNN with SMOTE
print("KNN with SMOTE")

classifier = KNeighborsClassifier()
t1 = time.clock()
classifier.fit(X_equalized,y_equalized)
t2 = time.clock()
print(t2-t1)
pred_knn = classifier.predict(X_test)
F1_KNN_SMOTE = f1_score(y_test,pred_knn,average='weighted')
print(classification_report(y_test, pred_knn, target_names=['class 0','class 1']))
kappa_KNN_SMOTE = cohen_kappa_score(y_test,pred_knn)

# MLP with SMOTE
print("MLP with SMOTE")

NN_sklearn = MLPClassifier(activation='relu', alpha=0.01,hidden_layer_sizes=(100,80,50), random_state=77,verbose=1)
t1 = time.clock()
NN_sklearn.fit(X_equalized,y_equalized)
t2 = time.clock()
print(t2-t1)
accuracy_MLP_SMOTE = NN_sklearn.score(X_test,y_test)
pred_mlp = NN_sklearn.predict(X_test)
F1_MLP_SMOTE = f1_score(y_test,pred_mlp,average='weighted')
print(classification_report(y_test,pred_mlp))
kappa_MLP_SMOTE = cohen_kappa_score(y_test,pred_mlp)

# RandomForest with SMOTE
print("RandomForest with SMOTE")

classifier = RandomForestClassifier(n_estimators=170)
t1 = time.clock()
classifier.fit(X_equalized,y_equalized)
t2 = time.clock()
print(t2-t1)
accuracy_RF_SMOTE = classifier.score(X_test,y_test)
rf_pred = classifier.predict(X_test)
F1_RF_SMOTE = f1_score(y_test,rf_pred,average='weighted')
print(classification_report(y_test, rf_pred))
kappa_RF_SMOTE = cohen_kappa_score(y_test,rf_pred)

# RandomForest with Undersampling 
print("RandomForest with Undersampling")

classifier = MLPClassifier(activation='relu', alpha=0.01,hidden_layer_sizes=(1000,500,100,50), random_state=77,verbose=1)
t1 = time.clock()
classifier.fit(X_under,y_under)
t2 = time.clock()
print(t2-t1)
accuracy_RF_UNDER = classifier.score(X_test,y_test)
rf_pred = classifier.predict(X_test)
F1_RF_UNDER = f1_score(y_test,rf_pred,average='weighted')
print(classification_report(y_test, rf_pred))
kappa_RF_UNDER = cohen_kappa_score(y_test,rf_pred)



















