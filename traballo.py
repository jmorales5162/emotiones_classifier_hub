import numpy as np
import time
from datetime import datetime
import pandas as pd
import os
import scipy.stats as stats
import matplotlib.pyplot as plt
import cv2
from keras.utils import to_categorical

from skimage import feature
from skimage.io import imread
from skimage.color import rgb2gray

from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

hora_actual = []
now = datetime.now() # current date and time
hora_actual.append(now.strftime("%d/%m/%Y %H:%M:%S"))

import xlsxwriter
workbook = xlsxwriter.Workbook('Metricas.xlsx')
worksheet = workbook.add_worksheet()
workbook.close()

def load_data(path):
    X, y = [], []
    #1205 es la menor cantidad de img en cada clase
    #En este caso solo se consideran 3740 img
    # de cada clase debido a que es la clase que menos imagenes posee
    for label in CLASSES:
        for img in os.listdir(os.path.join(path, label))[:100]: # OJO for img in os.listdir(os.path.join(path, label))[:1466]:
            full_path = os.path.join(path, label, img)
            image = cv2.imread(full_path)
            image = cv2.resize(image, SIZE)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.asarray(image).astype(np.float32) / 255.0
            X.append(image)
            y.append(CLASSES.index(label))

    y = to_categorical(y, num_classes=NUM_CLASSES)
    y = np.argmax(y, axis=1)
    return np.array(X), np.array(y)


def calculate_metrics(classifier, X_train, y_train, cv):
    scoring = ['accuracy','precision_macro', 'recall_macro', 'precision_micro', 'recall_micro', 'f1_macro']
    scores = cross_validate(classifier, X_train, y_train, cv=20, scoring=scoring)
    mean_accuracy = np.mean(scores['test_accuracy'])
    mean_precision_macro = np.mean(scores['test_precision_macro'])
    mean_recall_macro = np.mean(scores['test_recall_macro'])
    mean_precision_micro = np.mean(scores['test_precision_micro'])
    mean_recall_micro = np.mean(scores['test_recall_micro'])
    mean_f1_macro = np.mean(scores['test_f1_macro'])

    metrics_dict = {
        'accuracy'          : mean_accuracy,
        'precision_macro'   : mean_precision_macro,
        'recall_macro'      : mean_recall_macro,
        'precision_micro'   : mean_precision_micro,
        'recall_micro'      : mean_recall_micro,
        'f1_macro'          : mean_f1_macro
    }

    return metrics_dict

def append_data_to_excel(excel_name, df):
    with pd.ExcelWriter(excel_name,
        mode="a",
        engine="openpyxl",
        if_sheet_exists="overlay") as writer:
        start_row = 0
        header = True
        if os.path.exists(excel_name):
            df_source = pd.read_excel(excel_name, engine="openpyxl").iloc[:,1:]
        if df_source is not None:
            n, m = df_source.shape
            header = False if n > 0 else True
            start_row = n + 1 if n > 0 else n

        df.to_excel(writer, sheet_name="Sheet1",startcol=0, startrow = start_row, header=header)

def KNN_model(vectores_hog , y_train):
    #n_neighbors = [1 , 5 , 31 , 61]
    n_neighbors = [1 , 5]
    for k in n_neighbors:
        tic = time.perf_counter()
        knn_model = KNeighborsClassifier(n_neighbors=k)
        metrics_knn = calculate_metrics(knn_model, vectores_hog, y_train, cv=20)
        toc = time.perf_counter()

        data_knn_model = {
            'accuracy'          : metrics_knn['accuracy'],
            'precision_macro'   : metrics_knn['precision_macro'],
            'recall_macro'      : metrics_knn['recall_macro'],
            'precision_micro'   : metrics_knn['precision_micro'],
            'recall_micro'      : metrics_knn['recall_micro'],
            'f1_macro'          : metrics_knn['f1_macro'],
            'tiempo (s)'        : [round((toc-tic), 2)]
        }

        df_knn_model = pd.DataFrame(data_knn_model, index = ["KNN n_neighbors = "+str(k)])
        append_data_to_excel('./Metricas.xlsx', df_knn_model)

        print('Metrics for KNN with k =', k)
        print(metrics_knn)

    return data_knn_model['accuracy']


def LR_model(vectores_hog , y_train):  
    solver = ['lbfgs', 'newton-cg'] 
    for label in solver:
      tic = time.perf_counter()
      LR_model = LogisticRegression(penalty=None, solver = label, max_iter=10000, multi_class='multinomial')
      #LR_model = LogisticRegression(penalty="l1", solver = label, max_iter=5000, multi_class='multinomial')
      metrics_lr = calculate_metrics(LR_model, vectores_hog, y_train, cv=20)
      toc = time.perf_counter()

      data_lr_model = {
          'accuracy'          : metrics_lr['accuracy'],
          'precision_macro'   : metrics_lr['precision_macro'],
          'recall_macro'      : metrics_lr['recall_macro'],
          'precision_micro'   : metrics_lr['precision_micro'],
          'recall_micro'      : metrics_lr['recall_micro'],
          'f1_macro'          : metrics_lr['f1_macro'],
          'tiempo (s)'        : [round((toc-tic), 2)]
      }

      df_lr_model = pd.DataFrame(data_lr_model, index = ["LR con solver = " + label])
      append_data_to_excel('./Metricas.xlsx', df_lr_model)

      print('Metrics for LR model:')
      print(df_lr_model)

    return data_lr_model['accuracy']

def SVM_model(vectores_hog , y_train): 
    kernel = ['linear', 'poly']
    for label in kernel:
        tic = time.perf_counter()
        svm_model = svm.SVC(kernel=label)
        metrics_svm = calculate_metrics(svm_model, vectores_hog, y_train, cv=20)
        toc = time.perf_counter()
        data_svm_model = {
            'accuracy'          : metrics_svm['accuracy'],
            'precision_macro'   : metrics_svm['precision_macro'],
            'recall_macro'      : metrics_svm['recall_macro'],
            'precision_micro'   : metrics_svm['precision_micro'],
            'recall_micro'      : metrics_svm['recall_micro'],
            'f1_macro'          : metrics_svm['f1_macro'],
            'tiempo (s)'        : [round((toc-tic), 2)]
        }
        df_svm_model = pd.DataFrame(data_svm_model, index = ["SVM para kernel " + label])
        append_data_to_excel('./Metricas.xlsx', df_svm_model)

        print('Metrics for SVM:')
        print(df_svm_model)
    return data_svm_model['accuracy']

def Random_Forest_model(vectores_hog, y_train):
    estimators = [10, 50]  
    for label in estimators:
        tic = time.perf_counter()
        rf_model = RandomForestClassifier(n_estimators=label)
        metrics_rf = calculate_metrics(rf_model, vectores_hog, y_train, cv=20)
        toc = time.perf_counter()

        data_rf_model = {
            'accuracy'          : metrics_rf['accuracy'],
            'precision_macro'   : metrics_rf['precision_macro'],
            'recall_macro'      : metrics_rf['recall_macro'],
            'precision_micro'   : metrics_rf['precision_micro'],
            'recall_micro'      : metrics_rf['recall_micro'],
            'f1_macro'          : metrics_rf['f1_macro'],
            'tiempo (s)'        : [round((toc-tic), 2)]
        }

        df_rf_model = pd.DataFrame(data_rf_model, index = ["Random Forest n_estimators = "+str(label)])
        append_data_to_excel('./Metricas.xlsx', df_rf_model)

        print('Metrics for Random Forest with n_estimators =', label)
        print(metrics_rf)

    return data_rf_model['accuracy']




TRAIN_PATH = './dataset'
SIZE = (112, 112)
CLASSES = []
for class_ in os.listdir(TRAIN_PATH):
    CLASSES.append(class_)
NUM_CLASSES = len(CLASSES)
print(CLASSES)

X_train, y_train = load_data(TRAIN_PATH)
cant_imag , ancho, alto = X_train.shape

vectores_hog=[]
for i in range(cant_imag):
    vector_caracteristicas = feature.hog(X_train[i], feature_vector=True)
    vectores_hog.append(vector_caracteristicas)
vectores_hog = np.array(vectores_hog)


print(vectores_hog.shape)
accuracy_test_RF = Random_Forest_model(vectores_hog,y_train)
#DUDA : accuracy, recall_macro ,precision_micro y recall_micro ,me dan exactamente iguales

accuracy_test_KNN = KNN_model(vectores_hog,y_train)
print("precision = " +str(accuracy_test_KNN))
#accuracy_test_SVM = SVM_model(vectores_hog , y_train)
#accuracy_test_LR = LR_model(vectores_hog , y_train)
#print(type(accuracy_test_RF))
#acc_KNN = []
#acc_KNN.append(accuracy_test_KNN)
#acc_LR = []
#acc_LR.append(accuracy_test_LR)
#acc_SVM = []
#acc_SVM.append(accuracy_test_SVM)
#acc_RF =[]
#acc_RF.append(accuracy_test_RF)
#data = [acc_LR, acc_KNN,acc_SVM,acc_RF]
#fig7, ax = plt.subplots()
#ax.set_title('Modelos')
#ax.boxplot(data,labels=['LR','KNN','SVM','RF']);
