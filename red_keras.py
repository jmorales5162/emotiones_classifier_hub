import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime

from random import seed
from random import randint
from keras import layers

import scipy.stats as stats
import cv2
from keras.utils import to_categorical

from skimage import feature
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.transform import resize

from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.metrics import ConfusionMatrixDisplay,confusion_matrix
from sklearn.model_selection import cross_validate

from keras.models import Sequential, Model
from keras import optimizers
from keras.applications import vgg16, mobilenet, resnet, xception
from keras.layers import Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Precision, Recall, TrueNegatives, FalsePositives, AUC
from tensorflow.keras import callbacks

from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison

import warnings
warnings.filterwarnings('ignore')
import random
IMG_SIZE = 112

def create_excel(func):
    def wrapper(excel_name, df):
        if not create_excel_file(excel_name):
            return  # No proceder si la creación del archivo falla
        return func(excel_name, df)
    return wrapper

def create_excel_file(excel_name):
    cwd = os.getcwd()  # Obtiene el directorio de trabajo actual
    print("Directorio de trabajo:\n", cwd)
    file_path = os.path.join(cwd, excel_name)  # Ruta del archivo Excel
    if os.path.exists(file_path):
        print("El archivo ya existe en el directorio.")
    else:
        try:
            # Crear un archivo Excel vacío
            with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
                pass  # No hace nada, simplemente crea el archivo
            print("Archivo Excel creado exitosamente en el directorio.")
        except Exception as e:
            print("Error al crear el archivo Excel:", e)
            return False
    return True

@create_excel
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

# Función para graficar progreso durante el entrenamiento de la red
def plot_history(history):
    plt.figure(figsize=(12,5))
    plt.plot(history.history['accuracy'], label='Entrenamiento')
    plt.plot(history.history['val_accuracy'], label = 'Validación')
    plt.xlabel('Iteración (epoch)')
    plt.ylabel('Exactitud (accuracy)')
    plt.ylim([0, 1])
    plt.grid()
    #plt.title('Modelo '+str(i+1))
    plt.title('Modelo '+str(1))
    plt.legend(loc='lower right')
    plt.show()
 

def load_data(path):
  X, y = [], []
  #1205 es la menor cantidad de img en cada clase
  for label in CLASSES:
      for img in os.listdir(os.path.join(path, label))[:50]: # OJO for img in os.listdir(os.path.join(path, label))[:1205]:
          full_path = os.path.join(path, label, img)
          image = cv2.imread(full_path)
          image = cv2.resize(image, (IMG_SIZE,IMG_SIZE))
          #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
          image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  #verificar si las imagenes estan en rgb
          image = np.asarray(image).astype(np.float32) / 255.0
          X.append(image)
          y.append(CLASSES.index(label))

  y = to_categorical(y, num_classes=NUM_CLASSES)
  #y = np.argmax(y, axis=1)
  return np.array(X), np.array(y)

#Distintas arquitectura para salida de la red neuronal
data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical")
])

def cnn4(NUM_CLASSES):
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(NUM_CLASSES))

    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    return model

TRAIN_PATH = './dataset'
CLASSES = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
NUM_CLASSES = len(CLASSES)
print(CLASSES)
X_train, y_train = load_data(TRAIN_PATH)
print(X_train.shape , y_train.shape)
train_images , test_images , train_labels , test_labels = train_test_split(X_train, y_train, test_size=0.3,shuffle=True)
y_train_labels = np.argmax(train_labels, axis=1)

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(CLASSES[y_train_labels[i]])
plt.show()

model4 = cnn4(NUM_CLASSES)
model4.summary()
# Entrenamiento CNN_4

epochs = 1
batch_size = 100
history , time_ = [[] for _ in range(2)]  #se crean dos listas vacias
n_iter_no_change = 3

min_delta = 0.5  # Definir el mínimo cambio deseado en la métrica

earlystop_callback = callbacks.EarlyStopping(monitor='val_accuracy', verbose=1, restore_best_weights=True, patience=n_iter_no_change, min_delta=min_delta)
#earlystop_callback = callbacks.EarlyStopping(monitor='val_accuracy', verbose = 1, restore_best_weights = True, patience=n_iter_no_change)

train_labels_ = np.argmax(train_labels, axis=1)
test_labels_  = np.argmax(test_labels , axis=1)

tic_train = time.time()
history = model4.fit(train_images, train_labels_, epochs=epochs, batch_size=batch_size, validation_data=(test_images, test_labels_) , callbacks = [earlystop_callback])
toc_train = time.time()
train_time = toc_train - tic_train

# Evaluación
tic_eval = time.time()
test_pred_probs = model4.predict(test_images)
test_pred = np.argmax(test_pred_probs, axis=1)
toc_eval = time.time()
eval_time = toc_eval - tic_eval

# Métricas
accuracy = accuracy_score(test_labels_, test_pred)
recall = tf.keras.metrics.Recall()(test_labels_, test_pred).numpy()
precision = tf.keras.metrics.Precision()(test_labels_, test_pred).numpy()
f1_score = 2 * (precision * recall) / (precision + recall)
specificity = tf.keras.metrics.SpecificityAtSensitivity(0.5)(test_labels, test_pred_probs).numpy() #este valor se ajusta SpecificityAtSensitivity(0.5)
auc = tf.keras.metrics.AUC()(test_labels, test_pred_probs).numpy()

test_data = {
        'base model'    : "cnn4",
        'out layers'    : 3,  #estas son las tres utlias capas de la estrcutura de la red que escogimos
        'extra params'  : model4.count_params(),
        'epochs'        : epochs,
        'batch size'    : batch_size,
        'accuracy'      : round(accuracy,2),
        'recall'        : round(recall,2),
        'specificity'   : round(specificity,2),
        'precision'     : round(precision,2),
        'f1_score'      : round(f1_score,2),
        'auc'           : round(auc,2),
        'train time'    : round(train_time,2),
        'eval time'     : round(eval_time,2)
        }

now = datetime.now() # current date and time
time_.append(now.strftime("%d/%m/%Y %H:%M:%S"))
df = pd.DataFrame(test_data, index = time_)

excel_name = "./Metricas_CNN.xlsx"
append_data_to_excel(excel_name, df)

print(df)
plot_history(history)

t_test_ = np.argmax(test_labels, axis=1)
t_pred = model4.predict(test_images)
t_pred_ = np.argmax(t_pred , axis=1)
cm = confusion_matrix(t_test_, t_pred_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=CLASSES)
disp.plot()
test_labels_ = np.argmax(test_labels, axis=1)


plt.figure(figsize=(12,12))
for i in range(25):
    index = random.randint(0, test_images.shape[0]) # Se elige un número de imagen al azar
    image = test_images[index:index+1]

    plt.subplot(5, 5, i+1)
    plt.imshow(image[0])       # Se muestra la imagen elegida al azar
    plt.axis('off')

    pred = model4.predict(image)  # Se obtiene la predicción del modelo para la imagen elegida
    class_pred = np.argmax(pred)

    if test_labels_[index] == class_pred:   # Si hay acierto en la clase predicha: se muestra en el título solo el nombre de clase
        print()
        plt.title(CLASSES[class_pred])
    else:                                  # Si hay un error en la clasificación: se muestra en rojo en el título ambas clases
        print()
        plt.title(CLASSES[class_pred] + "!=" + CLASSES[test_labels_[index]], color='#ff0000')
plt.show()
