import os
SIZE = (112, 112)
datasetName = "datasetFULL"
path = os.path.join(os.path.dirname(__file__), datasetName)
NC = len(os.listdir(path)) # Numero de clases
TRAIN_ITEMS = 50 # Numero de imaxes que colle de cada clase para adestrar (Max 1200)
SEED = None

##############################
# Classic methods parameters #
##############################

CV = 10
#kNN
#NEIGHBORS = [3,4,5]
NEIGHBORS = [3]
# SVM
#KERNEL = ["poly", "linear"] 
# LR
#SOLVER = ['lbfgs', 'newton-cg']
SOLVER = ['lbfgs']
MAX_ITER = 10000
MULTI_CLASS = "multinomial"
#RF
#N_ESTIMATORS = [10,30,50]
#metrics = ['accuracy','recall','precision','f1','roc_auc']

##############################
#     Metrics parameters     #
##############################
# accuracy = exactitud  ( (TP+TN)/(TP+FP+FN+TN) ) ACC
# precision_score = precision ( TP/(TP+FP) ) PPV
# recall_score = sensibilidad (  TP/(TP+FN) ) TPR
# f1_score = F1 ( 2 * ( precision * sensibilidad )) / ( precision + sensibilidad )
# Especificidade = ( TN / (TN+FP) )   TNR
mparams = ['accuracy', 'precision_micro', 'recall_micro', 'f1_macro', 'roc_auc_ovr']


knnParams = {'type': "knn", 'cv': 10}
svmParams = {'type': "svm", 'cv': 10}
lrParams = {'type': "lr", 'cv': 10}
rfParams = {'type': "rf", 'cv': 10}

##############################
# Neural Networks parameters #
##############################

params0 = {'min_delta':0.002, \
    'patience':6, \
    'inputSize': None, \
    'classes': NC, \
    'type': "ANN"}

params1 = {'min_delta':0.002, \
    'patience':6, \
    'inputSize':SIZE, \
    'classes': NC,
    'type': "simple"}


# fine tunned networks parameters
params2 = {'min_delta':0.002, \
    'patience':6, \
    'inputSize':SIZE, \
    'classes': NC,
    'type': "ResNet50"}

params3 = {'min_delta':0.002, \
    'patience':6, \
    'inputSize':SIZE, \
    'classes': NC,
    'type': "VGG19"}


trainParams = {'iterations': 3, \
               'test_split': 0.1, \
               'seed': None, \
               'nc': NC, \
               'epochs': 30, \
               'batch_tam': 16, \
               'val_split': 0.1, \
               'cv': 10 }
                    

sparams = {}

# Training
#ITERATIONS=3
#NUM_DIM_ULTCAPA=1
#BASE_ROUTE=""
#LEARNING_RATE=0.002
#EPOCHS=30
#BATCH_TAM=16 # Potencia de dous!
#TRAIN_SPLIT=0.8
#VALIDATION_SPLIT=0.1
#TEST_SPLIT=0.1
#MIN_DELTA=0.002
#PATIENCE=6
#CLASES=["Estradas", "Edificios", "Libre"]
