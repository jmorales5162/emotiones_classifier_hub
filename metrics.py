import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import os, random, shutil
from datetime import datetime
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
matplotlib.use('TkAgg')


class Metrics:
    def __init__(self, params, path, seed):
        self.seed = seed
        self.listMetrics = params
        self.metricas = {"ACC": np.array([]), "TN": np.array([]), "FP": np.array([]),
                        "FN": np.array([]), "TP": np.array([]), "TPR": np.array([]), 
                        "TNR": np.array([]), "PPV": np.array([]), "F1": np.array([]),
                        "NPV": np.array([]), "FPR": np.array([]), "FNR": np.array([]), 
                        "FDR": np.array([]) } 
        self.resultados = [] # Metricas por cada clase
        self.path = None
        self.kappas = np.array([])
        self.clases = os.listdir(path)
        for clase in self.clases:
            self.resultados.append(self.metricas.copy())
        self.y_orixinal = None
        self.y_predecida = None
        self.folderName = "training_"+datetime.now().strftime("%d_%m_%Y_%H_%M")
        self.cwd = os.path.dirname(__file__)
        self.accuracyBoxPlot = []
        self.listResults = []
        os.mkdir(os.path.join(self.cwd, self.folderName))


    def log(self, y_orixinal, y_predecida):
        self.y_orixinal = y_orixinal; self.y_predecida = y_predecida
        cnf_matrix = self.compute_confusion_matrix(y_orixinal, y_predecida)
        metrica = self.compute_total_classification_metrics(cnf_matrix)
        self.kappas = np.append(self.kappas, cohen_kappa_score(y_orixinal, y_predecida))
        for i,key in enumerate(self.metricas):
            for j,clase in enumerate(self.clases):
                self.resultados[j][key] = np.append(self.resultados[j][key], metrica[i][j])


    def calc_mean_dt(self,params):
        print(self.resultados)
        medias = []; dt = []
        for clase in self.clases:
            medias.append(np.array([])); dt.append(np.array([]));
        for metrica in self.metricas:
            for i,clase in enumerate(self.clases):
                medias[i] = np.append(medias[i], np.average(self.resultados[i][metrica]))
                dt[i] = np.append(dt[i], np.std(self.resultados[i][metrica]))
        
        metrics = {}
        metrics['Nombre'] = params['type']
        metrics['Tiempo (s)'] = 0
        metrics['Exactitud'] = np.mean( [self.resultados[n]['ACC'] for n in range(len(self.clases))] ) #np.mean(r['test_accuracy'])
        metrics['Precision'] = np.mean( [self.resultados[n]['PPV'] for n in range(len(self.clases))] ) #r.pop('test_precision_micro', None))
        metrics['Sensibilidad'] = np.mean( [self.resultados[n]['TPR'] for n in range(len(self.clases))] ) #np.mean(r.pop('test_recall_micro', None))
        metrics['F1'] = np.mean( [self.resultados[n]['F1'] for n in range(len(self.clases))] ) #np.mean(r.pop('test_f1_macro', None))
        metrics['Especificidad'] = np.mean( [self.resultados[n]['TNR'] for n in range(len(self.clases))] ) #r.pop('test_precision_micro', None))#np.mean(TNR)
        
        df = None
        if params['type'] == "ANN":
            df = pd.DataFrame(metrics, index = ["ANN"])
        if params['type'] == "simple":
            df = pd.DataFrame(metrics, index = ["simple"])
        if params['type'] == "ResNet50":
            df = pd.DataFrame(metrics, index = ["ResNet50"])
        if params['type'] == "VGG19":
            df = pd.DataFrame(metrics, index = ["VGG19"])
        self.listResults.append(df)

        #df.to_excel(os.path.join(self.path, self.folderName, 'metrics.xlsx'))


        self.path = os.path.join(self.cwd, self.folderName, params['type'])
        os.mkdir(self.path)
        f = open(os.path.join(self.path,"metricas.txt"), "w")
        #df = pd.DataFrame()
        #df = pd.concat([df, ])
        f.write("Metricas acadadas no adestramento (promedio de iteracions)\n")
        for ic, clase in enumerate(self.clases):
            f.write("\nClase (" + clase + "): \n\t\t\t media\t\t\t\tdesviacion tipica\n-------------------------------------------\n")
            for metrica in self.metricas:
                f.write("\t" + metrica + ":  " + str(medias[ic][list(self.metricas.keys()).index(metrica)]))
                f.write(",\t" + str(dt[ic][list(self.metricas.keys()).index(metrica)]) + "\n")
        
        f.write("\n\nKappa: media = " + str(np.average(self.kappas)) + ", desviacion tipica: " + str(np.std(self.kappas)))
        f.close()
        df.to_excel(os.path.join(self.path, 'metrics.xlsx'))
        tmp = np.array([self.resultados[n]['ACC'] for n in range(len(self.clases))])
        print("tmp: " + str(tmp))
        acc = tuple([np.mean(tmp[:,j]) for j in range(len(tmp[0]))])
        self.accuracyBoxPlot.append(( acc , params['type']))
        

    def compute_total_classification_metrics(self, cnf_matrix):
        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)
        FP = FP.astype(float);FN = FN.astype(float);TP = TP.astype(float);TN = TN.astype(float)
        
        # Sensibilidade, "hit rate", "recall", ou "true positive rate"
        TPR = TP / (TP + FN)
        # Especificidade ou "true negative rate"
        TNR = TN / (TN + FP)
        # Precision ou "positive predictive value"
        PPV = TP / (TP + FP)         
        # F1
        F1 = 2 * (PPV * TPR) / (PPV + TPR)
        # "Negative predictive value"
        NPV = TN / (TN + FN)
        # "Fall out" ou "false positive rate"
        FPR = FP / (FP + TN)
        # "False negative rate"
        FNR = FN / (TP + FN)
        # "False discovery rate"
        FDR = FP / (TP + FP)
        # Precision promedia
        ACC = (TP + TN) / (TP + FP + FN + TN)
        return [ACC, TN, FP, FN, TP, TPR, TNR, PPV, F1, NPV, FPR, FNR, FDR]

    def calc_smetrics(self, cnf_matrix):
        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)
        FP = FP.astype(float);FN = FN.astype(float);TP = TP.astype(float);TN = TN.astype(float)
        # Especificidade ou "true negative rate"
        TNR = TN / (TN + FP)



    def compute_confusion_matrix(self, test_orig, test_predicted):
        print("test_orig: " + str(test_orig) + "   //   " + "test_predicted" + str(test_predicted))
        num_classes = len(np.unique(test_orig))
        matrix = np.zeros((num_classes,num_classes), int)
    
        for t1, t2 in zip(test_orig,test_predicted):
            matrix[t1,t2] += 1
        print(matrix)
        return matrix


    def initSeed(self):
        if self.seed == None:
            self.seed = np.random.randint(1, 255)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        import tensorflow as tf
        if tf.__version__ < '2.0.0':
            tf.set_random_seed(self.seed)
        else:
            import tensorflow.compat.v1 as tf
            tf.set_random_seed(self.seed)
        from tensorflow.python.keras import backend as K
        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)
        with open(os.path.join(os.path.dirname(__file__),"session_seed.txt"), 'w') as seed_file:
            seed_file.write(str(self.seed) + '\n')
            seed_file.close()

    def simpleModel(self, params, classifier):
        print(classifier.__class__.__name__)
        if classifier.__class__.__name__ == "KNeighborsClassifier":
            self.path = os.path.join(self.cwd, self.folderName, params['type']+str(classifier.n_neighbors))
        elif classifier.__class__.__name__ == "SVC":
            self.path = os.path.join(self.cwd, self.folderName, params['type']+str(classifier.kernel))
        elif classifier.__class__.__name__ == "LogisticRegression":
            self.path = os.path.join(self.cwd, self.folderName, params['type']+str(classifier.solver))
        elif classifier.__class__.__name__ == "RandomForestClassifier":
            self.path = os.path.join(self.cwd, self.folderName, params['type']+str(classifier.n_estimators))
        os.mkdir(self.path)

    def write(self, r, params, test, classifier):
        cnf_matrix = self.compute_confusion_matrix(test[0], test[1])
        self.gardar_cm(test[0], test[1])
        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)
        FP = FP.astype(float);FN = FN.astype(float);TP = TP.astype(float);TN = TN.astype(float)
        # Especificidade ou "true negative rate"
        TNR = TN / (TN + FP)
        metrics = {}
        metrics['Nombre'] = params['type']
        metrics['Tiempo (s)'] = np.mean(r.pop('fit_time', None)); r.pop('score_time', None)
        metrics['Exactitud'] = np.mean(r['test_accuracy'])
        metrics['Precision'] = np.mean(r.pop('test_precision_micro', None))
        metrics['Sensibilidad'] = np.mean(r.pop('test_recall_micro', None))
        metrics['F1'] = np.mean(r.pop('test_f1_macro', None))
        metrics['Especificidad'] = np.mean(TNR)
        
        df = None
        print(classifier.__class__.__name__)
        if classifier.__class__.__name__ == "KNeighborsClassifier":
            df = pd.DataFrame(metrics, index = ["KNN n_neighbors = "+str(classifier.n_neighbors)])
            self.accuracyBoxPlot.append((r['test_accuracy'], params['type']+str(classifier.n_neighbors)))
        elif classifier.__class__.__name__ == "CalibratedClassifierCV":
            df = pd.DataFrame(metrics, index = ["SVM Kernel = "+str(classifier.estimator.kernel)])
            self.accuracyBoxPlot.append((r['test_accuracy'], params['type']+"-"+str(classifier.estimator.kernel)))
        elif classifier.__class__.__name__ == "LogisticRegression":
            df = pd.DataFrame(metrics, index = ["LR solver = "+str(classifier.solver)])
            self.accuracyBoxPlot.append((r['test_accuracy'], params['type']+"-"+str(classifier.solver)))
        elif classifier.__class__.__name__ == "RandomForestClassifier":
            df = pd.DataFrame(metrics, index = ["RF = "+str(classifier.n_estimators)])
            self.accuracyBoxPlot.append((r['test_accuracy'], params['type']+str(classifier.n_estimators)))

        self.listResults.append(df)
        df.to_excel(os.path.join(self.path, 'metrics.xlsx'))

    def writeResults(self):
        path = os.path.join(self.cwd, self.folderName, 'Metricas.xlsx')
        pd.concat(self.listResults).to_excel(path)
        fig, ax = plt.subplots()
        results, models = zip(*self.accuracyBoxPlot)
        ax.set_title('Modelos')
        ax.boxplot(list(results))
        ax.set_xticklabels(list(models), rotation=315)
        fig.savefig(os.path.join(self.cwd, self.folderName, 'boxplot.png'))

    def gardar_cm(self, test_orig, test_predicted):
        plt.figure(figsize=(10,6));
        import seaborn as sns
        cm = confusion_matrix(test_orig, test_predicted)
        #print(cm)
        graf = sns.heatmap(cm, annot=True, cmap="Blues", fmt='g')
        graf.set_title("Matriz de confusion\n")
        graf.set_xlabel("\nValores predecidos")
        graf.set_ylabel("\nValores actuais")
        graf.xaxis.set_ticklabels(self.clases)
        graf.yaxis.set_ticklabels(self.clases)

        plt.savefig(os.path.join(self.path, "matriz_cm.png"))
        #plt.show()

    def training_graph(self, progress):
        loss = progress.history["loss"]
        val_loss = progress.history["val_loss"]
        epochs = range(1, len(loss) + 1)
        acc = progress.history['accuracy']
        val_acc = progress.history['val_accuracy']

        plt.figure(figsize=(15,5));
        plot1 = plt.subplot2grid((1,2), (0,0), colspan=1)
        plot2 = plt.subplot2grid((1,2), (0,1), colspan=1)
        plot1.plot(epochs, loss, 'y', label='loss adestramento')
        plot1.plot(epochs, val_loss, 'r', label='loss validacion')
        plot1.set_xlabel('Epochs'); plot1.set_ylabel('Loss')
        plot1.legend()
        
        plot2.plot(epochs, acc, 'y', label='Precision adestramento')
        plot2.plot(epochs, val_acc, 'r', label='Precision validacion')
        plot2.set_xlabel('Epochs'); plot2.set_ylabel('Precision')
        plot2.legend()
        plt.savefig(os.path.join(self.path, "graf.png"))
        plt.clf()
        self.gardar_cm(self.y_orixinal, self.y_predecida)


    def getMetrics(self):
        return self.listMetrics



