from models.model import Model
from skimage import feature
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import tensorflow as tf
from keras.models import Sequential
from keras import layers
import numpy as np

class Classifier(Model):

    def __init__(self, metrics):
        self.metrics = metrics
        self.metrics.initSeed()
        self.classifier = KNeighborsClassifier(5)


    def train(self, x, y, params):
        if self.classifier.__class__.__name__ == "Networks":
            for i in range(params['iterations']):
                X_train, X_test, Y_train, Y_test = \
                    train_test_split(x, y, test_size=params['test_split'], \
                    random_state=params['seed'])
                
                Y_train = tf.keras.utils.to_categorical(Y_train, num_classes=params['nc'])
                Y_test = tf.keras.utils.to_categorical(Y_test, num_classes=params['nc'])

                print("X_train: " + str(X_train.shape))
                print("Y_train: " + str(Y_train.shape))

                progress = self.classifier.net.fit(X_train, Y_train, epochs=params['epochs'], \
                    batch_size=params['batch_tam'], validation_split=params['val_split'], \
                    callbacks=self.classifier.callbacks)
                predict = self.classifier.net.predict(X_test, batch_size=8)
                self.metrics.log(np.argmax(Y_test, axis=1), np.argmax(predict,axis=1))


            self.metrics.calc_mean_dt(self.classifier.params) # Compute mean and typical deviation
            self.metrics.training_graph(progress)

        else:
            self.metrics.simpleModel(params, self.classifier)
            if self.classifier.__class__.__name__ == "SVC":
                self.classifier = CalibratedClassifierCV(self.classifier)
            y_pred = cross_val_predict(self.classifier, x, y, cv=params['cv'])
            r = cross_validate(self.classifier, x, y, cv=params['cv'], scoring=self.metrics.getMetrics())
            self.metrics.write(r, params, (y,y_pred), self.classifier)

    def setClassifier(self, newClassifier):
        self.classifier = newClassifier

    def writeResults(self):
        self.metrics.writeResults()



    def predict(self, img):
        print(obj)

