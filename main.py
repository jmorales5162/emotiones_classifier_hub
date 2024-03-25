import os, cv2, Config
import numpy as np
import Config
from models.classifier import Classifier
from models.networks import Networks
from metrics import Metrics
from keras.utils import to_categorical
from skimage import feature 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def load_data(path):
    x = []; y = []
    for i,label in enumerate(os.listdir(path)):
        for file in (os.listdir(os.path.join(path, label))[:Config.TRAIN_ITEMS]):
            filepath = os.path.join(path,label,file)
            img = cv2.resize(cv2.imread(filepath), Config.SIZE)
            x.append(img); y.append(i)

    return (np.array(x), np.array(y))


def process_data(x,y):
    px = []; py = []
    for img in x:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.asarray(img).astype(np.float32) / 255.0
        img = feature.hog(img, feature_vector=True)  
        px.append(img)
    py = np.argmax(to_categorical(y,num_classes=Config.NC), axis=1)
    return (np.array(px), np.array(py))


if __name__ == "__main__":
    x,y = load_data(Config.path)

    px,py = process_data(x,y) # HoG vectors

    c = Classifier(Metrics(Config.mparams, Config.path, Config.SEED))
    
    # Classic models
   
    for i in Config.NEIGHBORS:
        c.setClassifier(KNeighborsClassifier(i))
        c.train(px,py,Config.knnParams)

    for i in Config.KERNEL:
        c.setClassifier(SVC(kernel=i))
        c.train(px,py,Config.svmParams)

    for i in Config.SOLVER:
        c.setClassifier(LogisticRegression(penalty=None, solver=i, max_iter=Config.MAX_ITER, multi_class=Config.MULTI_CLASS))
        c.train(px,py,Config.lrParams)

    for i in Config.N_ESTIMATORS:
        c.setClassifier(RandomForestClassifier(n_estimators=i))
        c.train(px,py,Config.rfParams)

    # Networks

    # ANN
    c.setClassifier(Networks(Config.params0, px.shape[1:]))
    c.train(px,py,Config.trainParams)

    # Simple CNN
    c.setClassifier(Networks(Config.params1))
    c.train(x,y,Config.trainParams)

    # Fine Tunning ResNet50
    c.setClassifier(Networks(Config.params2))
    c.train(x,y,Config.trainParams)
    
    # Fine Tunning VGG19
    c.setClassifier(Networks(Config.params3))
    c.train(x,y,Config.trainParams)

    c.writeResults()
    
