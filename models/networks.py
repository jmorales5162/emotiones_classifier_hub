import tensorflow as tf
from keras.models import Sequential
from keras import layers

class Networks:
    def __init__(self, params, input_hog=None):
        self.params = params
        self.min_delta = params['min_delta']
        self.patience = params['patience']
        self.inputSize = params['inputSize']
        self.classes = params['classes']
        self.type = params['type']
        self.net = None
        if self.type == "ANN":
            self.net = self.ann(input_hog)
        elif self.type == "simple":
            self.net = self.simple()
        elif self.type == "ResNet50":
            self.net = self.fineTunning("ResNet50")
        elif self.type == "VGG19":
            self.net = self.fineTunning("VGG19")



        self.callbacks = []
        self.callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss',
            min_delta=self.min_delta, patience=self.patience,
            verbose=1, mode='min'))

        self.callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath='./pesos.h5',
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            mode='min'))

    def fineTunning(self, base):
        if base == "ResNet50":
            baseModel = tf.keras.applications.resnet50.ResNet50(weights="imagenet",
                        input_shape=(self.inputSize[0],self.inputSize[1],3),
                        include_top=False)
        else:
            baseModel = tf.keras.applications.VGG19(weights="imagenet",
                        input_shape=(self.inputSize[0],self.inputSize[1],3),
                        include_top=False)

        for i in range(len(baseModel.layers) - 1):
            baseModel.layers[i].trainable = False

        model = baseModel.layers[len(baseModel.layers) - 2].output
        model = tf.keras.layers.Flatten()(model)
        model = tf.keras.layers.Dense(400, activation='relu')(model)
        model = tf.keras.layers.Dense(self.classes, activation='softmax')(model)
        model = tf.keras.models.Model(baseModel.input,  model)
        model.compile(optimizer="adam", \
            loss=tf.keras.losses.CategoricalCrossentropy(), \
            metrics=['accuracy'])
        return model


    def simple(self):
        model = Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', \
            input_shape=(self.inputSize[0], self.inputSize[1], 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(self.classes, activation='softmax'))
        model.compile(optimizer="adam", \
            loss=tf.keras.losses.CategoricalCrossentropy(), \
            metrics=['accuracy'])
        return model

    def ann(self, input_hog):
        model = Sequential()
        model.add(layers.Flatten(input_shape=input_hog))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(self.classes, activation='softmax'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(self.classes, activation='softmax'))
        model.compile(optimizer="adam", \
            loss=tf.keras.losses.CategoricalCrossentropy(), \
            metrics=['accuracy'])
        return model

    