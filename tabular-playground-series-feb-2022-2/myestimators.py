from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from keras.layers.advanced_activations import PReLU
from keras.layers.core import Activation, Dense, Dropout
from tensorflow.keras.layers import BatchNormalization
from keras.models import Sequential, load_model
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
import os
import matplotlib.pyplot as plt


class NN(BaseEstimator, ClassifierMixin):
    def __init__(self, patience=10, layers=3, dropout=0.1, units=100, n_epoch=100, batch_size=256):
        self.patience = patience
        self.layers = layers
        self.dropout = dropout
        self.units = units
        self.n_epoch = n_epoch
        self.batch_size = batch_size

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        model = Sequential()
        model.add(Dense(self.units, input_shape=(X.shape[1], )))
        model.add(PReLU())
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout))

        for l in range(self.layers - 1):
            model.add(Dense(self.units))
            model.add(PReLU())
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout))

        model.add(Dense(self.classes_.size))
        model.add(Activation('softmax'))
        early_stopping = EarlyStopping(
            monitor='loss', patience=self.patience, verbose=0, restore_best_weights=True)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adagrad',  metrics=['accuracy'])

        self.history = model.fit(X, np_utils.to_categorical(y, num_classes=self.classes_.size),
                                 epochs=self.n_epoch, batch_size=self.batch_size, verbose=0, callbacks=[early_stopping])
        self.model = model
        return self

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        pred = self.model.predict(X)
        return pred

    def plot(self):
        plt.plot(self.history.epoch,
                 self.history.history["accuracy"], label="acc")
        plt.plot(self.history.epoch,
                 self.history.history["loss"], label="loss")
        plt.xlabel("epoch")
        plt.legend()
