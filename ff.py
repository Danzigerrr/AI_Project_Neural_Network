from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from eli5.sklearn import PermutationImportance
from sklearn.model_selection import train_test_split
from keras.models import Sequential # for creating a linear stack of layers for our Neural Network 
from keras import Input # for instantiating a keras tensor 
from keras.layers import Dense # for creating regular densely-connected NN layer.
import math
import pandas as pd
import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
from variables import labels, data, names
import shap
import tensorflow as tf
from metrics import permutationImportance, plotImportant

from variables import X_train, X_test, y_train, y_test

def create_model(d):
    #d --> data
    model = Sequential(name="A")
    model.add(Input(shape=(d.shape[1],), name="Input"))
    model.add(Dense(math.sqrt(d.shape[1]), activation='sigmoid', use_bias=True, name='Hidden-Layer'))
    model.add(Dense(math.sqrt(d.shape[1])/2, activation='sigmoid', use_bias=True, name='Hidden1'))
    model.add(Dense(math.sqrt(d.shape[1])/4, activation='sigmoid', use_bias=True, name='Hidden2'))
    # model.add(Dense(250, activation='softplus', use_bias=True, name='Hidden3'))
    model.add(Dense(1, activation='sigmoid', use_bias=True, name='Output'))
    model.compile(optimizer='adam', # default='rmsprop', an algorithm to be used in backpropagation
                  loss='binary_crossentropy', # Loss function to be optimized. A string (name of loss function), or a tf.keras.losses.Loss instance.
                  metrics=['accuracy', 'Precision', 'Recall'], # List of metrics to be evaluated by the model during training and testing. 
                 )
    return model


def keras(d):
    #data

    tf.compat.v1.disable_v2_behavior()
    model = create_model(d)
    X_train, X_test, y_train, y_test = train_test_split(np.asarray(d.toarray()), np.asarray(labels), shuffle=True)
    model.fit(X_train, y_train, batch_size=2, epochs=300, verbose='auto', shuffle=True, class_weight={0: 0.3, 1: 0.7}, initial_epoch=0)


    return model
def keras_shap(model):
    #tornado
    exp = shap.DeepExplainer(model, X_train)
    shapVal = exp.shap_values(X_train)
    print(shapVal[0][0])
    shap.summary_plot(shapVal[0], X_train.astype("float"), names, len(names))

def keras_classify(model):
    #tabelka
    pred_labels_tr = (model.predict(X_train) > 0.5).astype(int)
    pred_labels_te = (model.predict(X_test) > 0.5).astype(int)


    print('---------- Evaluation on Training Data ----------')
    print(classification_report(y_train, pred_labels_tr))
    print("")

    print('---------- Evaluation on Test Data ----------')
    print(classification_report(y_test, pred_labels_te))
    print("")

def keras_permutations(d):
    # permutajce
    # a czego dokaldnie to sobie zobacz
    model = KerasClassifier(build_fn=lambda:create_model(d), batch_size=2, epochs=200, shuffle=True)
    model.fit(X_train, y_train, batch_size=2, epochs=300, verbose='auto', shuffle=True, class_weight={0: 0.3, 1: 0.7}, initial_epoch=0)

    imp, names = permutationImportance(model)
    plotImportant(names, imp)

if __name__ == "__main__":
    model = keras(data)
    keras_shap(model)
    keras_classify(model)

    keras_permutations(data)

