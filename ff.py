from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from eli5.sklearn import PermutationImportance
from sklearn.model_selection import train_test_split
from keras.models import Sequential  # for creating a linear stack of layers for our Neural Network
from keras import Input  # for instantiating a keras tensor
from keras.layers import Dense  # for creating regular densely-connected NN layer.
import math
import pandas as pd
import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
from variables import labels, data, names
import shap
import tensorflow as tf
from metrics import permutationImportance, plotImportant

from variables import X_train, X_test, y_train, y_test


def create_model(inputData):
    """
    This functions creates neural network model.

    Parameters
        :inputData: vectorized_dt (text array transformed to document-term matrix)

    Returns
        :model: neural network model with layers ready to train
    """

    model = Sequential(name="A")
    model.add(Input(shape=(inputData.shape[1],), name="Input"))
    model.add(Dense(math.sqrt(inputData.shape[1]), activation='sigmoid', use_bias=True, name='Hidden1'))
    model.add(Dense(math.sqrt(inputData.shape[1]) / 2, activation='sigmoid', use_bias=True, name='Hidden2'))
    model.add(Dense(math.sqrt(inputData.shape[1]) / 4, activation='sigmoid', use_bias=True, name='Hidden3'))
    # model.add(Dense(250, activation='softplus', use_bias=True, name='Hidden3'))
    model.add(Dense(1, activation='sigmoid', use_bias=True, name='Output'))
    model.compile(optimizer='adam',  # default='rmsprop', an algorithm to be used in backpropagation
                  loss='binary_crossentropy',
                  # Loss function to be optimized. A string (name of loss function), or a tf.keras.losses.Loss instance.
                  metrics=['accuracy', 'Precision', 'Recall'],
                  # List of metrics to be evaluated by the model during training and testing.
                  )
    return model


def keras(inputData):
    """
    This functions creates and trains neural network model.

    Parameters
        :inputData: vectorized_dt (text array transformed to document-term matrix)

    Returns
        :model: trained neural network model
    """

    tf.compat.v1.disable_v2_behavior()  # It switches all global behaviors that are different between TensorFlow 1.x and 2.x to behave as intended for 1.x.
    model = create_model(inputData)
    X_train, X_test, y_train, y_test = train_test_split(np.asarray(inputData.toarray()), np.asarray(labels),
                                                        shuffle=True)
    model.fit(X_train, y_train, batch_size=2, epochs=300, verbose='auto', shuffle=True, class_weight={0: 0.3, 1: 0.7},
              initial_epoch=0)

    return model


def keras_shap(model):
    """
    This functions plots SHAP value for each word on a graph.

    Parameters
         :model: trained neural network model
    """
    # TODO change the legend visualisation
    exp = shap.DeepExplainer(model, X_train)
    shapVal = exp.shap_values(X_train)
    # -----
    # shap.summary_plot(shapVal[0], X_train.astype("float"), names, max_display=20, show=False) # max_display = len(names)
    # plt.gcf().axes[-1].set_aspect(1000)
    #plt.gcf().axes[-1].set_box_aspect(1000)
    plt.rc('font', size=14)
    shap.summary_plot(shapVal[0], X_train.astype("float"), names, max_display=20, show=False, color_bar=False, auto_size_plot=True)
    # -----
    #print(shapVal[0][0])
    exp = shap.DeepExplainer(model, X_train)
    shapVal = exp.shap_values(X_train)
    # print(shapVal[0][0])
    #shap.summary_plot(shapVal[0], X_train.astype("float"), names, 20)
    plt.colorbar(label="Feature value")
    plt.savefig("SavedVisualisation/FF_plot_20words.png")
    plt.show()

def keras_classify(model):
    """
    This functions prints two classification reports:
        - for training data.
        - for testing data

    Parameters
         :model: trained neural network model
    """
    pred_labels_tr = (model.predict(X_train) > 0.5).astype(int)
    pred_labels_te = (model.predict(X_test) > 0.5).astype(int)

    print('---------- Evaluation on Training Data ----------')
    print(classification_report(y_train, pred_labels_tr))
    print("")

    print('---------- Evaluation on Test Data ----------')
    print(classification_report(y_test, pred_labels_te))
    print("")


def keras_permutations(inputData):
    """
    This functions creates KerasClassifier and trains on training data.
    Permutation importance is calculated for each word in the model

    Parameters
         :inputData: trained neural network model
    """
    model = KerasClassifier(build_fn=lambda: create_model(inputData), batch_size=2, epochs=200, shuffle=True)
    model.fit(X_train, y_train, batch_size=2, epochs=300, verbose='auto', shuffle=True, class_weight={0: 0.3, 1: 0.7},
              initial_epoch=0)

    imp, names = permutationImportance(model)
    print(names)
    plotImportant(names, imp)


if __name__ == "__main__":
    model = keras(data)

    keras_shap(model)
    keras_classify(model)

    keras_permutations(data)
