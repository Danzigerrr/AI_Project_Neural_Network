from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from variables import X_train, y_train, X_test, y_test
import sys
from func import get_important
from metrics import permutationImportance, plotImportant


def createSvmClassifier():
    """
    This functions creates C-Support Vector Classifier.

    Returns:
        :svc: C-Support Vector Classifier fitted to training data
    """
    svc = svm.SVC(kernel='linear', verbose=True)  # C-Support Vector Classification.
    svc.fit(X_train, y_train)  # Fit the SVM model according to the given training data
    return svc


def classify(model):
    """
    This function performs classification on samples in model, which is svm.SVC (C-Support Vector Classification).
    After classification confusion matrix and classification report are printed.

    Parametrs:
        :model: data model --> svm.SVC (C-Support Vector Classification)
    """
    y_pred = model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


def getImportantData():
    """
    This function creates svm.SVC (C-Support Vector Classification) and performs classification on samples in this model.
    It gets important names basing on their value of importance.

    Returns:
        :importantData: array of texts containing only important words
    """
    svc = createSvmClassifier()
    _, snames = permutationImportance(svc)
    importantData, _ = get_important(snames)
    return importantData


if __name__ == "__main__":
    svc = createSvmClassifier()
    simp, snames = permutationImportance(svc)
    classify(svc)
    plotImportant(snames, simp)
