from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from variables import X_train, y_train, X_test, y_test
import sys
from func import get_important
from metrics import permutationImportance, plotImportant


def createSvmClassifier(X_train, y_train):
    """
    This functions creates C-Support Vector Classifier.

    Returns:
        :svc: C-Support Vector Classifier fitted to training data
    """
    svc = svm.SVC(kernel='linear', verbose=True, max_iter=300)  # C-Support Vector Classification.
    svc.fit(X_train, y_train)  # Fit the SVM model according to the given training data
    return svc


def classify(model, X_train, y_train, X_test, y_test):
    """
    This function performs classification on samples in model, which is svm.SVC (C-Support Vector Classification).
    After classification confusion matrix and classification report are printed.

    Parameters:
        :model: data model --> svm.SVC (C-Support Vector Classification)
    """
    y_pred = model.predict(X_train)  # Perform classification on samples in X_test
    print("---------- Evaluation on Train Data SVM ----------")
    print(confusion_matrix(y_train, y_pred))
    print(classification_report(y_train, y_pred))
    y_pred = model.predict(X_test)  # Perform classification on samples in X_test
    print("---------- Evaluation on Test Data SVM ----------")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


def getImportantData():
    """
    This function creates svm.SVC (C-Support Vector Classification) and performs classification on samples in this model.
    It gets important names basing on their value of importance.

    Returns:
        :importantData: array of texts containing only important words
    """
    svc = createSvmClassifier(X_train, y_train)
    _, snames = permutationImportance(svc) # get only words
    importantData, _ = get_important(snames) # get vectorizer_dt (documenent transformed into document-term matrix)
    return importantData


if __name__ == "__main__":
    svc = createSvmClassifier(X_train, y_train)
    simp, snames = permutationImportance(svc)
    plotImportant(snames, simp)
    classify(svc, X_train, y_train, X_test, y_test)
