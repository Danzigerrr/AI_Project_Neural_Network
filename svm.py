from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from variables import  X_train, y_train, X_test, y_test
import sys
from func import get_important
from metrics import permutationImportance, plotImportant

def svmClassifier():
    svc = svm.SVC(kernel='linear', verbose=True)
    svc.fit(X_train, y_train)
    return svc

def classify(model):
    y_pred = model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def getImportantData():
    svc = svmClassifier()
    _, snames = permutationImportance(svc)
    importantData, _ = get_important(snames)
    return importantData

if __name__ == "__main__":
    svc = svmClassifier()
    simp, snames = permutationImportance(svc)
    classify(svc)
    plotImportant(snames, simp)
