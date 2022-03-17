from eli5.sklearn import PermutationImportance
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from variables import names, X_train, y_train, X_test, y_test
from word import Word


def svmClassifier():
    svc = svm.SVC(kernel='linear', verbose=True)
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    perm = PermutationImportance(svc).fit(X_test, y_test)

    c = 0
    importances = []
    for i in perm.feature_importances_:
        importances.append(Word(names[c], i))
        c += 1
    ss = sorted(importances, key=lambda x:x.importance)

    snames = []
    simp = []
    for i in ss:
        if i.importance != 0:
            snames.append(i.name)
            simp.append(i.importance)

    fig, ax = plt.subplots()
    y_size = np.arange(len(snames))
    ax.barh(y_size, simp)
    ax.set_yticks(y_size, labels=snames)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    plt.show()


if __name__ == "__main__":
    svmClassifier()
