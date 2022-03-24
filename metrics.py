from variables import X_test, y_test, names
from eli5.sklearn import PermutationImportance
from word import Word
import sys
import matplotlib.pyplot as plt
import numpy as np
def permutationImportance(model):
    perm = PermutationImportance(model).fit(X_test, y_test)
    importances = []
    c = 0
    for i in perm.feature_importances_:
        importances.append(Word(names[c], i))
        c+=1
    snames = []
    simp = []
    ss = sorted(importances, key=lambda x:x.importance)
    for i in ss:
        if i.importance > sys.float_info.epsilon or i.importance < -sys.float_info.epsilon:
            snames.append(i.name)
            simp.append(i.importance)
    return simp, snames
def plotImportant(names, importances):
    _, ax = plt.subplots()
    y_size = np.arange(len(names))
    ax.barh(y_size, importances)
    ax.set_yticks(y_size, labels=names)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    plt.show()
