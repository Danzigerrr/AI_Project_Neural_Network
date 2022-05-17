from variables import X_test, y_test, names
from eli5.sklearn import PermutationImportance
from word import Word
import sys
import matplotlib.pyplot as plt
import numpy as np


def permutationImportance(model):
    """
    This functions calculates permutation importance for provided model.

    Depending on using "takeXMostSignificantFeatures(impArray, namesArray, ss, numberOfFeatures=10)" or
    "takeAllFeatures(impArray, namesArray, ss)" the essential data will be returned from this function.

    Parameters:
        :model: data model, for example svc(C-Support Vector Classifier fitted to training data)

    Returns:
        :impArray: array containing the importance of each word
        :namesArray: array containing all words

    """
    # PermutationImportance --> Meta-estimator which computes ``feature_importances_`` attribute
    #                           based on permutation importance (also known as mean score decrease)
    # .fit --> Compute ``feature_importances_`` attribute and optionally
    #          fit the base estimator.
    perm = PermutationImportance(model).fit(X_test, y_test)
    importances = []
    counter = 0
    for impOfWord in perm.feature_importances_:
        importances.append(Word(names[counter], impOfWord))
        counter += 1
        
    namesArray = []
    impArray = []
    sortedWords = sorted(importances, key=lambda x: x.importance)
    
    # takeXMostSignificantFeatures(impArray, namesArray, ss, numberOfFeatures=10)
    # or:
    takeAllFeatures(impArray, namesArray, sortedWords)

    return impArray, namesArray


def takeAllFeatures(impArray, namesArray, sortedWords):
    """
    This functions takes all the important features.
    
    Parameters:
        :impArray: array to store value of importance of words
        :namesArray: array to store value of words
        :sortedWords: array with words sorted by the importance value  
    """
    for i in sortedWords:
        if i.importance > sys.float_info.epsilon or i.importance < -sys.float_info.epsilon:
            namesArray.append(i.name)
            impArray.append(i.importance)


def takeXMostSignificantFeatures(impArray, namesArray, sortedWords, numberOfFeatures):
    """
    This functions takes numberOfFeatures from the beginning and numberOfFeatures from the end of the set of words.

    Parameters:
        :impArray: array to store value of importance of words
        :namesArray: array to store value of words
        :sortedWords: array with words sorted by the importance value  
        :numberOfFeatures: integer value        
    """
    
    # get features from the beginning
    for i in range(0, numberOfFeatures):
        namesArray.append(sortedWords[i].name)
        impArray.append(sortedWords[i].importance)

    # get features from the end
    for i in range(len(names) - numberOfFeatures - 1, len(names) - 1):
        namesArray.append(sortedWords[i].name)
        impArray.append(sortedWords[i].importance)


def plotImportant(namesArray, impArray):
    """
    This functions creates a plot of importances of provided words.

    Parameters:
        :impArray: array to store value of importance of words
        :namesArray: array to store value of words
    """
    _, ax = plt.subplots()
    y_size = np.arange(len(namesArray))
    ax.barh(y_size, impArray)
    ax.set_yticks(y_size, labels=namesArray)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    plt.show()
