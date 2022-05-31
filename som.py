from minisom import MiniSom  # https://github.com/JustGlowing/minisom
from matplotlib.pylab import bone, pcolor, colorbar, plot, show, legend, text
from variables import labels, data, names
import random
import copy
import numpy as np
import pandas as pd
import math


def som(data, learning_rate=.1):
    """
    This functions creates and trains SOM basing on provided data.

    Parameters
        :data: array of importance of words

    Returns
        :vectorizer_dt: TF-IDF is a numerical statistic that is intended to reflect how important a word is to a document
        :labels save: info about isAntiVaccine
        :featureNames: feature names from vectorizer_dt got from texts basing on TF-IDF method
    """

    som_size = 10
    som = MiniSom(som_size, som_size, data.shape[1], sigma=8, learning_rate=learning_rate,
                  neighborhood_function='triangle', random_seed=10)
    som.pca_weights_init(data)
    som.train_random(data, 500, verbose=True)
    return som


def somWithClassInfo(data):
    """
    SOM with additional column representing if the text is for or against vaccination.

    Parameters:
        :data: array of words from vectorizer

    Returns:
        :model: trained SOM model

    """
    data_copy = copy.copy(data)
    # for i in range(0, d.shape[0]):
    df = pd.DataFrame(data_copy.toarray())
    df["Labels"] = [0 if i == False else 1 for i in labels]

    model = som(df.values)

    plotSom(model, df.to_numpy(), labels)
    return model


def plotSom(som, data, labels, featureNames=None, LR=0):
    """
    This function is used to plot som. Colors, sizes and data can be set here.

    Parameters:
        :som: trained SOM model
        :data: words in an array
    """
    bone()
    pcolor(som.distance_map().T)
    colorbar()
    colors = ['r', 'g']

    dataPositions = []

    for i, x in enumerate(data):
        ox = random.random()
        oy = random.random()
        w = som.winner(x)

        xpos = w[0] + ox
        ypos = w[1] + oy
        dataPositions.append([xpos, ypos, labels[i]])
        plot(xpos,
             ypos,
             'o',
             markeredgecolor=colors[labels[i]],
             markerfacecolor=colors[labels[i]],
             markersize=6,
             markeredgewidth=2)
        if featureNames is not None:
            text(w[0] + ox, w[1] + oy, featureNames[i], c=colors[labels[i]])

    result = calculateDistBetweenNeurons(dataPositions)

    show()

    result.insert(0, LR)
    return result


def calcDistOfLabel(pos, currNeur, labelType):
    distancesWithAllOthers = 0
    for randNeur in pos:
        if randNeur[2] == labelType:
            Xdiff = abs(randNeur[0] - currNeur[0])
            Ydiff = abs(randNeur[1] - currNeur[1])
            distancesWithAllOthers += math.sqrt(pow(Xdiff, 2) + pow(Ydiff, 2))
    return distancesWithAllOthers


def calculateDistBetweenNeurons(positionData):
    sumFor = 0
    nFor = 0
    sumAgainst = 0
    nAgainst = 0
    for currNeur in positionData:
        nodeLabel = currNeur[2]
        if nodeLabel == 0:
            sumFor += calcDistOfLabel(positionData, currNeur, 0)
            nFor += 1
        elif nodeLabel == 1:
            sumAgainst += calcDistOfLabel(positionData, currNeur, 1)
            nAgainst += 1

    sumFor = round(sumFor, 2)
    avgFor = round(sumFor / nFor, 2)
    sumAgainst = round(sumAgainst, 2)
    avgAgainst = round(sumAgainst / nAgainst, 2)

    result = [sumFor, avgFor, sumAgainst, avgAgainst]

    return result


if __name__ == "__main__":
    # from svm import getImportantData
    # data = getImportantData()
    somModel = somWithClassInfo(data)
