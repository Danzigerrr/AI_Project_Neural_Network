from som import som, plotSom
# from ff import keras
# from svm import getImportantData
from func import get_word_embeddings2, get_word_embeddings
import numpy as np


"""
    SOM map is generated. Data to model is provided by the getImportantData() function. 
"""

if __name__ == "__main__":
    _, data, labels, names = get_word_embeddings(.0, .8)
    _, data2, labels, names = get_word_embeddings(.8, 1.0)
    _, data3, labels, names = get_word_embeddings()
    sommodel = som(data)
    plotSom(sommodel, data, labels, names)
    sommodel.train_random(data2, 50)
    plotSom(sommodel, data3, labels, names)
    # keras(data)

    sommodel = som(data, .01)
    plotSom(sommodel, data, labels, names)
    sommodel.train_random(data2, 50)
    plotSom(sommodel, data3, labels, names)

    sommodel = som(data, .001)
    plotSom(sommodel, data, labels, names)
    sommodel.train_random(data2, 50)
    plotSom(sommodel, data3, labels, names)
