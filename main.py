from som import som, plotSom
# from ff import keras
# from svm import getImportantData
from func import get_word_embeddings2
import numpy as np


"""
    SOM map is generated. Data to model is provided by the getImportantData() function. 
"""

if __name__ == "__main__":
    data, labels, names = get_word_embeddings2(.8)
    data2, labels, names = get_word_embeddings2()
    d = data2 - data
    sommodel = som(data)
    plotSom(sommodel, data, labels, names)
    sommodel.train_random(d, 50)
    plotSom(sommodel, data2, labels, names)
    # keras(data)

    sommodel = som(data, .01)
    plotSom(sommodel, data, labels, names)
    sommodel.train_random(d, 50)
    plotSom(sommodel, data2, labels, names)

    sommodel = som(data, .001)
    plotSom(sommodel, data, labels, names)
    sommodel.train_random(d, 50)
    plotSom(sommodel, data2, labels, names)
