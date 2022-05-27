from som import som, plotSom
# from ff import keras
# from svm import getImportantData
from func import get_word_embeddings2
import numpy as np


"""
    SOM map is generated. Data to model is provided by the getImportantData() function. 
"""

if __name__ == "__main__":
    data, labels, names = get_word_embeddings2()
    somModel = som(data)
    plotSom(somModel, data, labels, names)
    # keras(data)

