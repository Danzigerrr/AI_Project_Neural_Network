from som import som, plotSom
from ff import keras
from svm import getImportantData


"""
    SOM map is generated. Data to model is provided by the getImportantData() function. 
"""

if __name__ == "__main__":
    data = getImportantData()
    somModel = som(data.toarray())
    plotSom(somModel, data.toarray())
    # keras(data)

