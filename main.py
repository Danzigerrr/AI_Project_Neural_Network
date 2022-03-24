from som import som, plotSom
from ff import keras
from svm import getImportantData


data = getImportantData()
somModel = som(data)
plotSom(somModel, data)
keras(data)

