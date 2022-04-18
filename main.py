from som import som, plotSom
from ff import keras
from svm import getImportantData
from func import get_data


data, labels, features = get_data()
print(features)
somModel = som(data.toarray())
plotSom(somModel, data.toarray())
# keras(data)

