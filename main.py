from som import som
from ff import keras
from svm import getImportantData


data = getImportantData()
som(data)
keras(data)

