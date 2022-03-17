from sklearn.model_selection import train_test_split
from functions import get_data
import numpy as np

data, labels, names = get_data()
X_train, X_test, y_train, y_test = train_test_split(np.asarray(data.toarray()), np.asarray(labels), shuffle=True)

