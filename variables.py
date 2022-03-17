from sklearn.model_selection import train_test_split
import numpy as np
from func import get_data

data, labels, names = get_data()
X_train, X_test, y_train, y_test = train_test_split(np.asarray(data.toarray()), np.asarray(labels), shuffle=True)
