from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from minisom import MiniSom #https://github.com/JustGlowing/minisom
from functions import classify, get_data

data, labels = get_data()

X_train, X_test, y_train, y_test = train_test_split(data.toarray(), labels, shuffle=False)

som = MiniSom(20, 20, data.shape[1], sigma=18, learning_rate=0.1,
              neighborhood_function='triangle', random_seed=10)
som.pca_weights_init(X_train)
som.train_random(X_train, 500, verbose=False)

print(classification_report(y_test, classify(som, X_test, X_train, y_train)))
