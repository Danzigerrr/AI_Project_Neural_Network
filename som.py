from minisom import MiniSom #https://github.com/JustGlowing/minisom
from matplotlib.pylab import bone, pcolor, colorbar, plot, show, legend
from variables import labels, data, names
import random
import copy
import numpy as np
import pandas as pd

def som(data):
    som_size = 15
    som = MiniSom(som_size, som_size, data.shape[1], sigma=5, learning_rate=0.1,
                  neighborhood_function='triangle', random_seed=10)
    som.pca_weights_init(data)
    som.train_random(data, 500, verbose=True)
    return som

def somWithClassInfo(data):
    d = copy.copy(data)
    # for i in range(0, d.shape[0]):
    df = pd.DataFrame(d.toarray())
    df["Labels"] = [0 if i == False else 1 for i in labels]

    model = som(df.values)


    plotSom(model, df.to_numpy())
    return model

def plotSom(som, data):
    bone()
    pcolor(som.distance_map().T)
    colorbar()
    colors = ['r', 'g']
    for i, x in enumerate(data):
        print("i: " + str(i) + "x: " + str(x))
        w = som.winner(x)
        plot(w[0] + random.random(),
            w[1] + random.random(),
            'o',
            markeredgecolor=colors[labels[i]],
            markerfacecolor = colors[labels[i]],
            markersize=6,
            markeredgewidth=2)
    show()

if __name__ == "__main__":
    somModel = somWithClassInfo(data)

