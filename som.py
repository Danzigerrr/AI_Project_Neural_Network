from minisom import MiniSom #https://github.com/JustGlowing/minisom
from matplotlib.pylab import bone, pcolor, colorbar, plot, show
from variables import labels
import random

def som(data):
    som_size = 15
    som = MiniSom(som_size, som_size, data.shape[1], sigma=5, learning_rate=0.1,
                  neighborhood_function='triangle', random_seed=10)
    som.pca_weights_init(data.toarray())
    som.train_random(data.toarray(), 500, verbose=True)
    return som

def plotSom(som, data):
    bone()
    pcolor(som.distance_map().T)
    colorbar()
    colors = ['r', 'g']
    for i, x in enumerate(data.toarray()):
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
    from variables import data
    somModel = som(data)
    plotSom(somModel, data)
