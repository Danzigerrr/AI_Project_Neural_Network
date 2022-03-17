from minisom import MiniSom #https://github.com/JustGlowing/minisom
from pylab import bone, pcolor, colorbar, plot, show
from variables import data, labels


def som():
    som = MiniSom(15, 15, data.shape[1], sigma=5, learning_rate=0.1,
                  neighborhood_function='triangle', random_seed=10)
    som.pca_weights_init(data.toarray())
    som.train_random(data.toarray(), 500, verbose=True)

    bone()
    pcolor(som.distance_map().T)
    colorbar()
    markers = ['o', 's']
    colors = ['r', 'g']
    for i, x in enumerate(data.toarray()):
        w = som.winner(x)
        plot(w[0] + 0.5,
             w[1] + 0.5,
             markers[labels[i]],
             markeredgecolor = colors[labels[i]],
             markerfacecolor = 'None',
             markersize = 10,
             markeredgewidth = 2)
    show()


if __name__ == "__main__":
    som()
