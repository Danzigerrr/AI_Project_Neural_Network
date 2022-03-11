from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from minisom import MiniSom #https://github.com/JustGlowing/minisom
from functions import classify, get_data
from pylab import bone,pcolor,colorbar, plot, show
from tensorflow import keras # for building Neural Networks
print('Tensorflow/Keras: %s' % keras.__version__) # print version
from keras.models import Sequential # for creating a linear stack of layers for our Neural Network
from keras import Input # for instantiating a keras tensor
from keras.layers import Dense # for creating regular densely-connected NN layer.
import numpy
import pandas
import math

data, labels = get_data()

X_train, X_test, y_train, y_test = train_test_split(numpy.asarray(data.toarray()), numpy.asarray(labels), shuffle=True)
som = MiniSom(10, 10, data.shape[1], sigma=3, learning_rate=0.1,
              neighborhood_function='triangle', random_seed=10)
som.pca_weights_init(data.toarray())
som.train_random(data.toarray(), 500, verbose=False)

model = Sequential(name="A")
model.add(Input(shape=(data.shape[1],), name="Input"))
model.add(Dense(math.sqrt(data.shape[1]), activation='sigmoid', use_bias=True, name='Hidden-Layer'))
# model.add(Dense(750, activation='softplus', use_bias=True, name='Hidden1'))
# model.add(Dense(500, activation='softplus', use_bias=True, name='Hidden2'))
# model.add(Dense(250, activation='softplus', use_bias=True, name='Hidden3'))
model.add(Dense(1, activation='sigmoid', use_bias=True, name='Output'))
model.compile(optimizer='adam', # default='rmsprop', an algorithm to be used in backpropagation
              loss='binary_crossentropy', # Loss function to be optimized. A string (name of loss function), or a tf.keras.losses.Loss instance.
              metrics=['Accuracy', 'Precision', 'Recall'], # List of metrics to be evaluated by the model during training and testing. Each of this can be a string (name of a built-in function), function or a tf.keras.metrics.Metric instance. 
              loss_weights=None, # default=None, Optional list or dictionary specifying scalar coefficients (Python floats) to weight the loss contributions of different model outputs.
              weighted_metrics=None, # default=None, List of metrics to be evaluated and weighted by sample_weight or class_weight during training and testing.
              run_eagerly=None, # Defaults to False. If True, this Model's logic will not be wrapped in a tf.function. Recommended to leave this as None unless your Model cannot be run inside a tf.function.
              steps_per_execution=None # Defaults to 1. The number of batches to run during each tf.function call. Running multiple batches inside a single tf.function call can greatly improve performance on TPUs or small models with a large Python overhead.
             )
model.fit(X_train, # input data
          y_train, # target data
          batch_size=32, # Number of samples per gradient update. If unspecified, batch_size will default to 32.
          epochs=30, # default=1, Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided
          verbose='auto', # default='auto', ('auto', 0, 1, or 2). Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. 'auto' defaults to 1 for most cases, but 2 when used with ParameterServerStrategy.
          callbacks=None, # default=None, list of callbacks to apply during training. See tf.keras.callbacks
          # validation_split=0.2, # default=0.0, Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. 
          #validation_data=(X_test, y_test), # default=None, Data on which to evaluate the loss and any model metrics at the end of each epoch. 
          shuffle=True, # default=True, Boolean (whether to shuffle the training data before each epoch) or str (for 'batch').
          class_weight={0 : 0.3, 1 : 0.7}, # default=None, Optional dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during training only). This can be useful to tell the model to "pay more attention" to samples from an under-represented class.
          sample_weight=None, # default=None, Optional Numpy array of weights for the training samples, used for weighting the loss function (during training only).
          initial_epoch=0, # Integer, default=0, Epoch at which to start training (useful for resuming a previous training run).
          steps_per_epoch=None, # Integer or None, default=None, Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch. When training with input tensors such as TensorFlow data tensors, the default None is equal to the number of samples in your dataset divided by the batch size, or 1 if that cannot be determined. 
          validation_steps=None, # Only relevant if validation_data is provided and is a tf.data dataset. Total number of steps (batches of samples) to draw before stopping when performing validation at the end of every epoch.
          validation_batch_size=None, # Integer or None, default=None, Number of samples per validation batch. If unspecified, will default to batch_size.
          validation_freq=3, # default=1, Only relevant if validation data is provided. If an integer, specifies how many training epochs to run before a new validation run is performed, e.g. validation_freq=2 runs validation every 2 epochs.
          max_queue_size=10, # default=10, Used for generator or keras.utils.Sequence input only. Maximum size for the generator queue. If unspecified, max_queue_size will default to 10.
          workers=1, # default=1, Used for generator or keras.utils.Sequence input only. Maximum number of processes to spin up when using process-based threading. If unspecified, workers will default to 1.
          use_multiprocessing=False, # default=False, Used for generator or keras.utils.Sequence input only. If True, use process-based threading. If unspecified, use_multiprocessing will default to False. 
          )


##### Step 6 - Use model to make predictions
# Predict class labels on training data
pred_labels_tr = (model.predict(X_train) > 0.5).astype(int)
# Predict class labels on a test data
pred_labels_te = (model.predict(X_test) > 0.5).astype(int)


##### Step 7 - Model Performance Summary
print("")
print('-------------------- Model Summary --------------------')
model.summary() # print model summary
print("")
print('-------------------- Weights and Biases --------------------')
for layer in model.layers:
    print("Layer: ", layer.name) # print layer name
    print("  --Kernels (Weights): ", layer.get_weights()[0]) # kernels (weights)
    print("  --Biases: ", layer.get_weights()[1]) # biases
print("")
print('---------- Evaluation on Training Data ----------')
print(classification_report(y_train, pred_labels_tr))
print("")

print('---------- Evaluation on Test Data ----------')
print(classification_report(y_test, pred_labels_te))
print("")

# crint(classification_report(y_test, classify(som, X_test, X_train, y_train)))

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



#------------
#notatki po drugich konsultacjach(sroda 9.03.2022):

#wizualiacja - rysowanie siatki SOM
#zobaczyc na TF a znie na cale TFIDF
#dredukowac z1500 do 200
#wyrac najsitotniejsze slowa
#part of speedch taging
#pokolorac wszgledem  0 i 1
#wpraowdzic nowa siec FitFowrawd  - wezmie polowe tych tekstow i sie przetrenuje (bedzie trnowana na tej zredukowanej tabeli)
#dolozyc tesktow
#ktore slowa sa klczowe do wskania ze dany tekst nalezy do konkretnej klasy


