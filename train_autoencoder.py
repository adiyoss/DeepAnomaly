from keras.layers import Input, Dense
from keras.models import Model
import cPickle as pickle
import numpy as np

with open("data/ECG5000_TRAIN_PHASE_2_CONTINUOUS_SIGNAL_1.pkl", "rb") as f:
    X = pickle.load(f)
win_size = 140
step = 140
docX = []
for i in range(0, X.shape[1] - win_size, step):
    docX.append(np.abs(X[0, i:i + win_size] - X[1, i:i + win_size]))
data = np.array(docX)

encoding_dim = 16
input_img = Input(shape=(data.shape[1],))
encoded = Dense(encoding_dim, activation='tanh')(input_img)

decoded = Dense(data.shape[1], activation='tanh')(encoded)

autoencoder = Model(input=input_img, output=decoded)
autoencoder.compile(optimizer='adam', loss='mae')

# this model maps an input to its encoded representation
encoder = Model(input=input_img, output=encoded)
# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

autoencoder.fit(data, data,
                nb_epoch=200,
                batch_size=32,
                shuffle=True)

encoded_imgs = encoder.predict(data)
decoded_imgs = decoder.predict(encoded_imgs)

import matplotlib.pyplot as plt

n = 4  # how many windows we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(3, n, i + 1)
    plt.plot(data[i])
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(3, n, i + 1 + n)
    plt.plot(decoded_imgs[i])
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(3, n, i + 1 + 2*n)
    plt.plot(X[0, i*win_size:i*win_size + win_size])
    plt.plot(X[1, i*win_size:i*win_size + win_size])
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)

plt.show()
