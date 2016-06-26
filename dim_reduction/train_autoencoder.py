import argparse

import os
from keras.layers import Input, Dense
from keras.models import Model
import cPickle as pickle

from common.utils import build_data_auto_encoder


def train(path, win_size, step, batch_size, enc_size, n_epochs, plot, save, model, save_dir):
    with open(path, "rb") as f:
        X = pickle.load(f)

    data = build_data_auto_encoder(X, step, win_size)

    encoding_dim = enc_size
    input_img = Input(shape=(data.shape[1],))
    encoded = Dense(encoding_dim, activation='tanh')(input_img)

    decoded = Dense(data.shape[1])(encoded)

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
                    nb_epoch=n_epochs,
                    batch_size=batch_size,
                    shuffle=True)
    encoded_imgs = encoder.predict(data)
    decoded_imgs = decoder.predict(encoded_imgs)

    if plot:
        import matplotlib.pyplot as plt
        offset = 1050
        n = 10  # how many windows we will display
        plt.figure(figsize=(20, 4))
        for i in range(n):
            # display original
            ax = plt.subplot(3, n, i + 1)
            ax.set_ylim([0, 5])
            plt.plot(data[i + offset])
            # ax.get_xaxis().set_visible(False)
            # ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(3, n, i + 1 + n)
            ax.set_ylim([0, 5])
            plt.plot(encoded_imgs[i + offset])
            # ax.get_xaxis().set_visible(False)
            # ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(3, n, i + 1 + 2 * n)
            plt.plot(X[0, (i + offset) * win_size:(i + offset) * win_size + win_size])
            # plt.plot(X[1, (i + offset) * win_size:(i + offset) * win_size + win_size])
            # ax.get_xaxis().set_visible(False)
            # ax.get_yaxis().set_visible(False)
        plt.show()

    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_dir = os.path.abspath(save_dir)+'/'
        encoder.save_weights(save_dir+model)
        features_path = save_dir+'features.pkl'
        encodings_path = save_dir+'encodings.pkl'
        with open(features_path, 'w') as f:
            pickle.dump(data, f)
        with open(encodings_path, 'w') as f:
            pickle.dump(encodings_path, f)

if __name__ == "__main__":
    # the first argument is the wav file path
    # the second argument is the TextGrid path
    # -------------MENU-------------- #
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="the path to the data",
                        default="data/ECG5000_TRAIN_PHASE_2_CONTINUOUS_SIGNAL_1.pkl")
    parser.add_argument("--win_size", help="the window size", default=140)
    parser.add_argument("--step", help="the step size", default=140)
    parser.add_argument("--batch_size", help="the batch size", default=32)
    parser.add_argument("--enc_size", help="the size of the encoder", default=32)
    parser.add_argument("--n_epochs", help="the number of epochs", default=10)
    parser.add_argument("--plot", help="visualize", default=False)
    parser.add_argument("--save", help="save the features", default=True)
    parser.add_argument("--model", help="model name", default='autoencoder_model.net')
    parser.add_argument("--save_dir", help="the path to the dir to save the data", default='files/')
    args = parser.parse_args()

    train(args.data_path, args.win_size, args.step, args.batch_size, args.enc_size, args.n_epochs, args.plot, args.save,
          args.model, args.save_dir)
