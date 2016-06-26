import argparse
import os

import numpy as np
from keras import callbacks
from keras.layers import LSTM, Dense
from keras.models import Sequential

from common.utils import create_sequences, store_prediction_and_ground_truth, read_data


def train_normal_model(path_train, input_size, hidden_size, batch_size, early_stopping_patience, val_percentage, save_dir, model_name, maxlen):

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    db = read_data(path_train)
    train_x = db[:-140]
    train_y = db[140:]

    X = create_sequences(train_x, 140, 140)
    y = create_sequences(train_y, 140, 140)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # preparing the callbacks
    check_pointer = callbacks.ModelCheckpoint(filepath=save_dir + model_name, verbose=1, save_best_only=True)
    early_stop = callbacks.EarlyStopping(patience=early_stopping_patience, verbose=1)

    # build the model: 1 layer LSTM
    print('Build model...')
    model = Sequential()
    model.add(LSTM(hidden_size, return_sequences=False, input_shape=(maxlen, input_size)))
    model.add(Dense(140))

    model.compile(loss='mse', optimizer='adam')
    model.summary()

    model.fit(X, y, batch_size=batch_size, nb_epoch=100, validation_split=val_percentage,
              callbacks=[check_pointer, early_stop])

    return model


if __name__ == "__main__":
    # the first argument is the wav file path
    # the second argument is the TextGrid path
    # -------------MENU-------------- #
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="the path to the data",
                        default='../data/ECG5000_TRAIN_CONTINUOUS_SIGNAL_1.pkl')
    parser.add_argument("--input_size", help="the input size", default=1)
    parser.add_argument("--hidden_size", help="the hidden layer size", default=128)
    parser.add_argument("--batch_size", help="the batch size", default=32)
    parser.add_argument("--early_stopping_patience", help="for early stopping", default=10)
    parser.add_argument("--save_dir", help="the folder to save the model", default='results/')
    parser.add_argument("--model_name", help="the name of the model to be saved", default='model_dense_26_06_16.net')
    parser.add_argument("--val_percentage", help="percentage for validation", default=0.1)
    parser.add_argument("--maxlen", help="the mas sequence length", default=140)
    args = parser.parse_args()

    model = train_normal_model(args.data_path, args.input_size, args.hidden_size, args.batch_size,
                               args.early_stopping_patience, args.val_percentage, args.save_dir, args.model_name,
                               args.maxlen)
    store_prediction_and_ground_truth(model)