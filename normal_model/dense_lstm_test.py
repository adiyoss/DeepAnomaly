import argparse

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential

from common.utils import create_sequences, read_data


def test(path_test, input_size, hidden_size, batch_size, save_dir, model_name, maxlen):
    db = read_data(path_test)

    X = create_sequences(db[:-maxlen], win_size=maxlen, step=maxlen)
    X = np.reshape(X, (X.shape[0], X.shape[1], input_size))

    # build the model: 1 layer LSTM
    print('Build model...')
    model = Sequential()
    model.add(LSTM(hidden_size, return_sequences=False, input_shape=(maxlen, input_size)))
    model.add(Dense(maxlen))

    model.load_weights(save_dir + model_name)
    model.compile(loss='mse', optimizer='adam')

    prediction = model.predict(X, batch_size, verbose=1)
    prediction = prediction.flatten()
    # prediction_container = np.array(prediction).flatten()
    Y = db[maxlen:]
    plt.plot(prediction, label='prediction')
    plt.plot(Y, label='true')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # the first argument is the wav file path
    # the second argument is the TextGrid path
    # -------------MENU-------------- #
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="the path to the data",
                        default='../data/ECG5000_TEST_CONTINUOUS_SIGNAL_1.pkl')
    parser.add_argument("--input_size", help="the input size", default=1)
    parser.add_argument("--hidden_size", help="the hidden layer size", default=128)
    parser.add_argument("--batch_size", help="the batch size", default=32)
    parser.add_argument("--save_dir", help="the folder to save the model", default='results/')
    parser.add_argument("--model_name", help="the name of the model to be saved", default='model_dense_26_06_16.net')
    parser.add_argument("--maxlen", help="the mas sequence length", default=140)
    args = parser.parse_args()

    test(args.data_path, args.input_size, args.hidden_size, args.batch_size, args.save_dir, args.model_name,
         args.maxlen)
