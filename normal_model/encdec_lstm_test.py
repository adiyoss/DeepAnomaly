import argparse

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, RepeatVector
from keras.layers import LSTM, TimeDistributed
from keras.models import Sequential

from common.utils import create_sequences, read_data, store_prediction_and_ground_truth


def test(path_test, input_size, hidden_size, batch_size, save_dir, model_name, maxlen):
    db = read_data(path_test)
    X = create_sequences(db, maxlen, maxlen)
    y = create_sequences(db, maxlen, maxlen)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    y = np.reshape(y, (y.shape[0], y.shape[1], 1))

    # build the model: 1 layer LSTM
    print('Build model...')
    model = Sequential()
    # "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE
    # note: in a situation where your input sequences have a variable length,
    # use input_shape=(None, nb_feature).
    model.add(LSTM(hidden_size, input_shape=(maxlen, input_size)))
    # For the decoder's input, we repeat the encoded input for each time step
    model.add(RepeatVector(maxlen))
    # The decoder RNN could be multiple layers stacked or a single layer
    model.add(LSTM(hidden_size, return_sequences=True))

    # For each of step of the output sequence, decide which character should be chosen
    model.add(TimeDistributed(Dense(1)))

    model.load_weights(save_dir + model_name)

    model.compile(loss='mae', optimizer='adam')
    model.summary()

    prediction = model.predict(X, batch_size, verbose=1, )
    prediction = prediction.flatten()
    # prediction_container = np.array(prediction).flatten()
    plt.plot(prediction.flatten()[:4000], label='prediction')
    plt.plot(y.flatten()[maxlen:4000 + maxlen], label='true')
    plt.legend()
    plt.show()

    store_prediction_and_ground_truth(model)


if __name__ == "__main__":
    # the first argument is the wav file path
    # the second argument is the TextGrid path
    # -------------MENU-------------- #
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="the path to the data",
                        default='../data/ECG5000_TEST_CONTINUOUS_SIGNAL_1.pkl')
    parser.add_argument("--input_size", help="the input size", default=1)
    parser.add_argument("--hidden_size", help="the hidden layer size", default=256)
    parser.add_argument("--batch_size", help="the batch size", default=32)
    parser.add_argument("--save_dir", help="the folder to save the model", default='results/')
    parser.add_argument("--model_name", help="the name of the model to be saved", default='model_encdec_26_06_16.net')
    parser.add_argument("--maxlen", help="the mas sequence length", default=100)
    args = parser.parse_args()

    test(args.data_path, args.input_size, args.hidden_size, args.batch_size, args.save_dir, args.model_name,
         args.maxlen)
