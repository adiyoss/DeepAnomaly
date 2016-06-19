from keras import callbacks
from keras.layers import LSTM, Dropout, Dense, TimeDistributed
from keras.models import Sequential
import numpy as np
import cPickle as pickle
import os
import reader
from utils import create_sequences

path_train = 'data/ECG5000_TRAIN_CONTINUOUS_SIGNAL_1.pkl'


def train_normal_model():
    input_size = 1
    hidden_size = 128
    dropout_rate = 0.5
    maxlen = 140
    batch_size = 32
    early_stopping_patience = 5
    save_dir = 'results/'
    model_name = 'model_1.net'
    val_percentage = 0.15

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    db = reader.read_data(path_train)
    train_x = db[:-140]
    train_y = db[140:]

    X = create_sequences(train_x, 140, 140)
    y = create_sequences(train_y, 140, 140)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    # y = np.reshape(y, (y.shape[0], y.shape[1], 1))
    #
    # preparing the callbacks
    check_pointer = callbacks.ModelCheckpoint(filepath=save_dir + model_name, verbose=1, save_best_only=True)
    early_stop = callbacks.EarlyStopping(patience=early_stopping_patience, verbose=1)

    # build the model: 1 layer LSTM
    print('Build model...')
    model = Sequential()
    model.add(LSTM(hidden_size, return_sequences=False, input_shape=(maxlen, input_size)))
    model.add(Dense(140))

    # model.add(Dropout(dropout_rate))
    # model.add(LSTM(hidden_size, return_sequences=False))
    # model.add(Dense(140))

    model.compile(loss='mse', optimizer='adam')

    model.fit(X, y, batch_size=batch_size, nb_epoch=100, validation_split=val_percentage,
              callbacks=[check_pointer, early_stop])

    return model


def store_prediction_and_ground_truth(model):
    input_size = 1
    maxlen = 140
    batch_size = 32

    db = reader.read_data('data/ECG5000_TEST_PHASE_1_CONTINUOUS_SIGNAL_1.pkl')
    X = create_sequences(db[:-140], win_size=maxlen, step=maxlen)
    X = np.reshape(X, (X.shape[0], X.shape[1], input_size))
    Y = create_sequences(db[140:], win_size=maxlen, step=maxlen).flatten()

    prediction = model.predict(X, batch_size, verbose=1)
    prediction = prediction.flatten()
    with open('data/ECG5000_TRAIN_PHASE_2_CONTINUOUS_SIGNAL_1.pkl', 'wb') as f:
        pickle.dump(np.stack((Y, prediction)), f)


if __name__ == "__main__":
    model = train_normal_model()
    store_prediction_and_ground_truth(model)
