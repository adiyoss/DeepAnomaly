from keras import callbacks
from keras.layers import LSTM, Dropout, Dense, TimeDistributed
from keras.models import Sequential
import numpy as np

import os
import reader


def create_sequences(data, win_size=20):
    """
    data should be pd.DataFrame()
    :param data:
    :param win_size:
    """
    docX = []
    for i in range(len(data)-win_size):
        docX.append(data[i:i+win_size])
    alsX = np.array(docX)

    return alsX

path_train = 'ECG5000_TRAIN_CONTINUOUS_SIGNAL_1.pkl'
path_test = 'ECG5000_TEST_CONTINUOUS_SIGNAL_1.pkl'

input_size = 1
hidden_size = 128
dropout_rate = 0.2
maxlen = 20
batch_size = 32
early_stopping_patience = 5
save_dir = 'results/'
model_name = 'model.net'
val_percentage = 0.15

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

db = reader.read_data(path_train)
train_x = db[:-1]
train_y = db[1:]

X = create_sequences(train_x)
y = create_sequences(train_y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))
y = np.reshape(y, (y.shape[0], y.shape[1], 1))

# preparing the callbacks
check_pointer = callbacks.ModelCheckpoint(filepath=save_dir+model_name, verbose=1, save_best_only=True)
early_stop = callbacks.EarlyStopping(patience=early_stopping_patience, verbose=1)

# build the model: 1 layer LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(hidden_size, return_sequences=True, input_shape=(maxlen, input_size)))
model.add(Dropout(dropout_rate))
model.add(TimeDistributed(Dense(input_size)))

model.compile(loss='mse', optimizer='adam')

model.fit(X, y, batch_size=batch_size, validation_split=val_percentage, callbacks=[check_pointer, early_stop])





# test_x = reader.read_data(path_test)