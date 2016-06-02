from keras import callbacks
from keras.layers import LSTM, Dropout, Dense, TimeDistributed
from keras.models import Sequential
import numpy as np

import os
import reader
from utils import create_sequences

path_train = 'data/ECG5000_TRAIN_CONTINUOUS_SIGNAL_1.pkl'

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
train_y = db[21:]

X = create_sequences(train_x)
y = train_y #create_sequences(train_y)
X = np.reshape(X, (X.shape[0], X.shape[1], input_size))
# y = np.reshape(y, (y.shape[0], y.shape[1], input_size))

# preparing the callbacks
check_pointer = callbacks.ModelCheckpoint(filepath=save_dir+model_name, verbose=1, save_best_only=True)
early_stop = callbacks.EarlyStopping(patience=early_stopping_patience, verbose=1)

# build the model: 1 layer LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(hidden_size, return_sequences=True, input_shape=(maxlen, input_size)))
model.add(Dropout(dropout_rate))
model.add(LSTM(hidden_size, return_sequences=False))
model.add(Dense(input_size))

model.compile(loss='mse', optimizer='adam')

model.fit(X, y, batch_size=batch_size, validation_split=val_percentage, callbacks=[check_pointer, early_stop])