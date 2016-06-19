from keras.layers import Dropout, Dense
from keras.layers import LSTM, TimeDistributed
from keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
import reader
from utils import create_sequences

input_size = 1
hidden_size = 128
dropout_rate = 0.2
maxlen = 140
batch_size = 32
early_stopping_patience = 5
save_dir = 'results/'
model_name = 'model_1.net'
val_percentage = 0.15

path_test = 'data/ECG5000_TEST_CONTINUOUS_SIGNAL_1.pkl'
db = reader.read_data(path_test)
X = create_sequences(db[:-140], win_size=maxlen, step=maxlen)

X = np.reshape(X, (X.shape[0], X.shape[1], input_size))

# build the model: 1 layer LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(hidden_size, return_sequences=False, input_shape=(maxlen, input_size)))
model.add(Dense(140))
# model.add(Dropout(dropout_rate))
# model.add(LSTM(hidden_size, return_sequences=False))
# model.add(Dense(input_size))

model.load_weights(save_dir + model_name)
model.compile(loss='mse', optimizer='adam')

# prediction_container = []
# value = X[0][0]
# # value = np.reshape(value, (1, value.shape[0], value.shape[1]))
# value = np.reshape(value, (1, 1, 1))

# for i in range(1000):
#     prediction = model.predict(value, verbose=1)
#     value = prediction
#     prediction_container.append(prediction[0])

prediction = model.predict(X, batch_size, verbose=1)
prediction = prediction.flatten()
# prediction_container = np.array(prediction).flatten()
Y = db[140:]
plt.plot(prediction, label='prediction')
plt.plot(Y, label='true')
plt.legend()
plt.show()
