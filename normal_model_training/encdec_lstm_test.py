from keras.layers import Dropout, Dense, RepeatVector
from keras.layers import LSTM, TimeDistributed
from keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
import reader
from utils import create_sequences

input_size = 1
hidden_size = 256
dropout_rate = 0.2
maxlen = 100
batch_size = 32
early_stopping_patience = 5
save_dir = '../results/'
model_name = 'model.net'
val_percentage = 0.1

path_test = '../data/ECG5000_TEST_CONTINUOUS_SIGNAL_1.pkl'
db = reader.read_data(path_test)
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
plt.plot(y.flatten()[maxlen:4000+maxlen], label='true')
plt.legend()
plt.show()
