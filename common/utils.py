import pickle
import numpy as np


def read_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def create_sequences(data, win_size=20, step=1):
    """
    :param data:
    :param win_size:
    """
    docX = []
    for i in range(0, len(data) - win_size, step):
        docX.append(data[i:i + win_size])
    alsX = np.array(docX)

    return alsX


def store_prediction_and_ground_truth(model):
    input_size = 1
    maxlen = 140
    batch_size = 32

    db = read_data('../data/ECG5000_TEST_PHASE_1_CONTINUOUS_SIGNAL_1.pkl')
    X = create_sequences(db[:-140], win_size=maxlen, step=maxlen)
    X = np.reshape(X, (X.shape[0], X.shape[1], input_size))
    Y = create_sequences(db[140:], win_size=maxlen, step=maxlen).flatten()

    prediction = model.predict(X, batch_size, verbose=1)
    prediction = prediction.flatten()
    with open('../data/ECG5000_TRAIN_PHASE_2_CONTINUOUS_SIGNAL_1.pkl', 'wb') as f:
        pickle.dump(np.stack((Y, prediction)), f)


def prepare_data():
    test_data = read_data("../data/ECG5000_TEST_CONTINUOUS_SIGNAL_1.pkl")
    test_data_half_len = int(len(test_data) / 2)

    with open("../data/ECG5000_TEST_PHASE_1_CONTINUOUS_SIGNAL_1.pkl", "wb") as f:
        pickle.dump(test_data[:test_data_half_len], f)

    with open("../data/ECG5000_TEST_PHASE_2_CONTINUOUS_SIGNAL_1.pkl", "wb") as f:
        pickle.dump(test_data[test_data_half_len:], f)


def build_data_auto_encoder(data, step, win_size):
    count = data.shape[1] / float(step)
    docX = np.zeros((count, 3, win_size))

    for i in range(0, data.shape[1] - win_size, step):
        c = i / step
        docX[c][0] = np.abs(data[0, i:i + win_size] - data[1, i:i + win_size])
        docX[c][1] = np.power(data[0, i:i + win_size] - data[1, i:i + win_size], 2)
        docX[c][2] = np.pad(
            (data[0, i:i + win_size - 1] - data[0, i + 1:i + win_size]) * (data[1, i:i + win_size - 1] - data[1, i + 1:i + win_size]),
            (0, 1), 'constant', constant_values=0)
    data = np.dstack((docX[:, 0], docX[:, 1], docX[:, 2])).reshape(docX.shape[0], docX.shape[1]*docX.shape[2])

    return data

