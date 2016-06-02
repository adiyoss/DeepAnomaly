import pickle
from matplotlib import pyplot as plt

def read_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


# dd = read_data('ECG5000_TRAIN_CONTINUOUS_SIGNAL_1.pkl')
# plt.plot(dd)
# plt.show()