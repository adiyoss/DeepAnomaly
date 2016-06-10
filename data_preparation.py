import reader
import cPickle as pickle
test_data = reader.read_data("data/ECG5000_TEST_CONTINUOUS_SIGNAL_1.pkl")
test_data_half_len = int(len(test_data)/2)

with open("data/ECG5000_TEST_PHASE_1_CONTINUOUS_SIGNAL_1.pkl", "wb") as f:
    pickle.dump(test_data[:test_data_half_len], f)

with open("data/ECG5000_TEST_PHASE_2_CONTINUOUS_SIGNAL_1.pkl", "wb") as f:
    pickle.dump(test_data[test_data_half_len:], f)