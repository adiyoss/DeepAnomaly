import numpy as np


def create_sequences(data, win_size=20, step=1):
    """
    :param data:
    :param win_size:
    """
    docX = []
    for i in range(0, len(data)-win_size, step):
        docX.append(data[i:i+win_size])
    alsX = np.array(docX)

    return alsX