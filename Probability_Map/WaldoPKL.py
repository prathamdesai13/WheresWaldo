from _pickle import dump as to_pkl, load as load_pkl
from os import listdir

import matplotlib.pyplot as plt
from numpy import array as np_array


def pkl_waldo_images():

    filelist = listdir("../Cropped Waldos/Waldos 30x30")

    x = np_array([np_array(plt.imread("../Cropped Waldos/Waldos 30x30/" + fname)) for fname in filelist])

    with open('./waldo.pkl', 'wb') as f:
        to_pkl(x, f, protocol=2)


def load_waldo_pkl():
    with open('./waldo.pkl', 'rb') as f:
        data = load_pkl(f)
    return data

if __name__ == "__main__":

    pkl_waldo_images()


