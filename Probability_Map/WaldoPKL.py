from _pickle import dump as to_pkl, load as load_pkl
from os import listdir

import matplotlib.pyplot as plt
from numpy import array as np_array


def pkl_waldo_images(dim="32"):

    filelist = listdir("../Cropped Waldos/Waldos "+dim+"x"+dim)

    x = np_array([np_array(plt.imread("../Cropped Waldos/Waldos "+dim+"x"+dim+"/" + fname)) for fname in filelist])

    with open('./waldo.pkl', 'wb') as f:
        to_pkl(x, f, protocol=2)

def pkl_not_waldo_images(dim="32"):

    filelist = listdir("../Cropped Waldos/Not Waldos "+dim+"x"+dim)

    x = np_array([np_array(plt.imread("../Cropped Waldos/Not Waldos "+dim+"x"+dim+"/" + fname)) for fname in filelist])

    with open('./not_waldo.pkl', 'wb') as f:
        to_pkl(x, f, protocol=2)


def load_waldo_pkl():
    with open('./waldo.pkl', 'rb') as f:
        data = load_pkl(f)
    return data

def load_not_waldo_pkl():
    with open('./not_waldo.pkl', 'rb') as f:
        data = load_pkl(f)
    return data

if __name__ == "__main__":

    pkl_waldo_images()
    pkl_not_waldo_images()


