import tensorflow as tf
import numpy as np
from Probability_Map.WaldoPKL import load_not_waldo_pkl, load_waldo_pkl
import Heat.gosharubsinky as gosha
import matplotlib.pyplot as plt

def load_data():

    waldos = load_waldo_pkl()
    not_waldos = load_not_waldo_pkl()

    data = np.zeros((waldos.shape[0] + not_waldos.shape[0], waldos.shape[1], waldos.shape[2], waldos.shape[3]))

    return data

def process_data(training_data):

    proccessed_data = np.ones_like(training_data)

    for im_i in range(training_data.shape[0]):

        proccessed_data[im_i, :, :, :] = gosha.process(filepath=None, im=training_data[im_i, :, :, :])

    return proccessed_data

if __name__ == '__main__':

    reg_data = load_data()
    data = process_data(reg_data)
    print(reg_data[0].shape)
    plt.imshow(reg_data[0])
    plt.show()

