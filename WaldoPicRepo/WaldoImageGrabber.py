from os import listdir
from time import time

from matplotlib.pyplot import imread
from numpy import array as np_array
from numpy.random import shuffle
from scipy import misc

from Heat.HeatMap import process


def process_images():
    path_list = listdir("./32/waldos")
    for image in path_list:
        img = misc.imread("./32/waldos/" + image) / 255.0
        imagio = process(filepath=None, im=img)
        misc.imsave('./32process/waldos/' + image, imagio)

# process_images()

waldo_dirs = ["../WaldoPicRepo/32process/waldos/"]  # , "../WaldoPicRepo/64bwprocess/"
not_waldo_dirs = ["../WaldoPicRepo/32process/notwaldos/"]  # , "../WaldoPicRepo/64bwprocess/"

def get_training_data():
    print("Starting Image Grab")
    start = time()

    waldo_data = []
    for dir in waldo_dirs:
        path_list = listdir(dir)
        for image in path_list:
            waldo_data.append([np_array(imread(dir+image)),
                               np_array([1, 0])])

    not_waldo_data = []
    for dir in not_waldo_dirs:
        path_list = listdir(dir)
        for image in path_list:
            not_waldo_data.append([np_array(imread(dir+image)),
                                   np_array([0, 1])])

    shuffle(waldo_data)
    shuffle(not_waldo_data)

    not_waldo_test = not_waldo_data[:300]
    not_waldo_train = not_waldo_data[300:]

    end = time()
    print("Took "+str(end-start)+" Seconds to grab images")

    return waldo_data, not_waldo_train, not_waldo_test



