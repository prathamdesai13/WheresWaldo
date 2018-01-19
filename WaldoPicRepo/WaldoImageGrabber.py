from os import listdir
from time import time

from numpy import array as np_array
from numpy.random import shuffle

from Heat.HeatMap import process

waldo_dirs = ["../WaldoPicRepo/64/waldopng/",
            "../WaldoPicRepo/64-bw/waldopng/",
            "../WaldoPicRepo/64-gray/waldopng/"]

not_waldo_dirs = ["../WaldoPicRepo/64/notwaldopng/",
                "../WaldoPicRepo/64-bw/notwaldopng/",
                "../WaldoPicRepo/64-gray/notwaldopng/"]

def get_training_data():
    start = time()

    data = []
    for dir in waldo_dirs:
        path_list = listdir(dir)
        for path in path_list:
            data.append([np_array(process(filepath=dir+path)),
                         np_array([1, 0])])
    print("Processed Waldo Images")
    for dir in not_waldo_dirs:
        path_list = listdir(dir)
        for path in path_list:
            data.append([np_array(process(filepath=dir+path)),
                         np_array([0, 1])])

    shuffle(data)

    test_data = data[0:100]
    training_data = data[100:]

    end = time()
    print("Took "+str(end-start)+" Seconds to grab images")

    return training_data, test_data



