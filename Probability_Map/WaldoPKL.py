from _pickle import dump as to_pkl, load as load_pkl
from os import listdir

import matplotlib.pyplot as plt
from numpy import array as np_array

from Heat.gosharubsinky import process

dim = "32x32"

def pkl_images(input_path, output_path):
    if not input_path.endswith("/"):
        input_path += "/"
    filelist = listdir(input_path)

    x = np_array([process(im=np_array(plt.imread(input_path + fname))) for fname in filelist])

    with open(output_path, 'wb') as f:
        to_pkl(x, f, protocol=2)

def load_waldo_pkl():
    with open('./waldo.pkl', 'rb') as f:
        data = load_pkl(f)
    return data

def load_not_waldo_pkl():
    with open('./not_waldo.pkl', 'rb') as f:
        data = load_pkl(f)
    return data

def load_test_waldo_pkl():
    with open('./test_waldo.pkl', 'rb') as f:
        data = load_pkl(f)
    return data

def load_test_not_waldo_pkl():
    with open('./test_not_waldo.pkl', 'rb') as f:
        data = load_pkl(f)
    return data

if __name__ == "__main__":

    pkl_images("../Cropped Waldos/Waldos "+dim+"/Waldos", "./waldo.pkl")
    pkl_images("../Cropped Waldos/Waldos "+dim+"/Not Waldos", "./not_waldo.pkl")
    pkl_images("../Cropped Waldos/Test Waldos "+dim+"/Waldos", "./test_waldo.pkl")
    pkl_images("../Cropped Waldos/Test Waldos "+dim+"/Not Waldos", "./test_not_waldo.pkl")


