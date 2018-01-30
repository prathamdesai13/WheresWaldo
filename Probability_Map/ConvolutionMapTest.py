from time import time

import matplotlib.pyplot as plt
import tensorflow as tf
from numpy import array as np_array, reshape, zeros, full, maximum

from Heat.ProcessMaps import read_processed_map

dim = 32

def convolve_map(map, waldo_save, stride=int(dim/2/2)):
    with tf.Session() as session:

        saver = tf.train.import_meta_graph(waldo_save+'.meta')
        saver.restore(session, tf.train.latest_checkpoint('./'))

        graph = tf.get_default_graph()
        input = graph.get_tensor_by_name("input:0")
        keep_prob = graph.get_tensor_by_name("keep_prob:0")
        network = graph.get_tensor_by_name("network:0")

        probability_map_input = [[], []]
        for i in range((map.shape[0]-dim)//stride):
            # probability_map_input.append([])
            for j in range((map.shape[1]-dim)//stride):
                probability_map_input[0].append([stride*i, stride*j])
                probability_map_input[1].append(map[stride*i:stride*i+dim, stride*j:stride*j+dim])

        probability_map_input[0] = np_array(probability_map_input[0])
        probability_map_input[1] = np_array(probability_map_input[1])
        print(probability_map_input[1].shape)
        # check_probability_map_input(map, probability_map_input)
        probability_map = []
        size = 500
        for i in range(0, probability_map_input[1].shape[0], size):
            # print(i,":",min(i+size, len(probability_map_input)))
            probability_map.extend(
                network.eval(feed_dict={
                    input: probability_map_input[1][i: min(i+size, probability_map_input[1].shape[0])],
                    keep_prob:1}))

        print(len(probability_map_input[1]))
        print(len(probability_map))
        assert len(probability_map_input[1]) == len(probability_map)
        assert len(probability_map_input[0]) == len(probability_map)
        return probability_map_input[0], np_array(probability_map)



if __name__ == "__main__":
    print("Starting Map Test")

    map = "3.png"
    save = "./CNN Waldo Recognizer___2018-01-29_21.18.48"
    r = 0.9
    r_1 = 1-r

    original_map = np_array(plt.imread("../Maps/Unprocessed/"+map))
    print(original_map.shape)
    # processed_map = read_processed_map("../Maps/Processed/"+map+".pkl")

    print("Analysing...")
    start = time()

    locations, probabilities = convolve_map(original_map, save)

    end = time()
    print("Took",(end-start),"Seconds to analyse")
    start = time()

    overlay = zeros(original_map.shape[0:2])
    print(overlay.shape)
    for l, x in zip(locations, probabilities):
        p = abs(x[0])/(abs(x[0])+abs(x[1]))
        p = p*r_1 if p > r else 0
        section = full((dim, dim), p)
        overlay[l[0]:l[0]+dim, l[1]:l[1]+dim] = maximum(overlay[l[0]:l[0]+dim, l[1]:l[1]+dim], section)

    end = time()
    print("Took",(end-start),"Seconds to process")


    plt.imshow(original_map, cmap='jet')
    plt.imshow(overlay, cmap='gray', alpha=0.75)
    plt.show()

