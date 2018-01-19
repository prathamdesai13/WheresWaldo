from time import time

import matplotlib.pyplot as plt
import tensorflow as tf
from numpy import array as np_array, reshape

from Heat.ProcessMaps import read_processed_map

dim = 64

def get_probability_map(map, waldo_save="./CNN Waldo Recognizer___2018-01-19_17.29.33"):
    with tf.Session() as session:

        saver = tf.train.import_meta_graph(waldo_save+'.meta')
        saver.restore(session, tf.train.latest_checkpoint('./'))

        graph = tf.get_default_graph()
        input = graph.get_tensor_by_name("input:0")
        keep_prob = graph.get_tensor_by_name("keep_prob:0")
        network = graph.get_tensor_by_name("network:0")

        probability_map_input = []
        for i in range(map.shape[0]//dim):
            # probability_map_input.append([])
            for j in range(map.shape[1]//dim):
                probability_map_input.append(map[dim*i:dim*(i+1), dim*j:dim*(j+1)])

        probability_map_input = np_array(probability_map_input)
        print(probability_map_input.shape)
        # check_probability_map_input(map, probability_map_input)
        probability_map = []
        size = 500
        for i in range(0, len(probability_map_input), size):
            # print(i,":",min(i+size, len(probability_map_input)))
            probability_map.extend(
                network.eval(feed_dict={
                    input: probability_map_input[i: min(i+size, len(probability_map_input))], keep_prob:1}))

        print(len(probability_map_input))
        print(len(probability_map))
        assert len(probability_map_input) == len(probability_map)
        return reshape(a=np_array(probability_map), newshape=(map.shape[0]//dim, map.shape[1]//dim, 2))

# def check_probability_map_input(map, probability_map_input):
#     print("Checking if Map matches Probability Map Input...")
#     for i in range(map.shape[0]//32*32):
#         i_ = i//32
#         imod = i%32
#         for j in range(map.shape[1]//32*32):
#             map_at = map[i][j]
#             probability_map_input_at = probability_map_input[i_, j//32, imod, j%32, :]
#             if map_at[0] != probability_map_input_at[0] or \
#                map_at[1] != probability_map_input_at[1] or \
#                map_at[2] != probability_map_input_at[2]:
#                 print(i, j)
#                 print(i_, j//32, imod, j%32)
#                 print(map_at)
#                 print(probability_map_input_at)
#                 exit(1)
#     print("Map matches Probability Map Input")

if __name__ == "__main__":
    print("Starting Map Test")

    map = "3.png"

    original_map = np_array(plt.imread("../Maps/Unprocessed/"+map))
    print(original_map.shape)
    processed_map = read_processed_map("../Maps/Processed/"+map+".pkl")

    print("Analysing...")
    start = time()
    analysed = get_probability_map(processed_map)
    probability_map = []
    r = 0.2
    for row in analysed:
        probability_map.append([])
        for x in row:
            p = abs(x[0])/(abs(x[0])+abs(x[1]))
            probability_map[-1].append(p*r if p > (1-r) else 0)

    overlay = []
    for i in range(original_map.shape[0]//dim*dim):
        overlay.append([])
        for j in range(original_map.shape[1]//dim*dim):
            overlay[-1].append(probability_map[i//dim][j//dim])

    plt.imshow(original_map, cmap='jet')
    plt.imshow(overlay, cmap='gray', alpha=0.75)
    plt.show()
    end = time()
    print("Took",(end-start),"Seconds to process")

