from _pickle import dump as to_pkl, load as load_pkl

import matplotlib.pyplot as plt

from Heat.gosharubsinky import process

def process_and_pkl_map(path, output_path):
    print("Processing:   ",path)
    processed = process(path)

    with open(output_path, 'wb') as f:
        to_pkl(processed, f, protocol=2)

def read_processed_map(path):
    with open(path, 'rb') as f:
        map = load_pkl(f)
    return map

if __name__ == "__main__":

    map = read_processed_map("../Maps/Processed/7.png.pkl")
    plt.imshow(map)
    plt.show()

    # map_list = listdir("../Maps/Unprocessed")
    #
    # for map in map_list:
    #     process_and_pkl_map("../Maps/Unprocessed/"+map, "../Maps/Processed/"+map+".pkl")
