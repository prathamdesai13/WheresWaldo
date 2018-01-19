import os
import tensorflow as tf
import numpy as np
from Probability_Map.WaldoPKL import load_waldo_pkl, load_not_waldo_pkl
import matplotlib.pyplot as plt
import Heat as heat
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


def load_data():

    waldos = load_waldo_pkl()
    not_waldos = load_not_waldo_pkl()

    data = []
    for i in range(waldos.shape[0]):
        data.append((waldos[i], 1))

    for i in range(not_waldos.shape[0]):
        data.append((not_waldos[i], 0))

    return data

def process_data(data):

    proccessed_data = []

    for im_i in range(len(data)):

        x, y = data[im_i]

        #x = np.reshape(x, [1, x.shape[0] * x.shape[1] * x.shape[2]])

        proccessed_data.append((heat.process(filepath=None, im=x), y))

        x, y = proccessed_data[im_i]

        proccessed_data[im_i] = (np.reshape(x, [1, x.shape[0] * x.shape[1] * x.shape[2]]), np.reshape(np.array([y]), (1, 1)))

    return proccessed_data

def flatten(processed_data):

    for i in range(len(processed_data)):

        x, y = processed_data[i]

        x = np.reshape(x, [1, 32 * 32 * 3])

        processed_data[i] = (x[0], y)

    return processed_data

def network(data, height, width, channels):

    vector = tf.placeholder(tf.float32, [1, height * width * channels])
    target = tf.placeholder(tf.float32, [1, 1])

    weight_1 = tf.Variable(tf.truncated_normal([height * width * channels, 10], stddev=0.1))
    bias_1 = tf.Variable(tf.zeros([10]))
    weight_2 = tf.Variable(tf.truncated_normal([10, 5], stddev=0.1))
    bias_2 = tf.Variable(tf.zeros([5]))
    weight_3 = tf.Variable(tf.truncated_normal([5, 1], stddev=0.1))
    bias_3 = tf.Variable(tf.zeros([1]))

    forward_1 = tf.nn.sigmoid(tf.matmul(vector, weight_1) + bias_1)
    forward_2 = tf.nn.sigmoid(tf.matmul(forward_1, weight_2) + bias_2)
    forward_3 = tf.matmul(forward_2, weight_3) + bias_3

    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=forward_3))

    training = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

    with tf.Session() as session:

        tf.global_variables_initializer().run()

        for _ in range(200):

            for x in data:
                input_vector, target_vector = x

                _, loss = session.run([training, cross_entropy], feed_dict={vector:input_vector, target:target_vector})

                print(loss)

        """test_vector = plt.imread("/Users/niravdesai/Desktop/WheresWaldo/Cropped Waldos/Test Waldos 32x32/Waldos/Waldo7.png")
        plt.imshow(test_vector)
        plt.show()
        test_vector = heat.process(filepath=None, im=test_vector)
        plt.imshow(test_vector)
        plt.show()
        test_vector = np.reshape(test_vector, (1, 32 * 32 * 3))
        print(sigmoid(forward_3.eval(feed_dict={vector: test_vector})[0]))"""

        map = plt.imread("/Users/niravdesai/Desktop/WheresWaldo/Maps/Unprocessed/1.png")
        plt.imshow(map)
        plt.show()
        map_vector = heat.process(filepath=None, im=map)
        plt.imshow(map_vector)
        plt.show()
        stride = 32
        prob_map = np.zeros_like(map)
        for height in range(map_vector.shape[0] - stride):
            for width in range(map_vector.shape[1] - stride):

                map_patch = map_vector[height:height + stride, width:width + stride, :]
                map_patch_vector = np.reshape(map_patch, (1, 32 * 32 * 3))

                probability = sigmoid(forward_3.eval(feed_dict={vector: map_patch_vector})[0])
                if probability > 0.96:
                    prob_map[height:height + stride, width:width + stride] = \
                        map[height:height + stride, width:width + stride, :]

        plt.imshow(prob_map)
        plt.show()

def sigmoid(x):

    return 1.0 / (1.0 + np.exp(-x))


if __name__ == '__main__':

    reg_data = load_data()
    data = process_data(reg_data)

    network(data, 32, 32, 3)
