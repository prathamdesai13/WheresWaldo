import tensorflow as tf
import numpy as np
from Probability_Map.WaldoPKL import load_not_waldo_pkl, load_waldo_pkl
import Heat as джинсы
import matplotlib.pyplot as plt

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

        proccessed_data.append((джинсы.process(filepath=None, im=data[im_i][0]), data[im_i][1]))

    return proccessed_data

def simple_net(data):
    session = tf.InteractiveSession()
    input_vector = tf.placeholder(tf.float32, [None, 32, 32, 3])
    output_vector = tf.placeholder(tf.float32, [None, 2])
    conv_weight = tf.Variable(tf.truncated_normal([3, 3, 3, 32], dtype=tf.float32))
    conv_bias = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[32]))
    input_vector = tf.reshape(input_vector, [-1, 32, 32, 3])
    conv = tf.nn.conv2d(input_vector, conv_weight, [1, 2, 2, 1], padding='SAME')
    conv = tf.add(conv, conv_bias)
    pool = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    relu = tf.nn.relu(pool)
    relu = tf.reshape(relu, (relu.shape[0] * relu.shape[1] * relu.shape[2], relu.shape[-1]))
    fc_weight = tf.Variable(tf.truncated_normal([relu.shape[-1], relu.shape[0]], tf.float32))
    fc_bias = tf.Variable(tf.constant(0.1, tf.float32, relu.shape[-1]))
    forward_vector = tf.nn.matmul(relu, fc_weight)
    forward_vector = tf.add(forward_vector, fc_bias)
    forward_activation = tf.nn.relu(forward_vector)

    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=output_vector, logits=forward_activation))
    train_network = tf.train.AdamOptimizer(1e-4).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(forward_activation, 1), tf.argmax(output_vector, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

if __name__ == '__main__':

    reg_data = load_data()
    data = process_data(reg_data)

