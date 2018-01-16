from time import time, strftime
import matplotlib.pyplot as plt
import tensorflow as tf
from numpy import array as np_array
from numpy.random import shuffle

from Probability_Map.WaldoPKL import load_waldo_pkl, load_not_waldo_pkl, \
                                     load_test_waldo_pkl, load_test_not_waldo_pkl


def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

input = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name="input")
output = tf.placeholder(tf.float32, shape=[None, 2], name="output")

weights1 = weight_variable([3, 3, 3, 64], "weights1")
biases1 = bias_variable([64], "biases1")

reshaped_input = tf.reshape(input, [-1, 32, 32, 3])

conv1 = tf.nn.relu(tf.add(conv2d(reshaped_input, weights1), biases1))
pool1 = max_pool_2x2(conv1)

weights2 = weight_variable([3, 3, 64, 128], "weights2")
biases2 = bias_variable([128], "biases2")

conv2 = tf.nn.relu(tf.add(conv2d(pool1, weights2), biases2))
pool2 = max_pool_2x2(conv2)

weights3 = weight_variable([3, 3, 128, 256], "weights3")
biases3 = bias_variable([256], "biases3")

conv3 = tf.nn.relu(tf.add(conv2d(pool2, weights3), biases3))
pool3 = max_pool_2x2(conv3)

fc_weights1 = weight_variable([4 * 4 * 256, 256], "fc_weights1")
fc_biases1 = bias_variable([256], "fc_biases1")

pool3_flat = tf.reshape(pool3, [-1, 4 * 4 * 256])
fc1 = tf.nn.relu(tf.add(tf.matmul(pool3_flat, fc_weights1), fc_biases1))

keep_prob = tf.placeholder(tf.float32, name="keep_prob")
fc1_drop = tf.nn.dropout(fc1, keep_prob)

fc_weights2 = weight_variable([256, 2], "fc_weights2")
fc_biases2 = bias_variable([2], "fc_biases2")

network = tf.add(tf.matmul(fc1_drop, fc_weights2), fc_biases2, name="network")

loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=output, logits=network))
train_network = tf.train.AdamOptimizer(1e-4).minimize(loss)

correct_prediction = tf.equal(tf.argmax(network, 1), tf.argmax(output, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")


def get_accuracy(training_data, label, set_name, forward=False):

    batches = [[], []]
    for data in training_data:
        batches[0].append(data)
        batches[1].append(label)

    a = accuracy.eval(feed_dict={input: batches[0], output: batches[1], keep_prob:1})
    print(set_name,"Accuracy:  ", (a * 100), "%","\n")
    if forward:
        print(network.eval(feed_dict={input:batches[0]}))

    return a

if __name__ == "__main__":

    # config=tf.ConfigProto(log_device_placement=True)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        training_data1 = load_waldo_pkl()
        training_data2 = load_not_waldo_pkl()
        epoch = 100
        start = time()
        for i in range(epoch):
            # if i%10 == 0:
            print("Epoch:  ",i)
            batches = [[], []]
            shuffle(training_data1)
            for i in range(25):
                batches[0].append(training_data1[i])
                batches[1].append(np_array([1, 0]))
            shuffle(training_data2)
            for i in range(25):
                batches[0].append(training_data2[i])
                batches[1].append(np_array([0, 1]))
            session.run(train_network, feed_dict={input: batches[0], output: batches[1], keep_prob:0.25})
        end = time()
        print("Took",(end-start),"Seconds to run",epoch,"Epochs")

        a1 = get_accuracy(training_data1, np_array([1, 0]), "Waldo")
        a2 = get_accuracy(training_data2, np_array([0, 1]), "Not Waldo")
        a3 = get_accuracy(load_test_waldo_pkl(), np_array([1, 0]), "Test Waldo")
        a4 = get_accuracy(load_test_not_waldo_pkl(), np_array([0, 1]), "Test Not Waldo")

        print("\nAverage Accuracy:  ", (a1 + a2 + a3 + a4) * 25, "%")

        if (a1+a2)/2 >= 0.95 and a3 > 0.5 and a4 > 0.5:
            saver = tf.train.Saver()
            path = "./CNN Waldo Recognizer___{}".format(strftime("%Y-%m-%d_%H.%M.%S"))

            print(saver.save(session, path))
