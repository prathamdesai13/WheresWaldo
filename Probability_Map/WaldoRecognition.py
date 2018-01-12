from time import time, strftime

import tensorflow as tf
from numpy import array as np_array
from numpy.random import shuffle

from Probability_Map.WaldoPKL import load_waldo_pkl, load_not_waldo_pkl


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

weights1 = weight_variable([5, 5, 3, 64], "weights1")
biases1 = bias_variable([64], "biases1")

reshaped_input = tf.reshape(input, [-1, 32, 32, 3])

conv1 = tf.nn.relu(tf.add(conv2d(reshaped_input, weights1), biases1))
pool1 = max_pool_2x2(conv1)

weights2 = weight_variable([5, 5, 64, 128], "weights2")
biases2 = bias_variable([128], "biases2")

conv2 = tf.nn.relu(tf.add(conv2d(pool1, weights2), biases2))
pool2 = max_pool_2x2(conv2)

weights3 = weight_variable([3, 3, 128, 256], "weights3")
biases3 = bias_variable([256], "biases3")

conv3 = tf.nn.relu(tf.add(conv2d(pool2, weights3), biases3))
pool3 = max_pool_2x2(conv3)

fc_weights1 = weight_variable([4 * 4 * 256, 256], "fc_weights1")
fc_biases1 = bias_variable([256], "fc_biases1")

h_pool3_flat = tf.reshape(pool3, [-1, 4 * 4 * 256])
h_fc1 = tf.nn.relu(tf.add(tf.matmul(h_pool3_flat, fc_weights1), fc_biases1))

fc_weights2 = weight_variable([256, 2], "fc_weights2")
fc_biases2 = bias_variable([2], "fc_biases2")

network = tf.add(tf.matmul(h_fc1, fc_weights2), fc_biases2, name="network")

loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=output, logits=network))
train_network = tf.train.AdamOptimizer(1e-4).minimize(loss)

correct_prediction = tf.equal(tf.argmax(network, 1), tf.argmax(output, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

if __name__ == "__main__":

    # config=tf.ConfigProto(log_device_placement=True)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        training_data1 = load_waldo_pkl()
        training_data2 = load_not_waldo_pkl()
        epoch = 7
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
            for i in range(30):
                batches[0].append(training_data2[i])
                batches[1].append(np_array([0, 1]))
            session.run(train_network, feed_dict={input: batches[0], output: batches[1]})
        end = time()
        print("Took",(end-start),"Seconds to run",epoch,"Epochs")

        test_data = [[], []]
        for data in training_data1:
            test_data[0].append(data)
            test_data[1].append(np_array([1, 0]))

        a1 = accuracy.eval(feed_dict={input: test_data[0], output: test_data[1]})
        print("\nWaldo Accuracy:  ", (a1 * 100), "%")

        test_data = [[], []]
        training_data2 = load_not_waldo_pkl()
        for data in training_data2:
            test_data[0].append(data)
            test_data[1].append(np_array([0, 1]))

        a2 = accuracy.eval(feed_dict={input: test_data[0], output: test_data[1]})
        print("\nNot Waldo Accuracy:  ", (a2 * 100), "%")

        print("\nAverage Accuracy:  ", (a1+a2)*50, "%")

        if (a1+a2)/2 > 0.9:
            saver = tf.train.Saver()
            path = "./CNN Waldo Recognizer___{}".format(strftime("%Y-%m-%d_%H.%M.%S"))

            print(saver.save(session, path))
        # for data in training_data1:
        #     print(wr.run(session, data))
