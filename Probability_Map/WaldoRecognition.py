from time import strftime

import tensorflow as tf
from numpy import array as np_array

from Probability_Map.WaldoPKL import load_waldo_pkl


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

class WaldoRecogonizer:

    def __init__(self):

        self.input = tf.placeholder(tf.float32, shape=[32, 32, 3])
        self.output = tf.placeholder(tf.float32, shape=[1])

        self.weights1 = weight_variable([5, 5, 3, 16])
        self.biases1 = bias_variable([16])

        self.reshaped_input = tf.reshape(self.input, [-1, 32, 32, 3])

        self.conv1 = tf.nn.sigmoid(conv2d(self.reshaped_input, self.weights1) + self.biases1)
        self.pool1 = max_pool_2x2(self.conv1)

        self.weights2 = weight_variable([5, 5, 16, 32])
        self.biases2 = bias_variable([32])

        self.conv2 = tf.nn.sigmoid(conv2d(self.pool1, self.weights2) + self.biases2)
        self.pool2 = max_pool_2x2(self.conv2)

        self.fc_weights1 = weight_variable([8 * 8 * 32, 512])
        self.fc_biases1 = bias_variable([512])

        self.h_pool2_flat = tf.reshape(self.pool2, [-1, 8 * 8 * 32])
        self.h_fc1 = tf.nn.sigmoid(tf.matmul(self.h_pool2_flat, self.fc_weights1) + self.fc_biases1)

        self.fc_weights2 = weight_variable([512, 1])
        self.fc_biases2 = bias_variable([1])

        self.network = tf.matmul(self.h_fc1, self.fc_weights2) + self.fc_biases2

    def run(self, session, input):
        output = session.run(self.network, {self.input: input})
        return output

    def train(self, session, training_data, labels, learning_rate=1e-2):

        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.output, logits=self.network))
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
        correct_prediction = tf.equal(self.network, self.output)
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        session.run(tf.global_variables_initializer())
        for x, y in zip(training_data, labels):
            session.run(train_step, feed_dict={self.input: x, self.output: y})



    def save_variables(self, session, path=None):
        if (path is None):
            path = "./Waldo Recognizer {}.ckpt".format(strftime("%Y-%m-%d  %H_%M_%S"))
        saver = tf.train.Saver()
        return saver.save(session, path)


if __name__ == "__main__":
    wr = WaldoRecogonizer()

    training_data = load_waldo_pkl()

    with tf.Session() as session:
        for _ in range(2):
            wr.train(session, training_data, [np_array([1]) for _ in range(len(training_data))])
        print(wr.save_variables(session))