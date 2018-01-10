from time import strftime

import tensorflow as tf
from numpy import array as np_array
from numpy.random import shuffle

def weight_variable(shape, name):
  initial = tf.truncated_normal(shape)
  return tf.Variable(initial, name=name)

def bias_variable(shape, name):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name=name)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

class WaldoRecogonizer:

    def __init__(self):

        self.input = tf.placeholder(tf.float32, shape=[32, 32, 3], name="input")
        self.output = tf.placeholder(tf.float32, shape=[1])

        self.weights1 = weight_variable([5, 5, 3, 16], "weights1")
        self.biases1 = bias_variable([16], "biases1")

        reshaped_input = tf.reshape(self.input, [1, 32, 32, 3])

        conv1 = tf.nn.relu(tf.add(conv2d(reshaped_input, self.weights1), self.biases1))
        pool1 = max_pool_2x2(conv1)

        self.weights2 = weight_variable([5, 5, 16, 32], "weights2")
        self.biases2 = bias_variable([32], "biases2")

        conv2 = tf.nn.relu(tf.add(conv2d(pool1, self.weights2), self.biases2))
        pool2 = max_pool_2x2(conv2)

        self.fc_weights1 = weight_variable([8 * 8 * 32, 512], "fc_weights1")
        self.fc_biases1 = bias_variable([512], "fc_biases1")

        h_pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 32])
        h_fc1 = tf.nn.tanh(tf.add(tf.matmul(h_pool2_flat, self.fc_weights1), self.fc_biases1))

        self.fc_weights2 = weight_variable([512, 1], "fc_weights2")
        self.fc_biases2 = bias_variable([1], "fc_biases2")

        self.network = tf.nn.sigmoid(tf.add(tf.matmul(h_fc1, self.fc_weights2), self.fc_biases2), name="network")

        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.output, logits=self.network))
        self.train_step = tf.train.AdamOptimizer(1e-6).minimize(self.loss)
        # correct_prediction = tf.equal(self.network, self.output)
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def run(self, session, input):
        output = session.run(self.network, {self.input: input})
        return output

    def train(self, session, training_data, labels=None, learning_rate=1e-2):

        if labels is None:
            for x in training_data:
                session.run(self.train_step, feed_dict={self.input: x[0], self.output: x[1]})
        else:
            for x, y in zip(training_data, labels):
                session.run(self.train_step, feed_dict={self.input: x, self.output: y})


if __name__ == "__main__":

    with tf.Session() as session:
        wr = WaldoRecogonizer()

        session.run(tf.global_variables_initializer())

        training_data1 = load_waldo_pkl()
        labels1 = [np_array([1]) for _ in range(len(training_data1))]
        training_data2 = load_not_waldo_pkl()
        labels2 = [np_array([0]) for _ in range(len(training_data2))]
        for _ in range(1):
            shuffle(training_data1)
            shuffle(training_data2)
            training_data = []
            for i in range(50):
                training_data.append([training_data1[i], np_array([1])])
                training_data.append([training_data2[i], np_array([0])])
            wr.train(session, training_data)

        saver = tf.train.Saver()
        path = "./CNN Waldo Recognizer  {}".format(strftime("%Y-%m-%d %H_%M_%S"))

        print(saver.save(session, path))
        test_waldo(path)
        # for data in training_data1:
        #     print(wr.run(session, data))
