from time import time, strftime

import tensorflow as tf
from numpy.random import shuffle

from WaldoPicRepo.WaldoImageGrabber import get_training_data


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

input_dim = 32
output_dim = int(input_dim/2/2)

input = tf.placeholder(tf.float32, shape=[None, input_dim, input_dim, 3], name="input")
output = tf.placeholder(tf.float32, shape=[None, 2], name="output")

weights1 = weight_variable([3, 3, 3, 64], "weights1")
biases1 = bias_variable([64], "biases1")

reshaped_input = tf.reshape(input, [-1, input_dim, input_dim, 3])

conv1 = tf.nn.relu(tf.add(conv2d(reshaped_input, weights1), biases1))
pool1 = max_pool_2x2(conv1)

weights2 = weight_variable([3, 3, 64, 128], "weights2")
biases2 = bias_variable([128], "biases2")

conv2 = tf.nn.relu(tf.add(conv2d(pool1, weights2), biases2))
pool2 = max_pool_2x2(conv2)

# weights3 = weight_variable([3, 3, 128, 256], "weights3")
# biases3 = bias_variable([256], "biases3")
#
# conv3 = tf.nn.relu(tf.add(conv2d(pool2, weights3), biases3))
# pool3 = max_pool_2x2(conv3)

fc_weights1 = weight_variable([output_dim * output_dim * 128, 128], "fc_weights1")
fc_biases1 = bias_variable([128], "fc_biases1")

pool2_flat = tf.reshape(pool2, [-1, output_dim * output_dim * 128])
fc1 = tf.nn.relu(tf.add(tf.matmul(pool2_flat, fc_weights1), fc_biases1))

keep_prob = tf.placeholder(tf.float32, name="keep_prob")
fc1_drop = tf.nn.dropout(fc1, keep_prob)

fc_weights2 = weight_variable([128, 2], "fc_weights2")
fc_biases2 = bias_variable([2], "fc_biases2")

network = tf.add(tf.matmul(fc1_drop, fc_weights2), fc_biases2, name="network")

loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=output, logits=network))
train_network = tf.train.AdamOptimizer(1e-4).minimize(loss)

correct_prediction = tf.equal(tf.argmax(network, 1), tf.argmax(output, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")


def get_accuracy(test_data, set_name, forward=False):

    test_batch = [[], []]
    for i in range(0, len(test_data)):
        test_batch[0].append(test_data[i][0])
        test_batch[1].append(test_data[i][1])

    a = accuracy.eval(feed_dict={input: test_batch[0], output: test_batch[1], keep_prob:1})
    print(set_name,"Accuracy:  ", (a * 100), "%","\n")
    if forward:
        print(network.eval(feed_dict={input:test_batch[0]}))

    return a

if __name__ == "__main__":

    # config=tf.ConfigProto(log_device_placement=True)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        waldo_data, not_waldo_train, not_waldo_test = get_training_data()
        epoch = 1000
        start = time()
        for i in range(epoch):
            if i%100 == 0:
                print("Epoch:  ",i)
            shuffle(waldo_data)
            shuffle(not_waldo_train)
            batches = [[], []]
            for i in range(10):
                batches[0].append(waldo_data[i][0])
                batches[1].append(waldo_data[i][1])
                for j in range(15):
                    batches[0].append(not_waldo_train[10*i+j][0])
                    batches[1].append(not_waldo_train[10*i+j][1])
            session.run(train_network, feed_dict={input: batches[0], output: batches[1], keep_prob:0.5})
        end = time()
        print("Took",(end-start),"Seconds to run",epoch,"Epochs")

        test_data = []
        for test in waldo_data:
            test_data.append([test[0], test[1]])

        a1 = get_accuracy(test_data, "Waldo Test")

        test_data = []
        for test in not_waldo_test:
            test_data.append([test[0], test[1]])

        a2 = get_accuracy(test_data, "Not Waldo Test")

        if a1 > 0.75 and a2 > 0.95:
            saver = tf.train.Saver()
            path = "./CNN Waldo Recognizer___{}".format(strftime("%Y-%m-%d_%H.%M.%S"))

            print(saver.save(session, path))
