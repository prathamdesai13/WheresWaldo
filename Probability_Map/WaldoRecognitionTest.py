import tensorflow as tf
from numpy import array as np_array

from Probability_Map.WaldoPKL import load_waldo_pkl, load_not_waldo_pkl, \
                                     load_test_waldo_pkl, load_test_not_waldo_pkl

input = None
output = None
network = None
accuracy = None

def test_waldo(path):

    with tf.Session() as session:

        saver = tf.train.import_meta_graph(path+'.meta')
        saver.restore(session, tf.train.latest_checkpoint('./'))

        # for x, p in zip(training_data1, probabilities):
        #     p.append(wr.run(session, x))

        graph = tf.get_default_graph()

        global input
        input = graph.get_tensor_by_name("input:0")
        global output
        output = graph.get_tensor_by_name("output:0")

        global network
        network = graph.get_tensor_by_name("network:0")
        global accuracy
        accuracy = graph.get_tensor_by_name("accuracy:0")

        a1 = get_accuracy(load_waldo_pkl(), np_array([1, 0]), "Waldo")
        a2 = get_accuracy(load_not_waldo_pkl(), np_array([0, 1]), "Not Waldo")
        a3 = get_accuracy(load_test_waldo_pkl(), np_array([1, 0]), "Test Waldo")
        a4 = get_accuracy(load_test_not_waldo_pkl(), np_array([0, 1]), "Test Not Waldo")

        print("\nAverage Accuracy:  ", (a1 + a2 + a3 + a4) * 25, "%")


def get_accuracy(training_data, label, set_name, forward=False):

    batches = [[], []]
    for data in training_data:
        batches[0].append(data)
        batches[1].append(label)

    a = accuracy.eval(feed_dict={input: batches[0], output: batches[1]})
    print(set_name,"Accuracy:  ", (a * 100), "%","\n")
    if forward:
        print(network.eval(feed_dict={input:batches[0]}))

    return a

if __name__ == "__main__":
    # train()
    test_waldo("CNN Waldo Recognizer___2018-01-11_21.22.59")