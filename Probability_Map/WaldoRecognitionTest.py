import tensorflow as tf
from numpy import array as np_array

from Probability_Map.WaldoPKL import load_waldo_pkl, load_not_waldo_pkl


def test_waldo(path):

    with tf.Session() as session:

        saver = tf.train.import_meta_graph(path+'.meta')
        saver.restore(session, tf.train.latest_checkpoint('./'))

        # for x, p in zip(training_data1, probabilities):
        #     p.append(wr.run(session, x))

        graph = tf.get_default_graph()

        input = graph.get_tensor_by_name("input:0")
        output = graph.get_tensor_by_name("output:0")

        network = graph.get_tensor_by_name("network:0")
        accuracy = graph.get_tensor_by_name("accuracy:0")

        batches = [[], []]
        training_data1 = load_waldo_pkl()
        for data in training_data1:
            batches[0].append(data)
            batches[1].append(np_array([1, 0]))

        a1 = accuracy.eval(feed_dict={input:batches[0], output:batches[1]})
        print("\nWaldo Accuracy:  ",(a1*100),"%")
        print(network.eval(feed_dict={input:batches[0]}))


        batches = [[], []]
        training_data2 = load_not_waldo_pkl()
        for data in training_data2:
            batches[0].append(data)
            batches[1].append(np_array([0, 1]))

        a2 = accuracy.eval(feed_dict={input:batches[0], output:batches[1]})
        print("\nNot Waldo Accuracy:  ",(a2*100),"%")
        print(network.eval(feed_dict={input:batches[0]}))

        print("\nAverage Accuracy:  ", (a1 + a2) * 50, "%")




if __name__ == "__main__":
    # train()
    test_waldo("CNN Waldo Recognizer___2018-01-11_21.22.59")