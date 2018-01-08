import tensorflow as tf

from Probability_Map.WaldoPKL import load_waldo_pkl, load_not_waldo_pkl


def test_waldo(path):

    with tf.Session() as session:
        training_data1 = load_waldo_pkl()
        training_data2 = load_not_waldo_pkl()

        saver = tf.train.import_meta_graph(path+'.meta')
        saver.restore(session, tf.train.latest_checkpoint('./'))

        # for x, p in zip(training_data1, probabilities):
        #     p.append(wr.run(session, x))

        graph = tf.get_default_graph()
        network = graph.get_tensor_by_name("network:0")

        input = graph.get_tensor_by_name("input:0")

        for x in training_data1:
            print(session.run(network, {input:x}))
        print("________________________________________________")
        for x in training_data2:
            print(session.run(network, {input:x}))



if __name__ == "__main__":
    # train()
    test_waldo("./Waldo Recognizer  2018-01-07 17_31_49")