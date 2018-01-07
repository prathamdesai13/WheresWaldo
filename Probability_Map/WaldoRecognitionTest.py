import tensorflow as tf

from Probability_Map.WaldoPKL import load_waldo_pkl
from Probability_Map.WaldoRecognition import WaldoRecogonizer


def test(path):

    with tf.Session() as session:
        wr = WaldoRecogonizer()

        training_data = load_waldo_pkl()

        tf.train.Saver().restore(session, path)

        session.run(tf.global_variables_initializer())

        # for x, p in zip(training_data, probabilities):
        #     p.append(wr.run(session, x))

        for x in training_data:
            print(wr.run(session, x))



if __name__ == "__main__":
    # train()
    test("./Waldo Recognizer 2018-01-06  21_40_45.ckpt")