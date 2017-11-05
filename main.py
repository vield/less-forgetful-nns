import argparse
from copy import deepcopy
import csv
import sys

import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


from simple_network import Network


FLAGS = None



def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    np.random.seed(1)
    perm = np.random.permutation(mnist.train._images.shape[1])
    permuted = deepcopy(mnist)
    permuted.train._images = permuted.train._images[:, perm]
    permuted.test._images = permuted.test._images[:, perm]
    permuted.validation._images = permuted.validation._images[:, perm]


    if FLAGS.mode == 'simple':
        network = Network()
    elif FLAGS.mode == 'mixed':
        raise NotImplementedError("Mixed mode not implemented yet!")
    elif FLAGS.mode == 'ewc':
        raise NotImplementedError("EWC mode not implemented yet!")
    else:
        raise Exception("Unrecognized mode: " + FLAGS.mode)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    with open(FLAGS.mode + '.csv', 'w') as f:
        fieldnames = ["Epoch", "Group", "TestAccuracy"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(1001):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(network.train_step, feed_dict={network.inputs: batch_xs, network.correct_labels: batch_ys})
            if i % 50 == 0:
                test_accuracy1 = sess.run(
                    network.accuracy,
                    feed_dict={
                        network.inputs: mnist.test.images,
                        network.correct_labels: mnist.test.labels
                    }
                )
                test_accuracy2 = sess.run(
                    network.accuracy,
                    feed_dict={
                        network.inputs: permuted.test.images,
                        network.correct_labels: permuted.test.labels
                    }
                )

                writer.writerow({'Epoch': i, 'TestAccuracy': test_accuracy1, 'Group': 1})
                writer.writerow({'Epoch': i, 'TestAccuracy': test_accuracy2, 'Group': 2})
                #print("Training error:",
                #      sess.run(network.accuracy,
                #               feed_dict={
                #                   network.inputs: mnist.train.images,
                #                   network.correct_labels: mnist.train.labels
                #               }))
                #print(" Testing error:",
                #      sess.run(network.accuracy,
                #               feed_dict={
                #                   network.inputs: mnist.test.images,
                #                   network.correct_labels: mnist.test.labels
                #               }))

        for i in range(1001, 2001):
            batch_xs, batch_ys = permuted.train.next_batch(100)
            sess.run(network.train_step, feed_dict={network.inputs: batch_xs, network.correct_labels: batch_ys})
            if i % 50 == 0:
                test_accuracy1 = sess.run(
                    network.accuracy,
                    feed_dict={
                        network.inputs: mnist.test.images,
                        network.correct_labels: mnist.test.labels
                    }
                )
                test_accuracy2 = sess.run(
                    network.accuracy,
                    feed_dict={
                        network.inputs: permuted.test.images,
                        network.correct_labels: permuted.test.labels
                    }
                )

                writer.writerow({'Epoch': i, 'TestAccuracy': test_accuracy1, 'Group': 1})
                writer.writerow({'Epoch': i, 'TestAccuracy': test_accuracy2, 'Group': 2})

        print(sess.run(network.accuracy, feed_dict={network.inputs: mnist.train.images, network.correct_labels: mnist.train.labels}))
        print(sess.run(network.accuracy, feed_dict={network.inputs: mnist.test.images,
                                      network.correct_labels: mnist.test.labels}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./MNIST_data',
                      help='Directory for storing input data')
    parser.add_argument('--mode', type=str, default="simple", choices=('simple', 'mixed', 'ewc'))
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)