import argparse
from copy import deepcopy
import csv

import numpy as np


def run_training(network, sess, training_datasets, evaluation_datasets, verbose=True):
    with open(options.mode + '.csv', 'w') as f:
        fieldnames = ["Epoch", "Group", "TestAccuracy"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        epoch = 0

        for data in training_datasets:
            total = 1000
            for i in range(total):
                batch_xs, batch_ys = data.train.next_batch(100)
                sess.run(network.train_step, feed_dict={network.inputs: batch_xs, network.correct_labels: batch_ys})
                if i % 50 == 0:
                    for j in range(len(evaluation_datasets)):
                        test_accuracy = sess.run(
                            network.accuracy,
                            feed_dict={
                                network.inputs: evaluation_datasets[j].test.images,
                                network.correct_labels: evaluation_datasets[j].test.labels
                            }
                        )
                        if verbose:
                            print(str(j+1), ':', test_accuracy)
                        writer.writerow({'Epoch': i + epoch, 'TestAccuracy': test_accuracy, 'Group': j+1})
            epoch += total


def main(options):
    from tensorflow.examples.tutorials.mnist import input_data
    import tensorflow as tf

    from simple_network import Network

    mnist = input_data.read_data_sets(options.data_dir, one_hot=True)

    np.random.seed(1)
    perm = np.random.permutation(mnist.train._images.shape[1])
    permuted = deepcopy(mnist)
    permuted.train._images = permuted.train._images[:, perm]
    permuted.test._images = permuted.test._images[:, perm]
    permuted.validation._images = permuted.validation._images[:, perm]

    combined = deepcopy(mnist)
    combined.train._images = np.concatenate((mnist.train._images, permuted.train._images))
    combined.train._labels = np.concatenate((mnist.train._labels, permuted.train._labels))
    combined.train._num_examples *= 2
    combined.test._images = np.concatenate((mnist.test._images, permuted.test._images))
    combined.test._labels = np.concatenate((mnist.test._labels, permuted.test._labels))
    combined.test._num_examples *= 2
    combined.validation._images = np.concatenate((mnist.validation._images, permuted.validation._images))
    combined.validation._labels = np.concatenate((mnist.validation._labels, permuted.validation._labels))
    combined.validation._num_examples *= 2

    if options.mode == 'simple':
        network = Network()
        training_datasets = [mnist, permuted]
        evaluation_datasets = [mnist, permuted]
    elif options.mode == 'mixed':
        network = Network()
        training_datasets = [combined, combined]
        evaluation_datasets = [mnist, permuted]
    elif options.mode == 'ewc':
        raise NotImplementedError("EWC mode not implemented yet!")
    else:
        raise Exception("Unrecognized mode: " + options.mode)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    run_training(network, sess, training_datasets, evaluation_datasets)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./MNIST_data',
                      help='Directory for storing input data')
    parser.add_argument('--mode', type=str, default="simple", choices=('simple', 'mixed', 'ewc'))

    options = parser.parse_args()

    main(options)