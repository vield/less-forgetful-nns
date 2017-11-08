from copy import deepcopy

import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# Very hacky code that works for now


def get_dataset_permutations(data_dir, num_permutations=2):
    # Could be implemented as a generator later
    # We need to keep all validation/testing sets for plotting though
    # so will need to think about it.
    mnist = input_data.read_data_sets(data_dir, one_hot=True)

    datasets = [mnist,]

    for seed in range(1, num_permutations):
        np.random.seed(seed)
        perm = np.random.permutation(mnist.train._images.shape[1])
        permuted = deepcopy(mnist)
        permuted.train._images = permuted.train._images[:, perm]
        permuted.test._images = permuted.test._images[:, perm]
        permuted.validation._images = permuted.validation._images[:, perm]
        datasets.append(permuted)

    return datasets


def merge_datasets(datasets):
    combined = deepcopy(datasets[0])

    for dataset in datasets[1:]:
        combined.train._images = np.concatenate((combined.train._images, dataset.train._images))
        combined.train._labels = np.concatenate((combined.train._labels, combined.train._labels))
        combined.train._num_examples += dataset.train._num_examples
        combined.test._images = np.concatenate((combined.test._images, dataset.test._images))
        combined.test._labels = np.concatenate((combined.test._labels, dataset.test._labels))
        combined.test._num_examples += dataset.test._num_examples
        combined.validation._images = np.concatenate((combined.validation._images, dataset.validation._images))
        combined.validation._labels = np.concatenate((combined.validation._labels, dataset.validation._labels))
        combined.validation._num_examples += dataset.validation._num_examples

    return combined