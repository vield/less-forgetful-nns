import argparse
import csv

from setup import TrainingSetup


def run_training(sess, setup, options):
    """Run experiment, sampling accuracy at fixed intervals."""

    with open(options.filename, 'w') as f:
        fieldnames = ["Epoch", "Group", "TestAccuracy"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        batches_completed = 0

        while setup.batches_left > 0:
            setup.train_batch(sess)

            if batches_completed % options.log_frequency == 0:

                accuracy_results = setup.check_accuracy(sess)
                for i in range(len(accuracy_results)):
                    writer.writerow({'Epoch': batches_completed, 'TestAccuracy': accuracy_results[i], 'Group': i+1})

                if options.verbose:
                    # Print current accuracy on all datasets.
                    # E.g.
                    # 0.95432  0.09800  0.10999
                    # means that the accuracy is 95.4% on the first dataset, and
                    # roughly equivalent to random guessing on the latter two.
                    print("  ".join("{:.5f}".format(acc) for acc in accuracy_results))

            batches_completed += 1


def main(options):
    # Conditional import so `main.py --help` stays fast
    import tensorflow as tf
    import time

    start = time.time()

    setup = TrainingSetup(options)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    run_training(sess, setup, options)
    end = time.time()
    print(options.mode,"running time: ",end - start)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=
"""Runs a single experiment to demonstrate catastrophic forgetting in 
neural networks, and Elastic Weight Consolidation to overcome it.
The results are recorded in .csv files so they can be plotted later.

You have the choice of three modes:

1. Simple
    This mode sequentially trains the network on different datasets
    of equal complexity.
    The network will "forget" what it learned about the first datasets 
    as it learns more.
2. Mixed
    This mode trains the network on different datasets that have
    been shuffled together before training.
    The network learns all datasets, but we have to have all of
    the data at the start.
3. EWC
    This mode sequentially trains the networks on different datasets,
    using Elastic Weight Consolidation to prevent the network from
    forgetting so much about previously learned data.
4. L2
    Like EWC, but not including the Fisher information, just a
    uniformly weighted quadratic penalty for moving away from the
    old mode.
    
The datasets are permutations of the MNIST handwritten digit dataset.
""", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--data_dir', type=str, default='./MNIST_data',
                      help='Directory for storing input data')
    parser.add_argument('--mode', type=str, default="simple",
                        choices=('simple', 'mixed', 'ewc', 'l2'),
                        help='Type of experiment to run')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size to use in training')
    parser.add_argument('--num_batches', type=int, default=3000,
                        help='Number of batches per dataset')
    parser.add_argument('--permutations', type=int, default=2,
                        help='Number of datasets to generate')

    options = parser.parse_args()

    # ...could be made configurable later... for now, we have some
    # pseudoparameters here so they're easier to pass around
    options.log_frequency = 50
    options.filename = options.mode + '.csv'
    options.verbose = True
    if options.mode == 'mixed':
        # The datasets will be combined, so for fairness we will let
        # the experiment run for as long as the sequential ones.
        # Plus, it makes the plots line up nicer.
        options.num_batches *= options.permutations

    main(options)