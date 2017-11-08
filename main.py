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

    setup = TrainingSetup(options)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    run_training(sess, setup, options)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./MNIST_data',
                      help='Directory for storing input data')
    parser.add_argument('--mode', type=str, default="simple", choices=('simple', 'mixed', 'ewc'))

    options = parser.parse_args()

    # ...could be made configurable later... for now, we have some
    # pseudoparameters here so they're easier to pass around
    options.permutations = 2
    options.batch_size = 100
    options.num_batches = 1000
    options.log_frequency = 50
    options.filename = options.mode + '.csv'
    options.verbose = True
    if options.mode == 'mixed':
        # The datasets will be combined, so for fairness we will let
        # the experiment run for as long as the sequential ones.
        # Plus, it makes the plots line up nicer.
        options.num_batches *= options.permutations

    main(options)