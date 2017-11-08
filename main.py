import argparse
import csv


def run_training(sess, network, training_datasets, evaluation_datasets, options, verbose=True):
    filename = options.mode + '.csv'
    with open(filename, 'w') as f:
        if verbose:
            print("Saving results into " + filename)

        # FIXME: "Epoch" is actually "Batch"
        # Need to change in the plotting code and then re-run all experiments with final settings
        fieldnames = ["Epoch", "Group", "TestAccuracy"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        batch = 0

        for i in range(len(training_datasets)):
            data = training_datasets[i]
            total = 1000

            for i in range(total):
                batch_xs, batch_ys = data.train.next_batch(100)

                feed_dict = {
                    network.inputs: batch_xs,
                    network.correct_labels: batch_ys
                }

                network.run_one_step_of_training(sess, feed_dict=feed_dict)

                if i % 50 == 0:
                    accuracy_results = []

                    for j in range(len(evaluation_datasets)):
                        feed_dict = {
                            network.inputs: evaluation_datasets[j].validation.images,
                            network.correct_labels: evaluation_datasets[j].validation.labels
                        }
                        accuracy = network.compute_accuracy(sess, feed_dict=feed_dict)
                        accuracy_results.append(accuracy)

                        writer.writerow({'Epoch': i + batch, 'TestAccuracy': accuracy, 'Group': j+1})

                    if verbose:
                        print(" ".join("{:.5f}".format(acc) for acc in accuracy_results))

            # Different datasets are chained for graphing purposes
            batch += total

            # FIXME: Update Fisher diagonal if running in EWC mode and if this is not the last training dataset


def main(options):
    import tensorflow as tf

    from data import get_dataset_permutations, merge_datasets
    from simple_network import Network, EWCNetwork

    permuted_datasets = get_dataset_permutations(options.data_dir, options.permutations)

    evaluation_datasets = permuted_datasets  # Same for all of them
    if options.mode == 'simple':
        network = Network()
        training_datasets = permuted_datasets
    elif options.mode == 'mixed':
        network = Network()
        combined = merge_datasets(permuted_datasets)
        training_datasets = (combined, combined)
    elif options.mode == 'ewc':
        network = EWCNetwork()
        training_datasets = permuted_datasets
    else:
        raise Exception("Unrecognized mode: " + options.mode)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    run_training(sess, network, training_datasets, evaluation_datasets, options)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./MNIST_data',
                      help='Directory for storing input data')
    parser.add_argument('--mode', type=str, default="simple", choices=('simple', 'mixed', 'ewc'))

    options = parser.parse_args()
    # Only two datasets are supported at this time.
    options.permutations = 2

    main(options)