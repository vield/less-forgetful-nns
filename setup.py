class TrainingSetup:
    """Wrapper for the experiment where we may use different networks or datasets.

    train_batch() runs the next training step (if we've completed training on the
    previous dataset, we will continue to the next dataset).

    check_accuracy() computes the accuracy on all the evaluation datasets that
    were defined.
    """
    def __init__(self, options):
        if options.mode not in ['simple', 'mixed', 'ewc', 'l2']:
            raise Exception("Unrecognized mode: " + options.mode)

        # Conditional imports, the TensorFlow imports are slow so we only
        # want to execute them if we've actually got this far.
        import tensorflow as tf

        from data import get_dataset_permutations, merge_datasets
        from network import Network, EWCNetwork

        permuted_datasets = get_dataset_permutations(options.data_dir, options.permutations)

        self.mode = options.mode

        self.training_datasets = permuted_datasets
        self.evaluation_datasets = permuted_datasets

        if options.mode not in ['ewc', 'l2']:
            # To prevent "_1" suffixes in variable names in the checkpoint...
            self.network = Network()

        if options.mode == 'mixed':
            combined = merge_datasets(permuted_datasets)
            self.training_datasets = (combined,)
        elif options.mode == 'ewc' or options.mode == 'l2':
            self.network = EWCNetwork()
            self.network.set_train_step()

        self.batch_size = options.batch_size
        self.batch_index = 0
        self.num_batches = options.num_batches

        self.current_dataset = 0
        self.num_datasets = len(self.training_datasets)

        self.batches_left = self.num_datasets * self.num_batches

        vars = self.network._var_list[:]
        if options.mode in ['ewc', 'l2']:
            vars += self.network._fisher_diagonal[:]

        self.saver = tf.train.Saver(vars)

    def train_batch(self, sess):
        """Run a single step of the experiment.

        Keeps track of where we are in terms of datasets/batches/etc."""
        if self.batches_left:
            data = self.training_datasets[self.current_dataset]

            batch_xs, batch_ys = data.train.next_batch(self.batch_size)
            feed_dict = {
                self.network.inputs: batch_xs,
                self.network.correct_labels: batch_ys
            }
            self.network.run_one_step_of_training(sess, feed_dict=feed_dict)

            self.batch_index += 1
            all_batches_run_for_current_dataset = self.batch_index >= self.num_batches

            if all_batches_run_for_current_dataset:
                self.current_dataset += 1
                self.batch_index = 0

                this_wasnt_the_last_dataset = self.current_dataset < self.num_datasets
                if this_wasnt_the_last_dataset:
                    if self.mode == 'ewc':
                        # Update Fisher diagonal and save old values if running in EWC mode
                        self.network.reset_fisher_diagonal(sess)
                        self.network.savepoint_current_vars(sess)
                        self.network.update_fisher_diagonal(sess, dataset=data.train)
                        self.network.set_train_step()
                    elif self.mode == 'l2':
                        self.network.set_uniform_fisher_diagonal(sess)
                        self.network.savepoint_current_vars(sess)
                        self.network.set_train_step(fisher_coeff=0.1)

                save_file = self.saver.save(sess, "./" + self.mode + str(self.current_dataset) + ".ckpt")
                print("Model saved in file: " + save_file)

            # Update overall counter
            self.batches_left -= 1
        else:
            raise StopIteration

    def check_accuracy(self, sess):
        """Compute accuracy on all evaluation datasets configured.

        Returns
        -------
            accuracy_results : list of floats between 0.0 and 1.0
                There is one value for each evaluation dataset.
        """
        accuracy_results = []

        for j in range(len(self.evaluation_datasets)):
            feed_dict = {
                self.network.inputs: self.evaluation_datasets[j].validation.images,
                self.network.correct_labels: self.evaluation_datasets[j].validation.labels
            }
            accuracy = self.network.compute_accuracy(sess, feed_dict=feed_dict)
            accuracy_results.append(accuracy)

        return accuracy_results
