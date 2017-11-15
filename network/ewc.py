import tensorflow as tf

from .mixins import ListOperationsMixin
from .base import Network


class EWCNetwork(ListOperationsMixin, Network):
    """Network with Elastic Weight Consolidation capabilities.

    Note that to use the special cost function, you have to tell
    the network to save the old weight/bias values and to compute
    the Fisher diagonal."""

    def __init__(self, fisher_coeff=1, fisher_batch_size=100, learning_rate=0.01, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # The computational graph is built to compute the Fisher diagonal
        # in batches of a fixed size.
        # E.g. if the batch size is 100 and there are 55000 datapoints
        # in the dataset we use for computing the Fisher diagonal, we'll
        # end up doing 550 batches (each batch can be parallelized somewhat
        # and it appears to be faster to not come back to Python as often).
        # Batching idea shamelessly copied from https://github.com/stokesj/EWC
        self.fisher_batch_size = fisher_batch_size
        self.fisher_coeff = fisher_coeff
        self.learning_rate = learning_rate

        # Savepointed versions of self._biases, self._weights and self._var_list
        self._old_biases = self._create_bias_shaped_variables(
            self._nodes_per_layer,
            name_prefix="OldBiases",
            trainable=False
        )
        self._old_weights = self._create_weight_shaped_variables(
            self._nodes_per_layer,
            name_prefix="OldWeights",
            trainable=False
        )
        self._old_var_list = self._old_biases + self._old_weights

        self._savepointed_vars_exist = False

        # self._var_list, self._old_var_list and self._fisher_diagonal all have
        # the same shape.
        # More specifically, they are lists of tensors shaped like
        # self._biases + self._weights.
        # They are combined to create the EWC penalty part of the cost function.
        self._fisher_diagonal = self._create_bias_shaped_variables(
            self._nodes_per_layer,
            name_prefix="FisherDiagonalBiases",
            trainable=False
        ) + self._create_weight_shaped_variables(
            self._nodes_per_layer,
            name_prefix="FisherDiagonalWeights",
            trainable=False
        )

        self._optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        #
        #
        # Fisher diagonal computations
        self._fisher_inputs = None
        self._fisher_correct_labels = None
        # _fisher_delta stores a partial result during computation
        self._fisher_delta = None
        # _fisher_diagonal_computed stores the most recent computed result
        # (this can then e.g. be directly assigned to _fisher_diagonal, which
        # is used by the cost function, or added to it with a weight, or any
        # other preprocessing).
        self._fisher_diagonal_computed = None

        self._fisher_sum_up_operation = None

        self._create_fisher_diagonal_computational_graph()

    def savepoint_current_vars(self, sess):
        """Store current self._var_list in self._old_var_list.

        The savepointed versions will be used as the "old mode" ("theta star")
        in the cost function that incorporates the Fisher information.

        You would typically also want to update the Fisher information
        around the time you savepoint vars so it is computed for the most
        recent state (or includes the most recent state, in case we're summing
        them up). (Look at the setup module for examples.)
        """
        assignments = []
        for i in range(len(self._var_list)):
            assignments.append(tf.assign(self._old_var_list[i], self._var_list[i]))

        sess.run(assignments)
        self._savepointed_vars_exist = True

    def reset_fisher_diagonal(self, sess):
        """Sets the Fisher information to all zeroes."""
        self.reset_vars(sess, self._fisher_diagonal)

    def set_uniform_fisher_diagonal(self, sess):
        """Sets the Fisher information to all ones.

        Useful for demonstrating how the L2 penalty does not work."""
        assignments = []
        for tensor in self._fisher_diagonal:
            assignments.append(tf.assign(tensor, tf.ones_like(tensor)))

        sess.run(assignments)

    def set_train_step(self, learning_rate=None, fisher_coeff=None):
        if learning_rate is None:
            learning_rate = self.learning_rate
        if fisher_coeff is None:
            fisher_coeff = self.fisher_coeff
        if not self._savepointed_vars_exist:
            fisher_coeff = 0.0

        self._squared_var_distances_scaled_by_fisher = []
        for var, old_var, fisher in zip(self._var_list, self._old_var_list, self._fisher_diagonal):
            self._squared_var_distances_scaled_by_fisher.append(
                tf.multiply(
                    fisher,
                    tf.square(tf.subtract(var, old_var), name="SquaredDist" + var.name.split(":")[0])
                )
            )

        # Sum of the above
        self._ewc_penalty = tf.add_n([tf.reduce_sum(svd) for svd in self._squared_var_distances_scaled_by_fisher])

        # (1/2) * lambda * sum of weighted squared distances
        self._ewc_cost = tf.add(
            self._cross_entropy,
            tf.multiply(
                tf.cast(tf.divide(fisher_coeff, 2), tf.float32),
                self._ewc_penalty
            ),
            name="EWCCost"
        )

        self._train_step = self._optimizer.minimize(self._ewc_cost, var_list=self._var_list)

    def update_fisher_diagonal(self, sess, dataset):
        """Computes the Fisher diagonal for the current self._var_list.

        The result will be added to the Fisher diagonal values used
        in the cost function.

        If you want to only use the most recent Fisher diagonal, reset
        the diagonal before calling this function. If you want a sum,
        skip the reset step.
        """
        print("Resetting Fisher diagonal.")
        self.reset_vars(sess, self._fisher_diagonal_computed)
        print("Done.")

        # It's the user's responsibility to initialize Fisher batch size to
        # a value that divides the number of images in their dataset if they
        # want to use them all exactly once
        print("Computing number of batches.")
        num_batches = len(dataset.images) // self.fisher_batch_size
        if num_batches == 0:
            raise Exception("The dataset does not contain enough data for the given batch size!")
        print("There will be {} batches.".format(num_batches))

        for batch in range(num_batches):
            print("In batch {}".format(batch+1))
            batch_xs, batch_ys = dataset.next_batch(self.fisher_batch_size)
            feed_dict = {self._fisher_inputs: batch_xs, self._fisher_correct_labels: batch_ys}
            sess.run(self._fisher_sum_up_operation, feed_dict=feed_dict)

        divisions = [
            tf.divide(self._fisher_diagonal_computed[i], num_batches * self.fisher_batch_size)
            for i in range(len(self._var_list))
        ]
        sess.run(divisions)

        assignations = [
            tf.assign_add(self._fisher_diagonal[i], self._fisher_diagonal_computed[i])
            for i in range(len(self._var_list))
        ]
        sess.run(assignations)

    #
    #
    # Helper functions

    def _create_fisher_diagonal_computational_graph(self):
        # Only run this function once at the start
        assert self._fisher_delta is None

        self._fisher_inputs = tf.placeholder(
            tf.float32,
            [self.fisher_batch_size, 784],
            name="FisherInputs"
        )
        self._fisher_correct_labels = tf.placeholder(
            tf.float32,
            [self.fisher_batch_size, 10],
            name="FisherCorrectLabels"
        )

        # The Fisher diagonal is computed as ~the sample variance of the gradient
        # of the log likelihood (is that the correct way to state it?)

        # Split the batch!
        fisher_inputs_one_by_one = [
            tf.reshape(fisher_input, shape=(1, 784))
            # The num and axis are actually just the default values but I wanted to
            # be more explicit about what is going on...
            for fisher_input in tf.unstack(self._fisher_inputs, num=self.fisher_batch_size, axis=0)
        ]
        fisher_labels_one_by_one = tf.unstack(self._fisher_correct_labels, num=self.fisher_batch_size, axis=0)

        fisher_biases =  []
        fisher_weights = []
        fisher_var_list =  []
        raw_fisher_outputs = []
        log_likelihoods = []

        for i in range(self.fisher_batch_size):
            # Create a copy of the neural network graph to compute raw outputs
            biases = [tf.identity(bias) for bias in self._biases]
            weights = [tf.identity(weight) for weight in self._weights]
            var_list = biases + weights
            raw_outputs = self._create_network_architecture(
                inputs=fisher_inputs_one_by_one[i],
                biases=biases,
                weights=weights
            )
            log_likelihood = tf.multiply(fisher_labels_one_by_one[i], tf.nn.log_softmax(raw_outputs))

            fisher_biases.append(biases)
            fisher_weights.append(weights)
            fisher_var_list.append(var_list)
            raw_fisher_outputs.append(raw_outputs)
            log_likelihoods.append(log_likelihood)

        batch_log_likelihood = tf.reduce_sum(log_likelihoods)

        # This will contain self.fisher_batch_size elements, each of them
        # in the shape of self._var_list.
        grads = [tf.gradients(batch_log_likelihood, vars) for vars in fisher_var_list]

        # When we sum them up, we will form one self._var_list-shaped total
        # sum
        sum_of_squared_grads = []
        for i in range(len(self._var_list)):
            squared_grads_for_current_var = [tf.square(g[i]) for g in grads]
            sum_of_squared_grads.append(tf.add_n(squared_grads_for_current_var, name="SumOfSquaredGrads"))

        self._fisher_delta = sum_of_squared_grads

        # Initialize the computed variables
        # These will temporarily store the computed Fisher diagonal before we
        # decide how we're going to add it into the variable used in the cost
        self._fisher_diagonal_computed = self._create_bias_shaped_variables(
            self._nodes_per_layer,
            name_prefix="FisherBiasesComputed",
            trainable=False
        ) + self._create_weight_shaped_variables(
            self._nodes_per_layer,
            name_prefix="FisherWeightsComputed",
            trainable=False
        )

        self._fisher_sum_up_operation = [
            tf.assign_add(fisher_tmp, fisher_delta)
            for fisher_tmp, fisher_delta in zip(self._fisher_diagonal_computed, self._fisher_delta)
        ]