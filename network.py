import tensorflow as tf

from mixins import ListOperationsMixin


class Network:

    def __init__(self, learning_rate=0.5, *args, **kwargs):
        """Initialize a classification network for the MNIST dataset.

        There's an attempt to make a distinction between internal variables
        and the external API: any functions and attributes not starting with
        an underscore are the ones you're meant to be interfacing with.

        This is mainly just to make the project code easier to read (you can
        skim over the internal details). :)

        Parameters
        ----------
            learning_rate : float
                Learning rate to pass to the optimizer.
        """
        #
        #
        # Placeholder variables to feed data into
        self.inputs = tf.placeholder(tf.float32, [None, 784], name="Inputs")
        self.correct_labels = tf.placeholder(tf.float32, [None, 10], name="CorrectLabels")

        # Gives a single number instead of the one-hot representation we
        # expect as input
        self._correct_labels_as_numbers = tf.argmax(self.correct_labels, axis=1, name="CorrectLabelsAsNumbers")

        #
        #
        # Create network architecture
        self._nodes_per_layer = [784, 100, 100, 10]  # Currently not configurable
        self._biases = self._create_bias_shaped_variables(self._nodes_per_layer, stddev=0.1)
        self._weights = self._create_weight_shaped_variables(self._nodes_per_layer, stddev=0.1)

        self._raw_outputs = self._create_network_architecture(
            inputs=self.inputs,
            biases=self._biases,
            weights=self._weights
        )
        self._var_list = self._biases + self._weights

        # "Soft" classification outputs are the softmax probabilities
        # for each input to be from a particular class, e.g. for a number
        # six we could see something like this in the output:
        #    0    1    2    3    4    5    6    7    8    9
        # [0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.7, 0.0, 0.1, 0.0]
        self._soft_classification_outputs = tf.nn.softmax(self._raw_outputs, name="SoftClassificationOutputs")
        # "Hard" classification outputs are just a single number for
        # each input, representing the class the network thinks the number
        # most likely belongs to (e.g. "6").
        self._classification_outputs = tf.argmax(self._raw_outputs, axis=1, name="ClassificationOutputs")

        #
        #
        # Initialize evaluation
        _correct_prediction = tf.equal(
            self._classification_outputs,
            self._correct_labels_as_numbers
        )
        # Ratio of correct classifications out of all classifications
        # (currently the only metric this class offers).
        self._accuracy = tf.reduce_mean(tf.cast(_correct_prediction, tf.float32), name="Accuracy")

        #
        #
        # Initialize learning
        self._optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

        self._cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.correct_labels, logits=self._raw_outputs)
        )
        self._train_step = self._optimizer.minimize(self._cross_entropy)

    def run_one_step_of_training(self, sess, feed_dict):
        assert self.inputs in feed_dict
        assert self.correct_labels in feed_dict

        return sess.run(self._train_step, feed_dict=feed_dict)

    def compute_accuracy(self, sess, feed_dict):
        assert self.inputs in feed_dict
        assert self.correct_labels in feed_dict

        return sess.run(self._accuracy, feed_dict=feed_dict)

    #
    #
    # Helper functions

    def _create_bias_shaped_variables(self, nodes_per_layer, mean=None, stddev=None, name_prefix="Biases", trainable=True):
        """Does what it says on the tin.

        Parameters
        ----------
        nodes_per_layer : list of integers
            E.g. [784, 100, 100, 10] means that there are 784 features (pixels)
            coming in, two hidden layers with 100 nodes each, and the output
            vector has length 10.
            The bias variables will be created to match this structure.
        mean : float
        stddev : float
            If set to a truthy value, the bias-shaped variable will be initialized
            from a truncated normal distribution with the given mean (default 0.0)
            and the stddev. Otherwise, it will be initialized to all zeroes.
        name_prefix : string
            Used to name the tensors.
        trainable : bool
            Passed into the Variable constructor to make the tensor trainable
            by default, or not if trainable=False.
        """
        biases = []

        for layer_idx in range(1, len(nodes_per_layer)):
            num_out = nodes_per_layer[layer_idx]
            shape = [num_out]

            if stddev:
                initial = tf.truncated_normal(shape=shape, stddev=stddev, mean=mean if mean else 0.0)
            else:
                initial = tf.constant(0.0, shape=shape)

            b = tf.Variable(
                initial,
                name=name_prefix + str(layer_idx),
                trainable=trainable
            )
            biases.append(b)

        return biases

    def _create_weight_shaped_variables(self, nodes_per_layer, mean=None, stddev=None, name_prefix="Weights", trainable=True):
        """Same as bias-shaped variables except this is for weights. See other docstring."""
        weights = []

        for layer_idx in range(1, len(nodes_per_layer)):
            num_in = nodes_per_layer[layer_idx-1]
            num_out = nodes_per_layer[layer_idx]
            shape = [num_in, num_out]

            if stddev:
                initial = tf.truncated_normal(shape=shape, stddev=stddev, mean=mean if mean else 0.0)
            else:
                initial = tf.constant(0.0, shape=shape)

            W = tf.Variable(
                initial,
                name=name_prefix + str(layer_idx),
                trainable=trainable
            )
            weights.append(W)

        return weights

    def _create_network_architecture(self, inputs, biases, weights):

        num_hidden_layers = len(self._nodes_per_layer) - 2

        prev = inputs
        for layer_idx in range(num_hidden_layers):
            b = biases[layer_idx]
            W = weights[layer_idx]

            y = tf.nn.relu(tf.matmul(prev, W) + b)
            prev = y

        # Last layer.
        # The difference is that we don't apply the ReLU activation function.
        # The softmax application function is applied later for optimization
        # and for exact classification, we just look at the magnitudes of the
        # raw values.
        layer_idx = str(len(self._nodes_per_layer)-1)

        b = biases[-1]
        W = weights[-1]

        outputs = tf.add(tf.matmul(prev, W), b, name="RawOutputs")

        return outputs


class EWCNetwork(ListOperationsMixin, Network):
    """Network with Elastic Weight Consolidation capabilities.

    Note that to use the special cost function, you have to tell
    the network to save the old weight/bias values and to compute
    the Fisher diagonal."""

    def __init__(self, fisher_batch_size=100, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # The computational graph is built to compute the Fisher diagonal
        # in batches of a fixed size.
        # E.g. if the batch size is 100 and there are 55000 datapoints
        # in the dataset we use for computing the Fisher diagonal, we'll
        # end up doing 550 batches (each batch can be parallelized somewhat
        # and it appears to be faster to not come back to Python as often).
        # Batching idea shamelessly copied from https://github.com/stokesj/EWC
        self.fisher_batch_size = fisher_batch_size

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

        # The "lambda" from the paper.
        # Sets the relative informance of the EWC penalty vs the new dataset.
        self._fisher_coeff = tf.Variable(initial_value=0.0, name="FisherCoefficient")

        self._squared_var_distances_scaled_by_fisher = []
        for var, old_var, fisher in zip(self._var_list, self._old_var_list, self._fisher_diagonal):
            self._squared_var_distances_scaled_by_fisher.append(
                tf.multiply(
                    fisher,
                    # Hacky way to get the first part of the name
                    # var.name will give something like "Biases1:0"
                    # We want "SquaredDistBiases1"
                    # Having to do this is probably a symptom of me
                    # misunderstanding the namespacing system...
                    # we can think about it later.
                    tf.square(tf.subtract(var, old_var), name="SquaredDist" + var.name.split(":")[0])
                )
            )

        # Sum of the above
        self._ewc_penalty = tf.add_n([tf.reduce_sum(svd) for svd in self._squared_var_distances_scaled_by_fisher])

        # (1/2) * lambda * sum of weighted squared distances
        self._ewc_cost = tf.add(
            self._cross_entropy,
            tf.multiply(
                tf.multiply(self._fisher_coeff, 0.5),
                self._ewc_penalty),
            name="EWCCost"
        )

        self._train_step = self._optimizer.minimize(self._ewc_cost)

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

    def reset_fisher_diagonal(self, sess):
        """Sets the Fisher information to all zeroes."""
        self.reset_vars(sess, self._fisher_diagonal)

    def update_fisher_coefficient(self, sess, new_coefficient):
        sess.run(tf.assign(self._fisher_coeff, new_coefficient))

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
            assignations = [tf.assign_add(self._fisher_diagonal_computed[i], self._fisher_delta[i]) for i in range(len(self._var_list))]
            sess.run(assignations, feed_dict=feed_dict)

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

    def print_test(self):
        print("I can print stuff!")

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
