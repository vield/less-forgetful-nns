import tensorflow as tf


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


class EWCNetwork(Network):
    """Network with Elastic Weight Consolidation capabilities.

    Note that to use the special cost function, you have to tell
    the network to save the old weight/bias values and to compute
    the Fisher diagonal."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

    def savepoint_current_vars(self, sess):
        assignments = []
        for i in range(len(self._var_list)):
            assignments.append(tf.assign(self._old_var_list[i], self._var_list[i]))

        sess.run(assignments)
