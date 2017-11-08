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
        self._biases, self._weights, self._raw_outputs = self._create_network_architecture(
            inputs=self.inputs,
            nodes_per_layer=[784, 100, 100, 10]
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

    def _create_network_architecture(self, inputs, nodes_per_layer):

        biases = []
        weights =[]

        prev = inputs
        for i in range(1, len(nodes_per_layer)-1):
            num_in = nodes_per_layer[i-1]
            num_out = nodes_per_layer[i]
            W = tf.Variable(tf.truncated_normal([num_in, num_out], stddev=0.1), name="Weights" + str(i))
            b = tf.Variable(tf.truncated_normal([num_out], stddev=0.1), name="Biases" + str(i))

            weights.append(W)
            biases.append(b)

            y = tf.nn.relu(tf.matmul(prev, W) + b)
            prev = y

        # Last layer.
        # The difference is that we don't apply the ReLU activation function.
        # The softmax application function is applied later for optimization
        # and for exact classification, we just look at the order of the values.
        layer_idx = str(len(nodes_per_layer)-1)

        W = tf.Variable(tf.truncated_normal([nodes_per_layer[-2], nodes_per_layer[-1]], stddev=0.1), name="Weights" + layer_idx)
        b = tf.Variable(tf.truncated_normal([nodes_per_layer[-1]], stddev=0.1), name="Biases" + layer_idx)

        weights.append(W)
        biases.append(b)

        outputs = tf.add(tf.matmul(prev, W), b, name="RawOutputs")

        return biases, weights, outputs


class EWCNetwork(Network):
    """Network with Elastic Weight Consolidation capabilities.

    Note that to use the special cost function, you have to tell
    the network to save the old weight/bias values and to compute
    the Fisher diagonal."""

    def __init__(self, fisher_coeff=10, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fisher_coeff = fisher_coeff
        raise NotImplementedError("EWC Network not implemented yet!")