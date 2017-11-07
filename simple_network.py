import numpy as np
import tensorflow as tf


class Network:

    def __init__(self):
        hidden1 = 100
        hidden2 = 100

        # Inputs
        self.inputs = tf.placeholder(tf.float32, [None, 784])

        # Hidden layer 1
        W1 = tf.Variable(tf.truncated_normal([784, hidden1], stddev=0.1))
        b1 = tf.Variable(tf.truncated_normal([hidden1], stddev=0.1))
        y1 = tf.nn.relu(tf.matmul(self.inputs, W1) + b1)

        # Hidden layer 2
        W2 = tf.Variable(tf.truncated_normal([hidden1, hidden2], stddev=0.1))
        b2 = tf.Variable(tf.truncated_normal([hidden2], stddev=0.1))
        y2 = tf.nn.relu(tf.matmul(y1, W2) + b2)

        # Output layer
        W3 = tf.Variable(tf.truncated_normal([hidden2, 10], stddev=0.1))
        b3 = tf.Variable(tf.truncated_normal([10], stddev=0.1))
        self.outputs = tf.matmul(y2, W3) + b3

        self.var_list = [W1, b1, W2, b2, W3, b3]

        self.correct_labels = tf.placeholder(tf.float32, [None, 10])

        self.cost = None
        self.train_step = None
        self.set_train_step()

        correct_prediction = tf.equal(tf.argmax(self.outputs, 1), tf.argmax(self.correct_labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def set_train_step(self):
        self.cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.correct_labels, logits=self.outputs)
        )
        self.train_step = tf.train.GradientDescentOptimizer(0.5).minimize(self.cost)


class EWCNetwork(Network):

    def __init__(self):
        super().__init__()

        # Has the same shape as var list; starts out as zero
        # That is, there is one "diagonal" value for each weight/bias variable
        self.fisher_diagonal = None
        self.reset_fisher_diagonal()

        self.ewc_penalty = 0.0
        self.fisher_coeff = 0.0

        self.old_var_list = None

    def reset_fisher_diagonal(self):
        """Set self.fisher_diagonal to all zeroes in the shape of var_list."""
        self.fisher_diagonal = []
        for i in range(len(self.var_list)):
            self.fisher_diagonal.append(np.zeros(self.var_list[i].shape))

    def compute_fisher(self, sess, dataset, num_samples=200):
        num_examples = dataset.images.shape[0]
        self.reset_fisher_diagonal()

        class_probs = tf.nn.softmax(self.outputs)
        sample_class = tf.to_int32(tf.multinomial(tf.log(class_probs), 1)[0][0])

        for i in range(num_samples):
            # Select random image from dataset
            random_index = np.random.randint(num_examples)
            feed_dict = {
                self.inputs: dataset.images[random_index:random_index+1]
            }

            # Sample class for this image from the output distribution
            # Then consider the probability that the example is in that class (given the var_list)
            # Compute the gradient (for the above probability function wrt each var in the var_list)
            prob_for_sample_class = class_probs[0, sample_class]
            grad = sess.run(
                tf.gradients(
                    tf.log(prob_for_sample_class),
                    self.var_list
                ),
                feed_dict=feed_dict
            )

            # Add squared gradients to total
            for i in range(len(self.fisher_diagonal)):
                self.fisher_diagonal[i] += np.square(grad[i])

        # Average to get (approximate) expected value
        for i in range(len(self.fisher_diagonal)):
            self.fisher_diagonal[i] /= num_samples

    def save_current_vars(self):
        self.old_var_list = []
        for i in range(len(self.var_list)):
            self.old_var_list.append(self.var_list[i].eval())

    def set_train_step(self, fisher_coeff=None):
        super().set_train_step()

        if fisher_coeff:
            self.fisher_coeff = fisher_coeff
            self.ewc_penalty = 0.0

            for i in range(len(self.var_list)):
                self.ewc_penalty += tf.reduce_sum(
                    tf.multiply(
                        self.fisher_diagonal[i].astype(np.float32),  # Types need to match
                        tf.square(self.var_list[i] - self.old_var_list[i])
                    )
                )

            ewc_cost = self.cost + (fisher_coeff / 2) * self.ewc_penalty
            self.train_step = tf.train.GradientDescentOptimizer(0.5).minimize(ewc_cost)
