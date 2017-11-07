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
        self.fisher_diagonal = []
        for i in range(len(self.var_list)):
            self.fisher_diagonal.append(
                tf.Variable(
                    tf.constant(0.0, shape=self.var_list[i].shape),
                    trainable=False)
            )

        self.ewc_penalty = 0.0
        self.fisher_coeff = 0.0

        self.old_var_list = []
        for i in range(len(self.var_list)):
            self.old_var_list.append(
                tf.Variable(
                    tf.constant(0.0, shape=self.var_list[i].shape),
                    trainable=False
                )
            )

    def update_fisher_diagonal(self, sess, dataset):
        dataset._index_in_epoch = 0  # ensures that all training examples are included without repetitions
        num_samples = 100  # FIXME -- Do not run on full dataset (yet) for speed

        # Reset Fisher diagonal values to zero
        sess.run([tf.assign(tensor, tf.zeros_like(tensor)) for tensor in self.fisher_diagonal])

        # Add them up
        print("adding up fisher diagonal")
        for i in range(num_samples):
            inputs, correct_labels = dataset.next_batch(1)
            #print("inputs, correct_labels")
            #print(inputs.shape, correct_labels.shape)
            # log probs
            log_likelihood = tf.reduce_sum(self.correct_labels * tf.nn.log_softmax(self.outputs))
            #print("log likelihood")
            #print(log_likelihood)
            # gradients
            grads = tf.gradients(log_likelihood, self.var_list)
            #print("gradients")
            #print(grads)
            # square them
            squared_gradients = [tf.square(grad) for grad in grads]
            #print("squared gradients")
            #print(squared_gradients)

            sums_of_squared_gradients = [tf.assign_add(f1, f2) for f1, f2 in zip(self.fisher_diagonal, squared_gradients)]
            #print("sums of squared gradients")
            #print(sums_of_squared_gradients)

            sess.run(sums_of_squared_gradients,
                     feed_dict={self.inputs: inputs, self.correct_labels: correct_labels})
        print("done")
        # Compute averages
        scale = 1.0 / num_samples
        sess.run([tf.assign(var, tf.multiply(scale, var)) for var in self.fisher_diagonal])

        # Save old vars
        sess.run([v1.assign(v2) for v1, v2 in zip(self.old_var_list, self.var_list)])

    def set_train_step(self, fisher_coeff=None):
        if not fisher_coeff:
            super().set_train_step()
        else:
            penalty = tf.add_n([tf.reduce_sum(tf.square(tf.subtract(w1, w2)) * f) for w1, w2, f
                                in zip(self.var_list, self.old_var_list, self.fisher_diagonal)])
            self.ewc_penalty = (fisher_coeff / 2) * penalty
            self.train_step = tf.train.GradientDescentOptimizer(0.5).minimize(self.cost + self.ewc_penalty)
