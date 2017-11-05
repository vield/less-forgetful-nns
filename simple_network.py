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


        self.correct_labels = tf.placeholder(tf.float32, [None, 10])

        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.correct_labels, logits=self.outputs)
        )
        self.train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cost)

        correct_prediction = tf.equal(tf.argmax(self.outputs, 1), tf.argmax(self.correct_labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))