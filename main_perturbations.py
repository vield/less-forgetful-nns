import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

from network import EWCNetwork


if __name__ == "__main__":

    network = EWCNetwork()
    mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    saver = tf.train.Saver(network._var_list + network._fisher_diagonal)
    # Restore variables from checkpoint
    #print_tensors_in_checkpoint_file(file_name='./ewc1.ckpt', tensor_name='', all_tensors=True)
    saver.restore(sess, './ewc1.ckpt')

    #for noise_stddev in [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]:
    for noise_stddev in np.logspace(np.log(0.001), np.log(1), num=50):

        uniform_assignations = []
        nullspace_assignations = []
        for i in range(len(network._var_list)):
            var = network._var_list[i]
            fish = network._fisher_diagonal[i]
            random_noise = tf.random_normal(shape=var.shape, mean=0.0, stddev=noise_stddev)

            nullspace_noise = tf.where(tf.equal(fish, 0.0), random_noise, tf.zeros_like(random_noise))

            assign_op = tf.assign_add(var, random_noise)
            uniform_assignations.append(assign_op)
            nullspace_assignations.append(tf.assign_add(var, nullspace_noise))

        saver.restore(sess, './ewc1.ckpt')
        sess.run(uniform_assignations)

        # Test accuracy on validation set
        acc = network.compute_accuracy(sess,
                                       feed_dict={
                                           network.inputs: mnist.validation.images,
                                           network.correct_labels: mnist.validation.labels
                                       }
                                       )

        saver.restore(sess, './ewc1.ckpt')
        sess.run(nullspace_assignations)

        null_acc = network.compute_accuracy(
            sess,
            feed_dict={
                network.inputs: mnist.validation.images,
                network.correct_labels: mnist.validation.labels
            }
        )

        print("{:.5f}  {:.5f}  {:.5f}".format(noise_stddev, acc, null_acc))