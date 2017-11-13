import tensorflow as tf


class ListOperationsMixin:
    """Helper operations to run an operation on one or more lists of tensors."""
    def reset_vars(self, sess, vars):
        """Set all tensors in vars to 0.0."""
        assignments = []
        for tensor in vars:
            assignments.append(tf.assign(tensor, tf.zeros_like(tensor)))

        sess.run(assignments)

    def copy_vars_into_vars(self, sess, to_vars, from_vars):
        """Copy values from a list of tensors to another.

        The tensors must have the same shape (or be broadcastable, I suppose)."""
        assignments = []
        for target, source in zip(to_vars, from_vars):
            assignments.append(tf.assign(target, source))

        sess.run(assignments)

    def multiply_vars(self, sess, vars, multiplier):
        assignments = []

        # If multiplying by single scalar (or broadcastable tensor), make zippable
        try:
            iter(multiplier)
        except TypeError:
            multiplier = [multiplier] * len(vars)

        for tensor, multiplier in zip(vars, multiplier):
            assignments.append(tf.multiply(tensor, multiplier))

        sess.run(assignments)

    def clone_vars(self, vars):
        # Is this the same as tf.identity_n?
        cloned_vars = []

        for var in vars:
            cloned_vars.append(tf.identity(var))

        return cloned_vars
