
import tensorflow as tf
from models.configurable import Configurable


class ModelBase(Configurable):
    """
    Abstract base class for models.

    Args:
        params: A dictionary of hyperparameter values
        name: A name for this model to be used as a variable scope
    """
    def __init__(self, params, mode, name):
        super(ModelBase, self).__init__(params, mode)
        self.name = name
        self.global_step = tf.Variable(0, trainable=False)
        self.saver = tf.train.Saver(max_to_keep=self.params['keep_checkpoint_max'])

    @staticmethod
    def default_params():
        """
        Returns a dictionary of default parameters for this model.
        """
        return {
            "optimizer": "SGD",
            "learning_rate": 1e-4,
            "clip_gradients": 5.0,
            "lr_decay_type": "",
            "lr_decay_steps": 100,
            "lr_decay_rate": 0.99,
            "lr_start_decay_at": 0,
            "lr_stop_decay_at": tf.int32.max,
            "lr_min_learning_rate": 1e-12,
            "lr_staircase": False,
            "keep_checkpoint_max": 5,
        }

    def save(self, sess, checkpoint_path):
        print("Saving model...")
        self.saver.save(sess, checkpoint_path, self.global_step)
        print("Model saved")

    def load(self, sess, checkpoint_path):
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            print("Model loaded")

    def _clip_gradients(self, grads_and_vars):
        """
        Clips gradients by global norm.
        """
        gradients, variables = zip(*grads_and_vars)
        clipped_gradients, _ = tf.clip_by_global_norm(
            gradients, self.params["clip_gradients"])
        return list(zip(clipped_gradients, variables))

    def _build_train_op(self, loss):
        """
        Creates the training operation
        """
        learning_rate_decay_fn = create_learning_rate_decay_fn(
            decay_type=self.params["lr_decay_type"] or None,
            decay_steps=self.params["lr_decay_steps"],
            decay_rate=self.params["lr_decay_rate"],
            start_decay_at=self.params["lr_start_decay_at"],
            stop_decay_at=self.params["lr_stop_decay_at"],
            min_learning_rate=self.params["lr_min_learning_rate"],
            staircase=self.params["lr_staircase"])

        name = self.params["optimizer"]
        optimizer = tf.contrib.layers.OPTIMIZER_CLS_NAMES[name](
            learning_rate=self.params["learning_rate"])

        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=self.global_step,
            learning_rate=self.params["learning_rate"],
            learning_rate_decay_fn=learning_rate_decay_fn,
            clip_gradients=self._clip_gradients,
            optimizer=optimizer,
            summaries=["learning_rate", "loss", "gradients", "gradient_norm"])

        return train_op

    def build_graph(self):
        """
        Subclasses should implement this method.
        """
        raise NotImplementedError


def create_learning_rate_decay_fn(decay_type,
                                  decay_steps,
                                  decay_rate,
                                  start_decay_at=0,
                                  stop_decay_at=1e9,
                                  min_learning_rate=None,
                                  staircase=False):
    """
    Creates a function that decays the learning rate.

    Args:
        decay_steps: How often to apply decay.
        decay_rate: A Python number. The decay rate.
        start_decay_at: Don't decay before this step
        stop_decay_at: Don't decay after this step
        min_learning_rate: Don't decay below this number
        decay_type: A decay function name defined in `tf.train`
        staircase: Whether to apply decay in a discrete staircase,
          as opposed to continuous, fashion.

    Returns:
        A function that takes (learning_rate, global_step) as inputs
        and returns the learning rate for the given step.
        Returns `None` if decay_type is empty or None.
    """
    if decay_type is None or decay_type == "":
        return None

    start_decay_at = tf.to_int32(start_decay_at)
    stop_decay_at = tf.to_int32(stop_decay_at)

    def decay_fn(learning_rate, global_step):
        """The computed learning rate decay function.
        """
        global_step = tf.to_int32(global_step)

        decay_type_fn = getattr(tf.train, decay_type)
        decayed_learning_rate = decay_type_fn(
            learning_rate=learning_rate,
            global_step=tf.minimum(global_step, stop_decay_at) - start_decay_at,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=staircase,
            name="decayed_learning_rate")

        final_lr = tf.train.piecewise_constant(
            x=global_step,
            boundaries=[start_decay_at],
            values=[learning_rate, decayed_learning_rate])

        if min_learning_rate:
            final_lr = tf.maximum(final_lr, min_learning_rate)

        return final_lr

    return decay_fn
