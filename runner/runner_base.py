import tensorflow as tf


class TrainerBase(object):
    """
    Base class for model trainer
    """
    def __init__(self, sess, model, data_generator, params, log_f):
        self.sess = sess
        self.model = model
        self.data_generator = data_generator
        self.params = params
        self.log_f = log_f
        # summary writer
        self.summary_file_writer = tf.summary.FileWriter(self.params['output_dir'], sess.graph)

    def train(self):
        """
        train function for model
        :return:
        """
        for epoch in range(self.params['max_epochs']):
            print('Training epoch %d' % epoch)
            self.train_epoch()

    def train_epoch(self):
        """
        implement the logic of epoch:
        """
        raise NotImplementedError

    def evaluate(self):
        """
        implement evaluation for model
        :return:
        """
        raise NotImplementedError


class InferBase(object):
    """
    Base class for model infer
    """
    def __init__(self, sess, model, data_generator, params, log_f):
        self.sess = sess
        self.model = model
        self.data_generator = data_generator
        self.params = params
        self.log_f = log_f

    def infer(self):
        """
        implement infer for model
        :return:
        """
        raise NotImplementedError
