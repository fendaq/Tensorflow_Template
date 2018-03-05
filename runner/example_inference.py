import numpy as np

from runner.runner_base import InferBase
from utils import misc_utils as utils


class ExampleInference(InferBase):
    """
    Example inference for model
    """
    def __init__(self, sess, model, data_generator, params, log_f):
        super(ExampleInference, self).__init__(sess, model, data_generator, params, log_f)

        checkpoint_path = self.params['output_dir']
        self.model.load(self.sess, checkpoint_path)

    def infer(self):
        """
        inference for model
        :return:
        """
        correct_count = 0
        total_count = 0
        for batch_data in self.data_generator.valid_minibatches():
            x, y = batch_data
            logits = self.model.infer(self.sess, batch_data)

            correct_prediction = np.equal(np.argmax(logits, 1), np.argmax(y, 1))
            total_count += correct_prediction.shape[0]
            correct_count += np.sum(correct_prediction)

        accuracy = float(correct_count) / total_count
        utils.print_out("accuracy: %f" % accuracy, self.log_f)
