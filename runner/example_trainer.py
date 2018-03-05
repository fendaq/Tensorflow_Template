import os
import numpy as np
import tensorflow as tf

from runner.runner_base import TrainerBase
from utils import misc_utils as utils


class ExampleTrainer(TrainerBase):
    """
    Example model trainer
    """
    def __init__(self, sess, model, data_generator, params, log_f):
        super(ExampleTrainer, self).__init__(sess, model, data_generator, params, log_f)
        self.sess.run(tf.global_variables_initializer())

    def train_epoch(self):
        """
        implement the logic of epoch
        """
        train_losses = []
        for batch_data in self.data_generator.train_minibatches():
            _, loss, global_step = self.model.train(self.sess, batch_data)
            train_losses.append(loss)
            if global_step % 10 == 0:
                # write summary
                self.summary_file_writer.add_summary(summaries, global_step)
                utils.print_out("  global step %d,  loss %.4f" % (global_step, np.mean(train_losses)), self.log_f)
                train_losses = []
        checkpoint_path = os.path.join(self.params['output_dir'], self.params['checkpoint_name'])
        self.model.save(self.sess, checkpoint_path)

        accuracy = self.evaluate()
        utils.print_out("accuracy: %f" % accuracy, self.log_f)

    def evaluate(self):
        """
        evaluation for model
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

        return float(correct_count) / total_count
