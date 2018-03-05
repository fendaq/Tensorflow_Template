import tensorflow as tf

from models.model_base import ModelBase


class ExampleModel(ModelBase):
    def __init__(self, params, mode="default", name="example_model"):
        super(ExampleModel, self).__init__(params, mode, name)

        # placeholders
        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        self.y = tf.placeholder(tf.float32, shape=[None, 10])

        self.logits, self.loss, self.train_op = self.build_graph()

    @staticmethod
    def default_params():
        params = ModelBase.default_params()
        params.update({
            "num_units": 512,
            "num_classes": 10
        })
        return params

    def build_graph(self):
        # define network architecture
        logits = tf.layers.dense(self.x, self.params["num_units"], activation=tf.nn.relu, name="dense2")
        logits = tf.layers.dense(logits, self.params["num_classes"])

        with tf.name_scope("loss"):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=logits))
            train_op = self._build_train_op(loss)
        return logits, loss, train_op

    def create_feed_dict(self, batch_data):
        """
        :param batch: list train/evaluate data
        :return: structured data to feed
        """
        x, y = batch_data
        feed_dict = {self.x: x, self.y: y}
        return feed_dict

    def train(self, sess, batch):
        """
        :param sess: session to runner the batch
        :param batch: a dict containing batch data
        :return: batch result, loss of the batch or logits
        """
        feed_dict = self.create_feed_dict(batch)
        return sess.run(
            [self.train_op, self.loss, self.global_step], feed_dict=feed_dict)

    def infer(self, sess, batch):
        """
        :param sess: session to runner the batch
        :param batch: a dict containing batch data
        :return: logits
        """
        feed_dict = self.create_feed_dict(batch)
        return sess.run(self.logits, feed_dict=feed_dict)
