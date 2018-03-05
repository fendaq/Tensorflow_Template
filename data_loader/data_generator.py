import numpy as np


class DataGenerator(object):
    """
    Data generator for training or evaluate
    """
    def __init__(self, params):
        self.params = params
        # load data here
        self.input = np.random.randn(500, 784)
        self.y = np.eye(10)[np.random.randint(0, 10, (500,))]

    def train_minibatches(self):
        """
        yield mini batch data
        :return: batch data
        """
        batch_size = self.params['batch_size']
        start_index = 0
        while start_index + batch_size < 500:
            end_index = start_index + batch_size
            yield self.input[start_index:end_index], self.y[start_index:end_index]
            start_index = end_index

    def valid_minibatches(self):
        """
        yield mini batch data
        :return: batch data
        """
        batch_size = self.params['batch_size']
        start_index = 0
        while start_index + batch_size < 500:
            end_index = start_index + batch_size
            yield self.input[start_index:end_index], self.y[start_index:end_index]
            start_index = end_index

    def test_minibatches(self):
        """
        yield mini batch data
        :return: batch data
        """
        batch_size = self.params['batch_size']
        start_index = 0
        while start_index + batch_size < 500:
            end_index = start_index + batch_size
            yield self.input[start_index:end_index], self.y[start_index:end_index]
            start_index = end_index
