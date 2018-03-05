import os, sys

import tensorflow as tf
import time

from utils import misc_utils as utils
from models.example_model import ExampleModel
from data_loader.data_generator import DataGenerator
from runner.example_trainer import ExampleTrainer
from runner.example_inference import ExampleInference
from arguments import load_parameters, parse_arguments, get_model_parametes


def main(argv=sys.argv):
    """
    main function
    :param argv:    Incoming parameters
    :return:
    """
    # parse and load parameters
    parameters = load_parameters('parameters.json')
    arguments = parse_arguments(argv[1:])
    parameters = utils.parse_params(arguments, parameters)
    utils.print_parametes('parameters', parameters)
    # get model parameters
    model_parametes = get_model_parametes(parameters)
    # log file
    log_file = os.path.join(parameters["output_dir"], "log_%d" % time.time())
    log_f = utils.get_log_f(log_file)
    # data generator
    data_generator = DataGenerator(parameters)
    # create, train and infer model
    with tf.Session() as sess:
        model = ExampleModel(model_parametes)
        trainer = ExampleTrainer(sess, model, data_generator, parameters, log_f)
        trainer.train()

        # inference
        inference = ExampleInference(sess, model, data_generator, parameters, log_f)
        inference.infer()

if __name__ == "__main__":
    main()
