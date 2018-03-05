import sys
import argparse
from argparse import RawTextHelpFormatter
import json


def load_parameters(parameters_filepath):
    """
    load parameters from ini file
    :param parameters_filepath:     parameters file path
    :return:                        loaded parameters
    """
    with open(parameters_filepath, 'r') as config_file:
        conf_parameters = json.load(config_file)
    return conf_parameters


def parse_arguments(arguments=None):
    """
    parse part parameters which changed frequently
    :param arguments:   Incoming parameters
    :return:            parsed dict parameters
    """
    parser = argparse.ArgumentParser(description='''Common parameters''', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--num_units', required=False, default=512, help='')
    try:
        arguments = parser.parse_args(args=arguments)
    except:
        parser.print_help()
        sys.exit(0)
    arguments = vars(arguments)
    return arguments


def get_model_parametes(parameters):
    """
    model parameters
    :param arguments:   parameters
    :return:            model parameters
    """
    model_params = dict()
    model_params["num_units"] = parameters["num_units"]
    model_params["optimizer"] = parameters["optimizer"]
    model_params["learning_rate"] = parameters["learning_rate"]
    model_params["keep_checkpoint_max"] = parameters["keep_checkpoint_max"]
    return model_params
