# -*- coding: utf-8 -*-
import six
from abc import ABCMeta
from utils.misc_utils import print_parametes, parse_params


class abstractstaticmethod(staticmethod):
    """
    Decorates a method as abstract and static
    """
    __slots__ = ()

    def __init__(self, function):
        super(abstractstaticmethod, self).__init__(function)
        function.__isabstractmethod__ = True

    __isabstractmethod__ = True


@six.add_metaclass(ABCMeta)
class Configurable(object):
    """
    Interface for all classes that are configurable
    via a parameters dictionary.

    Args:
        params: A dictionary of parameters.
        mode: A value in tf.contrib.learn.ModeKeys
    """
    def __init__(self, params, mode):
        self._params = parse_params(params, self.default_params())
        self._mode = mode
        self._print_params()

    def _print_params(self):
        """
        Logs parameter values
        """
        classname = self.__class__.__name__
        print("Creating %s in mode=%s" % (classname, self._mode))
        print_parametes(classname, self.params)

    @property
    def mode(self):
        """
        Returns a value in tf.contrib.learn.ModeKeys.
        """
        return self._mode

    @property
    def params(self):
        """
        Returns a dictionary of parsed parameters.
        """
        return self._params

    @abstractstaticmethod
    def default_params():
        """
        Returns a dictionary of default parameters. The default parameters
        are used to define the expected type of passed parameters. Missing
        parameter values are replaced with the defaults returned by this method.
        """
        raise NotImplementedError
