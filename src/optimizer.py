from enum import Enum
import tensorflow as tf
from torch import optim
import logging


class OptimizerType(Enum):
    ADAM = "adam"
    ADAGRAD = "adagrad"
    ADADELTA = "adadelta"
    SGD = "sgd"


def get_optimizer(optimizer_type, model_params, learning_rate):

    params = filter(lambda p: p.requires_grad, model_params)
    optimizer_args = (params, learning_rate)

    if optimizer_type == OptimizerType.ADAM.value:
        return optim.Adam(*optimizer_args)
    elif optimizer_type == OptimizerType.ADAGRAD.value:
        return optim.Adagrad(*optimizer_args)
    elif optimizer_type == OptimizerType.ADADELTA.value:
        return optim.Adadelta(*optimizer_args)
    elif optimizer_type == OptimizerType.SGD.value:
        return optim.SGD(*optimizer_args)
    else:
        raise ValueError("invalid optimizer name")


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
