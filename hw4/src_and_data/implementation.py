import numpy as np


def counting_heuristic(x_inputs, y_outputs, feature_index, classes):
    """
    Calculate the total number of correctly classified instances for a given
    feature index, using the counting heuristic.

    :param x_inputs: numpy array of shape (n_samples, n_features), containing the input data
    :param y_outputs: numpy array of shape (n_samples,), containing the output data (class labels)
    :param feature_index: int, index of the feature to be evaluated
    :param classes: container, unique set of class labels.
           e.g. [0,1] for two classes or [0,1,2] for 3 classes
    :return: int, total number of correctly classified instances
    """

    total_correct = 0  # TODO: fix me

    return total_correct


def set_entropy(x_inputs, y_outputs, classes):
    """Calculate the entropy of the given input-output set.

    :param x_inputs: numpy array of shape (n_samples, n_features), containing the input data
    :param y_outputs: numpy array of shape (n_samples,), containing the output data (class labels)
    :param classes: container, unique set of class labels.
           e.g. [0,1] for two classes or [0,1,2] for 3 classes
    :return: float, entropy value of the set
    """

    entropy = 0  # TODO: fix me

    return entropy


def information_remainder(x_inputs, y_outputs, feature_index, classes):
    """Calculate the information remainder after splitting the input-output set based on the
given feature index.


    :param x_inputs: numpy array of shape (n_samples, n_features), containing the input data
    :param y_outputs: numpy array of shape (n_samples,), containing the output data (class labels)
    :param feature_index: int, index of the feature to be evaluated
    :param classes: container, unique set of class labels.
           e.g. [0,1] for two classes or [0,1,2] for 3 classes
    :return: float, information remainder value
    """

    # Calculate the entropy of the overall set
    overall_entropy = set_entropy(x_inputs, y_outputs, classes)

    # Calculate the entropy of each split set
    set_entropies = []  # TODO: fix me

    # Calculate the remainder
    remainder = 0  # TODO: fix me

    gain = 0  # TODO: fix me

    return gain
