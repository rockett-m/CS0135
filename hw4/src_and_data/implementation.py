from math import log2
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

    totals = [ 0 for x in range(len(classes)) ]
    x_inputs = x_inputs.astype(int)

    for idx, y_num in enumerate(y_outputs):
        # print(f'{idx = } : {y_num = } : {x_inputs[idx][feature_index] = }')
        if y_num == 1: # indicates correct based on y-value
            # increase that count for the respective class
            totals[x_inputs[idx][feature_index]] += 1

    total_correct = int(np.max(totals)) # TODO: fix me
    print(f'{total_correct = }')
    return total_correct


def set_entropy(x_inputs, y_outputs, classes):
    """Calculate the entropy of the given input-output set.

    :param x_inputs: numpy array of shape (n_samples, n_features), containing the input data
    :param y_outputs: numpy array of shape (n_samples,), containing the output data (class labels)
    :param classes: container, unique set of class labels.
           e.g. [0,1] for two classes or [0,1,2] for 3 classes
    :return: float, entropy value of the set
    """
    # pi = 1/k
    # k = 2^len(classes)
    entropy = 0  # TODO: fix me

    """
    # https://www.cs.tufts.edu/comp/135/2020s/lectures/slides/lec08.pdf
    for idx_r, x_val in enumerate(x_inputs): # 0,1,2,3
        for idx_c, cls in enumerate(classes): # 0,1
            # print(f'{idx_c = } : {x_val = } : {x_val[idx_c] = } : {entropy = }')
            if x_val[idx_c] != 0: # 0.0 default otherwise
                entropy += -x_val[idx_c] * log2(x_val[idx_c])
    """

    # k = len(classes)
    # pi = 1/k
    # # https://datascience.stackexchange.com/questions/58565/conditional-entropy-calculation-in-python-hyx
    # unique, count = np.unique(y_outputs, return_counts=True, axis=0)
    # prob = count / len(y_outputs)
    # entropy = -np.sum(prob * log2(prob))
    # for it in range(len(classes)):
    #     k = len(classes)
    #     pi = 1/k
    #     if pi > 0:
    #         entropy += -np.sum(pi * log2(pi))
    #

    entropy += -np.sum(np.multiply(prob, np.log2(prob)))
    print(f'{entropy = }')
    return entropy  # between 0.0 and 1.0


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
    print(f'{x_inputs = }')
    print(f'{y_outputs = }')
    print(f'{feature_index = }')
    print(f'{classes = }')

    # x_inputs[idx][feature_index]

    # Calculate the entropy of the overall set
    overall_entropy = set_entropy(x_inputs, y_outputs, classes)

    # Calculate the entropy of each split set
    set_entropies = []  # TODO: fix me

    # Calculate the remainder
    remainder = 0  # TODO: fix me

    gain = 0  # TODO: fix me
    print(f'{gain = }')
    return gain
