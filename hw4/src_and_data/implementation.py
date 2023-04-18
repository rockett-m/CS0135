from math import log2
import numpy as np
from collections import OrderedDict

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

    total_correct = 0

    x_col = np.array(x_inputs[:, feature_index]) # select all rows vals from a single column (A or B)
    unique_x_vals, counts = np.unique(x_col, return_counts=True)
    # print(f'{unique_x_vals = } : {counts = }')

    x_to_y_count_dict = OrderedDict() # see mapping of x vals to y vals
    x_to_y_count_dict = {key: 0 for key in unique_x_vals} # 0=>0; 1=>0

    # for each unique x value, create a key in a dict with the value being
    # counts of corresponding y values for the same row index
    # then take the max of the values and add it to the total correct value
    for unq_x_val in list(unique_x_vals): # 1

        relevant_row_indices = [] # where is x equal to 0 in the full col?
        for row_idx, row_val in enumerate(x_col): # 0,1,2,3,4,5,6,7; 1,1,0,0,0,0,0,0

            if unq_x_val == row_val:
                relevant_row_indices.append(row_idx)

        # in relevant rows, total up the y-value counts in a dict
        for good_row in relevant_row_indices:
            if y_outputs[good_row] not in x_to_y_count_dict.keys():
                x_to_y_count_dict[y_outputs[good_row]] = 1
            else:
                x_to_y_count_dict[y_outputs[good_row]] += 1

        # add the max value prediction to the running total
        total_correct += max(x_to_y_count_dict.values())
        x_to_y_count_dict = {key: 0 for key in unique_x_vals}  # 0=>0; 1=>0

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
    # entropy += -np.sum(pi * log2(pi))
    k = len(classes)

    # for each class, prob of sample being in class given y data
    for k in range(len(classes)):
        if k > 0:
            pi = 1/k
            entropy += -np.sum(pi * log2(pi))

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
    # ((# samples in  split) / (total # of samples)) * (entropy of split)
    gain = 0  # TODO: fix me
    print(f'{gain = }')
    return gain
