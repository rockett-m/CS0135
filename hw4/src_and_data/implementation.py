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

    entropy = 0  # TODO: fix me

    unique_y_vals, counts = np.unique(y_outputs, return_counts=True)

    # y count over length of y for pi prob
    for y, y_count in zip(unique_y_vals, counts):

        pi = y_count / len(y_outputs)

        entropy += -(pi * log2(pi))

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

    # Calculate the entropy of the overall set
    overall_entropy = set_entropy(x_inputs, y_outputs, classes)

    # Calculate the entropy of each split set
    x_col = np.array(x_inputs[:, feature_index]) # select all rows vals from a single column (A or B)

    set_entropies = []  # TODO: fix me

    # Calculate the remainder
    remainder = 0  # TODO: fix me
    # ((# samples in  split) / (total # of samples)) * (entropy of split)
    for cls_idx, cls in enumerate(classes):  # 0,1,2

        y_rows_relevant = np.where(x_col == cls)[0]
        y_output_relevant = y_outputs[y_rows_relevant]

        set_ent = set_entropy(x_inputs=x_inputs, y_outputs=y_output_relevant, classes=classes)
        set_entropies.append(set_ent)

        remainder += ( (len(y_rows_relevant) / (len(y_outputs))) * set_entropies[cls] )

    gain = overall_entropy - remainder # TODO: fix me

    return gain
