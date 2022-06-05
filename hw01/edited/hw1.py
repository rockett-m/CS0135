"""
hw1.py
Author: Morgan Rockett

Tufts CS 135 Intro ML

Commands:
conda activate ml135_env_su22
python3 hw1.py
"""

import numpy as np
import math
import sys


def split_into_train_and_test(x_all_LF, frac_test=0.5, random_state=None):
    """ Divide provided array into train and test set along first dimension

    User can provide a random number generator object to ensure reproducibility.

    Args
    ----
    x_all_LF : 2D array, shape = (n_total_examples, n_features) (L, F)
        Each row is a feature vector
    frac_test : float, fraction between 0 and 1
        Indicates fraction of all L examples to allocate to the "test" set
    random_state : np.random.RandomState instance or integer or None
        If int, code will create RandomState instance with provided value as seed
        If None, defaults to the current numpy random number generator np.random

    Returns
    -------
    x_train_MF : 2D array, shape = (n_train_examples, n_features) (M, F)
        Each row is a feature vector
        Should be a separately allocated array, NOT a view of any input array

    x_test_NF : 2D array, shape = (n_test_examples, n_features) (N, F)
        Each row is a feature vector
        Should be a separately allocated array, NOT a view of any input array

    Post Condition
    --------------
    This function should be side-effect free. The provided input array x_all_LF
    should not change at all (not be shuffled, etc.)

    Examples
    --------
    >>> x_LF = np.eye(10)
    >>> xcopy_LF = x_LF.copy() # preserve what input was before the call
    >>> train_MF, test_NF = split_into_train_and_test(
    ...     x_LF, frac_test=0.3, random_state=np.random.RandomState(0))
    >>> train_MF.shape
    (7, 10)
    >>> test_NF.shape
    (3, 10)
    >>> print(train_MF)
    [[0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
     [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
     [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]]
    >>> print(test_NF)
    [[0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
     [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]]

    ## Verify that input array did not change due to function call
    >>> np.allclose(x_LF, xcopy_LF)
    True

    References
    ----------
    For more about RandomState, see:
    https://stackoverflow.com/questions/28064634/random-state-pseudo-random-numberin-scikit-learn
    """

    if random_state is None:
        random_state_gen = np.random.default_rng(seed=42)
    else:
        random_state_gen = np.random.default_rng(random_state)

    x_all_LF_copy = x_all_LF.copy()  # preserve what input was before the call

    random_state_gen.shuffle(x_all_LF_copy)  # pseudo-randomize input arr

    rows, cols = x_all_LF_copy.shape  # (10, 10)

    test_rows =  math.ceil(rows * frac_test)  # (3, 10)
    train_rows = rows - test_rows             # (7, 10)

    x_train_MF = x_all_LF_copy[0:train_rows]     # cols implied (7, 10)
    x_test_NF =  x_all_LF_copy[train_rows:rows]  # cols implied (3, 10)

    # for arr in [x_all_LF, x_all_LF_copy, x_train_MF, x_test_NF]:
    #     print(f'arr.shape:\n{arr.shape}\narr:\n{arr}\n')

    np.allclose(x_all_LF, x_all_LF_copy)  # Verify that input array did not change due to function call

    return x_train_MF, x_test_NF


def euclidean_distance(row_a, row_b):
    dist = np.linalg.norm(row_a, row_b)
    return dist


def calc_k_nearest_neighbors(data_NF, query_QF, K=1):
    ''' Compute and return k-nearest neighbors under Euclidean distance

    Any ties in distance may be broken arbitrarily.

    Args
    ----
    data_NF : 2D array, shape = (n_examples, n_features) aka (N, F)
        Each row is a feature vector for one example in dataset
    query_QF : 2D array, shape = (n_queries, n_features) aka (Q, F)
        Each row is a feature vector whose neighbors we want to find
    K : int, positive (must be >= 1)
        Number of neighbors to find per query vector

    Returns
    -------
    neighb_QKF : 3D array, (n_queries, n_neighbors, n_feats) (Q, K, F)
        Entry q,k is feature vector of the k-th neighbor of the q-th query
    '''
    # TODO fixme

    for q in range(len(query_QF)):
        distance_list = []
        for d in range(len(data_NF)):
            distance = np.linalg.norm(data_NF[d] - query_QF[q])
            distance_list.append(distance)
            print(query_QF[q], data_NF[d], distance)

        distance_list = np.array(distance_list)

    sys.exit()
    for row_a, row_b in zip(data_NF, query_QF):  # iterate through rows of both 2D arrays
        print(f'row_a: {row_a}\nrow_b: {row_b}\n')
        for i, j in zip(row_a, row_b):
            print(i, j)
            dist = np.linalg.norm(row_a[i] - row_b[j])
            print(dist)

    sys.exit()

    neighb_QKF = ''

    return neighb_QKF


if __name__ == '__main__':

    train_MF, test_NF = split_into_train_and_test(x_all_LF=np.eye(10), frac_test=0.3, random_state=420)

    neighb_QKF = calc_k_nearest_neighbors(data_NF=np.eye(10), query_QF=np.eye(10), K=1)