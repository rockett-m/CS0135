"""
Simple Classifiers for Cancer-Risk Screening

Problem Statement:
You have been given a data set containing some medical history information for
patients at risk for cancer [1]. This data has been split into various training and
testing sets; each set is given in CSV form, and is divided into inputs (x) and
outputs (y).

Each patient in the data set has been biopsied to determine their actual cancer
status. This is represented as a boolean variable, cancer in the y data sets,
where 1 means the patient has cancer and 0 means they do not. You will build
classifiers that seek to predict whether a patient has cancer, based on other
features of that patient. (The idea is that if we could avoid painful biopsies,
this would be preferred.)

[1] A. Vickers, Memorial Sloan Kettering Cancer Center
https://www.mskcc.org/sites/default/files/node/4509/documents/dca-tutorial-2015-2-26.pdf

================================================================================
See PDF for details of implementation and expected responses.
"""
import os
import numpy as np
import pandas as pd

import warnings

import sklearn.linear_model
import sklearn.metrics
import sklearn.calibration
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8')  # pretty matplotlib plots


def calc_confusion_matrix_for_threshold(y_true_N, y_proba1_N, thresh=0.5):
    ''' Compute the confusion matrix for a given probabilistic classifier and threshold

    Args
    ----
    y_true_N : 1D array of floats
        Each entry represents the binary value (0 or 1) of 'true' label of one example
        One entry per example in current dataset
    y_proba1_N : 1D array of floats
        Each entry represents a probability (between 0 and 1) that correct label is positive (1)
        One entry per example in current dataset
        Needs to be same size as ytrue_N
    thresh : float
        Scalar threshold for converting probabilities into hard decisions
        Calls an example "positive" if yproba1 >= thresh
        Default value reflects a majority-classification approach (class is the one that gets
        the highest probability)

    Returns
    -------
    cm_df : Pandas DataFrame
        Can be printed like print(cm_df) to easily display results
    '''
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true_N, y_proba1_N >= thresh)
    cm_df = pd.DataFrame(data=cm, columns=[0, 1], index=[0, 1])
    cm_df.columns.name = 'Predicted'
    cm_df.index.name = 'True'
    return cm_df


def calc_binary_metrics(y_true_N, y_hat_N):
    """
    Calculate confusion metrics.
    Args
    ----
    y_true_N : 1D array of floats
        Each entry represents the binary value (0 or 1) of 'true' label of one example
        One entry per example in current dataset
    y_hat_N : 1D array of floats
        Each entry represents a predicted binary value (either 0 or 1).
        One entry per example in current dataset.
        Needs to be same size as y_true_N.

    Returns
    -------
    TP : int
        Number of true positives
    TN : int
        Number of true negatives
    FP : int
        Number of false positives
    FN : int
        Number of false negatives
    """

    # TODO
    TP = 0.0
    TN = 0.0
    FP = 0.0
    FN = 0.0

    for ytrue, yhat in zip(y_true_N, y_hat_N):
        if ytrue == yhat and ytrue == 0: # no cancer and prediction correct
            TN += 1
        elif ytrue == yhat and ytrue == 1: # cancer and prediction correct
            TP += 1
        elif ytrue != yhat and ytrue == 0: # no cancer and prediction incorrect
            FP += 1
        else: # ytrue != yhat and ytrue == 1: # cancer and prediction incorrect
            FN += 1

    return TP, TN, FP, FN


def calc_percent_cancer(labels, cancer_label=1):
    """
    Calculate the number of instances that are equal to `cancer_label`
    :param labels: target variables of training of testing set
    :param cancer_label: the value indicating cancer positive
    :return: Percentage of labels marked as cancerous.
    """
    # TODO
    cancerous = sum([int(x) == cancer_label for x in labels])

    pct_cancerous = (cancerous / len(labels)) * 100

    return pct_cancerous


def predict_0_always_classifier(X):
    """
    Implement a classifier that predicts 0 always.
    :param X: Samples to classify
    :return: predictions from the always-0 classifier
    """
    # TODO
    predictions = np.zeros(len(X))

    return predictions  # floating point values between 0.0 and 1.0


def calc_accuracy(tp, tn, fp, fn):
    """
    Calculate the accuracy via confusion metrics.
    :param tp: Number of true positives
    :param tn: Number of true negative
    :param fp: Number of false negative
    :param fn: Number of false negative
    :return: Accuracy value from 0.0 to 1.0
    """
    # TODO
    accuracy_val = ((tp + tn) / (tp + tn + fp + fn))

    return accuracy_val


def standardize_data(X_train, X_test):
    """
    Transform features by scaling each feature to a given range.

    This estimator scales and translates each feature individually such that it
    is in the given range on the training set, e.g. between zero and one.

    :param X_train: training features
    :param X_test: testing features
    :return: standardize training features and testing features
    """
    from sklearn.preprocessing import MinMaxScaler
    # TODO
    # don't need to change family history since it is boolean already [0,1]
    scaler = MinMaxScaler(feature_range=(0, 1))

    # normalize each column using scaler.fit_transform
    for i in range(X_train.shape[1]):
        X_train[:, i] = scaler.fit_transform(X_train[:, i].reshape(-1, 1)).flatten()

    for i in range(X_test.shape[1]):
        X_test[:, i] = scaler.fit_transform(X_test[:, i].reshape(-1, 1)).flatten()

    return X_train, X_test


def calc_perf_metrics_for_threshold(y_true_N, y_proba1_N, thresh=0.5):  # check if correct
    """
    Compute performance metrics for a given probabilistic classifier and threshold
    Args
    ----
    y_true_N : 1D array of floats
        Each entry represents the binary value (0 or 1) of 'true' label of one example
        One entry per example in current dataset
    y_proba1_N : 1D array of floats
        Each entry represents a probability (between 0 and 1) that correct label is positive (1)
        One entry per example in current dataset
        Needs to be same size as y_true_N
    thresh : float
        Scalar threshold for converting probabilities into hard decisions
        Calls an example "positive" if y_proba1 >= thresh
        Default value reflects a majority-classification approach (class is the one that gets
        the highest probability)

    Returns
    -------
    acc : accuracy of predictions
    tpr : true positive rate of predictions
    tnr : true negative rate of predictions
    ppv : positive predictive value of predictions
    npv : negative predictive value of predictions
    """
    # TODO
    # tp, tn, fp, fn = calc_binary_metrics(y_true_N, y_hat_N)

    y_hat_N = np.array()  # replace with probabilities with 0 or 1
    for idx, ypn in enumerate(y_proba1_N):
        if ypn >= thresh: # we set equal to 1 if >= thresh
            y_hat_N[idx] = 1
        else:
            y_hat_N[idx] = 0

    tp, tn, fp, fn = calc_binary_metrics(y_true_N, y_hat_N)  # replaced y_hat_N 0,1

    acc = 0.0
    tpr = 0.0
    tnr = 0.0
    ppv = 0.0
    npv = 0.0

    # TODO
    acc = calc_accuracy(tp, tn, fp, fn)
    tpr = (tp / (tp + fn))  # recall # sensitivity
    tnr = (tn / (tn + fp))  # specificity
    ppv = (tp / (tp + fp))  # precision # positive pred and has cancer
    npv = (tn / (tn + fn))  # negative pred and does not have cancer

    return acc, tpr, tnr, ppv, npv


def perceptron_classifier(x_train, y_train, x_test, y_test,
                          penalty="l2", alpha=0, random_state=42):
    """
    Trains a perceptron classifier on the given training data and returns the
    predicted values on both training and test data.
    
    Args
    ----
    x_train : 1D array of floats
    y_train : 1D array of floats
    x_test : 1D array of floats
    y_test : 1D array of floats
    random_state : Seed used by the random number generator
    
    Returns
    -------
    tuple: A tuple of predicted values for the training and test data.
    """
    # Basic Perceptron Models
    # Fit a perceptron to the training data.
    # Print out accuracy on this data, as well as on testing data.
    # Print out a confusion matrix on the testing data.

    pred_train = 0.0
    pred_test = 0.0

    # TODO: Use penalty, alpha, random_state for your perceptron classifier
    # BE SURE TO SET RANDOM SEED FOR CLASSIFIER TO BE DETERMINISTIC TO PASS TEST

    # model with default values
    model = Perceptron(penalty=penalty, alpha=alpha, random_state=random_state)

    # fit to training data
    model.fit(x_train, y_train)
    # make predictions on the test data
    pred_train = model.predict(x_train)

    # fit to training data
    model.fit(x_test, y_test)
    # make predictions on the test data
    pred_test = model.predict(x_test)

    return pred_train, pred_test


def series_of_preceptrons(x_train, y_train, x_test, y_test, alphas):
    """
    Trains a series of perceptron classifiers with different regularization
    strengths and returns the accuracies of each model on both training and test data.
    
    Parameters:
    x_train (array-like): Training input samples.
    y_train (array-like): Target values for the training input samples.
    x_test (array-like): Test input samples.
    y_test (array-like): Target values for the test input samples.
    alphas (array-like): Array of regularization strengths to be used.
    
    Returns:
    tuple: A tuple of lists containing the accuracies of each model on both training and test data.
    """

    """Generate a series of regularized perceptron models

    Each model will use a different `alpha` value, multiplying that by the L2 penalty. 
    """
    # TODO
    train_accuracy_list = list()
    test_accuracy_list = list()

    for idx, alpha in enumerate(alphas):
        model = Perceptron(penalty="l2", alpha=alpha, random_state=42)

        model.fit(x_train, y_train)
        pred_train = model.predict(x_train)
        train_acc = accuracy_score(y_train, pred_train)
        train_accuracy_list.append(train_acc)

        model.fit(x_test, y_test)
        pred_test = model.predict(x_test)
        test_acc = accuracy_score(y_test, pred_test)
        test_accuracy_list.append(test_acc)

    return train_accuracy_list, test_accuracy_list


def calibrated_perceptron_classifier(x_train, y_train, x_test, y_test,
                                     penalty="l2", alpha=0, random_state=42):
    """
    Calibrate preceptron classifier
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :param penalty:
    :param alpha:
    :param random_state:
    :return:
    """
    pred_train = 0.0
    pred_test = 0.0

    # TODO: Use penalty, alpha, random_state for your perceptron classifier
    # BE SURE TO SET RANDOM SEED FOR CLASSIFIER TO BE DETERMINISTIC TO PASS TEST
    from sklearn.calibration import CalibratedClassifierCV
    cppn = Perceptron(penalty=penalty, alpha=alpha, random_state=random_state)

    cppn.fit(x_train, y_train)
    pred_train = cppn.decision_function(x_train)

    cppn.fit(x_test, y_test)
    pred_test = cppn.decision_function(x_test)

    return pred_train, pred_test


def find_best_thresholds(y_test, pred_prob_test):
    """ Compare the probabilistic classifier across multiple decision thresholds
    Args
    ----
    y_test : 1D array of floats
    pred_prob_test : 1D array of floats

    Returns
    -------
    best_TPR : best true positive rate
    best_PPV_for_best_TPR : best positive predictive value for best true positive rate
    best_TPR_threshold : best true positive rate threshold
    best_PPV : best positive predictive value
    best_TPR_for_best_PPV : best true positive rate for best positive predictive value
    best_PPV_threshold : best positive predictive value  threshold
    """
    # Try a range of thresholds for classifying data into the positive class (1).
    # For each threshold, compute the true postive rate (TPR) and positive
    # predictive value (PPV).  Record the best value of each metric, along with
    # the threshold that achieves it, and the *other*

    best_TPR = 0.0
    best_PPV_for_best_TPR = 0.0
    best_TPR_threshold = 0.0

    best_PPV = 0.0
    best_TPR_for_best_PPV = 0.0
    best_PPV_threshold = 0.0

    # TODO: test different thresholds to compute these values
    # thresholds = np.linspace(0, 1.001, 51)

    return best_TPR, best_PPV_for_best_TPR, best_TPR_threshold, best_PPV, best_TPR_for_best_PPV, best_PPV_threshold


# You can use this function later to make printing results easier; don't change it.
def print_perf_metrics_for_threshold(y_true, y_proba1, thresh=0.5):
    """
    Pretty print perf. metrics for a given probabilistic classifier and threshold.

    See calc_perf_metrics_for_threshold() for parameter descriptions.
    """
    acc, tpr, tnr, ppv, npv = calc_perf_metrics_for_threshold(y_true, y_proba1, thresh)

    # Pretty print the results
    print("%.3f ACC" % acc)
    print("%.3f TPR" % tpr)
    print("%.3f TNR" % tnr)
    print("%.3f PPV" % ppv)
    print("%.3f NPV" % npv)


if __name__ == '__main__':

    dir_data = "./data"

    all0 = np.zeros(10)
    all1 = np.ones(10)

    # Testing code
    # The following four calls to the function above test your results.
    TP, TN, FP, FN = calc_binary_metrics(all0, all1)
    print(f"TP: {TP}\t\tTN: {TN}\t\tFP: {FP}\tFN: {FN}\t")

    calc_binary_metrics(all1, all0)
    print(f"TP: {TP}\t\tTN: {TN}\t\tFP: {FP}\tFN: {FN}\t")

    calc_binary_metrics(all1, all1)
    print(f"TP: {TP}\t\tTN: {TN}\t\tFP: {FP}\tFN: {FN}\t")

    calc_binary_metrics(all0, all0)
    print(f"TP: {TP}\t\tTN: {TN}\t\tFP: {FP}\tFN: {FN}\t\n")

    # Load the dataset: creates arrays with the 2- or 3-feature input datasets.
    # Load the x-data and y-class arrays
    x_train = np.loadtxt(f'{dir_data}/x_train.csv', delimiter=',', skiprows=1)

    x_test = np.loadtxt(f'{dir_data}/x_test.csv', delimiter=',', skiprows=1)

    y_train = np.loadtxt(f'{dir_data}/y_train.csv', delimiter=',', skiprows=1)

    y_test = np.loadtxt(f'{dir_data}/y_test.csv', delimiter=',', skiprows=1)

    feat_names = np.loadtxt(f'{dir_data}/x_train.csv', delimiter=',', dtype=str, max_rows=1)

    target_name = np.loadtxt(f'{dir_data}/y_train.csv', delimiter=',', dtype=str, max_rows=1)

    print(f"features: {feat_names}\n")

    df_sampled_data = pd.DataFrame(x_test, columns=feat_names)
    df_sampled_data[str(target_name)] = y_test
    print(f'{df_sampled_data.sample(15)}\n')

    # 2: Compute the fraction of patients with cancer.
    # Compute values for train and test sets (i.e., don't hand-count and print).
    # percent_cancer_train = calc_percent_cancer(y_train)
    # percent_cancer_test = calc_percent_cancer(y_test)
    # print("Percent of data that has_cancer on TRAIN: %.2f" %
    #       percent_cancer_train if percent_cancer_train else 0.0)
    # print("Percent of data that has_cancer on TEST : %.2f\n" %
    #       percent_cancer_test if percent_cancer_test else 0.0)

    # The predict-0-always baseline
    # Compute and print the accuracy of the always-0 classifier on val and test.
    # y_train_pred = predict_0_always_classifier(x_train)
    # y_test_pred = predict_0_always_classifier(x_test)
    # acc_always0_train = calc_accuracy(*calc_binary_metrics(y_train, y_train_pred))
    # acc_always0_test = calc_accuracy(*calc_binary_metrics(y_test, y_test_pred))
    # print("acc on TRAIN: %.3f" %
    #       acc_always0_train if acc_always0_train else 0.0)
    # print("acc on TEST : %.3f\n" %
    #       acc_always0_test if acc_always0_test else 0.0)

    # Print a confusion matrix for the always-0 classifier.
    # Generate a confusion matrix for the always-0 classifier on the test set.
    # print(calc_confusion_matrix_for_threshold(y_test, np.zeros_like(y_test)))

    # Basic Perceptron Models
    # Fit a perceptron to the training data.
    # Print out accuracy on this data, as well as on testing data.
    # Print out a confusion matrix on the testing data.
    # pred_train, pred_test = perceptron_classifier()
    # print("acc on TRAIN: %.3f" % accuracy_score(y_train, pred_train))
    # print("acc on TEST : %.3f" % accuracy_score(y_test, pred_test))
    # print("")
    # print("Confusion matrix for TEST:")
    # print(calc_confusion_matrix_for_threshold(y_test, pred_test))

    # Generate a series of regularized perceptron models
    # Each model will use a different `alpha` value multiplied by the L2 penalty.
    # Record and plot the accuracy of each model on both training and test data.

    # train_accuracy_list, test_accuracy_list = series_of_preceptrons(
    # alphas=np.logspace(-5, 5, base=10, num=100))
    # plt.plot(alphas, train_accuracy_list, label='Accuracy on training')
    # plt.plot(alphas, test_accuracy_list, label='Accuracy on testing')
    # plt.legend(loc='lower right')
    # plt.xscale('log')
    # plt.xlabel('log_10(alpha)')
    # plt.ylabel('Accuracy')
    # plt.show()

    # Decision functions and probabilistic predictions
    #
    # Create two new sets of predictions
    #
    # Fit `Perceptron` and `CalibratedClassifierCV` models to the data.
    # Use their predictions to generate ROC curves.

    # _, decisions_test = perceptron_classifier()
    # _, pred_prob_test = calibrated_perceptron_classifier()

    # fpr, tpr, thr = sklearn.metrics.roc_curve(y_test, decisions_test)
    # plt.plot(fpr, tpr, label='Decision function version')

    # fpr2, tpr2, thr2 = sklearn.metrics.roc_curve(y_test, pred_prob_test)
    # plt.plot(fpr2, tpr2, label='Probabilistic version')

    # plt.legend(loc='lower right')
    # plt.xlabel('False positive rate (FPR)')
    # plt.ylabel('True positive rate (TPR)')
    #
    # plt.show()
    #
    # print("AUC on TEST for Perceptron: %.3f" % sklearn.metrics.roc_auc_score(
    #     y_test, decisions_test))
    # print(
    #     "AUC on TEST for probabilistic model: %.3f" % sklearn.metrics.roc_auc_score(
    #         y_test, pred_prob_test))
    #
    # # Compare the probabilistic classifier across multiple decision thresholds
    # #
    # # Try a range of thresholds for classifying data into the positive class (1).
    # # For each threshold, compute the true postive rate (TPR) and positive
    # # predictive value (PPV).  Record the best value of each metric, along with
    # # the threshold that achieves it, and the *other*
    # # TODO: test different thresholds to compute these values
    # best_TPR, best_PPV_for_best_TPR, \
    #     best_TPR_threshold, best_PPV, \
    #     best_TPR_for_best_PPV, \
    #     best_PPV_threshold = find_best_thresholds(y_test, pred_prob_test)
    #
    # print("TPR threshold: %.4f => TPR: %.4f; PPV: %.4f" % (
    #     best_TPR_threshold, best_TPR, best_PPV_for_best_TPR))
    # print("PPV threshold: %.4f => PPV: %.4f; TPR: %.4f" % (
    #     best_PPV_threshold, best_PPV, best_TPR_for_best_PPV))
    #
    # # #### (e) Exploring different thresholds
    # #
    # # #### (i) Using default 0.5 threshold.
    # #
    # # Generate confusion matrix and metrics for probabilistic classifier, using threshold 0.5.
    # best_thr = 0.5
    # print("ON THE TEST SET:")
    # print("Chosen best threshold = %.4f" % best_thr)
    # print("")
    # print(calc_confusion_matrix_for_threshold(y_test, pred_prob_test, best_thr))
    # print("")
    # print_perf_metrics_for_threshold(y_test, pred_prob_test, best_thr)
    #
    # #  Using threshold with highest TPR.
    # #
    # # Generate confusion matrix and metrics for probabilistic classifier, using
    # # threshold that maximizes TPR.
    # best_thr = best_TPR_threshold
    # print("ON THE TEST SET:")
    # print("Chosen best threshold = %.4f" % best_thr)
    # print("")
    # print(calc_confusion_matrix_for_threshold(y_test, pred_prob_test, best_thr))
    # print("")
    # print_perf_metrics_for_threshold(y_test, pred_prob_test, best_thr)
    #
    # # #### (iii) Using threshold with highest PPV.
    # #
    # # Generate confusion matrix and metrics for probabilistic classifier, using threshold that maximizes PPV.
    # best_thr = best_PPV_threshold
    # print("ON THE TEST SET:")
    # print("Chosen best threshold = %.4f" % best_thr)
    # print("")
    # print(calc_confusion_matrix_for_threshold(y_test, pred_prob_test, best_thr))
    # print("")
    # print_perf_metrics_for_threshold(y_test, pred_prob_test, best_thr)
