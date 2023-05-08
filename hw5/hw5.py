import os

import numpy as np
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the movie review dataset and preprocess it.

    Args:
        file_path (str): The path to the dataset file.

    Returns:
        DataFrame: A pandas DataFrame containing the preprocessed dataset.
    """
    # Load dataset
    data = pd.read_csv(file_path)

    data = pd.DataFrame(data)
    # Lowercase text (review column)
    data['review'] = data['review'].str.lower()

    return data


def split_data(data, test_size=0.2, random_seed=42) -> Tuple[np.array, np.array, np.array, np.array]:
    """
    Split the dataset into training and testing sets.

    Use random_seed.
    Args:
        data (DataFrame): The preprocessed movie review dataset.
        test_size (float): The proportion of the dataset to include in the test set.
        random_seed (int): Seed to set split function with.

    Returns:
        tuple: The training and testing datasets.
    """
    X_train, X_test, y_train, y_test = None, None, None, None
    print(data.columns)

    X_train, X_test, y_train, y_test = train_test_split(data, test_size=test_size, random_state=random_seed)

    return X_train, X_test, y_train, y_test


def create_naive_bayes_classifier() -> MultinomialNB:
    """
    Create a Naive Bayes classifier for sentiment analysis.

    Returns:
        MultinomialNB: A Naive Bayes classifier object.
    """

    clf = None
    return clf


def train_classifier(clf: MultinomialNB, X_train: pd.Series,
                     y_train: pd.Series) -> CountVectorizer:
    """
    Train the Naive Bayes classifier using the training dataset.

    Args:
        clf (MultinomialNB): The Naive Bayes classifier object.
        X_train (Series): The training dataset.
        y_train (Series): The target variable for the training dataset.
    """
    # Vectorize count (stop_words=english)
    vectorizer = None
    # fit and transform

    # fit

    return vectorizer


def test_classifier(clf: MultinomialNB, vectorizer: CountVectorizer,
                    X_test: pd.Series) -> np.ndarray:
    """
    Test the Naive Bayes classifier using the testing dataset.

    Args:
        clf (MultinomialNB): The Naive Bayes classifier object.
        vectorizer (CountVectorizer): The vectorizer object used for transforming the dataset.
        X_test (Series): The testing dataset.

    Returns:
        np.ndarray: The predicted sentiment labels for the testing dataset.
    """
    # Transform using vectorize
    X_test_vectorized = None
    # Predict
    y_pred = None

    return y_pred


def evaluate_classifier(y_test: pd.Series, y_pred: np.array):
    """
    Evaluate the performance of the Naive Bayes classifier.

    Args:
        y_test (Series): The true sentiment labels for the testing dataset.
        y_pred (np.ndarray): The predicted sentiment labels for the testing dataset.
    """
    acc = None
    conf = None
    clss_report = None
    print("Accuracy:", acc)
    print("\nConfusion Matrix:\n", conf)
    print("\nClassification Report:\n", clss_report)


if __name__ == "__main__":

    data = load_data(file_path="movie-reviews.csv")

    X_train, X_test, y_train, y_test = split_data(data)

    clf = create_naive_bayes_classifier()

    vectorizer = train_classifier(clf, X_train, y_train)

    y_pred = test_classifier(clf, vectorizer, X_test)

    evaluate_classifier(y_test, y_pred)
