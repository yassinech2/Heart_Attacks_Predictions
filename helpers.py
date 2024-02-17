# -*- coding: utf-8 -*-
"""Some helper functions for project 1."""
import csv
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import seaborn as sns


def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x if std_x.all() != 0 else x
    return x, mean_x, std_x


def build_model_data(yb, input_data):
    """Form (y,tX) to get data in matrix form."""
    x = input_data
    num_samples = len(yb)
    tx = np.c_[np.ones(num_samples), x]
    return yb, tx


def sigmoid(t):
    """apply sigmoid function on t."""
    """apply sigmoid function on t."""
    t = np.where(t > 500, 500, t)
    t = np.where(t < -500, -500, t)
    return 1.0 / (1.0 + np.exp(-t))


def load_csv_data(data_path, sub_sample=False):
    """
    This function loads the data and returns the respectinve numpy arrays.
    Remember to put the 3 files in the same folder and to not change the names of the files.

    Args:
        data_path (str): datafolder path
        sub_sample (bool, optional): If True the data will be subsempled. Default to False.

    Returns:
        x_train (np.array): training data
        x_test (np.array): test data
        y_train (np.array): labels for training data in format (-1,1)
        train_ids (np.array): ids of training data
        test_ids (np.array): ids of test data
    """
    y_train = np.genfromtxt(
        os.path.join(data_path, "y_train.csv"),
        delimiter=",",
        skip_header=1,
        dtype=int,
        usecols=1,
    )
    x_train = np.genfromtxt(
        os.path.join(data_path, "x_train.csv"), delimiter=",", skip_header=1
    )
    x_test = np.genfromtxt(
        os.path.join(data_path, "x_test.csv"), delimiter=",", skip_header=1
    )

    train_ids = x_train[:, 0].astype(dtype=int)
    test_ids = x_test[:, 0].astype(dtype=int)
    x_train = x_train[:, 1:]
    x_test = x_test[:, 1:]

    # sub-sample
    if sub_sample:
        y_train = y_train[::50]
        x_train = x_train[::50]
        train_ids = train_ids[::50]

    return x_train, x_test, y_train, train_ids, test_ids


def create_csv_submission(ids, y_pred, name):
    """
    This function creates a csv file named 'name' in the format required for a submission in Kaggle or AIcrowd.
    The file will contain two columns the first with 'ids' and the second with 'y_pred'.
    y_pred must be a list or np.array of 1 and -1 otherwise the function will raise a ValueError.

    Args:
        ids (list,np.array): indices
        y_pred (list,np.array): predictions on data correspondent to indices
        name (str): name of the file to be created
    """
    # Check that y_pred only contains -1 and 1
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})


def print_results(yb_train, pred_train):
    """Print the accuracy, recall, precision, f1 score and confusion matrix."""
    assert len(yb_train) == len(
        pred_train
    ), "yb_train and pred_train must have the same length"
    # assert np.unique(yb_train).tolist() == [-1, 1], "yb_train must only contain -1 and 1"
    # assert np.unique(pred_train).tolist() == [-1, 1], "pred_train must only contain -1 and 1"

    print("Accuracy: ", accuracy(yb_train, pred_train))
    print("Recall: ", recall(yb_train, pred_train))
    print("Precision: ", precision(yb_train, pred_train))
    print("F1 score: ", f1_score(yb_train, pred_train))
    print("Confusion Matrix: \n", confusion_matrix(yb_train, pred_train))


def accuracy(y_true, y_pred):
    """Calculate the accuracy of predicted labels."""
    return np.mean(y_true == y_pred)


def recall(y_true, y_pred):
    """Calculate recall given true labels and predicted labels"""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == -1))
    return tp / (tp + fn)


def precision(y_true, y_pred):
    """Calculate precision given true labels and predicted labels"""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == -1) & (y_pred == 1))
    return tp / (tp + fp)


def f1_score(y_true, y_pred):
    """Calculate f1 score given true labels and predicted labels"""
    rec = recall(y_true, y_pred)
    prec = precision(y_true, y_pred)
    return 2 * (rec * prec) / (rec + prec)


def confusion_matrix(y_true, y_pred):
    """
    Compute the confusion matrix given true labels and predicted labels.
    Args:
        y_true (numpy.ndarray): True labels.
        y_pred (numpy.ndarray): Predicted labels.

    Returns:
        numpy.ndarray: The confusion matrix.
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == -1) & (y_pred == 1))
    tn = np.sum((y_true == -1) & (y_pred == -1))
    fn = np.sum((y_true == 1) & (y_pred == -1))
    return np.array([[tn, fp], [fn, tp]])


def SMOTE(X, y, k=5, ratio=1.0):
    """oversampling using the SMOTE algorithm\n
    This method oversamples only the minority class by creating synthetic samples.\n
    Args:
        X (np.array): training data
        y (np.array): labels of training data
        k (int, optional): number of neighbors to be considered. Defaults to 5.
        ratio (float, optional): ratio of newly created minority class/ minority class. (Defaults to 1.0 doubles the minority class).
    Returns:
        X_resampled (np.array): resampled training data
        y_resampled (np.array): resampled labels of training data
    """
    minority_samples = X[y == 1]
    n_samples = int(len(minority_samples) * ratio)

    synthetic_samples = np.zeros((n_samples, X.shape[1]))

    for i in tqdm(range(n_samples)):
        sample = minority_samples[np.random.choice(len(minority_samples))]
        distances = np.linalg.norm(minority_samples - sample, axis=1)
        k_neighbors = minority_samples[np.argsort(distances)[: k + 1][1:]]
        random_neighbor = k_neighbors[np.random.choice(k)]
        difference = random_neighbor - sample
        synthetic_samples[i] = sample + np.random.rand() * difference
    X_resampled = np.vstack([X, synthetic_samples])
    y_resampled = np.hstack([y, np.ones(n_samples)])

    return X_resampled, y_resampled


def undersample_majority(X, y, ratio=4):
    """
    Undersample the majority class by the specified ratio.
    Args:
        X (np.array): training data
        y (np.array): labels of training data
        ratio (float, optional): ratio of majority class/minority class. Defaults to 4.
    Returns:
        X_resampled (np.array): resampled training data
        y_resampled (np.array): resampled labels of training data
    """
    X_majority = X[y == -1]

    X_minority = X[y == 1]
    y_minority = y[y == 1]
    minority_samples = X[y == 1]

    n_samples = int(len(minority_samples) * ratio)

    indices = np.random.choice(X_majority.shape[0], n_samples, replace=False)
    X_majority_undersampled = X_majority[indices]

    X_resampled = np.vstack((X_minority, X_majority_undersampled))
    y_resampled = np.hstack((y_minority, [-1] * n_samples))

    return X_resampled, y_resampled


def train_test_split(X, y, test_size=0.2, random_state=None):
    """Split the data into train and test sets."""
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = len(X)
    n_test = int(test_size * n_samples)

    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test


def one_hot_encoder(x_train, column_names, relevant_features):
    """One hot encode the relevant features
    NOTE: This function replaces all 7 and 9 values with the median of the feature\n
    Args:
        x_train (np.array): training data
        column_names (np.array): list of column names of x_train
        relevant_features (np.array): names of the relevant features to be one hot encoded
    Returns:
            x_expanded (np.array): one hot encoded features
    """
    assert x_train.shape[1] == len(
        column_names
    ), "x_train and column_names must have the same length"
    x_expanded = np.zeros((x_train.shape[0], 0))
    for feature in relevant_features:
        values = np.unique(x_train[:, column_names.tolist().index(feature)])
        median = np.median(values)
        # replace all 7 and 9 values with median
        x_train[:, column_names.tolist().index(feature)][
            x_train[:, column_names.tolist().index(feature)] == 7
        ] = median
        x_train[:, column_names.tolist().index(feature)][
            x_train[:, column_names.tolist().index(feature)] == 9
        ] = median
        # delete 7 and 9 from values
        values = np.delete(values, np.where(values == 7))
        values = np.delete(values, np.where(values == 9))
        for value in values:
            # print("Feature {} with value {} encoded".format(feature, value))
            x_expanded = np.c_[
                x_expanded,
                (x_train[:, column_names.tolist().index(feature)] == value).astype(int),
            ]
    return x_expanded


def concat_features(x, x_one_hot, relevant_non_cat_features, column_names_clean):
    """Concatenate the one hot encoded features (categorical) with the non categorical features (continious)
    Args:
        x (np.array): training data
        x_one_hot (np.array): one hot encoded features
        relevant_non_cat_features (np.array): names of the relevant non categorical features
        column_names_clean (np.array): list of column names of x_train (should contain the names of the non categorical features)
    Returns:
            x_one_hot (np.array): one hot encoded features concatenated with non categorical features
    """
    for feature in relevant_non_cat_features:
        x_one_hot = np.concatenate(
            (x_one_hot, x[:, column_names_clean == feature]), axis=1
        )
    return x_one_hot


def feature_expansion(x, y, desired_number_of_features=190):
    """Feature expansion using the top 190 interaction terms
    Args:
        x_stan_train (np.array): training data
        x_stan_val (np.array): validation data
        x_stan_test (np.array): test data
        y_train_under (np.array): labels of training data
        desired_number_of_features (int, optional): number of total interaction terms to be selected. Defaults to 190.
    Returns:
            x_stan_train_extended (np.array): training data with interaction terms
            selected_interactions (np.array): indices of the selected interaction terms [(i,j),...]\n
    Examples:
    You can use it as follows: \n
    selected_interaction_terms = np.column_stack([x[:, i] * x[:, j] for i, j in selected_interactions]) \n
    This will output only the features expanded (without the original features)
    You will need to add the original features to the expanded features as follows: \n
    x = np.column_stack((x, selected_interaction_terms))
    """

    n_features = x.shape[1]  # number of features/columns
    upper_tri_indices = np.triu_indices(
        n_features, 1
    )  # indices of upper triangular matrix

    correlations = []  # list of correlations
    all_interactions = []  # list of all interactions between columns

    # Loop to calculate the correlation between each pair of columns
    for i, j in zip(upper_tri_indices[0], upper_tri_indices[1]):
        interaction = x[:, i] * x[:, j]
        correlation = np.corrcoef(interaction, y)[0, 1]

        # Append to our lists
        correlations.append(correlation)
        all_interactions.append((i, j))

    # Select top interactions
    top_indices = np.argsort(np.abs(correlations))[-desired_number_of_features:]

    # Store only the top interactions
    selected_interactions = [all_interactions[i] for i in top_indices]

    # Extract selected interaction terms for training set
    selected_interaction_terms_train = np.column_stack(
        [x[:, i] * x[:, j] for i, j in selected_interactions]
    )

    # Add Selected Interaction Terms to Original Features
    x_extended = np.column_stack((x, selected_interaction_terms_train))

    return x_extended, selected_interactions


def predict(tx, w, threshold=0.5):
    """
    Predict labels given weights and features in Logoistic Regression
    Args:
        tx (np.array): features
        w (np.array): weights of the logistic regression model
        threshold (float, optional): threshold to classify the labels. Defaults to 0.5.
    Returns:
        y_pred (np.array): predicted labels in format (-1,1)
    """
    y_pred = sigmoid(np.dot(tx, w))
    y_pred[y_pred <= threshold] = -1
    y_pred[y_pred > threshold] = 1
    return y_pred


def prepare_data(x_test, column_to_remove):
    """Clean the data by removing the necessary features & replcaing missing values with median\n
     Args:
        x_test (np.array): test data
        column_to_remove (np.array): indices of features to be removed

    Returns:
        x_test_clean (np.array): cleaned test data"""
    # remove features with more than 70% of missing values
    x_test_clean = np.delete(x_test, column_to_remove, axis=1)
    median = np.nanmedian(x_test_clean, axis=0)
    # replace missing values with median
    x_test_clean[np.isnan(x_test_clean)] = np.take(
        median, np.isnan(x_test_clean).nonzero()[1]
    )

    return x_test_clean


def feature_engineer(
    x,
    columns_clean,
    relevant_cat_features,
    relevant_non_cat_features,
    selected_interactions,
):
    """Feature engineering which includes : \n
    1. One hot encoding of categorical features \n
    2. Concatenating categorical features with non categorical features \n
    3. Expanding features using the selected interaction terms \n
    Args:
        x (np.array): training OR testing data
        columns_clean (np.array): names of the existing columns
        relevant_cat_features (np.array): names of the relevant categorical features
        relevant_non_cat_features (np.array): names of the relevant non categorical features
        selected_interactions (np.array): indices of the selected interaction terms [(i,j),...]\n
    Returns:
        x_final (np.array): Training OR Testing data with interaction terms
    """
    # One hot encode the relevant features
    # expand matrix by one hot encoding
    x_final = one_hot_encoder(
        x, columns_clean, relevant_cat_features
    )  # one hot encoding
    x_final = concat_features(
        x, x_final, relevant_non_cat_features, columns_clean
    )  # concatenate non categorical features with categorical features
    # Expand the features
    selected_interaction_terms = np.column_stack(
        [x_final[:, i] * x_final[:, j] for i, j in selected_interactions]
    )
    # Add Selected Interaction Terms to Original Features
    x_final = np.column_stack((x_final, selected_interaction_terms))
    return x_final


def remove_highly_correlated_columns(data, y, threshold=0.9):
    # Calculate the correlation matrix
    corr_matrix = np.corrcoef(data, rowvar=False)

    # Find pairs of highly correlated columns
    n = corr_matrix.shape[0]
    correlated_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(corr_matrix[i, j]) > threshold:
                correlated_pairs.append((i, j))

    # Columns to keep
    columns_to_keep = list(range(n))

    # Remove the least correlated column to y from each correlated pair
    columns_clean = []
    for pair in correlated_pairs:
        # Find the correlation of each column to y
        corr_to_y = [abs(np.corrcoef(data[:, col], y)[0, 1]) for col in pair]
        # print(corr_to_y)
        # Remove the column with the least correlation to y if it hasn't been removed already
        col_to_remove = pair[np.argmin(corr_to_y)]
        # print(col_to_remove)
        # for col in reversed(col_to_remove):
        if col_to_remove in columns_to_keep:
            columns_to_keep.remove(col_to_remove)
            columns_clean.append(col_to_remove)

    # Extract the columns to keep from the original data
    data_filtered = data[:, columns_to_keep]

    return data_filtered, columns_to_keep, columns_clean


def best_threshold(tx_test, y_test_split, w):
    # This cell plots the accuracy and f1 score for different thresholds
    best_threshold = 0.5
    best_f1 = 0
    best_accuracy = 0
    accuracy_list = []
    f1_score_list = []

    # Iterate over potential thresholds
    for threshold in np.arange(0.510, 0.99, 0.01):
        y_pred_val = predict(tx_test, w, threshold=threshold)
        f1 = f1_score(y_test_split, y_pred_val)
        acc = accuracy(y_test_split, y_pred_val)
        accuracy_list.append(acc)
        f1_score_list.append(f1)
        # Update if we found a better threshold
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_accuracy = acc

    print(f"Best Threshold: {best_threshold:.2f}")
    print(f"Accuracy for Validation Set at Best Threshold: {best_accuracy}")
    print(f"F1 score for Validation Set at Best Threshold: {best_f1}")

    sns.set_style("darkgrid")
    sns.set_palette("Set2")

    # Plot the accuracy and f1 score lists against the threshold values
    thresholds = np.arange(0.51, 0.99, 0.01)
    plt.figure(figsize=(5, 4))
    plt.plot(thresholds, accuracy_list, label="Accuracy")
    plt.plot(thresholds, f1_score_list, label="F1 Score")
    plt.scatter(
        best_threshold,
        best_f1,
        color="red",
        label=f"Best Threshold: {best_threshold:.2f}",
    )
    plt.legend()
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Accuracy and F1 Score vs. Threshold")
    plt.legend()
    plt.show()


# def select_best_threshold(y_val_split,val_tx,w):


#     """Select the best threshold for the logistic regression model. \n
#     The best threshold is the one that maximizes the f1 score.\n
#     Args:
#         y_val_split (np.array): validation labels
#         val_tx (np.array): validation features
#         w (np.array): weights of the logistic regression model
#     Returns:
#         best_threshold (float): best threshold
#         best_f1 (float): best f1 score
#         best_accuracy (float): best accuracy
#     """
#     best_threshold = 0.51
#     best_f1 = 0
#     best_accuracy = 0

#     # Iterate over potential thresholds
#     for threshold in np.arange(0.51, 0.81, 0.01):
#         y_pred_val = predict(val_tx, w, threshold=threshold)
#         y_pred_val = (y_pred_val + 1) / 2
#         f1 = f1_score(y_val_split, y_pred_val)
#         acc = accuracy(y_val_split, y_pred_val)

#         # Update if we found a better threshold
#         if f1 > best_f1:
#             best_f1 = f1
#             best_threshold = threshold
#             best_accuracy = acc

#     print(f"Best Threshold: {best_threshold:.2f}")
#     return best_threshold, best_f1, best_accuracy
