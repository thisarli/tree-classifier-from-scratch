import numpy as np

from tree_utils import traverse


def predict(x_test, tree):
    y_pred = []
    for instance in x_test:
        y_pred.append(traverse(instance, tree))
    return np.array(y_pred)


def accuracy(y_test, y_pred):
    """ Compute the accuracy given the ground truth and predictions

    Args:
        y_test (np.ndarray): the correct ground truth/gold standard labels
        y_pred (np.ndarray): the predicted labels

    Returns:
        float : the accuracy
    """

    assert len(y_test) == len(y_pred)
    return np.sum(y_test == y_pred) / len(y_test)


def confusion_matrix(y_gold, y_prediction, class_labels=None):
    """ Compute the confusion matrix.

    Args:
        y_gold (np.ndarray): the correct ground truth/gold standard labels
        y_prediction (np.ndarray): the predicted labels
        class_labels (np.ndarray): a list of unique class labels.
                               Defaults to the union of y_gold and y_prediction.

    Returns:
        np.array : shape (C, C), where C is the number of classes.
                   Rows are ground truth per class, columns are predictions
    """

    # if no class_labels are given, we obtain the set of unique class labels from
    # the union of the ground truth annotation and the prediction
    if not class_labels:
        class_labels = np.unique(np.concatenate((y_gold, y_prediction)))

    confusion = np.zeros((len(class_labels), len(class_labels)), dtype=np.int)

    # for each correct class (row),
    # compute how many instances are predicted for each class (columns)
    for (i, label) in enumerate(class_labels):
        # get predictions where the ground truth is the current class label
        indices = (y_gold == label)
        gold = y_gold[indices]
        predictions = y_prediction[indices]

        # quick way to get the counts per label
        (unique_labels, counts) = np.unique(predictions, return_counts=True)

        # convert the counts to a dictionary
        frequency_dict = dict(zip(unique_labels, counts))

        # fill up the confusion matrix for the current row
        for (j, class_label) in enumerate(class_labels):
            confusion[i, j] = frequency_dict.get(class_label, 0)

    return confusion


def precision(y_gold, y_prediction):
    """ Compute the precision score per class given the ground truth and predictions

    Also return the macro-averaged precision across classes.

    Args:
        y_gold (np.ndarray): the correct ground truth/gold standard labels
        y_prediction (np.ndarray): the predicted labels

    Returns:
        tuple: returns a tuple (precisions, macro_precision) where
            - precisions is a np.ndarray of shape (C,), where each element is the
              precision for class c
            - macro-precision is macro-averaged precision (a float)
    """

    confusion = confusion_matrix(y_gold, y_prediction)
    p = np.zeros((len(confusion),))
    for c in range(confusion.shape[0]):
        if np.sum(confusion[:, c]) > 0:
            p[c] = confusion[c, c] / np.sum(confusion[:, c])

    # # Compute the macro-averaged precision
    # macro_p = 0.
    # if len(p) > 0:
    #     macro_p = np.mean(p)

    return p


def recall(y_gold, y_prediction):
    """ Compute the recall score per class given the ground truth and predictions

    Also return the macro-averaged recall across classes.

    Args:
        y_gold (np.ndarray): the correct ground truth/gold standard labels
        y_prediction (np.ndarray): the predicted labels

    Returns:
        tuple: returns a tuple (recalls, macro_recall) where
            - recalls is a np.ndarray of shape (C,), where each element is the
                recall for class c
            - macro-recall is macro-averaged recall (a float)
    """

    confusion = confusion_matrix(y_gold, y_prediction)
    r = np.zeros((len(confusion),))
    for c in range(confusion.shape[0]):
        if np.sum(confusion[c, :]) > 0:
            r[c] = confusion[c, c] / np.sum(confusion[c, :])

    # # Compute the macro-averaged recall
    # macro_r = 0.
    # if len(r) > 0:
    #     macro_r = np.mean(r)

    return r


def f1_score(y_gold, y_prediction):
    """ Compute the F1-score per class given the ground truth and predictions

    Also return the macro-averaged F1-score across classes.

    Args:
        y_gold (np.ndarray): the correct ground truth/gold standard labels
        y_prediction (np.ndarray): the predicted labels

    Returns:
        tuple: returns a tuple (f1s, macro_f1) where
            - f1s is a np.ndarray of shape (C,), where each element is the
              f1-score for class c
            - macro-f1 is macro-averaged f1-score (a float)
    """

    precisions = precision(y_gold, y_prediction)
    recalls = recall(y_gold, y_prediction)

    # just to make sure they are of the same length
    assert len(precisions) == len(recalls)

    f = np.zeros((len(precisions),))
    for c, (p, r) in enumerate(zip(precisions, recalls)):
        if p + r > 0:
            f[c] = 2 * p * r / (p + r)

    return f