import numpy as np
from numpy.random import default_rng

from data_utils import split_dataset_x_y
from tree_utils import DecisionTreeBuilder


def k_fold_split(n_splits, n_instances, random_generator=default_rng()):
    """ Split n_instances into n mutually exclusive splits at random.

    Args:
        n_splits (int): Number of splits
        n_instances (int): Number of instances to split
        random_generator (np.random.Generator): A random generator

    Returns:
        list: a list (length n_splits). Each element in the list should contain a
            numpy array giving the indices of the instances in that split.
    """

    # generate a random permutation of indices from 0 to n_instances
    shuffled_indices = random_generator.permutation(n_instances)

    # split shuffled indices into almost equal sized splits
    split_indices = np.array_split(shuffled_indices, n_splits)

    return split_indices

# # For quick testing
# k_fold_split(3, 20, rg)
#k_fold_split(10, np.shape(data)[0], random_generator=default_rng())


def train_test_k_fold(n_folds, n_instances, random_generator=default_rng()):
    """ Generate train and test indices at each fold.

    Args:
        n_folds (int): Number of folds
        n_instances (int): Total number of instances
        random_generator (np.random.Generator): A random generator

    Returns:
        list: a list of length n_folds. Each element in the list is a list (or tuple)
            with two elements: a numpy array containing the train indices, and another
            numpy array containing the test indices.
    """

    # split the dataset into k splits
    split_indices = k_fold_split(n_folds, n_instances, random_generator)

    folds = []
    for k in range(n_folds):
        # pick k as test
        test_indices = split_indices[k]

        # combine remaining splits as train
        # this solution is fancy and worked for me
        # feel free to use a more verbose solution that's more readable
        train_indices = np.hstack(split_indices[:k] + split_indices[k + 1:])

        folds.append([train_indices, test_indices])

    return folds

def traverse(instance, tree):
    node = tree
    label = node.label
    while node.label is None:
        attribute = node.attribute
        value = node.value
        if instance[attribute] < value:
            node = node.left
            label = traverse(instance, node)
        else:
            node = node.right
            label = traverse(instance, node)
    return label

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

# Running the validation


def run_simple_cross_validation(n_folds, data, rg=default_rng()):
    accuracies = []
    confusion_matrices = []
    precisions = []
    recalls = []
    f1_scores = []
    for i, (train_indices, test_indices) in enumerate(train_test_k_fold(n_folds, len(data), rg)):
        print(f'------------{i}--------')
        # get the dataset from the correct splits
        train = data[train_indices, :]
        test = data[test_indices, :]
        # Train the KNN (we'll use one nearest neighbour)
        tree_classifier = DecisionTreeBuilder()
        model = tree_classifier.build(train, 0)
        y_test = test[:, -1]
        y_pred = predict(test[:, :-1], model[0])
        acc = accuracy(y_test, y_pred)
        accuracies += [acc]
        cm = confusion_matrix(y_test, y_pred)
        confusion_matrices += [cm]
        pr = precision(y_test, y_pred)
        precisions += [pr]
        rc = recall(y_test, y_pred)
        recalls += [rc]
        f1 = f1_score(y_test, y_pred)
        f1_scores += [f1]

    return np.mean(np.array(accuracies)), np.mean(np.array(confusion_matrices), axis=0), \
           np.mean(np.array(precisions), axis=0), np.mean(np.array(recalls), axis=0), np.mean(np.array(f1_scores), axis=0)

