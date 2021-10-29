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
        y_gold (np.ndarray): the correct ground truth/gold standard labels
        y_prediction (np.ndarray): the predicted labels

    Returns:
        float : the accuracy
    """

    assert len(y_test) == len(y_pred)  
    
    return np.sum(y_test == y_pred) / len(y_test)
    

# Running the validation

def run_simple_cross_validation(n_folds, data, rg=default_rng()): # TODO make this work
    accuracies = []
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
    return np.array(accuracies)