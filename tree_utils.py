import copy

import matplotlib as plt
import numpy as np
from numpy.random import default_rng

from Node import Node


def find_split(data):
    # Get number of features and examples in data
    num_examples, num_features = np.shape(data[:, :-1])

    best_splits_for_each_feature = []
    # Iterate through each feature column, i.e. exclude last column which holds labels
    for feature in range(num_features):
        # Dictionary to hold split points as key and information gain as values
        split_points = {}
        sorted_data = data[np.argsort(data[:, feature])]  # TODO check this actually sorts the other columns properly
        if len(np.unique(sorted_data[:, feature])) == 1: # if all values in this features are the same
            # the split point will be at the beginning or the end of the sorted dataset 
            # will result in a 0 information gain
            split_points[sorted_data[0, feature]] = 0 
        else:
            spitting_indices = [index for index in range(len(sorted_data[:,0])) if sorted_data[index, feature] != sorted_data[index-1, feature]][1:]
            for split_index in spitting_indices:
                left_label = sorted_data[0:(split_index), -1]
                right_label = sorted_data[(split_index):, -1]
                split_value = np.mean(sorted_data[split_index-1:split_index+1, feature])
                split_points[split_value] = information_gain(left_label, right_label, data[:, -1])
                
        # Return key (i.e. split value) for the maximum information gain
        feature_best_split_value = max(split_points, key=split_points.get)
        best_splits_for_each_feature.append([feature_best_split_value, split_points[feature_best_split_value]])

    # Find the feature with best split value
    best_split_feature = np.argmax([split_point[1] for split_point in best_splits_for_each_feature])
    best_split_value = best_splits_for_each_feature[best_split_feature][0]

    return best_split_value, best_split_feature


def split_dataset_by_split_point(dataset, attribute, value):
    """ Helper function to split a dataset by a specified split point.

    :param dataset:
    :param attribute:
    :param value:
    :return:
    """
    sorted_dataset = dataset[np.argsort(dataset[:, attribute])]
    right_dataset = sorted_dataset[np.where(sorted_dataset[:, attribute] > value)[0]]
    left_dataset = sorted_dataset[np.where(sorted_dataset[:, attribute] < value)[0]]
    return left_dataset, right_dataset


def information_entropy(label_column):
    """
    TODO
    """
    labels, frequencies = np.unique(label_column, return_counts=True)

    # labels_frequencies_dict = {labels[i]: frequencies[i] for i in range(len(labels))}

    probabilities = frequencies / np.sum(frequencies)

    return -1 * np.sum(np.log2(probabilities) * probabilities)


def information_gain(left_child_label_column, right_child_label_column, parent_label_column):
    """
    TODO
    """
    parent_entropy = information_entropy(parent_label_column)
    left_child_entropy = information_entropy(left_child_label_column)
    right_child_entropy = information_entropy(right_child_label_column)
    length_parent = len(parent_label_column)
    proportion_left_child = len(left_child_label_column) / length_parent
    proportion_right_child = len(right_child_label_column) / length_parent

    return parent_entropy - proportion_left_child * left_child_entropy - proportion_right_child * right_child_entropy


def is_pure_node(dataset):
    """Implement stopping condition if either all labels in a dataset are the same, or if there are no examples
    in the dataset.

    :param dataset:
    :return:
    """
    is_pure = False
    label = None
    node_count = None
    label_types, counts = np.unique(dataset[:, -1], return_counts=True)
    label_types = np.array([int(label_type) for label_type in label_types])
    if len(label_types) == 1:
        is_pure = True
        label = int(label_types[0])
        node_count = [label_types, counts]
    else:
        if dataset[:, :-1].std(axis=0).sum() == 0:
            label = int(label_types[np.argmax(counts)])
            is_pure = True
            node_count = [label_types, counts]
    
    return is_pure, label, node_count


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


def get_node_dict_from_tree(tree):
    """
    Retrieves a dictionary of nodes from a tree, i.e. a trained model, or output from .build() method
    in DecisionTreeBuilder class
    """
    nodes_to_process = [copy.deepcopy(tree)]
    node_dict = {}

    while nodes_to_process:
        # Get neighbors of current node and append to nodes_to_process list
        active_node = nodes_to_process[0]
        if active_node.label is None:
            nodes_to_process.append(active_node.left)
            nodes_to_process.append(active_node.right)

            node_dict.update({active_node.id: active_node})
            node_dict[active_node.id].left = active_node.left.id
            node_dict[active_node.id].right = active_node.right.id

        else:
            node_dict.update({active_node.id: active_node})
        # if active_node.label is None:
        #     list_of_trees.append({'id': active_node.id, 'attribute': active_node.attribute, 'value': active_node.value,
        #                           'left': active_node.left.id, 'right': active_node.right.id})
        # else:
        #     list_of_trees.append({'id': active_node.id, 'attribute': active_node.attribute, 'value': active_node.value,
        #                           'left': None, 'right': None})
        nodes_to_process.pop(0)
    return node_dict


def get_tree_from_dict(node_dict, node_id=1):
    # make a deepcopy before calling the function
    tree = node_dict[node_id]
    if tree.right == None and tree.left == None:
        return tree
    else:
        tree.left = get_tree_from_dict(node_dict, tree.left)
        tree.right = get_tree_from_dict(node_dict, tree.right)
        return tree


