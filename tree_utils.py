import matplotlib as plt
import numpy as np


def find_split(data):
    # Get number of features and examples in data
    num_examples, num_features = np.shape(data[:, :-1])

    best_splits_for_each_feature = []
    # Iterate through each feature column, i.e. exclude last column which holds labels
    for feature in range(num_features):
        # Dictionary to hold split points as key and information gain as values
        split_points = {}
        sorted_data = data[np.argsort(data[:, feature])]  # TODO check this actually sorts the other columns properly
        for example in range(num_examples - 1):
            if sorted_data[example, feature] != sorted_data[example+1, feature]:
                left_child_label_column = sorted_data[0:(example + 1), -1]
                right_child_label_column = sorted_data[(example + 1):, -1]
                split_value = np.mean(sorted_data[example:example+2, feature])
                split_points[split_value] = information_gain(left_child_label_column, right_child_label_column, data[:, -1])
        # Return key (i.e. split value) for the maximum information gain
        try:
            feature_best_split_value = max(split_points, key=split_points.get)
            best_splits_for_each_feature.append([feature_best_split_value, split_points[feature_best_split_value]])
        except ValueError:
            continue

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
    return len(np.unique(dataset[:, -1])) == 1 or dataset[:, :-1].std(axis=0).sum() == 0


class Node:
    def __init__(self, attribute, value, left=None, right=None, is_leaf=True):
        self.attribute = attribute
        self.value = value
        self.left = left
        self.right = right
        self.is_leaf = is_leaf

    # def is_leaf(self):
    #     return self.left is None and self.right is None

    def create(self):
        branch_dict = {'attribute': self.attribute,
                       'value': self.value,
                       'left': self.left,
                       'right': self.right,
                       'leaf': self.is_leaf}
        return branch_dict

    def __repr__(self):
        return f"Node({self.attribute}, {self.value},{self.left}, {self.right} )"


class DecisionTreeBuilder:

    def build(self, dataset, depth):
        attribute = None
        value = None
        if is_pure_node(dataset):
            print('pure')
            return Node(attribute, value), depth
        else:
            print(' splitting...')
            value, attribute = find_split(dataset)  # returns best_split_value and best_split_feature
            print(value, attribute)
            print('start to make a node,')
            node = Node(attribute, value)
            print(node)
            left_dataset, right_dataset = split_dataset_by_split_point(dataset, attribute, value)
            print(np.shape(left_dataset), np.shape(right_dataset))

            node.left, l_depth = self.build(left_dataset, depth+1)
            print('left ds: ', node.left, l_depth)
            node.right, r_depth = self.build(right_dataset, depth+1)
            print('left ds: ', node.right, r_depth)
            return node, max(l_depth, r_depth)
