import matplotlib as plt
import numpy as np


def find_split():
    pass


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
