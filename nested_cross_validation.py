import copy

import numpy as np

from DecisionTreeBuilder import DecisionTreeBuilder
from metrics import predict, accuracy, f1_score, recall, precision, confusion_matrix
from tree_utils import get_tree_from_dict, train_test_k_fold, get_node_dict_from_tree


def pruning(tree, validation_set):
    """
    Function to prune a given tree based on a given validation dataset.
    It finds the leaves of the tree, finds their parent nodes (pruning candidates) and executes pruning for each of
    the pruning candidates. The resulting pruned tree with the greatest improvement in accuracy on the validation set
    will now become the new tree, which will be pruned again (i.e. find all leaves, parent nodes as candidates, etc.).
    The pruning stops when none of the pruned trees (i.e. for each pruning candidate) result in an improved accuracy.
    When a branch is pruned, the new leaf node will be assigned the class label of the majority class of the datapoints
    below.

    :param tree: dict {id : Node}
    :param validation_set: validation dataset including class labels (np.array)
    :return: pruned tree (Node), depth of pruned tree (int)
    """
    tree_copy = copy.deepcopy(tree)
    node_tree = get_tree_from_dict(tree_copy)
    predictions = predict(validation_set[:, :-1], node_tree)
    labels = validation_set[:, -1]
    original_accuracy = accuracy(labels, predictions)

    # the candidates for pruning are the nodes only connected to leaf nodes
    leaves_id = [node_id for node_id in tree if tree[node_id].label is not None]
    # Get the parent nodes where both children are leaf nodes (i.e. both children are in leaves_id)
    candidates = [node for node in tree.values() if ((node.left in leaves_id) and (node.right in leaves_id))]
    while len(candidates) > 0:
        pruning_accuracies = {}  # key : id of node where we prune, value: accuracy of pruned tree wrt validation set

        for candidate in candidates:
            update_to_leaf_node(tree, candidate)
            tree_copy_can = copy.deepcopy(tree)
            node_tree = get_tree_from_dict(tree_copy_can)
            new_predictions = predict(validation_set[:, :-1], node_tree)
            pruned_accuracy = accuracy(new_predictions, labels)
            # Add to pruning dictionary
            pruning_accuracies[candidate.id] = pruned_accuracy
            # Convert back to un-pruned tree to evaluate other pruning options against the original tree
            tree[candidate.id].label = None
            tree[candidate.id].count = None

        best_candidate = tree[max(pruning_accuracies, key=pruning_accuracies.get)]

        if pruning_accuracies[best_candidate.id] > original_accuracy:
            update_to_leaf_node(tree, best_candidate)

            # Update original accuracy to accuracy of pruned tree
            original_accuracy = pruning_accuracies[best_candidate.id]

            # Remove children of best_candidate from tree
            del tree[best_candidate.left]
            del tree[best_candidate.right]
            tree[best_candidate.id].right = None
            tree[best_candidate.id].left = None

            # Update leaves_id and candidates
            leaves_id = [node_id for node_id in tree if tree[node_id].label is not None]
            candidates = [node for node in tree.values() if ((node.left in leaves_id) and (node.right in leaves_id))]
        else:
            break

    pruned_tree_depth = get_depth(tree)
    tree_copy = copy.deepcopy(tree)

    return get_tree_from_dict(tree_copy), pruned_tree_depth


def alternative_pruning(tree, validation_set):
    """
    Not currently used.

    An alternative implementation of pruning given tree based on a given validation dataset.
    It finds the leaves of the tree, finds their parent nodes (pruning candidates) and executes pruning for each of
    the pruning candidates. As soon as a pruned tree leads to an improvement in accuracy on the validation set, this
    will now become the new tree, which will be pruned again (i.e. find all leaves, parent nodes as candidates, etc.).
    The pruning stops when none of the pruned trees (i.e. for each pruning candidate) result in an improved accuracy.

    :param tree: dict {id : Node}
    :param validation_set: validation dataset including class labels (np.array)
    :return: pruned tree (Node), depth of pruned tree (int)
    """
    tree_copy = copy.deepcopy(tree)
    node_tree = get_tree_from_dict(tree_copy)
    predictions = predict(validation_set[:, :-1], node_tree)
    labels = validation_set[:, -1]
    original_accuracy = accuracy(labels, predictions)

    leaves_id = [node_id for node_id in tree if tree[node_id].label is not None]
    candidates = [node for node in tree.values() if ((node.left in leaves_id) and (node.right in leaves_id))]
    
    while len(candidates) > 0:
        candidate = candidates[0]
        update_to_leaf_node(tree, candidate)
        tree_copy_can = copy.deepcopy(tree)
        node_tree = get_tree_from_dict(tree_copy_can)
        new_predictions = predict(validation_set[:, :-1], node_tree)
        pruned_accuracy = accuracy(new_predictions, labels)
        # If a single leaf reduces the validation error, then the node in pruned and replaced by a single leaf.
        if pruned_accuracy > original_accuracy:
            update_to_leaf_node(tree, candidate)

            # Update original accuracy to accuracy of pruned tree
            original_accuracy = pruned_accuracy

            # Remove children of best_candidate from tree
            del tree[candidate.left]
            del tree[candidate.right]
            tree[candidate.id].right = None
            tree[candidate.id].left = None

            # Update leaves_id and candidates
            leaves_id = [node_id for node_id in tree if tree[node_id].label is not None]
            candidates = [node for node in tree.values() if ((node.left in leaves_id) and (node.right in leaves_id))]
        
        else :
            # we restore the node if the validation accuracy didn't improve with pruning
            tree[candidate.id].label = None
            tree[candidate.id].count = None

        candidates.pop(0)

    pruned_tree_depth = get_depth(tree)
    tree_copy = copy.deepcopy(tree)

    return get_tree_from_dict(tree_copy), pruned_tree_depth


def get_depth(tree):
    """
    Retrieves the depth of the passed tree

    :param tree: tree in dictionary form (dict)
    :return: maximum depth of the tree (int)
    """
    max_depth = 0
    for node_id, node in tree.items():
        if node.depth > max_depth:
            max_depth = node.depth
    return max_depth


def update_to_leaf_node(tree, node):
    """
    Updates the node in the same tree dictionary to a leaf node by assigning it the majority label, and
    instance count for each class.

    :param tree: tree in dict form (dict of Nodes)
    :param node: id of the node to be turned into a leaf node (int)
    :return: None (just updates the passed tree)
    """
    # Assert the passed Node is not currently a leaf node
    assert tree[node.left].label is not None and tree[node.right].label is not None

    # Count is a list of arrays with count[0] the labels and count[1] the number of instances for each label
    left_count = tree[node.left].count
    right_count = tree[node.right].count

    count_dict = {left_count[0][i]: left_count[1][i] for i in range(len(left_count[0]))}

    for i in range(len(right_count[0])):
        if right_count[0][i] not in count_dict:
            count_dict[right_count[0][i]] = right_count[1][i]
        else:
            count_dict[right_count[0][i]] += right_count[1][i]
    # count_dict has key:label, value:total instances with this label in the children nodes
    majority_label = max(count_dict, key=count_dict.get)

    tree[node.id].label = majority_label
    labels = np.array([label for label in count_dict.keys()])
    counts = np.array([count for count in count_dict.values()])
    tree[node.id].count = [labels, counts]
    return


def nested_cv_for_pruning(dataset, n_fold):
    """
    Runs the nested cross-validation to get the evaluation metrics for the ID3-pruning algorithm

    Arg:
        dataset(array): each row is an instance, each column in the value for one feature. 
                        the last column in the class for this instance
        n_fold(int): number of folds this cross validation will implement 
    
    Returns: 
        a tuple of two nested list (original_tree_output, pruned_tree_output). 
        the first list has all averaged metrics for original decision tree;
        the second list has all averaged metrics for all pruned trees

        Each list has the following metrics (in order): [accuracy, depth, confusion matrix, recall, precision, f1]

        for recall, precision, f1, each of them is a (1, 4) list, each element corresponds to each room given in the
        question

    """
    # return the index of outer cross validation
    outer_folds = train_test_k_fold(n_fold, len(dataset))
    outer_original_accuracies = []
    outer_pruned_accuracies = []
    original_depths = []
    outer_pruned_depths = []

    # metrics of all repeats trained below
    f1_pruned = []
    conf_matrix_pruned = []
    recall_pruned = []
    precision_pruned = []

    f1_original = []
    conf_matrix_original = []
    recall_original = []
    precision_original = []

    for i, outer_fold in enumerate(outer_folds):
        print(f'============Nest CV Outer={i}=============')
        test_index = outer_fold[1]
        train_val_index = outer_fold[0]
        test_ds = dataset[test_index]
        train_val_ds = dataset[train_val_index]

        inner_folds = train_test_k_fold(n_fold, len(train_val_ds))
        inner_pruned_accuracies = []
        inner_original_accuracies = []
        inner_pruned_depths = []
        
        for i, inner_fold in enumerate(inner_folds):
            print(f'---------------Nest CV Inner-{i}---------------')
            val_index = inner_fold[1]
            train_index = inner_fold[0]
            val_ds = train_val_ds[val_index]
            train_ds = train_val_ds[train_index]
            y_gold = test_ds[:, -1]
            
            # unpruned tree
            print('>>> training the unpruned tree ...')
            decisiontree = DecisionTreeBuilder()
            trained_tree, original_depth = decisiontree.build(train_ds)
            predictions_original_tree = predict(test_ds[:, :-1], trained_tree)
            
            inner_original_accuracies.append(accuracy(y_gold, predictions_original_tree))
            original_depths += [original_depth]
            f1_original += [f1_score(y_gold, predictions_original_tree)]
            conf_matrix_original += [confusion_matrix(y_gold, predictions_original_tree)]
            recall_original += [recall(y_gold, predictions_original_tree)]
            precision_original += [precision(y_gold, predictions_original_tree)]

            # pruned tree
            print('>>> training the pruned tree ...')
            trained_tree = get_node_dict_from_tree(trained_tree)
            pruned_tree, pruned_tree_depth = pruning(trained_tree, val_ds)
            inner_pruned_depths += [pruned_tree_depth]
            predictions_pruned_tree = predict(test_ds[:, :-1], pruned_tree)

            inner_pruned_accuracies.append(accuracy(test_ds[:, -1], predictions_pruned_tree))
            f1_pruned += [f1_score(y_gold, predictions_pruned_tree)]
            conf_matrix_pruned += [confusion_matrix(y_gold, predictions_pruned_tree)]
            recall_pruned += [recall(y_gold, predictions_pruned_tree)]
            precision_pruned += [precision(y_gold, predictions_pruned_tree)]

        outer_pruned_accuracies.append(np.mean(inner_pruned_accuracies))
        outer_original_accuracies.append(np.mean(inner_original_accuracies))
        outer_pruned_depths += [np.mean(inner_pruned_depths)]

    pruned_tree_output = [np.mean(outer_pruned_accuracies), np.mean(outer_pruned_depths),
                            np.mean(conf_matrix_pruned, axis=0), np.mean(recall_pruned, axis=0), 
                            np.mean(precision_pruned, axis=0), np.mean(f1_pruned, axis=0)]
    original_tree_output = [np.mean(outer_original_accuracies), np.mean(original_depths),
                            np.mean(conf_matrix_original, axis=0), np.mean(recall_original, axis=0),
                            np.mean(precision_original, axis=0), np.mean(f1_original, axis=0)]
    print("Here are the results:")

    return original_tree_output, pruned_tree_output
