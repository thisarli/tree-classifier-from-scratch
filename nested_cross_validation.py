# tree has been trained on training set
# function from dictionary to node tree

# This function will be called 10 times within 1 fold (=1 "held-out" test set) so 100 times in total
import copy

import numpy as np

from DecisionTreeBuilder import DecisionTreeBuilder
from metrics import predict, accuracy
from tree_utils import get_tree_from_dict, train_test_k_fold, get_node_dict_from_tree


def pruning(tree, validation_set):
    """
    Args:
      tree (dict {id : Node})
      validation_set (np.array) including class labels
  """
    tree_copy = copy.deepcopy(tree)
    print('>>> outer copy')
    node_tree = get_tree_from_dict(tree_copy)
    predictions = predict(validation_set[:, :-1], node_tree)
    labels = validation_set[:, -1]
    original_accuracy = accuracy(labels, predictions)

    # the candidates for pruning are the nodes only connected to leaf nodes
    leaves_id = [node_id for node_id in tree if tree[node_id].label is not None]
    print(leaves_id)
    # Get the parent nodes where both children are leaf nodes (i.e. both children are in leaves_id)
    candidates = [node for node in tree.values() if ((node.left in leaves_id) and (node.right in leaves_id))]
    print(candidates)
    while len(candidates) > 0:
        pruning_accuracies = {}  # key : id of node where we prune, value: accuracy of pruned tree wrt validation set

        for candidate in candidates:
            print(candidates)

            print(f'>>>>> candidate tree {candidate} ...')
            # pruning is simulated by assigning majority class label to the candidate
            # (when we predict the evaluation set on the pruned tree, the DecisionTreeBuilder.traverse() function will stop at this node with a not-None label)
            # tree[candidate.id].label = get_majority_label(tree, candidate)
            # tree_copy = copy.deepcopy(tree)
            update_to_leaf_node(tree, candidate)
            tree_copy_can = copy.deepcopy(tree)
            print('tree_copy_can', tree_copy_can)
            node_tree = get_tree_from_dict(tree_copy_can)
            new_predictions = predict(validation_set[:, :-1], node_tree)  # THIS SUPPOSES THAT tree IS NOW A Node
            pruned_accuracy = accuracy(new_predictions, labels)
            pruning_accuracies[candidate.id] = pruned_accuracy
            # Convert back to un-pruned tree to evaluate other pruning options against the original tree
            tree[candidate.id].label = None
            tree[candidate.id].count = None

        print('outtttt')
        best_candidate = tree[max(pruning_accuracies, key=pruning_accuracies.get)]
        print('best_candidate:', best_candidate)

        if pruning_accuracies[best_candidate.id] > original_accuracy:
            print("The candidate accuracy is better than the original")
            # tree[best_candidate.id].label = get_majority_label(tree, best_candidate)
            # tree[best_candidate.id].count = get_total_children_instances(tree, best_candidate)
            update_to_leaf_node(tree, best_candidate)

            # Update original accuracy to accuracy of pruned tree
            original_accuracy = pruning_accuracies[best_candidate.id]
            print("we updated the original accuracy")

            # Remove children of best_candidate from tree
            del tree[best_candidate.left]
            del tree[best_candidate.right]
            tree[best_candidate.id].right = None
            tree[best_candidate.id].left = None

            print("we removed the candidates children")

            # Update leaves_id and candidates
            leaves_id = [node_id for node_id in tree if tree[node_id].label is not None]
            print("Find new leafs")
            print(leaves_id)
            candidates = [node for node in tree.values() if ((node.left in leaves_id) and (node.right in leaves_id))]
            print('yayyyy we got new candidates', candidates)
        else:
            break

    pruned_tree_depth = get_depth(tree)
    tree_copy = copy.deepcopy(tree)

    return get_tree_from_dict(tree_copy), pruned_tree_depth


def get_depth(tree):
    max_depth = 0
    for node_id, node in tree.items():
        if node.depth > max_depth:
            max_depth = node.depth
    return max_depth

def update_to_leaf_node(tree, node):
    """
    Updates the node in the same tree dictionary to a leaf node by assigning it a majority label and
    instance count for each class

    tree (dict of Nodes)
    node (Node)
    """
    assert tree[node.left].label is not None and tree[node.right].label is not None

    # count is a list of arrays with count[0] labels and count[1] the number of instances for each label
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
    Runs the nested cross-validation to get the mean accuracy for the ID3-pruning algorithm
    """
    # return the index of outer cross validation
    outer_folds = train_test_k_fold(n_fold, len(dataset))
    outer_original_accuracies = []
    outer_pruned_accuracies = []
    original_depths = []
    outer_pruned_depths = []
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

            decisiontree = DecisionTreeBuilder()
            trained_tree, original_depth = decisiontree.build(train_ds)
            predictions_original_tree = predict(test_ds[:, :-1], trained_tree)
            inner_original_accuracies.append(accuracy(test_ds[:, -1], predictions_original_tree))
            original_depths += [original_depth]
            trained_tree = get_node_dict_from_tree(trained_tree)
            pruned_tree, pruned_tree_depth = pruning(trained_tree, val_ds)
            inner_pruned_depths += [pruned_tree_depth]
            predictions_pruned_tree = predict(test_ds[:, :-1], pruned_tree)
            inner_pruned_accuracies.append(accuracy(test_ds[:, -1], predictions_pruned_tree))

        outer_pruned_accuracies.append(np.mean(inner_pruned_accuracies))
        outer_pruned_depths += [np.mean(inner_pruned_depths)]
        outer_original_accuracies.append(np.mean(inner_original_accuracies))

    return np.mean(outer_original_accuracies), np.mean(outer_pruned_accuracies), np.mean(original_depths), np.mean(outer_pruned_depths)
