import numpy as np

from DecisionTreeBuilder import DecisionTreeBuilder
from tree_utils import train_test_k_fold


def run_simple_cross_validation(n_folds, data):
    accuracies = []
    confusion_matrices = []
    precisions = []
    recalls = []
    f1_scores = []
    depths = []
    for i, (train_indices, test_indices) in enumerate(train_test_k_fold(n_folds, len(data))):
        print(f'------------{i}--------')
        # get the dataset from the correct splits
        train = data[train_indices, :]
        test = data[test_indices, :]
        # Train the KNN (we'll use one nearest neighbour)
        tree_classifier = DecisionTreeBuilder()
        model, depth = tree_classifier.build(train)
        y_test = test[:, -1]
        y_pred = tree_classifier.predict(test[:, :-1], model)

        acc = tree_classifier.accuracy(y_test, y_pred)
        accuracies += [acc]

        cm = tree_classifier.confusion_matrix(y_test, y_pred)
        confusion_matrices += [cm]

        pr = tree_classifier.precision(y_test, y_pred)
        precisions += [pr]

        rc = tree_classifier.recall(y_test, y_pred)
        recalls += [rc]

        f1 = tree_classifier.f1_score(y_test, y_pred)
        f1_scores += [f1]

        depths.append(depth)

    return np.mean(np.array(accuracies)), np.mean(np.array(confusion_matrices), axis=0), \
           np.mean(np.array(precisions), axis=0), np.mean(np.array(recalls), axis=0), \
           np.mean(np.array(f1_scores), axis=0), np.mean(depths)
