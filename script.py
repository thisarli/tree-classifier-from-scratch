from cross_validation import run_simple_cross_validation
from data_utils import load_dataset
from nested_cross_validation import nested_cv_for_pruning

N_FOLDS = 10
FILEPATH = "data/noisy_dataset.txt"


def run_wifi_localisation():
    # Load the data
    data = load_dataset(FILEPATH)
    print(f"Loaded {FILEPATH} successfully")

    # Evaluate the ID3 tree-making algorithm without pruning (Task 3)
    print("Running the ID3 algorithm...")
    mean_accuracy, mean_cm, mean_precision, mean_recall, mean_f1_score, mean_depth = \
        run_simple_cross_validation(N_FOLDS, data)
    print("Successfully ran ID3 algorithm and evaluation")
    print(f"The mean accuracy of the ID3 algorithm is {mean_accuracy}")
    print(f"The mean confusion matrix of the ID3 algorithm is: \n {mean_cm}")
    print(f"The mean precision of the ID3 algorithm per class is: \n {mean_precision}")
    print(f"The mean recall of the ID3 algorithm per class is: \n {mean_recall}")
    print(f"The mean F1-Score of the ID3 algorithm per class is: \n {mean_f1_score}")
    print(f"The mean depth of the ID3 algorithm is {mean_depth}")

    # Evaluate the ID3-pruning algorithm
    print("Running ID3-Pruning algorithm. Please hold, this could take a while...")
    unpruned_mean_accuracy, pruned_mean_accuracy, unpruned_mean_depth, pruned_mean_depth = \
        nested_cv_for_pruning(data, N_FOLDS)
    print("Successfully ran ID3-Pruning algorithm and evaluation")
    print(f"The un-pruned mean accuracy of the ID3-Pruning algorithm is {unpruned_mean_accuracy}")
    print(f"The pruned_mean_accuracy of the ID3-Pruning algorithm is: \n {pruned_mean_accuracy}")
    print(f"The un-pruned mean depth of the ID3-Pruning algorithm is {unpruned_mean_depth}")
    print(f"The pruned mean depth of the ID3-Pruning algorithm is {pruned_mean_depth}")

    return


if __name__ == "__main__":
    run_wifi_localisation()
