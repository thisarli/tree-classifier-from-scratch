from cross_validation import run_simple_cross_validation
from data_utils import load_dataset

N_FOLDS = 10


def run_wifi_localisation():
    data = load_dataset("data/noisy_dataset.txt")
    run_simple_cross_validation(N_FOLDS, data)
    pass


if __name__ == "__main__":
    run_wifi_localisation()