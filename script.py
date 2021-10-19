import matplotlib as plt
import numpy as np

from data_utils import load_dataset, split_dataset_x_y


def run_wifi_localisation():
    data = load_dataset("data/clean_dataset.txt")
    x, y = split_dataset_x_y(data)
    pass


if __name__ == "__main__":
    run_wifi_localisation()