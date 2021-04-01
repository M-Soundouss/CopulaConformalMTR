import json
from pathlib import Path
import sys
import numpy as np

sys.path.append("..")


def get_box(data, index, label, epsilon="0.1"):
    avg_data = data["AVG"][epsilon]
    std_data = data["STD"][epsilon]
    return str(label) + " & " + str(np.format_float_scientific(avg_data["MEDIAN"][index], 3)) + " \pm " + \
           str(np.format_float_scientific(std_data["MEDIAN"][index], 3))


def get_val_box(data, index, label, epsilons):
    sum_avg = 0
    sum_std = 0
    for epsilon in epsilons:
        epsilon_g = 0.1 * (epsilons.index(epsilon) + 1)
        avg_data = data["AVG"][str(epsilon)]
        std_data = data["STD"][str(epsilon)]
        sum_avg = sum_avg + (avg_data[index] / 100) - (1 - epsilon_g)
        sum_std = sum_std + (std_data[index] / 100)
    moy_avg = np.format_float_scientific((sum_avg / len(epsilons)) * 100, 3)
    moy_std = np.format_float_scientific((sum_std / len(epsilons)) * 100, 3)
    return str(label) + " & " + str(moy_avg) + " \pm " + str(moy_std)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", default="../input/music_origin/")
    parser.add_argument("--cal_size", default=0.1, type=float)
    args = parser.parse_args()

    cal_size = args.cal_size

    base_path = Path(args.base_path)
    path_gumbel = str(cal_size) + "_results_gumbel_nn_conf_levels.json"
    path_empirical = str(cal_size) + "_results_empirical_nn_conf_levels.json"
    path_independent = str(cal_size) + "_results_independent_nn_conf_levels.json"
    config_path = "config.json"

    with open(base_path / config_path) as f:
        config = json.load(f)

    with open(base_path / path_independent, "r") as f:
        data_independent = json.load(f)

    with open(base_path / path_gumbel, "r") as f:
        data_gumbel = json.load(f)

    with open(base_path / path_empirical, "r") as f:
        data_empirical = json.load(f)

    columns_len = len(config["targets"])

    print("VALIDITY =========================================================")
    epsilons = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    epsilons_corr = []
    for epsilon in epsilons:
        epsilons_corr.append(1 - pow(1 - epsilon, 1 / columns_len))
    s_i = "hypercube"
    print(get_val_box(data_independent, s_i, label="Independent", epsilons=epsilons_corr))
    print(get_val_box(data_gumbel, s_i, label="Gumbel", epsilons=epsilons))
    print(get_val_box(data_empirical, s_i, label="Empirical", epsilons=epsilons))

    path_independent = str(cal_size) + "_results_independent_nn_conf_interval.json"
    path_gumbel = str(cal_size) + "_results_gumbel_nn_conf_interval.json"
    path_empirical = str(cal_size) + "_results_empirical_nn_conf_interval.json"
    config_path = "config.json"

    with open(base_path / config_path) as f:
        config = json.load(f)

    with open(base_path / path_independent, "r") as f:
        data_independent = json.load(f)

    with open(base_path / path_gumbel, "r") as f:
        data_gumbel = json.load(f)

    with open(base_path / path_empirical, "r") as f:
        data_empirical = json.load(f)

    print("EFFICIENCY =========================================================")
    epsilons = ["0.1"]
    s_i = "hypercube"
    for epsilon in epsilons:
        epsilons_corr = str(1 - pow(1 - float(epsilon), 1 / columns_len))
        print(epsilon)
        print(get_box(data_independent, s_i, label="Independent", epsilon=epsilons_corr))
        print(get_box(data_gumbel, s_i, label="Gumbel", epsilon=epsilon))
        print(get_box(data_empirical, s_i, label="Empirical", epsilon=epsilon))