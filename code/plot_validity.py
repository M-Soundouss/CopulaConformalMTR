import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 22})
sys.path.append("..")


def get_accuracies(data, epsilons):
    avg = data["AVG"]
    return [avg[str(eps)]['hypercube'] for eps in epsilons]


def get_std(data, epsilons):
    avg = data["STD"]
    return [avg[str(eps)]['hypercube'] for eps in epsilons]


def plot_accuracies(data_independent, data_gumbel, data_empirical, model, name):
    epsilon_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    one_minus_epsilon = [1 - x for x in epsilon_values]

    columns_len = len(config["targets"])
    epsilon_ind = [1 - pow(1 - x, 1 / columns_len) for x in epsilon_values]

    fig = plt.figure(figsize=(12, 12))
    plt.plot(epsilon_values, [100 * x for x in epsilon_values], label="Calibration line", color="brown", linestyle='--',
             linewidth=3)
    plt.plot(one_minus_epsilon, get_accuracies(data_independent, epsilon_ind), label="Independent Copula",
             color="r", marker="v")
    plt.errorbar(one_minus_epsilon, get_accuracies(data_independent, epsilon_ind),
                 yerr=get_std(data_independent, epsilon_ind), color="r", marker="v")
    plt.plot(one_minus_epsilon, get_accuracies(data_gumbel, epsilon_values), label="Gumbel Copula",
             color="yellowgreen", marker="o")
    plt.errorbar(one_minus_epsilon, get_accuracies(data_gumbel, epsilon_values),
                 yerr=get_std(data_gumbel, epsilon_values), color="yellowgreen", marker="o")
    plt.plot(one_minus_epsilon, get_accuracies(data_gumbel, epsilon_values),
             label="Empirical Copula", color="c", marker="s")
    plt.errorbar(one_minus_epsilon, get_accuracies(data_empirical, epsilon_values),
                 yerr=get_std(data_empirical, epsilon_values), color="c", marker="s")

    plt.legend()
    plt.xlabel(r'$1 - \epsilon_g$')
    plt.ylabel('Validity (%)')
    plt.title('%s : Empirical validity for %s' % (model, name))
    plt.savefig(base_path / ("%s_validity_%s.eps" % (model, name)), format='eps')
    plt.close()


def plot_accuracies_empirical(data_1, data_2, data_3, model, name):
    epsilon_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    one_minus_epsilon = [1 - x for x in epsilon_values]

    fig = plt.figure(figsize=(12, 12))
    plt.plot(epsilon_values, [100 * x for x in epsilon_values], label="Calibration line", color="brown", linestyle='--',
             linewidth=3)
    plt.plot(one_minus_epsilon, get_accuracies(data_1, epsilon_values), label="1 %",
             color="r", marker="v")
    plt.errorbar(one_minus_epsilon, get_accuracies(data_1, epsilon_values),
                 yerr=get_std(data_1, epsilon_values), color="r", marker="v")
    plt.plot(one_minus_epsilon, get_accuracies(data_2, epsilon_values), label="5 %",
             color="yellowgreen", marker="o")
    plt.errorbar(one_minus_epsilon, get_accuracies(data_2, epsilon_values),
                 yerr=get_std(data_2, epsilon_values), color="yellowgreen", marker="o")
    plt.plot(one_minus_epsilon, get_accuracies(data_2, epsilon_values),
             label="10 %", color="c", marker="s")
    plt.errorbar(one_minus_epsilon, get_accuracies(data_3, epsilon_values),
                 yerr=get_std(data_3, epsilon_values), color="c", marker="s")

    plt.legend()
    plt.xlabel(r'$1 - \epsilon_g$')
    plt.ylabel('Validity (%)')
    plt.title('%s : Empirical validity for Empirical Copula (%s)' % (model, name))
    plt.savefig(base_path / ("%s_all_empirical_validity_%s.eps" % (model, name)), format='eps')
    plt.close()


def plot_accuracies_gumbel(data_1, data_2, data_3, model, name):
    epsilon_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    one_minus_epsilon = [1 - x for x in epsilon_values]

    fig = plt.figure(figsize=(12, 12))
    plt.plot(epsilon_values, [100 * x for x in epsilon_values], label="Calibration line", color="brown", linestyle='--',
             linewidth=3)
    plt.plot(one_minus_epsilon, get_accuracies(data_1, epsilon_values), label="1 %",
             color="r", marker="v")
    plt.errorbar(one_minus_epsilon, get_accuracies(data_1, epsilon_values),
                 yerr=get_std(data_1, epsilon_values), color="r", marker="v")
    plt.plot(one_minus_epsilon, get_accuracies(data_2, epsilon_values), label="5 %",
             color="yellowgreen", marker="o")
    plt.errorbar(one_minus_epsilon, get_accuracies(data_2, epsilon_values),
                 yerr=get_std(data_2, epsilon_values), color="yellowgreen", marker="o")
    plt.plot(one_minus_epsilon, get_accuracies(data_2, epsilon_values),
             label="10 %", color="c", marker="s")
    plt.errorbar(one_minus_epsilon, get_accuracies(data_3, epsilon_values),
                 yerr=get_std(data_3, epsilon_values), color="c", marker="s")

    plt.legend()
    plt.xlabel(r'$1 - \epsilon_g$')
    plt.ylabel('Validity (%)')
    plt.title('%s : Empirical validity for Gumbel Copula (%s)' % (model, name))
    plt.savefig(base_path / ("%s_all_gumbel_validity_%s.eps" % (model, name)), format='eps')
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", default="../input/music_origin")
    parser.add_argument("--name", default="music_origin")
    parser.add_argument("--cal_size", default=0.1, type=float)
    args = parser.parse_args()

    cal_size = args.cal_size

    base_path = Path(args.base_path)
    name = args.name
    path_gumbel = str(cal_size) + "_results_gumbel_rf_conf_levels.json"
    path_empirical = str(cal_size) + "_results_empirical_rf_conf_levels.json"
    path_independent = str(cal_size) + "_results_independent_rf_conf_levels.json"
    config_path = "config.json"

    with open(base_path / path_gumbel, "r") as f:
        data_gumbel = json.load(f)

    with open(base_path / path_empirical, "r") as f:
        data_empirical = json.load(f)

    with open(base_path / path_independent, "r") as f:
        data_independent = json.load(f)

    with open(base_path / config_path) as f:
        config = json.load(f)

    plot_accuracies(data_independent, data_gumbel, data_empirical, "RF", name)

    path_gumbel = str(cal_size) + "_results_gumbel_nn_conf_levels.json"
    path_empirical = str(cal_size) + "_results_empirical_nn_conf_levels.json"
    path_independent = str(cal_size) + "_results_independent_nn_conf_levels.json"

    with open(base_path / path_gumbel, "r") as f:
        data_gumbel = json.load(f)

    with open(base_path / path_empirical, "r") as f:
        data_empirical = json.load(f)

    with open(base_path / path_independent, "r") as f:
        data_independent = json.load(f)

    plot_accuracies(data_independent, data_gumbel, data_empirical, "NN", name)

    path_1 = "0.01_results_empirical_nn_conf_levels.json"
    path_2 = "0.05_results_empirical_nn_conf_levels.json"
    path_3 = "0.1_results_empirical_nn_conf_levels.json"

    with open(base_path / path_1, "r") as f:
        data_1 = json.load(f)

    with open(base_path / path_2, "r") as f:
        data_2 = json.load(f)

    with open(base_path / path_3, "r") as f:
        data_3 = json.load(f)

    plot_accuracies_empirical(data_1, data_2, data_3, "NN", name)

    path_1 = "0.01_results_empirical_rf_conf_levels.json"
    path_2 = "0.05_results_empirical_rf_conf_levels.json"
    path_3 = "0.1_results_empirical_rf_conf_levels.json"

    with open(base_path / path_1, "r") as f:
        data_1 = json.load(f)

    with open(base_path / path_2, "r") as f:
        data_2 = json.load(f)

    with open(base_path / path_3, "r") as f:
        data_3 = json.load(f)

    plot_accuracies_empirical(data_1, data_2, data_3, "RF", name)

    path_1 = "0.01_results_gumbel_nn_conf_levels.json"
    path_2 = "0.05_results_gumbel_nn_conf_levels.json"
    path_3 = "0.1_results_gumbel_nn_conf_levels.json"

    with open(base_path / path_1, "r") as f:
        data_1 = json.load(f)

    with open(base_path / path_2, "r") as f:
        data_2 = json.load(f)

    with open(base_path / path_3, "r") as f:
        data_3 = json.load(f)

    plot_accuracies_gumbel(data_1, data_2, data_3, "NN", name)

    path_1 = "0.01_results_gumbel_rf_conf_levels.json"
    path_2 = "0.05_results_gumbel_rf_conf_levels.json"
    path_3 = "0.1_results_gumbel_rf_conf_levels.json"

    with open(base_path / path_1, "r") as f:
        data_1 = json.load(f)

    with open(base_path / path_2, "r") as f:
        data_2 = json.load(f)

    with open(base_path / path_3, "r") as f:
        data_3 = json.load(f)

    plot_accuracies_gumbel(data_1, data_2, data_3, "RF", name)
