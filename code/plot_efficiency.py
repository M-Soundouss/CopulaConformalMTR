import matplotlib.pyplot as plt
import json
from pathlib import Path
import sys

plt.rcParams.update({'font.size': 22})
sys.path.append("..")


def get_box(data, index, label, epsilon="0.1"):
    sub_data = data["AVG"][epsilon]
    return {
        "label": label,
        "whislo": max(sub_data["MIN"][index], sub_data["Q1"][index]
                      - 1.5 * (sub_data["Q3"][index] - sub_data["Q1"][index])),
        "q1": sub_data["Q1"][index],
        "med": sub_data["MEDIAN"][index],
        "q3": sub_data["Q3"][index],
        "whishi": min(sub_data["MAX"][index], sub_data["Q3"][index]
                      + 1.5 * (sub_data["Q3"][index] - sub_data["Q1"][index])),
        "fliers": [],
    }


def plot_interval_size(data_independent, data_gumbel, data_empirical, model, name):
    medianprops = dict(linestyle='-', linewidth=3, color='firebrick')
    columns_len = len(config["targets"])
    epsilon = str(1 - pow(1 - 0.1, 1 / columns_len))
    columns = list(data_independent["AVG"][epsilon]["AVG"])[:-1]

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(12, 12), sharey=True)
    s_i = "hypercube"
    boxes = [
        get_box(data_independent, s_i, label="Independent Copula", epsilon=epsilon),
        get_box(data_gumbel, s_i, label="Gumbel Copula", epsilon='0.1'),
        get_box(data_empirical, s_i, label="Empirical Copula", epsilon='0.1'),
    ]
    axs.bxp(boxes, medianprops=medianprops)
    axs.set_title("%s : Hyper-rectangle median volume for %s" % (model, name))

    plt.savefig(base_path / ("%s_Hyperrectangle_box_plot_%s.eps" % (model, name)), format='eps')
    plt.close()


def plot_interval_size_both(data_independent_mlp, data_gumbel_mlp, data_empirical_mlp,
                            data_independent_rf, data_gumbel_rf, data_empirical_rf, name):
    medianprops = dict(linestyle='-', linewidth=3, color='firebrick')
    columns_len = len(config["targets"])
    epsilon = str(1 - pow(1 - 0.1, 1 / columns_len))
    columns = list(data_independent_mlp["AVG"][epsilon]["AVG"])[:-1]

    fig, [axes1, axes2, axes3] = plt.subplots(nrows=1, ncols=3, figsize=(12, 12), sharey=True)
    plt.yscale("log")
    s_i = "hypercube"
    boxes_independent = [
        get_box(data_independent_mlp, s_i, label="NN", epsilon=epsilon),
        get_box(data_independent_rf, s_i, label="RF", epsilon=epsilon)
    ]
    boxes_gumbel = [
        get_box(data_gumbel_mlp, s_i, label="NN", epsilon='0.1'),
        get_box(data_gumbel_rf, s_i, label="RF", epsilon='0.1')
    ]
    boxes_empirical = [
        get_box(data_empirical_mlp, s_i, label="NN", epsilon='0.1'),
        get_box(data_empirical_rf, s_i, label="RF", epsilon='0.1')
    ]
    axes1.bxp(boxes_independent, medianprops=medianprops)
    axes1.set_title("Independent")
    axes2.bxp(boxes_gumbel, medianprops=medianprops)
    axes2.set_title("Hyper-rectangle median volume for %s (log) \n Gumbel" % (name))
    axes3.bxp(boxes_empirical, medianprops=medianprops)
    axes3.set_title("Empirical")

    plt.savefig(base_path / ("Hyperrectangle_box_plot_%s.eps" % (name)), format='eps')
    plt.close()


def plot_interval_size_empirical(data_1, data_2, data_3, model, name):
    medianprops = dict(linestyle='-', linewidth=3, color='firebrick')
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(12, 12), sharey=True)
    s_i = "hypercube"
    boxes = [
        get_box(data_1, s_i, label="1 %", epsilon='0.1'),
        get_box(data_2, s_i, label="5 %", epsilon='0.1'),
        get_box(data_3, s_i, label="10 %", epsilon='0.1'),
    ]
    axs.bxp(boxes, medianprops=medianprops)
    axs.set_title("%s : Hyper-rectangle median volumes for Empirical (%s)" % (model, name))

    plt.savefig(base_path / ("%s_all_empirical_Hyperrectangle_box_plot_%s.eps" % (model, name)), format='eps')
    plt.close()


def plot_interval_size_gumbel(data_1, data_2, data_3, model, name):
    medianprops = dict(linestyle='-', linewidth=3, color='firebrick')
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(12, 12), sharey=True)
    s_i = "hypercube"
    boxes = [
        get_box(data_1, s_i, label="1 %", epsilon='0.1'),
        get_box(data_2, s_i, label="5 %", epsilon='0.1'),
        get_box(data_3, s_i, label="10 %", epsilon='0.1'),
    ]
    axs.bxp(boxes, medianprops=medianprops)
    axs.set_title("%s : Hyper-rectangle median volumes for Gumbel (%s)" % (model, name))

    plt.savefig(base_path / ("%s_all_gumbel_Hyperrectangle_box_plot_%s.eps" % (model, name)), format='eps')
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", default="../input/music_origin/")
    parser.add_argument("--name", default="music_origin")
    parser.add_argument("--cal_size", default=0.1, type=float)
    args = parser.parse_args()

    cal_size = args.cal_size

    base_path = Path(args.base_path)
    name = Path(args.name)
    path_independent = str(cal_size) + "_results_independent_nn_conf_interval.json"
    path_gumbel = str(cal_size) + "_results_gumbel_nn_conf_interval.json"
    path_empirical = str(cal_size) + "_results_empirical_nn_conf_interval.json"
    config_path = "config.json"

    with open(base_path / config_path) as f:
        config = json.load(f)

    with open(base_path / path_independent, "r") as f:
        data_independent_mlp = json.load(f)

    with open(base_path / path_gumbel, "r") as f:
        data_gumbel_mlp = json.load(f)

    with open(base_path / path_empirical, "r") as f:
        data_empirical_mlp = json.load(f)

    plot_interval_size(data_independent_mlp, data_gumbel_mlp, data_empirical_mlp, "NN", name)

    path_independent = str(cal_size) + "_results_independent_rf_conf_interval.json"
    path_gumbel = str(cal_size) + "_results_gumbel_rf_conf_interval.json"
    path_empirical = str(cal_size) + "_results_empirical_rf_conf_interval.json"

    with open(base_path / path_independent, "r") as f:
        data_independent_rf = json.load(f)

    with open(base_path / path_gumbel, "r") as f:
        data_gumbel_rf = json.load(f)

    with open(base_path / path_empirical, "r") as f:
        data_empirical_rf = json.load(f)

    plot_interval_size(data_independent_rf, data_gumbel_rf, data_empirical_rf, "RF", name)

    plot_interval_size_both(data_independent_mlp, data_gumbel_mlp, data_empirical_mlp,
                            data_independent_rf, data_gumbel_rf, data_empirical_rf, name)

    path_1 = "0.01_results_empirical_nn_conf_interval.json"
    path_2 = "0.05_results_empirical_nn_conf_interval.json"
    path_3 = "0.1_results_empirical_nn_conf_interval.json"

    with open(base_path / path_1, "r") as f:
        data_1 = json.load(f)

    with open(base_path / path_2, "r") as f:
        data_2 = json.load(f)

    with open(base_path / path_3, "r") as f:
        data_3 = json.load(f)

    plot_interval_size_empirical(data_1, data_2, data_3, "NN", name)

    path_1 = "0.01_results_empirical_rf_conf_interval.json"
    path_2 = "0.05_results_empirical_rf_conf_interval.json"
    path_3 = "0.1_results_empirical_rf_conf_interval.json"

    with open(base_path / path_1, "r") as f:
        data_1 = json.load(f)

    with open(base_path / path_2, "r") as f:
        data_2 = json.load(f)

    with open(base_path / path_3, "r") as f:
        data_3 = json.load(f)

    plot_interval_size_empirical(data_1, data_2, data_3, "RF", name)

    path_1 = "0.01_results_gumbel_nn_conf_interval.json"
    path_2 = "0.05_results_gumbel_nn_conf_interval.json"
    path_3 = "0.1_results_gumbel_nn_conf_interval.json"

    with open(base_path / path_1, "r") as f:
        data_1 = json.load(f)

    with open(base_path / path_2, "r") as f:
        data_2 = json.load(f)

    with open(base_path / path_3, "r") as f:
        data_3 = json.load(f)

    plot_interval_size_gumbel(data_1, data_2, data_3, "NN", name)

    path_1 = "0.01_results_gumbel_rf_conf_interval.json"
    path_2 = "0.05_results_gumbel_rf_conf_interval.json"
    path_3 = "0.1_results_gumbel_rf_conf_interval.json"

    with open(base_path / path_1, "r") as f:
        data_1 = json.load(f)

    with open(base_path / path_2, "r") as f:
        data_2 = json.load(f)

    with open(base_path / path_3, "r") as f:
        data_3 = json.load(f)

    plot_interval_size_gumbel(data_1, data_2, data_3, "RF", name)