import numpy as np
import pandas as pd
import math
from tools.models import predict_per_target
from copulae import GumbelCopula
from copulae.core import pseudo_obs

pd.options.mode.chained_assignment = None


def empirical_copula_loss(x, data, epsilon):
    pseudo_data = pseudo_obs(data)
    return np.fabs(np.mean(np.all(np.less_equal(pseudo_data, np.array([x] * pseudo_data.shape[1])), axis=1)
                           ) - 1 + epsilon)


def gumbel_copula_loss(x, cop, data, epsilon):
    return np.fabs(cop.cdf([x] * data.shape[1]) - 1 + epsilon)


def aggregate_metrics_across_folds(xs):
    N = len(xs)
    epsilons = list(xs[0].keys())
    metrics = list(xs[0][epsilons[0]].keys())
    columns = list(xs[0][epsilons[0]][metrics[0]].keys())

    average = dict()
    for _epsilon in epsilons:
        average_epsilon = {}
        for m in metrics:
            average_epsilon[m] = {}
            for col in columns:
                values = [xs[a][_epsilon][m][col] for a in range(N)]
                average_epsilon[m][col] = np.mean(values)

        average[_epsilon] = average_epsilon

    xs["AVG"] = average

    std = dict()
    for _epsilon in epsilons:
        std_epsilon = {}
        for m in metrics:
            std_epsilon[m] = {}
            for col in columns:
                values = [xs[a][_epsilon][m][col] for a in range(N)]
                std_epsilon[m][col] = np.std(values)

        std[_epsilon] = std_epsilon

    xs["STD"] = std
    return xs


def aggregate_across_folds(xs):
    N = len(xs)
    epsilons = list(xs[0].keys())
    columns = list(xs[0][epsilons[0]].keys())

    average = dict()
    for _epsilon in epsilons:
        average[_epsilon] = {}
        for col in columns:
            values = [xs[a][_epsilon][col] for a in range(N)]
            average[_epsilon][col] = np.mean(values)

    xs["AVG"] = average

    std = dict()
    for _epsilon in epsilons:
        std[_epsilon] = {}
        for col in columns:
            values = [xs[a][_epsilon][col] for a in range(N)]
            std[_epsilon][col] = np.std(values)

    xs["STD"] = std

    return xs


def independent_norm_conf_all_targets_alpha_s(Y_cal, Y_cal_pred, epsilon, mu, beta):
    results_alpha_s = dict()

    for i in range(0, Y_cal.shape[1]):
        alphas_cal = np.abs(Y_cal[:, i] - Y_cal_pred[:, i]) / (np.exp(mu[:, i]) + beta)
        sorted_alphas = sorted(alphas_cal)
        quantile = math.ceil(((len(sorted_alphas) * (1 - epsilon)) - epsilon))
        results_alpha_s[i] = sorted_alphas[min(quantile, len(sorted_alphas)-1)]
    return results_alpha_s


def gumbel_norm_conf_all_targets_alpha_s(Y_cal, Y_cal_pred, epsilon, mu, beta):
    results_alpha_s = dict()

    alphas = np.abs(Y_cal - Y_cal_pred) / (np.exp(mu) + beta)
    mapping = {i: sorted(alphas[:, i].tolist()) for i in range(alphas.shape[1])}

    cop = GumbelCopula(dim=Y_cal.shape[1])
    cop.fit(alphas)

    x_candidates = np.linspace(0.0001, 0.999, num=300)
    x_fun = [gumbel_copula_loss(x, cop, alphas, epsilon) for x in x_candidates]

    x_sorted = sorted(list(zip(x_fun, x_candidates)))

    quantile = np.array([mapping[i][int(x_sorted[0][1] * alphas.shape[0])] for i in range(alphas.shape[1])])

    for i in range(0, Y_cal.shape[1]):
        results_alpha_s[i] = quantile[i]
    return results_alpha_s


def empirical_norm_conf_all_targets_alpha_s(Y_cal, Y_cal_pred, epsilon, mu, beta):
    results_alpha_s = dict()

    alphas = np.abs(Y_cal - Y_cal_pred) / (np.exp(mu) + beta)
    mapping = {i: sorted(alphas[:, i].tolist()) for i in range(alphas.shape[1])}

    x_candidates = np.linspace(0.0001, 0.999, num=300)
    x_fun = [empirical_copula_loss(x, alphas, epsilon) for x in x_candidates]

    x_sorted = sorted(list(zip(x_fun, x_candidates)))

    quantile = np.array([mapping[i][int(x_sorted[0][1] * alphas.shape[0])] for i in range(alphas.shape[1])])

    for i in range(0, Y_cal.shape[1]):
        results_alpha_s[i] = quantile[i]
    return results_alpha_s


# normalized conformal prediction
def prepare_norm_data(model, X_train, X_val, Y_train, Y_val):
    # predict
    Y_train_pred = model.predict(X_train)
    Y_val_pred = model.predict(X_val)

    # get new normalisation y = error difference
    Y_norm_train = np.log(np.abs(Y_train - Y_train_pred) + 1e-2)
    Y_norm_val = np.log(np.abs(Y_val - Y_val_pred) + 1e-2)
    return Y_norm_train, Y_norm_val


def prepare_norm_data_per_target(models, X_train, X_val, Y_train, Y_val):
    # predict
    Y_train_pred = predict_per_target(models, X_train, Y_train)
    Y_val_pred = predict_per_target(models, X_val, Y_val)

    # get new normalisation y = error difference
    Y_norm_train = np.log(np.abs(Y_train - Y_train_pred))
    Y_norm_val = np.log(np.abs(Y_val - Y_val_pred))
    return Y_norm_train, Y_norm_val


def norm_conf_predict(Y_cal_pred, mu, alphas, beta):
    Y_alphas = np.zeros(Y_cal_pred.shape)
    results = dict()
    for i in range(0, Y_cal_pred.shape[1]):
        alpha_s = alphas.get(i)
        Y_alphas[:, i] = alpha_s * (np.exp(mu[:, i]) + beta)
    for i in range(0, Y_cal_pred.shape[1]):
        results[i] = pd.DataFrame(
            list(
                zip(
                    Y_cal_pred[:, i] - Y_alphas[:, i],
                    Y_cal_pred[:, i],
                    Y_cal_pred[:, i] + Y_alphas[:, i],
                    Y_alphas[:, i],
                ),
            ),
            columns=["value_min", "value", "value_max", "alpha_normalized"],
        )
    return results


def check_conf_level(conf_preds, Y_true):
    Y_results = np.zeros(Y_true.shape)
    results = dict()
    for i in range(0, Y_results.shape[1]):
        value_min = conf_preds.get(i)["value_min"]
        value_max = conf_preds.get(i)["value_max"]
        for j in range(0, Y_results.shape[0]):
            if value_min[j] <= Y_true[j, i] <= value_max[j]:
                Y_results[j, i] = True
        results[i] = (np.count_nonzero(Y_results[:, i]) / Y_results.shape[0]) * 100

    results["hypercube"] = (np.count_nonzero(Y_results.prod(axis=-1)) / Y_results.shape[0]) * 100
    return results


def aggregates_interval_size(dict_df):
    avg = dict()
    std = dict()
    mini = dict()
    q1 = dict()
    med = dict()
    q3 = dict()
    maxi = dict()
    aggregates = dict()

    hypercube_volume = np.ones((dict_df[0].shape[0],))

    for i in dict_df:
        interval_size_i = dict_df[i]["value_max"] - dict_df[i]["value_min"]
        avg[i] = np.mean(interval_size_i)
        std[i] = np.std(interval_size_i)
        mini[i] = np.min(interval_size_i)
        q1[i] = np.percentile(interval_size_i, 25)
        med[i] = np.median(interval_size_i)
        q3[i] = np.percentile(interval_size_i, 75)
        maxi[i] = np.max(interval_size_i)

        hypercube_volume *= interval_size_i

    i = "hypercube"
    avg[i] = np.mean(hypercube_volume)
    std[i] = np.std(hypercube_volume)
    mini[i] = np.min(hypercube_volume)
    q1[i] = np.percentile(hypercube_volume, 25)
    med[i] = np.median(hypercube_volume)
    q3[i] = np.percentile(hypercube_volume, 75)
    maxi[i] = np.max(hypercube_volume)

    aggregates["AVG"] = avg
    aggregates["STD"] = std
    aggregates["MIN"] = mini
    aggregates["Q1"] = q1
    aggregates["MEDIAN"] = med
    aggregates["Q3"] = q3
    aggregates["MAX"] = maxi
    return aggregates
