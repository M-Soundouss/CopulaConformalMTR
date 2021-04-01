import json
import sys
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import time

sys.path.append("../")
sys.path.append("../../")

from tools.preprocessing_utilities import (
    learn_scalers,
    apply_scalers,
    get_continuous_arrays,
    get_targets_arrays
)
from tools.models import (
    two_model_mlp,
    fit_per_target,
    predict_per_target
)
from tools.conformal_utilities import (
    aggregate_metrics_across_folds,
    aggregate_across_folds
)
from tools.nonconformity_predictor import (
    multi_target_nn_nonconformity,
    single_target_rf_nonconformity
)

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", default="../input/music_origin/")
    parser.add_argument("--data_path", default="music_origin.csv")
    parser.add_argument("--cal_size", default=0.1, type=float)
    parser.add_argument("--nb_epoch", default=400, type=int)
    parser.add_argument("--layer_size", default=1024, type=int)
    parser.add_argument("--embed_size", default=64, type=int)
    parser.add_argument("--dropout_rate", default=0.3, type=float)
    parser.add_argument("--batch_size", default=256, type=int)

    args = parser.parse_args()

    nb_epoch = args.nb_epoch
    batch_size = args.batch_size
    embed_size = args.embed_size
    layer_size = args.layer_size
    dropout_rate = args.dropout_rate
    base_path = Path(args.base_path)
    data_path = args.data_path
    config_path = "config.json"
    data = pd.read_csv(base_path / data_path, sep="|")
    with open(base_path / config_path) as f:
        config = json.load(f)
    n_cont = len(config["continuous_variables"])
    n_out = len(config["targets"])
    n_kfold = 10
    cal_size = args.cal_size
    epsilon_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # update embed_size
    embed_size = min(embed_size, max(n_cont // 2 + 1, 10))

    independent_nn_conf_levels = dict()
    independent_nn_conf_interval = dict()
    gumbel_nn_conf_levels = dict()
    gumbel_nn_conf_interval = dict()
    empirical_nn_conf_levels = dict()
    empirical_nn_conf_interval = dict()

    independent_rf_conf_levels = dict()
    independent_rf_conf_interval = dict()
    gumbel_rf_conf_levels = dict()
    gumbel_rf_conf_interval = dict()
    empirical_rf_conf_levels = dict()
    empirical_rf_conf_interval = dict()

    i = 0

    # prepare cross validation
    kfold = KFold(n_splits=n_kfold, shuffle=True, random_state=2)
    # enumerate splits
    for train, test in kfold.split(data):
        train = data.iloc[train]
        test = data.iloc[test]
        train, cal = train_test_split(train, test_size=cal_size, random_state=1337)
        scaler_continuous, scaler_targets = learn_scalers(train_df=train, config=config)

        (train, cal, test) = apply_scalers(
            list_df=[train, cal, test],
            scaler_continuous=scaler_continuous,
            scaler_targets=scaler_targets,
            config=config,
        )

        X_train_cont, X_cal_cont, X_test_cont = get_continuous_arrays(
            list_df=[train, cal, test], config=config
        )
        Y_train, Y_cal, Y_test = get_targets_arrays(
            list_df=[train, cal, test], config=config
        )

        X_train_cont, X_val_cont, Y_train, Y_val = train_test_split(
            X_train_cont, Y_train, test_size=0.1, random_state=1337
        )

        checkpoint = ModelCheckpoint(
            "mlp.h5", monitor="val_loss", verbose=1, save_best_only=True, mode="min"
        )
        early = EarlyStopping(monitor="val_loss", mode="min", patience=20, verbose=1)
        redonplat = ReduceLROnPlateau(
            monitor="val_loss", mode="min", patience=10, verbose=2
        )
        callbacks_list = [checkpoint, early, redonplat]

        # normalized conformal prediction for multi
        print("Normalized conformal prediction for MULTI NN %s" % i)

        start_multi = time.time()

        model, model_repr = two_model_mlp(
            n_continuous=n_cont,
            n_outputs=n_out,
            learning_rate=0.0001,
            layer_size=layer_size,
            embed_size=embed_size,
            dropout_rate=dropout_rate,
            n_layers=5,
        )
        model.fit(
            X_train_cont,
            Y_train,
            validation_data=(X_val_cont, Y_val),
            epochs=nb_epoch,
            verbose=2,
            callbacks=callbacks_list,
            batch_size=batch_size,
        )
        model.load_weights("mlp.h5")

        # predict
        Y_test_pred = model.predict(X_test_cont)
        Y_cal_pred = model.predict(X_cal_cont)

        (
            independent_nn_conf_levels[i],
            independent_nn_conf_interval[i],
            gumbel_nn_conf_levels[i],
            gumbel_nn_conf_interval[i],
            empirical_nn_conf_levels[i],
            empirical_nn_conf_interval[i]
        ) = multi_target_nn_nonconformity(
            model,
            X_train_cont,
            X_val_cont,
            Y_train,
            Y_val,
            X_cal_cont,
            X_test_cont,
            Y_cal,
            Y_test,
            Y_cal_pred,
            Y_test_pred,
            epsilon_values,
            n_cont,
            n_out,
            nb_epoch,
            layer_size,
            dropout_rate,
            batch_size,
        )
        print("Time multi NN", time.time() - start_multi)

        # normalized conformal prediction for single RF
        print("Normalized conformal prediction for SINGLE RF %s" % i)

        start_multi = time.time()

        rf_models = dict()
        for nb_model in range(n_out):
            rf_models[nb_model] = RandomForestRegressor(max_depth=10, random_state=1337)

        # train models
        fit_per_target(
            rf_models,
            X_train_cont,
            Y_train
        )

        # predict
        Y_cal_pred = predict_per_target(rf_models, X_cal_cont, Y_cal)
        Y_test_pred = predict_per_target(rf_models, X_test_cont, Y_test)

        (
            independent_rf_conf_levels[i],
            independent_rf_conf_interval[i],
            gumbel_rf_conf_levels[i],
            gumbel_rf_conf_interval[i],
            empirical_rf_conf_levels[i],
            empirical_rf_conf_interval[i]
        ) = single_target_rf_nonconformity(
            rf_models,
            X_train_cont,
            X_val_cont,
            Y_train,
            Y_val,
            X_cal_cont,
            X_test_cont,
            Y_cal,
            Y_test,
            Y_cal_pred,
            Y_test_pred,
            epsilon_values,
            n_cont,
            n_out,
            nb_epoch,
            layer_size,
            dropout_rate,
            batch_size,
        )
        print("Time single rf", time.time() - start_multi)

        i = i + 1

    results_path = str(cal_size) + "_results_independent_nn_conf_levels.json"
    independent_nn_conf_levels = aggregate_across_folds(independent_nn_conf_levels)
    with open(base_path / results_path, "w") as f:
        json.dump(independent_nn_conf_levels, f, indent=4)

    results_path = str(cal_size) + "_results_independent_nn_conf_interval.json"
    independent_nn_conf_interval = aggregate_metrics_across_folds(independent_nn_conf_interval)
    with open(base_path / results_path, "w") as f:
        json.dump(independent_nn_conf_interval, f, indent=4)

    results_path = str(cal_size) + "_results_gumbel_nn_conf_levels.json"
    gumbel_nn_conf_levels = aggregate_across_folds(gumbel_nn_conf_levels)
    with open(base_path / results_path, "w") as f:
        json.dump(gumbel_nn_conf_levels, f, indent=4)

    results_path = str(cal_size) + "_results_gumbel_nn_conf_interval.json"
    gumbel_nn_conf_interval = aggregate_metrics_across_folds(gumbel_nn_conf_interval)
    with open(base_path / results_path, "w") as f:
        json.dump(gumbel_nn_conf_interval, f, indent=4)

    results_path = str(cal_size) + "_results_empirical_nn_conf_levels.json"
    empirical_nn_conf_levels = aggregate_across_folds(empirical_nn_conf_levels)
    with open(base_path / results_path, "w") as f:
        json.dump(empirical_nn_conf_levels, f, indent=4)

    results_path = str(cal_size) + "_results_empirical_nn_conf_interval.json"
    empirical_nn_conf_interval = aggregate_metrics_across_folds(empirical_nn_conf_interval)
    with open(base_path / results_path, "w") as f:
        json.dump(empirical_nn_conf_interval, f, indent=4)

    results_path = str(cal_size) + "_results_independent_rf_conf_levels.json"
    independent_rf_conf_levels = aggregate_across_folds(independent_rf_conf_levels)
    with open(base_path / results_path, "w") as f:
        json.dump(independent_rf_conf_levels, f, indent=4)

    results_path = str(cal_size) + "_results_independent_rf_conf_interval.json"
    independent_rf_conf_interval = aggregate_metrics_across_folds(independent_rf_conf_interval)
    with open(base_path / results_path, "w") as f:
        json.dump(independent_rf_conf_interval, f, indent=4)

    results_path = str(cal_size) + "_results_gumbel_rf_conf_levels.json"
    gumbel_rf_conf_levels = aggregate_across_folds(gumbel_rf_conf_levels)
    with open(base_path / results_path, "w") as f:
        json.dump(gumbel_rf_conf_levels, f, indent=4)

    results_path = str(cal_size) + "_results_gumbel_rf_conf_interval.json"
    gumbel_rf_conf_interval = aggregate_metrics_across_folds(gumbel_rf_conf_interval)
    with open(base_path / results_path, "w") as f:
        json.dump(gumbel_rf_conf_interval, f, indent=4)

    results_path = str(cal_size) + "_results_empirical_rf_conf_levels.json"
    empirical_rf_conf_levels = aggregate_across_folds(empirical_rf_conf_levels)
    with open(base_path / results_path, "w") as f:
        json.dump(empirical_rf_conf_levels, f, indent=4)

    results_path = str(cal_size) + "_results_empirical_rf_conf_interval.json"
    empirical_rf_conf_interval = aggregate_metrics_across_folds(empirical_rf_conf_interval)
    with open(base_path / results_path, "w") as f:
        json.dump(empirical_rf_conf_interval, f, indent=4)