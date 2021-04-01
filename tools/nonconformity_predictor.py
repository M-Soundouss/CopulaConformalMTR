import sys
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

sys.path.append("../")
sys.path.append("../../")

from tools.models import simple_mlp
from tools.conformal_utilities import (
    norm_conf_predict,
    prepare_norm_data,
    prepare_norm_data_per_target,
    check_conf_level,
    independent_norm_conf_all_targets_alpha_s,
    gumbel_norm_conf_all_targets_alpha_s,
    empirical_norm_conf_all_targets_alpha_s,
    aggregates_interval_size
)


def multi_target_nn_nonconformity(
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
        batch_size
):
    # 1 prepare training data
    Y_norm_train, Y_norm_val = prepare_norm_data(
        model, X_train_cont, X_val_cont, Y_train, Y_val
    )

    checkpoint = ModelCheckpoint(
        "multi.h5", monitor="val_loss", verbose=1, save_best_only=True, mode="min"
    )
    early = EarlyStopping(monitor="val_loss", mode="min", patience=20, verbose=1)
    redonplat = ReduceLROnPlateau(
        monitor="val_loss", mode="min", patience=10, verbose=2
    )
    callbacks_list = [checkpoint, early, redonplat]

    # 2 prepare model
    norm_model = simple_mlp(
        n_continuous=n_cont,
        n_outputs=n_out,
        n_layers=3,
        layer_size=layer_size,
        learning_rate=0.00001,
        dropout_rate=dropout_rate
    )

    # 3 train model
    norm_model.fit(
        X_train_cont,
        Y_norm_train,
        validation_data=(X_val_cont, Y_norm_val),
        epochs=nb_epoch,
        verbose=2,
        callbacks=callbacks_list,
        batch_size=batch_size
    )

    try:
        norm_model.load_weights("multi.h5")
    except:
        pass

    # 4 predict cal values (mu)
    mu_cal = norm_model.predict(X_cal_cont)
    mu_test = norm_model.predict(X_test_cont)

    # 5 normalized conformal prediction
    independent_norm_conf_levels = {}
    independent_norm_conf_interval = {}

    for _epsilon in epsilon_values:
        _epsilon = 1 - pow(1 - _epsilon, 1 / n_out)
        alphas = independent_norm_conf_all_targets_alpha_s(
            Y_cal, Y_cal_pred, epsilon=_epsilon, mu=mu_cal, beta=0.1
        )

        norm_conf_test_preds = norm_conf_predict(Y_test_pred, mu_test, alphas, beta=0.1)
        independent_norm_conf_levels[_epsilon] = check_conf_level(norm_conf_test_preds, Y_test)
        independent_norm_conf_interval[_epsilon] = aggregates_interval_size(norm_conf_test_preds)

    # 6 copula normalized conformal prediction
    gumbel_norm_conf_levels = {}
    gumbel_norm_conf_interval = {}

    for _epsilon in epsilon_values:
        alphas = gumbel_norm_conf_all_targets_alpha_s(
            Y_cal, Y_cal_pred, epsilon=_epsilon, mu=mu_cal, beta=0.1
        )

        norm_conf_test_preds = norm_conf_predict(Y_test_pred, mu_test, alphas, beta=0.1)
        gumbel_norm_conf_levels[_epsilon] = check_conf_level(norm_conf_test_preds, Y_test)
        gumbel_norm_conf_interval[_epsilon] = aggregates_interval_size(norm_conf_test_preds)

    # 7 empirical copula normalized conformal prediction
    empirical_norm_conf_levels = {}
    empirical_norm_conf_interval = {}

    for _epsilon in epsilon_values:
        alphas = empirical_norm_conf_all_targets_alpha_s(
            Y_cal, Y_cal_pred, epsilon=_epsilon, mu=mu_cal, beta=0.1
        )

        norm_conf_test_preds = norm_conf_predict(Y_test_pred, mu_test, alphas, beta=0.1)
        empirical_norm_conf_levels[_epsilon] = check_conf_level(norm_conf_test_preds, Y_test)
        empirical_norm_conf_interval[_epsilon] = aggregates_interval_size(norm_conf_test_preds)

    return independent_norm_conf_levels, independent_norm_conf_interval, gumbel_norm_conf_levels, \
           gumbel_norm_conf_interval, empirical_norm_conf_levels, empirical_norm_conf_interval


def single_target_rf_nonconformity(
        models,
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
        batch_size
):
    # 1 prepare training data
    Y_norm_train, Y_norm_val = prepare_norm_data_per_target(
        models, X_train_cont, X_val_cont, Y_train, Y_val
    )

    checkpoint = ModelCheckpoint(
        "multi.h5", monitor="val_loss", verbose=1, save_best_only=True, mode="min"
    )
    early = EarlyStopping(monitor="val_loss", mode="min", patience=20, verbose=1)
    redonplat = ReduceLROnPlateau(
        monitor="val_loss", mode="min", patience=10, verbose=2
    )
    callbacks_list = [checkpoint, early, redonplat]

    # 2 prepare model
    norm_model = simple_mlp(
        n_continuous=n_cont,
        n_outputs=n_out,
        n_layers=3,
        layer_size=layer_size,
        learning_rate=0.00001,
        dropout_rate=dropout_rate
    )

    # 3 train model
    norm_model.fit(
        X_train_cont,
        Y_norm_train,
        validation_data=(X_val_cont, Y_norm_val),
        epochs=nb_epoch,
        verbose=2,
        callbacks=callbacks_list,
        batch_size=batch_size
    )

    try:
        norm_model.load_weights("multi.h5")
    except:
        pass

    # 4 predict cal values (mu)
    mu_cal = norm_model.predict(X_cal_cont)
    mu_test = norm_model.predict(X_test_cont)

    # 5 normalized conformal prediction
    independent_rfs_conf_levels = {}
    independent_rfs_conf_interval = {}

    for _epsilon in epsilon_values:
        _epsilon = 1 - pow(1 - _epsilon, 1 / n_out)
        alphas = independent_norm_conf_all_targets_alpha_s(
            Y_cal, Y_cal_pred, epsilon=_epsilon, mu=mu_cal, beta=0.1
        )

        norm_conf_test_preds = norm_conf_predict(Y_test_pred, mu_test, alphas, beta=0.1)
        independent_rfs_conf_levels[_epsilon] = check_conf_level(norm_conf_test_preds, Y_test)
        independent_rfs_conf_interval[_epsilon] = aggregates_interval_size(norm_conf_test_preds)

    # 6 copula normalized conformal prediction
    gumbel_rfs_conf_levels = {}
    gumbel_rfs_conf_interval = {}

    for _epsilon in epsilon_values:
        alphas = gumbel_norm_conf_all_targets_alpha_s(
            Y_cal, Y_cal_pred, epsilon=_epsilon, mu=mu_cal, beta=0.1
        )

        norm_conf_test_preds = norm_conf_predict(Y_test_pred, mu_test, alphas, beta=0.1)
        gumbel_rfs_conf_levels[_epsilon] = check_conf_level(norm_conf_test_preds, Y_test)
        gumbel_rfs_conf_interval[_epsilon] = aggregates_interval_size(norm_conf_test_preds)

    # 7 empirical copula normalized conformal prediction
    empirical_rfs_conf_levels = {}
    empirical_rfs_conf_interval = {}

    for _epsilon in epsilon_values:
        alphas = empirical_norm_conf_all_targets_alpha_s(
            Y_cal, Y_cal_pred, epsilon=_epsilon, mu=mu_cal, beta=0.1
        )

        norm_conf_test_preds = norm_conf_predict(Y_test_pred, mu_test, alphas, beta=0.1)
        empirical_rfs_conf_levels[_epsilon] = check_conf_level(norm_conf_test_preds, Y_test)
        empirical_rfs_conf_interval[_epsilon] = aggregates_interval_size(norm_conf_test_preds)

    return independent_rfs_conf_levels, independent_rfs_conf_interval, gumbel_rfs_conf_levels, \
           gumbel_rfs_conf_interval, empirical_rfs_conf_levels, empirical_rfs_conf_interval
