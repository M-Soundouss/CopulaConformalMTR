from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    Add,
    Embedding,
    Flatten,
    Concatenate,
    BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mae, mse
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import relu, linear
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1


def simple_mlp(
        n_continuous,
        n_outputs,
        dropout_rate=0.1,
        learning_rate=0.0001,
        n_layers=3,
        layer_size=64,
):
    in_layer = Input(shape=(n_continuous,))

    x = Dense(layer_size, activation="selu")(in_layer)

    for _ in range(n_layers):
        a = Dropout(dropout_rate)(x)
        a = Dense(layer_size, activation="selu")(a)
        x = Add()([a, x])

    out = Dense(n_outputs, activation=linear)(x)

    model = Model(inputs=[in_layer], outputs=[out])

    model.compile(loss=mae, optimizer=Adam(learning_rate=learning_rate), kernel_regularizer=l1(l=0.0001))

    model.summary()

    return model


def two_model_mlp(
        n_continuous,
        n_outputs,
        dropout_rate=0.1,
        learning_rate=0.0001,
        n_layers=3,
        layer_size=64,
        embed_size=32,
):
    in_layer = Input(shape=(n_continuous,))

    x = Dense(layer_size, activation="selu")(in_layer)

    for i in range(n_layers):
        a = Dropout(dropout_rate)(x)
        a = Dense(layer_size, activation="selu")(a)
        x = Add()([a, x])

    internal_repr = Dense(embed_size, activation="selu")(x)

    out = Dense(n_outputs, activation=linear, kernel_regularizer=l1(l=0.0001))(internal_repr)

    model = Model(inputs=[in_layer], outputs=[out])

    model.compile(loss=mae, optimizer=Adam(learning_rate=learning_rate))

    model.summary()

    model_repr = Model(inputs=[in_layer], outputs=[internal_repr])
    model_repr.compile(loss=mae, optimizer=Adam(learning_rate=learning_rate))

    model_repr.summary()

    return model, model_repr


def embedding_mlp(
        n_continuous,
        categorical_sizes,
        n_outputs,
        dropout_rate=0.1,
        learning_rate=0.0001,
        n_layers=3,
        layer_size=64,
):
    in_cont_layer = Input(shape=(n_continuous,))

    in_cats = [Input(shape=(1,)) for _ in range(len(categorical_sizes))]
    embeds = [Embedding(in_size, out_size) for in_size, out_size in categorical_sizes]

    cats = [L(x) for L, x in zip(embeds, in_cats)]
    cats = Concatenate(axis=-1)(cats)
    cats = Flatten()(cats)
    concat = Concatenate(axis=-1)([in_cont_layer, cats])

    x = Dense(layer_size, activation=relu)(concat)

    for _ in range(n_layers):
        a = Dropout(dropout_rate)(x)
        a = Dense(layer_size, activation=relu)(a)
        x = Add()([a, x])

    out = Dense(n_outputs, activation=linear)(x)

    model = Model(inputs=[in_cont_layer] + in_cats, outputs=[out])

    model.compile(loss=mae, optimizer=Adam(learning_rate=learning_rate))

    model.summary()

    return model


def fit_per_target(models, X_train_cont, Y_train):
    for i in range(0, Y_train.shape[1]):
        models[i].fit(X_train_cont, Y_train[:, i]
        )


def predict_per_target(models, X_data, Y_data):
    Y_pred = np.zeros(Y_data.shape)
    for i in range(0, Y_data.shape[1]):
        model = models.get(i)
        Y_pred[:, i] = np.array(model.predict(X_data)).ravel()
    return Y_pred
