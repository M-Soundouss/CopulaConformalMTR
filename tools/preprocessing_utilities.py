import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

pd.options.mode.chained_assignment = None


def learn_scalers(train_df, config):
    scaler_continuous = StandardScaler()

    scaler_continuous.fit(train_df[config["continuous_variables"]])

    scaler_targets = StandardScaler()

    scaler_targets.fit(train_df[config["targets"]])

    return scaler_continuous, scaler_targets


def apply_scalers(list_df, scaler_continuous, scaler_targets, config):
    output_df = []
    for df in list_df:
        df[config["continuous_variables"]] = scaler_continuous.transform(
            df[config["continuous_variables"]]
        )
        df[config["targets"]] = scaler_targets.transform(df[config["targets"]])
        output_df.append(df)

    return output_df


def get_continuous_arrays(list_df, config):
    output_df = []
    for df in list_df:
        output_df.append(np.array(df[config["continuous_variables"]]))
    return output_df


def get_targets_arrays(list_df, config):
    output_df = []
    for df in list_df:
        output_df.append(np.array(df[config["targets"]]))
    return output_df


def get_categorical_arrays(list_df, config):
    output_df = []
    for df in list_df:
        output_df.append(
            [np.array(df[x])[..., np.newaxis] for x in config["categorical_variables"]]
        )
    return output_df


class CategoricalFeaturesMapper:
    def __init__(self, min_frequency=1, max_feature_size=10000):
        self.min_frequency = min_frequency
        self.max_feature_size = max_feature_size
        self.mapping_dicts = {}
        self.categorical_feature_names = None
        self.sizes = None

    def fit(self, df, categorical_feature_names):
        self.categorical_feature_names = categorical_feature_names
        self.mapping_dicts = {}
        for feature in self.categorical_feature_names:
            value_counts = df[feature].value_counts()
            tuples_count = [
                x for x in value_counts.items() if x[1] >= self.min_frequency
            ]
            tuples_count.sort(key=lambda x: x[1], reverse=True)
            tuples_count = tuples_count[: self.max_feature_size]
            mapping = {k: i + 1 for i, (k, v) in enumerate(tuples_count)}
            self.mapping_dicts[feature] = mapping

        self.sizes = [
            len(self.mapping_dicts[feature]) + 1
            for feature in self.categorical_feature_names
        ]

    def transform(self, df):
        for feature in self.categorical_feature_names:
            mapping = self.mapping_dicts[feature]
            df[feature] = df[feature].apply(lambda x: mapping.get(x, 0))

        return df


def learn_mapper(train_df, config):
    cat_mapper = CategoricalFeaturesMapper(min_frequency=2)
    cat_mapper.fit(train_df, config["categorical_variables"])

    return cat_mapper


def apply_mapper(list_df, cat_mapper):
    output_df = []
    for df in list_df:
        df = cat_mapper.transform(df)

        output_df.append(df)

    return output_df


if __name__ == "__main__":
    df = pd.DataFrame(
        {
            "A": ["X", "X", "X", "X", "X", "Y", "Y", "Y", "Y", "Y", "Y", "Z"],
            "B": [0] * 12,
            "C": ["G", "G", "G", "G", "G", "R", "R", "D", "R", "R", "R", "C"],
        }
    )

    catmapper = CategoricalFeaturesMapper(min_frequency=2)
    catmapper.fit(df, ["A", "C"])

    print(catmapper.transform(df))
