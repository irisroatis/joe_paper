#!/usr/bin/env python3
"""Real-data regression experiments with leakage-free NN validation."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import pickle
from functools import partial

import numpy as np
import pandas as pd
import tensorflow as tf
from category_encoders.glmm import GLMMEncoder
from category_encoders.ordinal import OrdinalEncoder
from openml.datasets import get_dataset
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from tensorflow.keras.layers import Dense, Input, Layer
from tensorflow.keras.models import Model


# ----------------------------- Configuration -----------------------------

# DATASET_IDS = [41211, 41445, 41210, 41267]
DATASET_IDS = [41211, 41445, 41210]
DATE_COLUMNS_TO_DROP = [
    "lpep_pickup_datetime",
    "lpep_dropoff_datetime",
    "datetime",
    "FL_DATE",
]
HOW_MANY_ITERATIONS = 20
TRAIN_SIZE = 1000
VALIDATION_SIZE = 1000
TOTAL_EPOCHS = 500
BATCH_SIZE = 32
RANDOM_SEED = 45
RESULTS_PATH = "/home/ir318/new_add_exp/results_real.pkl"

WHICH_METHODS = [
    "model_nocat",
    "model_tarreg",
    "GLMM",
    "OneHot",
    "JoeChar",
    "JoeOhe",
    "ORD",
]


def calculate_entropy(series):
    value_counts = series.value_counts(normalize=True)
    cardinality = len(value_counts)
    if cardinality <= 1:
        return 0.0
    entropy = -np.sum(value_counts * np.log2(value_counts))
    return entropy / np.log2(cardinality)


def characteristics_about_dataset(X_train, categorical_features, X_total=None):
    rows = []
    for feature in categorical_features:
        row = {
            "Feature": feature,
            "Cardinality": X_train[feature].nunique(),
            "Entropy": calculate_entropy(X_train[feature]),
        }
        if X_total is not None:
            total_cardinality = X_total[feature].nunique()
            row["Proportion"] = (
                X_train[feature].nunique() / total_cardinality
                if total_cardinality
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def split_inputs(x, y, categorical_variables):
    """Split [means, standard deviations, frequencies, continuous values]."""
    number_of_categories = len(categorical_variables)
    inputs = {}
    for index, feature in enumerate(categorical_variables):
        inputs[feature] = [
            x[index],
            x[number_of_categories + index],
            x[2 * number_of_categories + index],
        ]
    inputs["rest"] = x[3 * number_of_categories :]
    return inputs, y


def split_inputs_onehot(x, y, categorical_variables, category_widths):
    inputs = {}
    start = 0
    for feature, width in zip(categorical_variables, category_widths):
        inputs[feature] = x[start : start + width]
        start += width
    inputs["rest"] = x[start:]
    return inputs, y


class MyLayer(Layer):
    def call(self, small_model_outputs, inputs):
        return tf.concat([*small_model_outputs.values(), inputs["rest"]], axis=-1)


class SmallNetwork(Layer):
    def __init__(self, units, activation="sigmoid", **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.dense_layers = [
            Dense(
                unit,
                activation=activation,
                kernel_initializer=tf.keras.initializers.GlorotUniform(
                    seed=RANDOM_SEED
                ),
            )
            for unit in units
        ]

    def call(self, inputs):
        output = inputs
        for layer in self.dense_layers:
            output = layer(output)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "activation": self.activation})
        return config


def make_callbacks():
    """Create independent callback state for each neural-network fit."""
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=30,
            restore_best_weights=True,
            verbose=0,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=15,
            min_lr=1e-5,
            verbose=0,
        ),
    ]


def regression_metrics(y_true, prediction):
    return {
        "Mean Squared Error": mean_squared_error(y_true, prediction),
        "R-squared": r2_score(y_true, prediction),
    }


# def sample_train_validation(df_train_all, iteration):
#     required = TRAIN_SIZE + VALIDATION_SIZE
#     if len(df_train_all) < required:
#         raise ValueError(
#             f"Need at least {required} non-test rows, but only "
#             f"{len(df_train_all)} are available. Reduce TRAIN_SIZE or "
#             "VALIDATION_SIZE."
#         )
#     sampled = df_train_all.sample(n=required, random_state=RANDOM_SEED + iteration)
#     return sampled.iloc[:TRAIN_SIZE].copy(), sampled.iloc[TRAIN_SIZE:].copy()

def sample_train_validation(df_train_all, iteration, dataset_id):
    """Create the requested dataset-specific train/validation split."""
    if dataset_id in [41211]:
        if len(df_train_all) <= TRAIN_SIZE:
            raise ValueError(
                f"Dataset {dataset_id} needs more than {TRAIN_SIZE} non-test "
                f"rows, but only {len(df_train_all)} are available."
            )
        df_train = df_train_all.sample(
            n=TRAIN_SIZE, random_state=RANDOM_SEED + iteration
        ).copy()
        df_val = df_train_all.drop(index=df_train.index).copy()
        return df_train, df_val

    required = TRAIN_SIZE + VALIDATION_SIZE
    if len(df_train_all) < required:
        raise ValueError(
            f"Dataset {dataset_id} needs at least {required} non-test rows for "
            f"{TRAIN_SIZE} training and {VALIDATION_SIZE} validation rows, but "
            f"only {len(df_train_all)} are available."
        )
    sampled = df_train_all.sample(
        n=required, random_state=RANDOM_SEED + iteration
    )
    return sampled.iloc[:TRAIN_SIZE].copy(), sampled.iloc[TRAIN_SIZE:].copy()

def build_joechar_frames(
    df_train,
    df_val,
    df_test,
    categorical_features,
    continuous_features,
    target_variable,
):
    """Create train-derived [category mean, std, frequency] inputs."""
    overall_mean = float(df_train[target_variable].mean())
    overall_std = float(df_train[target_variable].std(ddof=0))
    transformed = {
        name: frame.drop(columns=target_variable).copy()
        for name, frame in {
            "train": df_train,
            "val": df_val,
            "test": df_test,
        }.items()
    }
    mean_columns, std_columns, frequency_columns = [], [], []

    for feature in categorical_features:
        # All category definitions and statistics are learned from training only.
        grouped = df_train.groupby(feature, dropna=False)[target_variable].agg(
            mean="mean", std=lambda values: values.std(ddof=0), size="size"
        )
        mean_map = grouped["mean"].to_dict()
        std_map = grouped["std"].fillna(0.0).to_dict()
        frequency_map = (grouped["size"] / len(df_train)).to_dict()

        mean_column = f"{feature}_P"
        std_column = f"{feature}_N"
        frequency_column = f"{feature}_FREQ"
        mean_columns.append(mean_column)
        std_columns.append(std_column)
        frequency_columns.append(frequency_column)

        for frame in transformed.values():
            original = frame[feature]
            frame[mean_column] = original.map(mean_map).fillna(overall_mean)
            frame[std_column] = original.map(std_map).fillna(overall_std)
            frame[frequency_column] = original.map(frequency_map).fillna(0.0)
            frame.drop(columns=feature, inplace=True)

    ordered_columns = (
        mean_columns + std_columns + frequency_columns + continuous_features
    )
    for name in transformed:
        transformed[name] = transformed[name][ordered_columns]
    return transformed


def make_joechar_datasets(frames, labels, categorical_features):
    mapper = partial(split_inputs, categorical_variables=categorical_features)
    datasets = {}
    for name, frame in frames.items():
        dataset = tf.data.Dataset.from_tensor_slices(
            (
                frame.to_numpy(dtype=np.float32),
                labels[name].to_numpy(dtype=np.float32),
            )
        ).map(mapper)
        if name == "train":
            dataset = dataset.shuffle(
                len(frame), seed=RANDOM_SEED, reshuffle_each_iteration=True
            )
        datasets[name] = dataset.batch(BATCH_SIZE)
    return datasets


def build_joeohe_frames(
    X_train, X_val, X_test, categorical_features, continuous_features
):
    """Fit every one-hot vocabulary on training only."""
    X_splits = {"train": X_train, "val": X_val, "test": X_test}
    blocks = {name: [] for name in X_splits}
    category_widths = []

    for feature in categorical_features:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoder.fit(X_train[[feature]])
        category_widths.append(len(encoder.categories_[0]))
        for name, frame in X_splits.items():
            blocks[name].append(encoder.transform(frame[[feature]]))

    output = {}
    for name, frame_blocks in blocks.items():
        continuous = X_splits[name][continuous_features].to_numpy(np.float32)
        output[name] = np.concatenate(frame_blocks + [continuous], axis=1)
    return output, category_widths


def make_joeohe_datasets(frames, labels, categorical_features, category_widths):
    mapper = partial(
        split_inputs_onehot,
        categorical_variables=categorical_features,
        category_widths=category_widths,
    )
    datasets = {}
    for name, values in frames.items():
        dataset = tf.data.Dataset.from_tensor_slices(
            (values.astype(np.float32), labels[name].to_numpy(dtype=np.float32))
        ).map(mapper)
        if name == "train":
            dataset = dataset.shuffle(
                len(values), seed=RANDOM_SEED, reshuffle_each_iteration=True
            )
        datasets[name] = dataset.batch(BATCH_SIZE)
    return datasets


def build_joe_model(
    categorical_features,
    continuous_features,
    input_widths,
    hidden_layers,
    encoding_activation,
):
    inputs, small_models = {}, {}
    for feature, width in zip(categorical_features, input_widths):
        inputs[feature] = Input(shape=(width,), name=str(feature))
        small_models[feature] = SmallNetwork(hidden_layers, encoding_activation)

    inputs["rest"] = Input(shape=(len(continuous_features),), name="rest")
    small_outputs = {
        feature: small_models[feature](inputs[feature])
        for feature in categorical_features
    }
    combined = MyLayer()(small_outputs, inputs)
    output = Dense(
        1,
        activation="linear",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=RANDOM_SEED),
    )(combined)
    model = Model(inputs=inputs, outputs=output)
    model.compile(
        loss="mse",
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
    )
    return model


def run_iteration(
    iteration,
    dataset_id,
    df_train_all,
    df_test,
    categorical_features,
    continuous_features,
    target_variable,
    total_epochs,
    df_full,
    which_methods,
):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    df_train, df_val = sample_train_validation(df_train_all, iteration, dataset_id)
    X_train, y_train = (
        df_train.drop(columns=target_variable),
        df_train[target_variable],
    )
    X_val, y_val = df_val.drop(columns=target_variable), df_val[target_variable]
    X_test, y_test = (
        df_test.drop(columns=target_variable),
        df_test[target_variable],
    )
    labels = {"train": y_train, "val": y_val, "test": y_test}

    summary_df = characteristics_about_dataset(
        X_train,
        categorical_features,
        X_total=df_full.drop(columns=target_variable),
    )
    predictions = {}

    if "model_nocat" in which_methods:
        model = LinearRegression().fit(X_train[continuous_features], y_train)
        predictions["model_nocat"] = model.predict(X_test[continuous_features])

    if "model_tarreg" in which_methods:
        encoder = TargetEncoder(smooth="auto", cv=5, target_type="continuous")
        encoder.fit(X_train[categorical_features], y_train)
        train_encoded, test_encoded = X_train.copy(), X_test.copy()
        train_encoded[categorical_features] = encoder.transform(
            X_train[categorical_features]
        )
        test_encoded[categorical_features] = encoder.transform(
            X_test[categorical_features]
        )
        model = LinearRegression().fit(train_encoded, y_train)
        predictions["model_tarreg"] = model.predict(test_encoded)

    if "GLMM" in which_methods:
        encoder = GLMMEncoder(
            cols=categorical_features,
            drop_invariant=False,
            verbose=0,
            binomial_target=False,
        )
        train_encoded, test_encoded = X_train.copy(), X_test.copy()
        train_encoded[categorical_features] = encoder.fit_transform(
            X_train[categorical_features], y_train
        )
        test_encoded[categorical_features] = encoder.transform(
            X_test[categorical_features]
        )
        model = LinearRegression().fit(train_encoded, y_train)
        predictions["GLMM"] = model.predict(test_encoded)

    if "ORD" in which_methods:
        encoder = OrdinalEncoder(cols=categorical_features)
        train_encoded, test_encoded = X_train.copy(), X_test.copy()
        train_encoded[categorical_features] = encoder.fit_transform(
            X_train[categorical_features], y_train
        )
        test_encoded[categorical_features] = encoder.transform(
            X_test[categorical_features]
        )
        model = LinearRegression().fit(train_encoded, y_train)
        predictions["ORD"] = model.predict(test_encoded)

    if "OneHot" in which_methods:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoder.fit(X_train[categorical_features])
        train_encoded = np.concatenate(
            [
                X_train[continuous_features].to_numpy(),
                encoder.transform(X_train[categorical_features]),
            ],
            axis=1,
        )
        test_encoded = np.concatenate(
            [
                X_test[continuous_features].to_numpy(),
                encoder.transform(X_test[categorical_features]),
            ],
            axis=1,
        )
        model = LinearRegression().fit(train_encoded, y_train)
        predictions["OneHot"] = model.predict(test_encoded)

    configurations = [
        ("3Sig", [3, 1], "sigmoid"),
        ("1Lin", [1], "linear"),
    ]

    if "JoeChar" in which_methods:
        frames = build_joechar_frames(
            df_train,
            df_val,
            df_test,
            categorical_features,
            continuous_features,
            target_variable,
        )
        datasets = make_joechar_datasets(frames, labels, categorical_features)
        for suffix, hidden_layers, activation in configurations:
            name = f"JoeChar{suffix}"
            model = build_joe_model(
                categorical_features,
                continuous_features,
                [3] * len(categorical_features),
                hidden_layers,
                activation,
            )
            model.fit(
                datasets["train"],
                validation_data=datasets["val"],
                epochs=total_epochs,
                callbacks=make_callbacks(),
                verbose=0,
            )
            predictions[name] = model.predict(
                datasets["test"], verbose=0
            ).flatten()

    if "JoeOhe" in which_methods:
        frames, category_widths = build_joeohe_frames(
            X_train,
            X_val,
            X_test,
            categorical_features,
            continuous_features,
        )
        datasets = make_joeohe_datasets(
            frames, labels, categorical_features, category_widths
        )
        for suffix, hidden_layers, activation in configurations:
            name = f"JoeOhe{suffix}"
            model = build_joe_model(
                categorical_features,
                continuous_features,
                category_widths,
                hidden_layers,
                activation,
            )
            model.fit(
                datasets["train"],
                validation_data=datasets["val"],
                epochs=total_epochs,
                callbacks=make_callbacks(),
                verbose=0,
            )
            predictions[name] = model.predict(
                datasets["test"], verbose=0
            ).flatten()

    results = {
        name: regression_metrics(y_test, prediction)
        for name, prediction in predictions.items()
    }
    results["summaries"] = summary_df
    results["train_indices"] = df_train.index.tolist()
    results["validation_indices"] = df_val.index.tolist()
    return results


def scale_zero_one(series):
    minimum, maximum = series.min(), series.max()
    difference = maximum - minimum
    return pd.Series(0.0, index=series.index) if difference == 0 else (
        series - minimum
    ) / difference


def prepare_dataset(dataset_id):
    dataset = get_dataset(dataset_id)
    X, y, _, _ = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )

    X = X.drop(
        columns=[column for column in DATE_COLUMNS_TO_DROP if column in X.columns]
    )
    categorical_features = list(
        X.select_dtypes(include=["category", "object", "string"]).columns
    )
    continuous_features = [
        column for column in X.columns if column not in categorical_features
    ]
    target_variable = y.name

    df = pd.concat([X, y], axis=1)
    df[continuous_features] = df[continuous_features].apply(
        pd.to_numeric, errors="coerce"
    )
    df[target_variable] = pd.to_numeric(df[target_variable], errors="coerce")
    df = df.dropna(subset=continuous_features + [target_variable]).copy()
    df[categorical_features] = df[categorical_features].astype(str)

    # Retains the original experiment's global scaling convention for comparability.
    df[target_variable] = scale_zero_one(df[target_variable])
    for feature in continuous_features:
        df[feature] = scale_zero_one(df[feature])

    return df, categorical_features, continuous_features, target_variable


def main():
    print("Real_regression")
    results = {}

    for dataset_id in DATASET_IDS:
        print(f"Loading OpenML dataset {dataset_id}...")
        df, categorical_features, continuous_features, target_variable = (
            prepare_dataset(dataset_id)
        )

        df_test = df.sample(frac=0.5, random_state=RANDOM_SEED).copy()
        df_train_all = df.drop(index=df_test.index).copy()
        methods = WHICH_METHODS.copy()
        if not continuous_features and "model_nocat" in methods:
            methods.remove("model_nocat")

        results[dataset_id] = []
        for iteration in range(HOW_MANY_ITERATIONS):
            print(
                f"Dataset {dataset_id}: iteration "
                f"{iteration + 1}/{HOW_MANY_ITERATIONS}"
            )
            results[dataset_id].append(
                run_iteration(
                    iteration,
                    dataset_id,
                    df_train_all,
                    df_test,
                    categorical_features,
                    continuous_features,
                    target_variable,
                    TOTAL_EPOCHS,
                    df,
                    methods,
                )
            )

        os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
        with open(RESULTS_PATH, "wb") as results_file:
            pickle.dump(results, results_file)
        print(f"Saved results through dataset {dataset_id} to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
