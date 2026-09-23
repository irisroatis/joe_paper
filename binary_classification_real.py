import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pickle
from functools import partial

import numpy as np
import pandas as pd
import tensorflow as tf
from category_encoders.glmm import GLMMEncoder
from category_encoders.ordinal import OrdinalEncoder
from openml.datasets import get_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from tensorflow.keras.layers import Dense, Input, Layer
from tensorflow.keras.models import Model


# ----------------------------- Configuration -----------------------------

DATASET_IDS = [41283, 1590, 4135, 41434, 981]
HOW_MANY_ITERATIONS = 20
TRAIN_SIZE = 1000
VALIDATION_SIZE = 1000
TOTAL_EPOCHS = 350
BATCH_SIZE = 32
RANDOM_SEED = 45
RESULTS_PATH = "/home/ir318/new_add_exp/results.pkl"

WHICH_METHODS = [
    "model_nocat",
    "model_tarreg",
    "ORD",
    "GLMM",
    "OneHot",
    "JoeChar",
    "JoeOhe",
]


def calculate_entropy(series):
    """Return entropy normalized by the number of observed categories."""
    value_counts = series.value_counts(normalize=True)
    entropy = -np.sum(value_counts * np.log2(value_counts))
    cardinality = len(value_counts)
    if cardinality <= 1:
        return 0.0
    return entropy / np.log2(cardinality)


def split_inputs(x, y, categorical_variables):
    """Split [category means, category proportions, continuous values]."""
    number_of_categorical_variables = len(categorical_variables)
    inputs = {}
    for index, feature in enumerate(categorical_variables):
        inputs[feature] = [
            x[index],
            x[number_of_categorical_variables + index],
        ]
    inputs["rest"] = x[2 * number_of_categorical_variables :]
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
        self.hidden_layers = [Dense(unit, activation=activation) for unit in units]

    def call(self, inputs):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "activation": self.activation})
        return config


def characteristics_about_dataset(
    X, categorical_indicator, want_plots=False, X_total=None
):
    """Summarize cardinality, normalized entropy, and train coverage."""
    del want_plots  # Retained in the signature for compatibility.
    rows = []
    for index, is_categorical in enumerate(categorical_indicator):
        if not is_categorical:
            continue
        series = X.iloc[:, index]
        row = {
            "Feature": X.columns[index],
            "Cardinality": series.nunique(),
            "Entropy": calculate_entropy(series),
        }
        if X_total is not None:
            total_cardinality = X_total.iloc[:, index].nunique()
            row["Proportion"] = (
                series.nunique() / total_cardinality if total_cardinality else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def make_callbacks():
    """Return fresh callbacks so callback state is never shared across models."""
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=30,
        restore_best_weights=True,
        verbose=0,
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=10,
        min_lr=1e-5,
        verbose=0,
    )
    return [early_stopping, reduce_lr]


def classification_metrics(y_true, y_pred, y_probability):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "ROC AUC Score": roc_auc_score(y_true, y_probability),
        "Brier Score": brier_score_loss(y_true, y_probability),
    }


def sample_train_validation(df_train_all, iteration):
    """Create disjoint train/validation samples without touching the test set."""
    required = TRAIN_SIZE + VALIDATION_SIZE
    if len(df_train_all) < required:
        raise ValueError(
            f"Need at least {required} non-test rows, but only {len(df_train_all)} "
            "are available. Reduce TRAIN_SIZE or VALIDATION_SIZE."
        )
    sampled = df_train_all.sample(n=required, random_state=RANDOM_SEED + iteration)
    df_train = sampled.iloc[:TRAIN_SIZE].copy()
    df_val = sampled.iloc[TRAIN_SIZE:].copy()
    return df_train, df_val


def build_joechar_frames(
    df_train,
    df_val,
    df_test,
    categorical_features,
    continuous_features,
    target_variable,
):
    """Build JoeChar inputs from training-derived category characteristics."""
    overall_mean = float(df_train[target_variable].mean())
    feature_tables = {}
    transformed = {}

    for split_name, split_df in {
        "train": df_train,
        "val": df_val,
        "test": df_test,
    }.items():
        transformed[split_name] = split_df.drop(columns=target_variable).copy()

    mean_columns = []
    proportion_columns = []

    for feature in categorical_features:
        # The category universe and both characteristics come only from training.
        grouped = df_train.groupby(feature, dropna=False)[target_variable].agg(
            ["mean", "size"]
        )
        mean_map = grouped["mean"].to_dict()
        proportion_map = (grouped["size"] / len(df_train)).to_dict()

        feature_tables[feature] = pd.DataFrame(
            {
                "cat": grouped.index,
                "mean": grouped["mean"].values,
                "proportion": (grouped["size"] / len(df_train)).values,
            }
        )

        mean_column = f"{feature}_P"
        proportion_column = f"{feature}_OM"
        mean_columns.append(mean_column)
        proportion_columns.append(proportion_column)

        for frame in transformed.values():
            original = frame[feature]
            frame[mean_column] = original.map(mean_map).fillna(overall_mean)
            frame[proportion_column] = original.map(proportion_map).fillna(0.0)
            frame.drop(columns=feature, inplace=True)

    ordered_columns = mean_columns + proportion_columns + continuous_features
    for split_name in transformed:
        transformed[split_name] = transformed[split_name][ordered_columns]

    return transformed, feature_tables


def make_joechar_datasets(
    frames, y_train, y_val, y_test, categorical_features
):
    datasets = {}
    labels = {"train": y_train, "val": y_val, "test": y_test}
    mapper = partial(split_inputs, categorical_variables=categorical_features)

    for split_name, frame in frames.items():
        dataset = tf.data.Dataset.from_tensor_slices(
            (
                frame.to_numpy(dtype=np.float32),
                labels[split_name].to_numpy(dtype=np.float32),
            )
        ).map(mapper)
        if split_name == "train":
            dataset = dataset.shuffle(
                len(frame), seed=RANDOM_SEED, reshuffle_each_iteration=True
            )
        datasets[split_name] = dataset.batch(BATCH_SIZE)
    return datasets


def build_joe_model(
    categorical_features,
    continuous_features,
    input_widths,
    hidden_layers,
    encoding_activation,
):
    small_models = {}
    inputs = {}
    for feature, width in zip(categorical_features, input_widths):
        small_models[feature] = SmallNetwork(hidden_layers, encoding_activation)
        inputs[feature] = Input(shape=(width,), name=str(feature))

    inputs["rest"] = Input(shape=(len(continuous_features),), name="rest")
    small_outputs = {
        feature: small_models[feature](inputs[feature])
        for feature in categorical_features
    }
    combined = MyLayer()(small_outputs, inputs)
    initializer = tf.keras.initializers.GlorotUniform(seed=RANDOM_SEED)
    output = Dense(1, activation="sigmoid", kernel_initializer=initializer)(combined)
    model = Model(inputs=inputs, outputs=output)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def make_joeohe_frames(
    X_train, X_val, X_test, categorical_features, continuous_features
):
    """One-hot encode from training only and keep each feature in its own block."""
    split_frames = {"train": [], "val": [], "test": []}
    category_widths = []

    for feature in categorical_features:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoder.fit(X_train[[feature]])
        category_widths.append(len(encoder.categories_[0]))
        for split_name, X_split in {
            "train": X_train,
            "val": X_val,
            "test": X_test,
        }.items():
            split_frames[split_name].append(encoder.transform(X_split[[feature]]))

    output = {}
    X_by_split = {"train": X_train, "val": X_val, "test": X_test}
    for split_name, blocks in split_frames.items():
        continuous = X_by_split[split_name][continuous_features].to_numpy(
            dtype=np.float32
        )
        output[split_name] = np.concatenate(blocks + [continuous], axis=1)
    return output, category_widths


def make_joeohe_datasets(
    frames, y_train, y_val, y_test, categorical_features, category_widths
):
    labels = {"train": y_train, "val": y_val, "test": y_test}
    mapper = partial(
        split_inputs_onehot,
        categorical_variables=categorical_features,
        category_widths=category_widths,
    )
    datasets = {}
    for split_name, values in frames.items():
        dataset = tf.data.Dataset.from_tensor_slices(
            (values.astype(np.float32), labels[split_name].to_numpy(np.float32))
        ).map(mapper)
        if split_name == "train":
            dataset = dataset.shuffle(
                len(values), seed=RANDOM_SEED, reshuffle_each_iteration=True
            )
        datasets[split_name] = dataset.batch(BATCH_SIZE)
    return datasets


def run_iteration(
    iteration,
    df_train_all,
    df_test,
    categorical_indicator,
    categorical_features,
    continuous_features,
    target_variable,
    total_epochs,
    df_full,
    which_methods,
):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(RANDOM_SEED + iteration)

    df_train, df_val = sample_train_validation(df_train_all, iteration)
    X_train = df_train.drop(columns=target_variable)
    y_train = df_train[target_variable]
    X_val = df_val.drop(columns=target_variable)
    y_val = df_val[target_variable]
    X_test = df_test.drop(columns=target_variable)
    y_test = df_test[target_variable]

    summary_df = characteristics_about_dataset(
        X_train,
        categorical_indicator,
        want_plots=False,
        X_total=df_full.drop(columns=target_variable),
    )
    predictions = {}

    if "model_nocat" in which_methods:
        model = LogisticRegression(penalty=None, max_iter=1000)
        model.fit(X_train[continuous_features], y_train)
        predictions["model_nocat"] = (
            model.predict(X_test[continuous_features]),
            model.predict_proba(X_test[continuous_features])[:, 1],
        )

    if "model_tarreg" in which_methods:
        encoder = TargetEncoder(target_type="binary", smooth="auto", cv=5)
        encoder.fit(X_train[categorical_features], y_train)
        train_encoded = X_train.copy()
        test_encoded = X_test.copy()
        train_encoded[categorical_features] = encoder.transform(
            X_train[categorical_features]
        )
        test_encoded[categorical_features] = encoder.transform(
            X_test[categorical_features]
        )
        model = LogisticRegression(penalty=None, max_iter=1000)
        model.fit(train_encoded, y_train)
        predictions["model_tarreg"] = (
            model.predict(test_encoded),
            model.predict_proba(test_encoded)[:, 1],
        )

    if "ORD" in which_methods:
        encoder = OrdinalEncoder(cols=categorical_features)
        train_encoded = X_train.copy()
        test_encoded = X_test.copy()
        train_encoded[categorical_features] = encoder.fit_transform(
            X_train[categorical_features], y_train
        )
        test_encoded[categorical_features] = encoder.transform(
            X_test[categorical_features]
        )
        model = LogisticRegression(penalty=None, max_iter=1000)
        model.fit(train_encoded, y_train)
        predictions["ORD"] = (
            model.predict(test_encoded),
            model.predict_proba(test_encoded)[:, 1],
        )

    if "GLMM" in which_methods:
        encoder = GLMMEncoder(cols=categorical_features, drop_invariant=False)
        train_encoded = X_train.copy()
        test_encoded = X_test.copy()
        train_encoded[categorical_features] = encoder.fit_transform(
            X_train[categorical_features], y_train
        )
        test_encoded[categorical_features] = encoder.transform(
            X_test[categorical_features]
        )
        model = LogisticRegression(penalty=None, max_iter=1000)
        model.fit(train_encoded, y_train)
        predictions["GLMM"] = (
            model.predict(test_encoded),
            model.predict_proba(test_encoded)[:, 1],
        )

    if "OneHot" in which_methods:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoder.fit(X_train[categorical_features])
        train_categories = encoder.transform(X_train[categorical_features])
        test_categories = encoder.transform(X_test[categorical_features])
        train_encoded = np.concatenate(
            [X_train[continuous_features].to_numpy(), train_categories], axis=1
        )
        test_encoded = np.concatenate(
            [X_test[continuous_features].to_numpy(), test_categories], axis=1
        )
        model = LogisticRegression(penalty=None, max_iter=1000)
        model.fit(train_encoded, y_train)
        predictions["OneHot"] = (
            model.predict(test_encoded),
            model.predict_proba(test_encoded)[:, 1],
        )

    if "JoeChar" in which_methods:
        frames, _ = build_joechar_frames(
            df_train,
            df_val,
            df_test,
            categorical_features,
            continuous_features,
            target_variable,
        )
        datasets = make_joechar_datasets(
            frames, y_train, y_val, y_test, categorical_features
        )
        configurations = [
            ("JoeChar3Sig", [3, 1], "sigmoid"),
            ("JoeChar1Lin", [1], "linear"),
        ]
        for name, hidden_layers, activation in configurations:
            model = build_joe_model(
                categorical_features,
                continuous_features,
                [2] * len(categorical_features),
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
            probabilities = model.predict(datasets["test"], verbose=0).flatten()
            predictions[name] = ((probabilities > 0.5).astype(int), probabilities)

    if "JoeOhe" in which_methods:
        frames, category_widths = make_joeohe_frames(
            X_train,
            X_val,
            X_test,
            categorical_features,
            continuous_features,
        )
        datasets = make_joeohe_datasets(
            frames,
            y_train,
            y_val,
            y_test,
            categorical_features,
            category_widths,
        )
        configurations = [
            ("JoeOhe3Sig", [3, 1], "sigmoid"),
            ("JoeOhe1Lin", [1], "linear"),
        ]
        for name, hidden_layers, activation in configurations:
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
            probabilities = model.predict(datasets["test"], verbose=0).flatten()
            predictions[name] = ((probabilities > 0.5).astype(int), probabilities)

    results = {
        name: classification_metrics(y_test, labels, probabilities)
        for name, (labels, probabilities) in predictions.items()
    }
    results["summaries"] = summary_df
    results["proportion_ones_train"] = float(y_train.mean())
    results["train_indices"] = df_train.index.tolist()
    results["validation_indices"] = df_val.index.tolist()
    return results


def prepare_dataset(dataset_id):
    dataset = get_dataset(dataset_id)
    X, y, categorical_indicator, _ = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )

    if dataset_id == 1590:
        y = y.replace({">50K": 1, "<=50K": 0})
    y = y.astype(int)

    categorical_indicator = np.asarray(categorical_indicator, dtype=bool)
    categorical_features = list(X.columns[categorical_indicator])
    continuous_features = list(X.columns[~categorical_indicator])
    target_variable = y.name

    df = pd.concat([X, y], axis=1)
    df[continuous_features] = df[continuous_features].apply(
        pd.to_numeric, errors="coerce"
    )
    df[categorical_features] = df[categorical_features].astype(str)
    df = df.dropna(subset=continuous_features).copy()

    # Scale using training statistics later would be ideal. This retains the original
    # experiment's scaling convention so results remain comparable.
    for feature in continuous_features:
        minimum = df[feature].min()
        feature_range = df[feature].max() - minimum
        df[feature] = 0.0 if feature_range == 0 else (df[feature] - minimum) / feature_range

    return (
        df,
        categorical_indicator,
        categorical_features,
        continuous_features,
        target_variable,
    )


def main():
    results = {}

    for dataset_id in DATASET_IDS:
        print(f"Loading OpenML dataset {dataset_id}...")
        (
            df,
            categorical_indicator,
            categorical_features,
            continuous_features,
            target_variable,
        ) = prepare_dataset(dataset_id)

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
            iteration_result = run_iteration(
                iteration,
                df_train_all,
                df_test,
                categorical_indicator,
                categorical_features,
                continuous_features,
                target_variable,
                TOTAL_EPOCHS,
                df,
                methods,
            )
            results[dataset_id].append(iteration_result)

        os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
        with open(RESULTS_PATH, "wb") as results_file:
            pickle.dump(results, results_file)
        print(f"Saved results through dataset {dataset_id} to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
