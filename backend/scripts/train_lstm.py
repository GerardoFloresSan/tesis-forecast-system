import argparse
import json
import math
import os
import random
import re
import unicodedata
from pathlib import Path

os.environ.pop("TF_USE_LEGACY_KERAS", None)

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sqlalchemy import create_engine
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.losses import Huber
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from app.core.config import settings

DEFAULT_CHANNEL_TO_TRAIN = "Choice"

TIME_STEPS = 32
EPOCHS = 100
BATCH_SIZE = 32
TRAIN_SPLIT = 0.8
VALIDATION_SPLIT_WITHIN_TRAIN = 0.2
RANDOM_SEED = 42
BIAS_CORRECTION_FACTOR = 0.35

TRAINABLE_CHANNELS = {"Choice", "España"}
CHANNEL_OPERATING_WINDOWS = {
    "Choice": ("00:00", "16:30"),
    "España": ("00:00", "16:30"),
}

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "data" / "models"


def _normalize_channel_key(channel: str) -> str:
    normalized = unicodedata.normalize("NFKD", (channel or "").strip())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = normalized.lower()
    normalized = " ".join(normalized.split())
    return normalized


def canonicalize_channel(channel: str) -> str:
    channel_key = _normalize_channel_key(channel)
    channel_aliases = {
        "choice": "Choice",
        "espana": "España",
    }
    canonical = channel_aliases.get(channel_key)
    if canonical is None or canonical not in TRAINABLE_CHANNELS:
        raise ValueError(
            f"Canal '{channel}' no permitido para entrenamiento. "
            f"Canales permitidos: {sorted(TRAINABLE_CHANNELS)}"
        )
    return canonical


def slugify_channel(channel: str) -> str:
    normalized = unicodedata.normalize("NFKD", channel.strip())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = normalized.lower()
    return re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")


def set_seeds(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    non_zero_mask = y_true != 0
    if not np.any(non_zero_mask):
        return 0.0
    return float(
        np.mean(
            np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])
        ) * 100
    )


def calculate_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denominator = np.sum(np.abs(y_true))
    if denominator == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / denominator * 100)


def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denominator = np.abs(y_true) + np.abs(y_pred)
    valid_mask = denominator > 0

    if not np.any(valid_mask):
        return 0.0

    return float(
        np.mean(
            2.0 * np.abs(y_pred[valid_mask] - y_true[valid_mask]) / denominator[valid_mask]
        ) * 100
    )


def calculate_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return float(np.mean(y_pred - y_true))


def create_sequences(features: np.ndarray, target: np.ndarray, time_steps: int):
    X, y = [], []

    for i in range(len(features) - time_steps):
        X.append(features[i : i + time_steps])
        y.append(target[i + time_steps])

    return np.array(X), np.array(y)


def time_str_to_minutes(value: str) -> int:
    parsed = pd.to_datetime(str(value), errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"No se pudo interpretar la hora: {value}")
    return int(parsed.hour * 60 + parsed.minute)


def apply_channel_business_rules(hist_df: pd.DataFrame, channel: str) -> pd.DataFrame:
    if channel not in TRAINABLE_CHANNELS:
        raise ValueError(
            f"El canal '{channel}' está fuera del entrenamiento por reglas de negocio. "
            f"Canales permitidos: {sorted(TRAINABLE_CHANNELS)}"
        )

    if channel not in CHANNEL_OPERATING_WINDOWS:
        return hist_df.copy()

    start_str, end_str = CHANNEL_OPERATING_WINDOWS[channel]
    start_minutes = time_str_to_minutes(start_str)
    end_minutes = time_str_to_minutes(end_str)

    df = hist_df.copy()
    parsed_times = pd.to_datetime(df["interval_time"].astype(str), errors="coerce")

    if parsed_times.isna().any():
        invalid_count = int(parsed_times.isna().sum())
        raise ValueError(
            f"Se encontraron {invalid_count} registros con interval_time inválido."
        )

    df["_interval_minutes"] = parsed_times.dt.hour * 60 + parsed_times.dt.minute

    before_count = len(df)
    df = df[
        (df["_interval_minutes"] >= start_minutes)
        & (df["_interval_minutes"] <= end_minutes)
    ].copy()
    after_count = len(df)

    removed = before_count - after_count
    print(
        f"Regla de negocio aplicada para {channel}: {start_str} a {end_str}. "
        f"Registros removidos fuera de horario: {removed}"
    )

    df.drop(columns=["_interval_minutes"], inplace=True, errors="ignore")
    return df


def load_and_prepare_dataset(engine, channel: str) -> pd.DataFrame:
    historical_query = """
        SELECT
            interaction_date,
            interval_time,
            channel,
            volume,
            aht
        FROM historical_interactions
        ORDER BY interaction_date, interval_time
    """

    external_query = """
        SELECT
            variable_date,
            variable_type,
            variable_value
        FROM external_variables
        ORDER BY variable_date
    """

    hist_df = pd.read_sql(historical_query, engine)
    ext_df = pd.read_sql(external_query, engine)

    if hist_df.empty:
        raise ValueError("No hay datos en historical_interactions.")

    hist_df = hist_df[hist_df["channel"] == channel].copy()

    if hist_df.empty:
        raise ValueError(f"No hay datos para el canal '{channel}'.")

    hist_df["interaction_date"] = pd.to_datetime(hist_df["interaction_date"]).dt.date
    hist_df["volume"] = pd.to_numeric(hist_df["volume"], errors="coerce").fillna(0.0)
    hist_df["aht"] = pd.to_numeric(hist_df["aht"], errors="coerce").fillna(0.0)

    hist_df = apply_channel_business_rules(hist_df, channel)

    if hist_df.empty:
        raise ValueError(
            f"No quedaron registros para el canal '{channel}' después de aplicar el horario operativo."
        )

    hist_df = hist_df.drop_duplicates(
        subset=["interaction_date", "interval_time", "channel", "volume", "aht"]
    )

    duplicate_slots = hist_df.duplicated(
        subset=["interaction_date", "interval_time", "channel"], keep=False
    ).sum()

    if duplicate_slots > 0:
        print(
            f"Advertencia: se detectaron {duplicate_slots} filas en slots duplicados. "
            "Se consolidarán por fecha/hora/canal."
        )
        hist_df = (
            hist_df.groupby(["interaction_date", "interval_time", "channel"], as_index=False)
            .agg(
                volume=("volume", "sum"),
                aht=("aht", "median"),
            )
        )

    holiday_columns = {
        "is_holiday": "is_holiday_peru",
        "is_holiday_peru": "is_holiday_peru",
        "is_holiday_spain": "is_holiday_spain",
        "is_holiday_mexico": "is_holiday_mexico",
        "campaign_day": "campaign_day",
        "absenteeism_rate": "absenteeism_rate",
    }

    external_features = pd.DataFrame(
        {"interaction_date": hist_df["interaction_date"].drop_duplicates().sort_values()}
    )

    if not ext_df.empty:
        ext_df["variable_date"] = pd.to_datetime(ext_df["variable_date"]).dt.date
        ext_df["variable_type"] = ext_df["variable_type"].astype(str).str.strip().str.lower()
        ext_df["variable_value"] = pd.to_numeric(ext_df["variable_value"], errors="coerce").fillna(0.0)
        ext_df["variable_type"] = ext_df["variable_type"].replace(holiday_columns)

        ext_df = ext_df[ext_df["variable_type"].isin(holiday_columns.values())].copy()

        if not ext_df.empty:
            pivot = (
                ext_df.pivot_table(
                    index="variable_date",
                    columns="variable_type",
                    values="variable_value",
                    aggfunc="max",
                    fill_value=0.0,
                )
                .reset_index()
                .rename(columns={"variable_date": "interaction_date"})
            )
            external_features = external_features.merge(
                pivot,
                on="interaction_date",
                how="left",
            )

    for column in holiday_columns.values():
        if column not in external_features.columns:
            external_features[column] = 0.0

    external_features["is_holiday_any"] = external_features[
        ["is_holiday_peru", "is_holiday_spain", "is_holiday_mexico"]
    ].max(axis=1)

    dataset = hist_df.merge(external_features, on="interaction_date", how="left")

    numeric_fill_columns = [
        "aht",
        "is_holiday_peru",
        "is_holiday_spain",
        "is_holiday_mexico",
        "is_holiday_any",
        "campaign_day",
        "absenteeism_rate",
    ]
    for column in numeric_fill_columns:
        dataset[column] = pd.to_numeric(dataset[column], errors="coerce").fillna(0.0)

    dataset["datetime"] = pd.to_datetime(
        dataset["interaction_date"].astype(str) + " " + dataset["interval_time"].astype(str)
    )
    dataset = dataset.sort_values("datetime").reset_index(drop=True)
    dataset["day_of_week"] = dataset["datetime"].dt.dayofweek
    dataset["month"] = dataset["datetime"].dt.month

    return dataset


def build_feature_matrix(dataset: pd.DataFrame):
    feature_columns = [
        "volume",
        "aht",
        "day_of_week",
        "month",
        "is_holiday_peru",
        "is_holiday_spain",
        "is_holiday_mexico",
        "is_holiday_any",
        "campaign_day",
        "absenteeism_rate",
    ]

    target_column = "volume"

    scaler = RobustScaler()
    scaled_features = scaler.fit_transform(dataset[feature_columns])

    X, y = create_sequences(
        features=scaled_features,
        target=dataset[target_column].values,
        time_steps=TIME_STEPS,
    )

    if len(X) == 0:
        raise ValueError(
            "No se pudieron generar secuencias para entrenamiento. "
            "Revisa volumen de datos y TIME_STEPS."
        )

    return X, y, scaler, feature_columns


def build_lstm_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=Huber(),
        metrics=["mae"],
    )
    return model


def save_artifacts(channel: str, model, scaler, metadata: dict, metrics: dict):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    channel_slug = slugify_channel(channel)
    model_path = MODEL_DIR / f"lstm_{channel_slug}.keras"
    scaler_path = MODEL_DIR / f"lstm_{channel_slug}_scaler.joblib"
    metadata_path = MODEL_DIR / f"lstm_{channel_slug}_metadata.joblib"
    metrics_path = MODEL_DIR / f"lstm_{channel_slug}_metrics.json"

    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(metadata, metadata_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return model_path, scaler_path, metadata_path, metrics_path


def main(channel: str):
    set_seeds()
    channel = canonicalize_channel(channel)

    print(f"Iniciando entrenamiento LSTM para canal: {channel}")

    engine = create_engine(settings.DATABASE_URL)
    dataset = load_and_prepare_dataset(engine, channel)

    print(f"Dataset preparado. Registros: {len(dataset)}")

    X, y, scaler, feature_columns = build_feature_matrix(dataset)

    train_size = max(1, int(len(X) * TRAIN_SPLIT))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    if len(X_test) == 0:
        X_train, X_test = X[:-1], X[-1:]
        y_train, y_test = y[:-1], y[-1:]

    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    validation_split = VALIDATION_SPLIT_WITHIN_TRAIN
    if len(X_train) < 10:
        validation_split = 0.0

    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1,
    )

    y_pred = model.predict(X_test, verbose=0).flatten()
    y_pred = np.maximum(y_pred, 0.0)

    bias = calculate_bias(y_test, y_pred)
    y_pred_corrected = np.maximum(y_pred - (bias * BIAS_CORRECTION_FACTOR), 0.0)

    mae = float(mean_absolute_error(y_test, y_pred_corrected))
    rmse = float(math.sqrt(mean_squared_error(y_test, y_pred_corrected)))
    mape = float(calculate_mape(y_test, y_pred_corrected))
    r2 = float(r2_score(y_test, y_pred_corrected)) if len(y_test) > 1 else 0.0
    wape = float(calculate_wape(y_test, y_pred_corrected))
    smape = float(calculate_smape(y_test, y_pred_corrected))

    metadata = {
        "channel": channel,
        "time_steps": TIME_STEPS,
        "feature_columns": feature_columns,
        "bias_correction_factor": BIAS_CORRECTION_FACTOR,
        "train_split": TRAIN_SPLIT,
        "validation_split_within_train": VALIDATION_SPLIT_WITHIN_TRAIN,
    }

    metrics = {
        "channel": channel,
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape": round(mape, 4),
        "r2": round(r2, 4),
        "wape": round(wape, 4),
        "smape": round(smape, 4),
        "bias": round(float(bias), 4),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "history_epochs": int(len(history.history.get("loss", []))),
    }

    model_path, scaler_path, metadata_path, metrics_path = save_artifacts(
        channel=channel,
        model=model,
        scaler=scaler,
        metadata=metadata,
        metrics=metrics,
    )

    metrics["model_path"] = str(model_path)
    metrics["scaler_path"] = str(scaler_path)
    metrics["metadata_path"] = str(metadata_path)
    metrics["metrics_path"] = str(metrics_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", default=DEFAULT_CHANNEL_TO_TRAIN, type=str)
    args = parser.parse_args()
    main(args.channel)