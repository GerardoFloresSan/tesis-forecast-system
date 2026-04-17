from __future__ import annotations

import argparse
import calendar
import json
import math
import os
import random
from pathlib import Path

os.environ.pop("TF_USE_LEGACY_KERAS", None)

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler, StandardScaler
from sqlalchemy import create_engine
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.losses import Huber
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from app.core.config import settings
from app.utils.channel_rules import (
    apply_business_hours_filter,
    canonicalize_channel,
    get_shift_label,
    get_slots_per_day,
    slugify_channel,
)

DEFAULT_CHANNEL_TO_TRAIN = "Choice"
EPOCHS = 60
BATCH_SIZE = 32
TRAIN_SPLIT = 0.8
VALIDATION_SPLIT_WITHIN_TRAIN = 0.2
RANDOM_SEED = 42
BIAS_CORRECTION_FACTOR = 0.20
NEGATIVE_BIAS_CORRECTION_FACTOR = 0.05
SLOT_BIAS_ADJUSTMENT_SHRINK = 0.35
SLOT_BIAS_ADJUSTMENT_LATE_MULTIPLIER = 1.15
MIN_SLOT_ADJUSTMENT_ABS = 1.0
MAX_SLOT_ADJUSTMENT_ABS = 6.0
MIN_SLOT_ADJUSTMENT_COUNT = 3
LATE_SLOT_WINDOW = 6
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "data" / "models"


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
        )
        * 100
    )


def calculate_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = float(np.sum(np.abs(y_true)))
    if denominator == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / denominator * 100)


def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.abs(y_true) + np.abs(y_pred)
    valid_mask = denominator > 0
    if not np.any(valid_mask):
        return 0.0
    return float(
        np.mean(2.0 * np.abs(y_pred[valid_mask] - y_true[valid_mask]) / denominator[valid_mask])
        * 100
    )


def calculate_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.array(y_pred, dtype=float) - np.array(y_true, dtype=float)))


def compute_effective_bias_adjustment(validation_bias: float) -> float:
    validation_bias = float(validation_bias or 0.0)
    factor = BIAS_CORRECTION_FACTOR if validation_bias >= 0 else NEGATIVE_BIAS_CORRECTION_FACTOR
    return float(validation_bias * factor)


def build_slot_bias_adjustments(
    slot_indices: np.ndarray,
    y_true: np.ndarray,
    y_pred_after_global_bias: np.ndarray,
    slots_per_day: int,
) -> dict[int, float]:
    if len(slot_indices) == 0 or len(y_true) == 0 or len(y_pred_after_global_bias) == 0:
        return {}

    aligned_n = min(len(slot_indices), len(y_true), len(y_pred_after_global_bias))
    residual_df = pd.DataFrame(
        {
            "slot_index": np.asarray(slot_indices[:aligned_n], dtype=int),
            "residual": np.asarray(y_true[:aligned_n], dtype=float)
            - np.asarray(y_pred_after_global_bias[:aligned_n], dtype=float),
        }
    )

    grouped = (
        residual_df.groupby("slot_index", as_index=False)
        .agg(mean_residual=("residual", "mean"), n=("residual", "size"))
        .sort_values("slot_index")
    )

    late_slot_start = max(0, int(slots_per_day) - LATE_SLOT_WINDOW)
    adjustments: dict[int, float] = {}
    for _, row in grouped.iterrows():
        slot_index = int(row["slot_index"])
        count = int(row["n"])
        mean_residual = float(row["mean_residual"])

        if count < MIN_SLOT_ADJUSTMENT_COUNT:
            continue
        if abs(mean_residual) < MIN_SLOT_ADJUSTMENT_ABS:
            continue

        slot_adjustment = mean_residual * SLOT_BIAS_ADJUSTMENT_SHRINK
        if slot_index >= late_slot_start:
            slot_adjustment *= SLOT_BIAS_ADJUSTMENT_LATE_MULTIPLIER

        slot_adjustment = float(
            np.clip(slot_adjustment, -MAX_SLOT_ADJUSTMENT_ABS, MAX_SLOT_ADJUSTMENT_ABS)
        )
        adjustments[slot_index] = slot_adjustment

    return adjustments


def apply_prediction_postprocess(
    y_pred_real: np.ndarray,
    validation_bias: float = 0.0,
    slot_indices: np.ndarray | None = None,
    slot_bias_adjustments: dict[int, float] | None = None,
) -> np.ndarray:
    adjusted = np.asarray(y_pred_real, dtype=float).copy()
    adjusted = adjusted - compute_effective_bias_adjustment(validation_bias)

    if slot_indices is not None and slot_bias_adjustments:
        slot_indices = np.asarray(slot_indices, dtype=int)
        aligned_n = min(len(adjusted), len(slot_indices))
        for index in range(aligned_n):
            adjusted[index] += float(slot_bias_adjustments.get(int(slot_indices[index]), 0.0))

    return np.maximum(adjusted, 0.0)


def create_sequences(features: np.ndarray, target: np.ndarray, time_steps: int):
    X, y = [], []
    for index in range(len(features) - time_steps):
        X.append(features[index : index + time_steps])
        y.append(target[index + time_steps])
    return np.array(X), np.array(y)


def inverse_target_transform(y_scaled: np.ndarray, y_scaler: StandardScaler) -> np.ndarray:
    y_scaled = np.array(y_scaled).reshape(-1, 1)
    y_log = y_scaler.inverse_transform(y_scaled).flatten()
    y_real = np.expm1(y_log)
    return np.maximum(y_real, 0.0)


def _normalize_external_variables(ext_df: pd.DataFrame) -> pd.DataFrame:
    if ext_df.empty:
        return ext_df

    ext_df = ext_df.copy()
    ext_df["variable_date"] = pd.to_datetime(ext_df["variable_date"]).dt.date
    ext_df["variable_type"] = ext_df["variable_type"].astype(str).str.strip().str.lower()
    ext_df["variable_value"] = pd.to_numeric(ext_df["variable_value"], errors="coerce").fillna(0.0)
    ext_df["variable_type"] = ext_df["variable_type"].replace({"is_holiday": "is_holiday_peru"})
    return ext_df


def load_and_prepare_dataset(engine, channel: str) -> pd.DataFrame:
    historical_query = """
        SELECT
            interaction_date,
            interval_time,
            channel,
            volume,
            aht
        FROM historical_interactions
        WHERE channel = %(channel)s
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

    hist_df = pd.read_sql(historical_query, engine, params={"channel": channel})
    ext_df = pd.read_sql(external_query, engine)

    if hist_df.empty:
        raise ValueError(f"No hay datos para el canal '{channel}'.")

    hist_df["interaction_date"] = pd.to_datetime(hist_df["interaction_date"]).dt.date
    hist_df["volume"] = pd.to_numeric(hist_df["volume"], errors="coerce").fillna(0.0)
    hist_df["aht"] = pd.to_numeric(hist_df["aht"], errors="coerce").fillna(0.0)
    hist_df["datetime"] = pd.to_datetime(
        hist_df["interaction_date"].astype(str) + " " + hist_df["interval_time"].astype(str)
    )

    hist_df = apply_business_hours_filter(hist_df, channel)
    hist_df = hist_df.drop_duplicates(
        subset=["interaction_date", "interval_time", "channel", "volume", "aht"]
    )

    duplicate_slots = hist_df.duplicated(
        subset=["interaction_date", "interval_time", "channel"], keep=False
    ).sum()
    if duplicate_slots > 0:
        hist_df = (
            hist_df.groupby(["interaction_date", "interval_time", "channel", "datetime"], as_index=False)
            .agg(volume=("volume", "sum"), aht=("aht", "median"))
            .sort_values("datetime")
            .reset_index(drop=True)
        )

    ext_df = _normalize_external_variables(ext_df)
    external_features = pd.DataFrame(
        {"interaction_date": hist_df["interaction_date"].drop_duplicates().sort_values()}
    )

    if not ext_df.empty:
        ext_df = ext_df[
            ext_df["variable_type"].isin(
                [
                    "is_holiday_peru",
                    "is_holiday_spain",
                    "is_holiday_mexico",
                    "campaign_day",
                    "absenteeism_rate",
                ]
            )
        ].copy()

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
            external_features = external_features.merge(pivot, on="interaction_date", how="left")

    for column in [
        "is_holiday_peru",
        "is_holiday_spain",
        "is_holiday_mexico",
        "campaign_day",
        "absenteeism_rate",
    ]:
        if column not in external_features.columns:
            external_features[column] = 0.0

    external_features["is_holiday_any"] = external_features[
        ["is_holiday_peru", "is_holiday_spain", "is_holiday_mexico"]
    ].max(axis=1)

    df = hist_df.merge(external_features, on="interaction_date", how="left")
    df["datetime"] = pd.to_datetime(
        df["interaction_date"].astype(str) + " " + df["interval_time"].astype(str)
    )
    df = df.sort_values("datetime").reset_index(drop=True)

    for column in [
        "volume",
        "aht",
        "is_holiday_peru",
        "is_holiday_spain",
        "is_holiday_mexico",
        "is_holiday_any",
        "campaign_day",
        "absenteeism_rate",
    ]:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    return df


def _add_calendar_features(enriched: pd.DataFrame) -> pd.DataFrame:
    unique_days = (
        enriched[["interaction_date", "is_holiday_any"]]
        .drop_duplicates(subset=["interaction_date"])
        .sort_values("interaction_date")
        .reset_index(drop=True)
    )

    unique_days["interaction_date"] = pd.to_datetime(unique_days["interaction_date"])
    unique_days["day_of_month"] = unique_days["interaction_date"].dt.day
    unique_days["week_of_month"] = ((unique_days["day_of_month"] - 1) // 7 + 1).astype(int)
    unique_days["days_in_month"] = unique_days["interaction_date"].dt.days_in_month
    unique_days["is_month_start"] = (unique_days["day_of_month"] <= 3).astype(int)
    unique_days["is_month_end"] = (
        unique_days["day_of_month"] >= (unique_days["days_in_month"] - 2)
    ).astype(int)
    unique_days["is_day_after_holiday"] = (unique_days["is_holiday_any"].shift(1).fillna(0.0) > 0).astype(int)

    merge_columns = [
        "interaction_date",
        "week_of_month",
        "is_month_start",
        "is_month_end",
        "is_day_after_holiday",
    ]

    enriched = enriched.merge(
        unique_days[merge_columns],
        on="interaction_date",
        how="left",
    )

    for column in merge_columns[1:]:
        enriched[column] = pd.to_numeric(enriched[column], errors="coerce").fillna(0).astype(int)

    return enriched


def _add_same_slot_history_features(enriched: pd.DataFrame) -> pd.DataFrame:
    same_slot_volume = enriched.groupby("slot_index")["volume"]
    enriched["same_slot_mean_3_day"] = (
        same_slot_volume.shift(1).rolling(window=3, min_periods=3).mean().reset_index(level=0, drop=True)
    )
    enriched["same_slot_mean_7_day"] = (
        same_slot_volume.shift(1).rolling(window=7, min_periods=7).mean().reset_index(level=0, drop=True)
    )
    return enriched


def add_time_and_lag_features(df: pd.DataFrame, channel: str) -> tuple[pd.DataFrame, int, int]:
    enriched = df.copy()
    slots_per_day = get_slots_per_day(channel)
    time_steps = slots_per_day
    lag_one_day = slots_per_day
    lag_two_day = slots_per_day * 2
    lag_three_day = slots_per_day * 3
    lag_one_week = slots_per_day * 7

    enriched["interaction_date"] = pd.to_datetime(enriched["interaction_date"])
    enriched["day_of_week"] = enriched["datetime"].dt.weekday
    enriched["month"] = enriched["datetime"].dt.month
    enriched["day"] = enriched["datetime"].dt.day
    enriched["hour"] = enriched["datetime"].dt.hour
    enriched["minute"] = enriched["datetime"].dt.minute
    enriched["is_weekend"] = (enriched["day_of_week"] >= 5).astype(int)
    enriched["minute_of_day"] = enriched["hour"] * 60 + enriched["minute"]
    enriched["slot_index"] = enriched.groupby(enriched["datetime"].dt.date).cumcount()
    enriched["slot_sin"] = np.sin(2 * np.pi * enriched["slot_index"] / slots_per_day)
    enriched["slot_cos"] = np.cos(2 * np.pi * enriched["slot_index"] / slots_per_day)
    enriched["is_opening_slot"] = (enriched["slot_index"] == 0).astype(int)
    enriched["is_closing_slot"] = (enriched["slot_index"] == (slots_per_day - 1)).astype(int)
    enriched["is_first_hour"] = (enriched["slot_index"] <= 1).astype(int)
    enriched["is_last_hour"] = (enriched["slot_index"] >= (slots_per_day - 2)).astype(int)

    enriched["shift_label"] = enriched["minute_of_day"].apply(lambda value: get_shift_label(channel, int(value)))
    enriched["is_morning_shift"] = (enriched["shift_label"] == "morning").astype(int)
    enriched["is_afternoon_shift"] = (enriched["shift_label"] == "afternoon").astype(int)

    enriched["dow_sin"] = np.sin(2 * np.pi * enriched["day_of_week"] / 7)
    enriched["dow_cos"] = np.cos(2 * np.pi * enriched["day_of_week"] / 7)
    enriched["month_sin"] = np.sin(2 * np.pi * enriched["month"] / 12)
    enriched["month_cos"] = np.cos(2 * np.pi * enriched["month"] / 12)
    enriched["minute_sin"] = np.sin(2 * np.pi * enriched["minute_of_day"] / 1440)
    enriched["minute_cos"] = np.cos(2 * np.pi * enriched["minute_of_day"] / 1440)

    enriched = _add_calendar_features(enriched)

    enriched["lag_volume_1"] = enriched["volume"].shift(1)
    enriched["lag_volume_2"] = enriched["volume"].shift(2)
    enriched["lag_volume_3"] = enriched["volume"].shift(3)
    enriched["lag_volume_1_day"] = enriched["volume"].shift(lag_one_day)
    enriched["lag_volume_2_day"] = enriched["volume"].shift(lag_two_day)
    enriched["lag_volume_3_day"] = enriched["volume"].shift(lag_three_day)
    enriched["lag_volume_1_week"] = enriched["volume"].shift(lag_one_week)

    enriched["rolling_mean_3"] = enriched["volume"].shift(1).rolling(window=3).mean()
    enriched["rolling_mean_6"] = enriched["volume"].shift(1).rolling(window=6).mean()
    enriched["rolling_mean_1_day"] = enriched["volume"].shift(1).rolling(window=lag_one_day).mean()
    enriched["rolling_mean_2_day"] = enriched["volume"].shift(1).rolling(window=lag_two_day).mean()
    enriched["rolling_mean_3_day"] = enriched["volume"].shift(1).rolling(window=lag_three_day).mean()

    enriched = _add_same_slot_history_features(enriched)

    enriched["lag_aht_1"] = enriched["aht"].shift(1)
    enriched["volume_diff_1"] = enriched["volume"].diff(1)
    enriched["volume_diff_1_day"] = enriched["volume"].diff(lag_one_day)

    enriched = enriched.dropna().reset_index(drop=True)
    if enriched.empty:
        raise ValueError("Después de generar lags y rolling, el dataset quedó vacío.")

    enriched["interaction_date"] = enriched["interaction_date"].dt.date
    return enriched, slots_per_day, time_steps


def build_feature_columns() -> list[str]:
    return [
        "volume",
        "aht",
        "is_holiday_peru",
        "is_holiday_spain",
        "is_holiday_mexico",
        "is_holiday_any",
        "campaign_day",
        "absenteeism_rate",
        "is_weekend",
        "is_morning_shift",
        "is_afternoon_shift",
        "is_opening_slot",
        "is_closing_slot",
        "is_first_hour",
        "is_last_hour",
        "week_of_month",
        "is_month_start",
        "is_month_end",
        "is_day_after_holiday",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        "minute_sin",
        "minute_cos",
        "slot_sin",
        "slot_cos",
        "lag_volume_1",
        "lag_volume_2",
        "lag_volume_3",
        "lag_volume_1_day",
        "lag_volume_2_day",
        "lag_volume_3_day",
        "lag_volume_1_week",
        "rolling_mean_3",
        "rolling_mean_6",
        "rolling_mean_1_day",
        "rolling_mean_2_day",
        "rolling_mean_3_day",
        "same_slot_mean_3_day",
        "same_slot_mean_7_day",
        "lag_aht_1",
        "volume_diff_1",
        "volume_diff_1_day",
    ]


def build_lstm_model(input_shape):
    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(16, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=0.001, clipnorm=1.0), loss=Huber())
    return model


def save_artifacts(channel: str, model, x_scaler, y_scaler, metadata: dict, metrics: dict):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    channel_slug = slugify_channel(channel)
    model_path = MODEL_DIR / f"lstm_{channel_slug}.keras"
    scaler_path = MODEL_DIR / f"lstm_{channel_slug}_scaler.joblib"
    metadata_path = MODEL_DIR / f"lstm_{channel_slug}_metadata.joblib"
    metrics_path = MODEL_DIR / f"lstm_{channel_slug}_metrics.json"

    model.save(model_path)
    joblib.dump({"x_scaler": x_scaler, "y_scaler": y_scaler}, scaler_path)
    joblib.dump(metadata, metadata_path)

    metrics["model_path"] = str(model_path)
    metrics["scaler_path"] = str(scaler_path)
    metrics["metadata_path"] = str(metadata_path)
    metrics["metrics_path"] = str(metrics_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return model_path, scaler_path, metadata_path, metrics_path


def main(channel: str):
    set_seeds()
    canonical_channel = canonicalize_channel(channel)

    print(f"Iniciando entrenamiento LSTM para canal: {canonical_channel}")
    engine = create_engine(settings.DATABASE_URL)

    base_df = load_and_prepare_dataset(engine, canonical_channel)
    df, slots_per_day, time_steps = add_time_and_lag_features(base_df, canonical_channel)
    feature_columns = build_feature_columns()

    dataset = df[feature_columns].copy()
    target = np.log1p(df[["volume"]].copy())
    calibration_meta = df[["slot_index"]].copy()

    split_index = int(len(dataset) * TRAIN_SPLIT)
    train_features = dataset.iloc[:split_index].copy()
    test_features = dataset.iloc[split_index:].copy()
    train_target = target.iloc[:split_index].copy()
    test_target = target.iloc[split_index:].copy()
    train_meta_base = calibration_meta.iloc[:split_index].copy()
    test_meta_base = calibration_meta.iloc[split_index:].copy()

    if len(train_features) <= time_steps or len(test_features) <= time_steps:
        raise ValueError("No hay suficientes datos para train/test con el TIME_STEPS actual.")

    x_scaler = RobustScaler()
    y_scaler = StandardScaler()
    train_features_scaled = x_scaler.fit_transform(train_features)
    test_features_scaled = x_scaler.transform(test_features)
    train_target_scaled = y_scaler.fit_transform(train_target)
    test_target_scaled = y_scaler.transform(test_target)

    X_train_full, y_train_full = create_sequences(train_features_scaled, train_target_scaled, time_steps)
    X_test, y_test = create_sequences(test_features_scaled, test_target_scaled, time_steps)

    if len(X_train_full) == 0 or len(X_test) == 0:
        raise ValueError("No hay suficientes datos para crear secuencias LSTM.")

    val_size = max(1, int(len(X_train_full) * VALIDATION_SPLIT_WITHIN_TRAIN))
    if val_size >= len(X_train_full):
        val_size = 1

    X_train = X_train_full[:-val_size]
    y_train = y_train_full[:-val_size]
    X_val = X_train_full[-val_size:]
    y_val = y_train_full[-val_size:]

    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))

    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5, verbose=1)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping, reduce_lr],
        verbose=1,
        shuffle=False,
    )

    y_val_pred_scaled = model.predict(X_val, verbose=0).flatten()
    y_val_pred_inv = inverse_target_transform(y_val_pred_scaled, y_scaler)
    y_val_inv = inverse_target_transform(y_val.flatten(), y_scaler)
    validation_bias = calculate_bias(y_val_inv, y_val_pred_inv)

    train_meta_df = train_meta_base.iloc[time_steps:].copy().reset_index(drop=True)
    val_meta_df = train_meta_df.iloc[-len(y_val_inv):].copy().reset_index(drop=True)
    val_slot_indices = val_meta_df["slot_index"].astype(int).values if not val_meta_df.empty else np.array([], dtype=int)

    y_val_pred_after_global_bias = apply_prediction_postprocess(
        y_val_pred_inv,
        validation_bias=validation_bias,
    )
    slot_bias_adjustments = build_slot_bias_adjustments(
        slot_indices=val_slot_indices,
        y_true=y_val_inv,
        y_pred_after_global_bias=y_val_pred_after_global_bias,
        slots_per_day=slots_per_day,
    )
    y_val_pred_post = apply_prediction_postprocess(
        y_val_pred_inv,
        validation_bias=validation_bias,
        slot_indices=val_slot_indices,
        slot_bias_adjustments=slot_bias_adjustments,
    )

    y_pred_scaled = model.predict(X_test, verbose=0).flatten()
    y_pred_inv = inverse_target_transform(y_pred_scaled, y_scaler)
    y_test_inv = inverse_target_transform(y_test.flatten(), y_scaler)

    test_meta_df = test_meta_base.iloc[time_steps:].copy().reset_index(drop=True)
    test_slot_indices = test_meta_df["slot_index"].astype(int).values if not test_meta_df.empty else np.array([], dtype=int)
    y_pred_inv = apply_prediction_postprocess(
        y_pred_inv,
        validation_bias=validation_bias,
        slot_indices=test_slot_indices,
        slot_bias_adjustments=slot_bias_adjustments,
    )

    mae = float(mean_absolute_error(y_test_inv, y_pred_inv))
    rmse = float(math.sqrt(mean_squared_error(y_test_inv, y_pred_inv)))
    r2 = float(r2_score(y_test_inv, y_pred_inv)) if len(y_test_inv) > 1 else 0.0
    mape = float(calculate_mape(y_test_inv, y_pred_inv))
    wape = float(calculate_wape(y_test_inv, y_pred_inv))
    smape = float(calculate_smape(y_test_inv, y_pred_inv))
    bias = float(calculate_bias(y_test_inv, y_pred_inv))

    baseline_test_df = df.iloc[split_index:].copy().reset_index(drop=True)
    baseline_real = baseline_test_df["volume"].iloc[time_steps:].values
    baseline_pred = baseline_test_df["lag_volume_1_day"].iloc[time_steps:].values
    baseline_pred = np.maximum(baseline_pred, 0.0)

    baseline_metrics = {
        "mae": round(float(mean_absolute_error(baseline_real, baseline_pred)), 4),
        "rmse": round(float(math.sqrt(mean_squared_error(baseline_real, baseline_pred))), 4),
        "mape": round(float(calculate_mape(baseline_real, baseline_pred)), 4),
        "wape": round(float(calculate_wape(baseline_real, baseline_pred)), 4),
        "smape": round(float(calculate_smape(baseline_real, baseline_pred)), 4),
        "bias": round(float(calculate_bias(baseline_real, baseline_pred)), 4),
    }

    best_val_loss = min(history.history.get("val_loss", []) or [0.0])
    model_version = f"lstm_{slugify_channel(canonical_channel)}_v4_feature_pruned_calibrated"

    metadata = {
        "channel": canonical_channel,
        "feature_columns": feature_columns,
        "time_steps": time_steps,
        "slots_per_day": slots_per_day,
        "epochs_configured": EPOCHS,
        "batch_size": BATCH_SIZE,
        "train_size": int(len(X_train)),
        "validation_size": int(len(X_val)),
        "test_size": int(len(X_test)),
        "best_val_loss": float(best_val_loss),
        "model_version": model_version,
        "target_transform": "log1p",
        "validation_bias": float(validation_bias),
        "effective_bias_adjustment": float(compute_effective_bias_adjustment(validation_bias)),
        "bias_correction_factor": float(BIAS_CORRECTION_FACTOR),
        "negative_bias_correction_factor": float(NEGATIVE_BIAS_CORRECTION_FACTOR),
        "slot_bias_adjustments": {str(k): float(v) for k, v in slot_bias_adjustments.items()},
        "slot_bias_adjustment_shrink": float(SLOT_BIAS_ADJUSTMENT_SHRINK),
        "slot_bias_adjustment_late_multiplier": float(SLOT_BIAS_ADJUSTMENT_LATE_MULTIPLIER),
        "late_slot_window": int(LATE_SLOT_WINDOW),
        "baseline_name": "naive_previous_operational_day",
        "feature_set_version": "v3_feature_pruned_calibrated",
        "postprocess_version": "v1_bias_slot_calibration",
    }

    metrics = {
        "channel": canonical_channel,
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape": round(mape, 4),
        "wape": round(wape, 4),
        "smape": round(smape, 4),
        "r2": round(r2, 4),
        "bias": round(bias, 4),
        "best_val_loss": round(float(best_val_loss), 6),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "time_steps": int(time_steps),
        "slots_per_day": int(slots_per_day),
        "model_version": model_version,
        "feature_set_version": "v3_feature_pruned_calibrated",
        "postprocess_version": "v1_bias_slot_calibration",
        "baseline_name": "naive_previous_operational_day",
        "baseline_metrics": baseline_metrics,
        "calibration_slots": len(slot_bias_adjustments),
        "effective_bias_adjustment": round(float(compute_effective_bias_adjustment(validation_bias)), 4),
    }

    save_artifacts(
        channel=canonical_channel,
        model=model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        metadata=metadata,
        metrics=metrics,
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", default=DEFAULT_CHANNEL_TO_TRAIN, type=str)
    args = parser.parse_args()
    main(args.channel)
