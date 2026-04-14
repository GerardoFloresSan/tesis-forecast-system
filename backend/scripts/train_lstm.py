import json
import math
import os
import random
import re
from pathlib import Path

# ── Compatibilidad TF 2.15 / Keras 3 en Windows ─────────────────────────────
# Fuerza el uso de la API Keras 2 (tf.keras) para evitar el crash de tracing
# en TF >= 2.11 sobre Windows sin GPU. Debe definirse ANTES de importar TF.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
# ─────────────────────────────────────────────────────────────────────────────

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

CHANNEL_TO_TRAIN = "Choice"

# Mantenemos la configuración estable para aislar el efecto del filtro horario
TIME_STEPS = 32
EPOCHS = 100
BATCH_SIZE = 32
TRAIN_SPLIT = 0.8
VALIDATION_SPLIT_WITHIN_TRAIN = 0.2
RANDOM_SEED = 42
BIAS_CORRECTION_FACTOR = 0.35

# Reglas de negocio de entrenamiento
TRAINABLE_CHANNELS = {"Choice", "España"}
CHANNEL_OPERATING_WINDOWS = {
    "Choice": ("00:00", "16:30"),
    "España": ("00:00", "16:30"),
    # Mexico se omite por ahora
}

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "data" / "models"


def slugify_channel(channel: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", channel.strip().lower()).strip("_")


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

    # Aplicar reglas de negocio del horario operativo
    hist_df = apply_channel_business_rules(hist_df, channel)

    if hist_df.empty:
        raise ValueError(
            f"No quedaron registros para el canal '{channel}' después de aplicar el horario operativo."
        )

    # Elimina duplicados exactos primero
    hist_df = hist_df.drop_duplicates(
        subset=["interaction_date", "interval_time", "channel", "volume", "aht"]
    )

    # Si aún existen múltiples filas por mismo slot, se consolidan
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

    if not ext_df.empty:
        ext_df["variable_date"] = pd.to_datetime(ext_df["variable_date"]).dt.date
        ext_df["variable_type"] = ext_df["variable_type"].astype(str).str.strip().str.lower()
        ext_df["variable_value"] = pd.to_numeric(
            ext_df["variable_value"], errors="coerce"
        ).fillna(0.0)

        # Compatibilidad hacia atrás
        ext_df["variable_type"] = ext_df["variable_type"].replace(
            "is_holiday", "is_holiday_peru"
        )

        ext_pivot = (
            ext_df.pivot_table(
                index="variable_date",
                columns="variable_type",
                values="variable_value",
                aggfunc="max",
                fill_value=0,
            )
            .reset_index()
        )
        ext_pivot.columns.name = None

        df = hist_df.merge(
            ext_pivot,
            how="left",
            left_on="interaction_date",
            right_on="variable_date",
        )
        df.drop(columns=["variable_date"], inplace=True, errors="ignore")
    else:
        df = hist_df.copy()

    for col in [
        "is_holiday_peru",
        "is_holiday_spain",
        "is_holiday_mexico",
        "absenteeism_rate",
        "campaign_day",
    ]:
        if col not in df.columns:
            df[col] = 0.0

    df["datetime"] = pd.to_datetime(
        df["interaction_date"].astype(str) + " " + df["interval_time"].astype(str)
    )

    df = df.sort_values("datetime").reset_index(drop=True)

    numeric_cols = [
        "volume",
        "aht",
        "is_holiday_peru",
        "is_holiday_spain",
        "is_holiday_mexico",
        "campaign_day",
        "absenteeism_rate",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["is_holiday_any"] = (
        (
            (df["is_holiday_peru"] > 0)
            | (df["is_holiday_spain"] > 0)
            | (df["is_holiday_mexico"] > 0)
        )
    ).astype(int)

    return df


def add_time_and_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["day_of_week"] = df["datetime"].dt.weekday
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["hour"] = df["datetime"].dt.hour
    df["minute"] = df["datetime"].dt.minute
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["minute_of_day"] = df["hour"] * 60 + df["minute"]

    # Slot diario aproximado
    df["slot_index"] = df.groupby(df["datetime"].dt.date).cumcount()
    df["slot_sin"] = np.sin(2 * np.pi * df["slot_index"] / 32)
    df["slot_cos"] = np.cos(2 * np.pi * df["slot_index"] / 32)

    # Turnos
    df["is_morning_shift"] = df["hour"].between(6, 13).astype(int)
    df["is_afternoon_shift"] = df["hour"].between(14, 21).astype(int)
    df["is_night_shift"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)

    # Codificación cíclica
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["minute_sin"] = np.sin(2 * np.pi * df["minute_of_day"] / 1440)
    df["minute_cos"] = np.cos(2 * np.pi * df["minute_of_day"] / 1440)

    # Señales de corto plazo
    df["lag_volume_1"] = df["volume"].shift(1)
    df["lag_volume_2"] = df["volume"].shift(2)

    # Señales de mismo slot histórico
    df["lag_volume_32"] = df["volume"].shift(32)
    df["lag_volume_224"] = df["volume"].shift(224)

    # Rolling usando solo pasado
    df["rolling_mean_3"] = df["volume"].shift(1).rolling(window=3).mean()
    df["rolling_mean_6"] = df["volume"].shift(1).rolling(window=6).mean()
    df["rolling_mean_32"] = df["volume"].shift(1).rolling(window=32).mean()

    df["rolling_std_6"] = df["volume"].shift(1).rolling(window=6).std()
    df["rolling_std_32"] = df["volume"].shift(1).rolling(window=32).std()
    df["rolling_max_32"] = df["volume"].shift(1).rolling(window=32).max()

    # AHT rezagado
    df["lag_aht_1"] = df["aht"].shift(1)

    # Variación
    df["volume_diff_1"] = df["volume"].diff(1)
    df["volume_diff_32"] = df["volume"].diff(32)

    # Relación contra nivel medio reciente
    df["volume_ratio_rolling32"] = df["volume"] / (df["rolling_mean_32"] + 1.0)

    df = df.dropna().reset_index(drop=True)

    if df.empty:
        raise ValueError("Después de generar lags/rolling, el dataset quedó vacío.")

    return df


def build_model(input_shape, run_eagerly: bool = False):
    """
    Construye el modelo LSTM secuencial.

    Cambios respecto a versiones anteriores:
    - Se usa Input(shape=...) como primera capa en lugar de pasar `input_shape`
      directamente al LSTM. Esto evita el UserWarning de Keras 3 y el crash de
      tracing en TensorFlow >= 2.11 sobre Windows sin GPU.
    - Se expone `run_eagerly` para diagnóstico: si el tracing sigue fallando,
      activarlo desactiva la compilación de grafos y muestra el error real.
    """
    model = Sequential(
        [
            Input(shape=input_shape),          # ← capa de entrada explícita
            LSTM(64, return_sequences=True),    # ← sin input_shape aquí
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(16, activation="relu"),
            Dense(1),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=0.001, clipnorm=1.0),
        loss=Huber(),
        run_eagerly=run_eagerly,               # ← False en producción, True para debug
    )
    return model


def inverse_target_transform(y_scaled: np.ndarray, y_scaler: StandardScaler) -> np.ndarray:
    y_scaled = np.array(y_scaled).reshape(-1, 1)
    y_log = y_scaler.inverse_transform(y_scaled).flatten()
    y_real = np.expm1(y_log)
    return np.maximum(y_real, 0.0)


def get_shift_label(hour: int) -> str:
    if 6 <= hour <= 13:
        return "Manana (06-13)"
    if 14 <= hour <= 21:
        return "Tarde (14-21)"
    return "Noche (22-05)"


def build_segment_metrics(df_eval: pd.DataFrame, segment_col: str) -> dict:
    results = {}

    for segment_value, group in df_eval.groupby(segment_col):
        y_true = group["real_volume"].values
        y_pred = group["predicted_volume"].values

        if len(group) == 0:
            continue

        results[str(segment_value)] = {
            "n": int(len(group)),
            "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
            "rmse": round(float(math.sqrt(mean_squared_error(y_true, y_pred))), 4),
            "mape": round(float(calculate_mape(y_true, y_pred)), 4),
            "wape": round(float(calculate_wape(y_true, y_pred)), 4),
            "smape": round(float(calculate_smape(y_true, y_pred)), 4),
            "bias": round(float(calculate_bias(y_true, y_pred)), 4),
        }

    return results


def main():
    set_seeds()

    print("Conectando a PostgreSQL...")
    engine = create_engine(settings.DATABASE_URL)

    print(f"Preparando dataset para canal: {CHANNEL_TO_TRAIN}")
    df = load_and_prepare_dataset(engine, CHANNEL_TO_TRAIN)
    df = add_time_and_lag_features(df)

    print(f"Total registros útiles del canal {CHANNEL_TO_TRAIN}: {len(df)}")
    print("\nResumen estadístico de volume:")
    print(df["volume"].describe())

    feature_columns = [
        "aht",
        "is_holiday_peru",
        "is_holiday_spain",
        "is_holiday_mexico",
        "is_holiday_any",
        "absenteeism_rate",
        "campaign_day",
        "is_weekend",
        "is_morning_shift",
        "is_afternoon_shift",
        "is_night_shift",
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
        "lag_volume_32",
        "lag_volume_224",
        "rolling_mean_3",
        "rolling_mean_6",
        "rolling_mean_32",
        "rolling_std_6",
        "rolling_std_32",
        "rolling_max_32",
        "lag_aht_1",
        "volume_diff_1",
        "volume_diff_32",
        "volume_ratio_rolling32",
        "volume",
    ]

    dataset = df[feature_columns].copy()
    target = np.log1p(df[["volume"]].copy())

    split_index = int(len(dataset) * TRAIN_SPLIT)

    train_features = dataset.iloc[:split_index].copy()
    test_features = dataset.iloc[split_index:].copy()

    train_target = target.iloc[:split_index].copy()
    test_target = target.iloc[split_index:].copy()

    if len(train_features) <= TIME_STEPS or len(test_features) <= TIME_STEPS:
        raise ValueError("No hay suficientes datos para train/test con el TIME_STEPS actual.")

    x_scaler = RobustScaler()
    y_scaler = StandardScaler()

    train_features_scaled = x_scaler.fit_transform(train_features)
    test_features_scaled = x_scaler.transform(test_features)

    train_target_scaled = y_scaler.fit_transform(train_target)
    test_target_scaled = y_scaler.transform(test_target)

    X_train_full, y_train_full = create_sequences(
        features=train_features_scaled,
        target=train_target_scaled,
        time_steps=TIME_STEPS,
    )

    X_test, y_test = create_sequences(
        features=test_features_scaled,
        target=test_target_scaled,
        time_steps=TIME_STEPS,
    )

    if len(X_train_full) == 0 or len(X_test) == 0:
        raise ValueError("No hay suficientes datos para crear secuencias LSTM.")

    val_size = int(len(X_train_full) * VALIDATION_SPLIT_WITHIN_TRAIN)

    if val_size == 0:
        raise ValueError("No hay suficientes datos para generar conjunto de validación.")

    X_train = X_train_full[:-val_size]
    y_train = y_train_full[:-val_size]
    X_val = X_train_full[-val_size:]
    y_val = y_train_full[-val_size:]

    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape  : {X_val.shape}")
    print(f"X_test shape : {X_test.shape}")

    model = build_model((X_train.shape[1], X_train.shape[2]))

    # Fallback automático: si el tracing falla en Windows sin GPU,
    # reintenta en modo eager para exponer el error real.
    # En producción Linux/Docker esto nunca se activa.
    _eager_fallback = False

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=12,
        restore_best_weights=True,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-5,
        verbose=1,
    )

    print("Entrenando modelo LSTM...")
    try:
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
    except Exception as e:
        # Si el tracing falla (problema Windows/Keras3), reintenta en modo eager
        # para obtener el mensaje de error real y continuar el entrenamiento.
        print(f"\n[AVISO] Falló el tracing de TensorFlow: {e}")
        print("[AVISO] Reintentando con run_eagerly=True (modo diagnóstico)...")
        _eager_fallback = True
        model = build_model((X_train.shape[1], X_train.shape[2]), run_eagerly=True)
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

    if _eager_fallback:
        print("\n[INFO] Entrenamiento completado en modo eager (run_eagerly=True).")
        print("[INFO] Para producción, verifica la versión de Keras/TF o usa Docker/WSL2.")

    print("Calculando bias de validación...")
    y_val_pred_scaled = model.predict(X_val, verbose=0).flatten()
    y_val_pred_inv = inverse_target_transform(y_val_pred_scaled, y_scaler)
    y_val_inv = inverse_target_transform(y_val.flatten(), y_scaler)

    val_bias = calculate_bias(y_val_inv, y_val_pred_inv)
    print(f"Bias de validación: {val_bias:.4f}")

    print("Generando predicciones...")
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()

    y_pred_inv = inverse_target_transform(y_pred_scaled, y_scaler)
    y_test_inv = inverse_target_transform(y_test.flatten(), y_scaler)

    # Corrección suave del sesgo
    y_pred_inv = np.maximum(y_pred_inv - (val_bias * BIAS_CORRECTION_FACTOR), 0.0)

    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    rmse = math.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    r2 = r2_score(y_test_inv, y_pred_inv)
    mape = calculate_mape(y_test_inv, y_pred_inv)
    wape = calculate_wape(y_test_inv, y_pred_inv)
    smape = calculate_smape(y_test_inv, y_pred_inv)
    bias = calculate_bias(y_test_inv, y_pred_inv)

    # Baseline naive diario
    baseline_test_df = df.iloc[split_index:].copy().reset_index(drop=True)
    baseline_real = baseline_test_df["volume"].iloc[TIME_STEPS:].values
    baseline_pred = baseline_test_df["lag_volume_32"].iloc[TIME_STEPS:].values
    baseline_pred = np.maximum(baseline_pred, 0)

    baseline_mae = mean_absolute_error(baseline_real, baseline_pred)
    baseline_rmse = math.sqrt(mean_squared_error(baseline_real, baseline_pred))
    baseline_r2 = r2_score(baseline_real, baseline_pred)
    baseline_mape = calculate_mape(baseline_real, baseline_pred)
    baseline_wape = calculate_wape(baseline_real, baseline_pred)
    baseline_smape = calculate_smape(baseline_real, baseline_pred)
    baseline_bias = calculate_bias(baseline_real, baseline_pred)

    # Dataset alineado para análisis segmentado
    eval_df = baseline_test_df.iloc[TIME_STEPS:].copy().reset_index(drop=True)
    eval_df["real_volume"] = y_test_inv
    eval_df["predicted_volume"] = y_pred_inv
    eval_df["error"] = eval_df["predicted_volume"] - eval_df["real_volume"]
    eval_df["shift_label"] = eval_df["hour"].apply(get_shift_label)

    shift_metrics = build_segment_metrics(eval_df, "shift_label")
    day_metrics = build_segment_metrics(eval_df, "day_of_week")
    hour_metrics = build_segment_metrics(eval_df, "hour")

    print("\n===== RESULTADOS LSTM =====")
    print(f"Canal : {CHANNEL_TO_TRAIN}")
    print(f"MAE   : {mae:.4f}")
    print(f"RMSE  : {rmse:.4f}")
    print(f"MAPE  : {mape:.4f}%")
    print(f"WAPE  : {wape:.4f}%")
    print(f"sMAPE : {smape:.4f}%")
    print(f"R2    : {r2:.4f}")
    print(f"Bias  : {bias:.4f}")

    print("\n===== BASELINE NAIVE DIARIO (lag_32) =====")
    print(f"MAE   : {baseline_mae:.4f}")
    print(f"RMSE  : {baseline_rmse:.4f}")
    print(f"MAPE  : {baseline_mape:.4f}%")
    print(f"WAPE  : {baseline_wape:.4f}%")
    print(f"sMAPE : {baseline_smape:.4f}%")
    print(f"R2    : {baseline_r2:.4f}")
    print(f"Bias  : {baseline_bias:.4f}")

    print("\n===== COMPARACIÓN LSTM vs BASELINE =====")
    print(
        f"LSTM     -> MAE: {mae:.4f} | RMSE: {rmse:.4f} | MAPE: {mape:.4f}% | "
        f"WAPE: {wape:.4f}% | sMAPE: {smape:.4f}% | R2: {r2:.4f} | Bias: {bias:.4f}"
    )
    print(
        f"BASELINE -> MAE: {baseline_mae:.4f} | RMSE: {baseline_rmse:.4f} | "
        f"MAPE: {baseline_mape:.4f}% | WAPE: {baseline_wape:.4f}% | "
        f"sMAPE: {baseline_smape:.4f}% | R2: {baseline_r2:.4f} | Bias: {baseline_bias:.4f}"
    )

    print("\n===== MÉTRICAS POR TURNO =====")
    for shift_name, metrics in shift_metrics.items():
        print(
            f"{shift_name} -> "
            f"N: {metrics['n']} | "
            f"MAE: {metrics['mae']:.4f} | "
            f"RMSE: {metrics['rmse']:.4f} | "
            f"MAPE: {metrics['mape']:.4f}% | "
            f"WAPE: {metrics['wape']:.4f}% | "
            f"sMAPE: {metrics['smape']:.4f}% | "
            f"Bias: {metrics['bias']:.4f}"
        )

    print("\nPrimeras 20 predicciones LSTM:")
    results_df = pd.DataFrame(
        {
            "real_volume": y_test_inv[:20],
            "predicted_volume": y_pred_inv[:20],
            "error": y_pred_inv[:20] - y_test_inv[:20],
        }
    )
    print(results_df)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    channel_slug = slugify_channel(CHANNEL_TO_TRAIN)
    model_version = "lstm_choice_v12_business_hours"

    model_path = MODEL_DIR / f"lstm_{channel_slug}.keras"
    scaler_path = MODEL_DIR / f"lstm_{channel_slug}_scaler.joblib"
    metadata_path = MODEL_DIR / f"lstm_{channel_slug}_metadata.joblib"
    metrics_path = MODEL_DIR / f"lstm_{channel_slug}_metrics.json"

    model.save(model_path)

    joblib.dump(
        {
            "x_scaler": x_scaler,
            "y_scaler": y_scaler,
        },
        scaler_path,
    )

    best_val_loss = min(history.history["val_loss"]) if "val_loss" in history.history else None

    joblib.dump(
        {
            "channel": CHANNEL_TO_TRAIN,
            "feature_columns": feature_columns,
            "time_steps": TIME_STEPS,
            "epochs_configured": EPOCHS,
            "batch_size": BATCH_SIZE,
            "train_size": int(len(X_train)),
            "validation_size": int(len(X_val)),
            "test_size": int(len(X_test)),
            "best_val_loss": float(best_val_loss) if best_val_loss is not None else None,
            "records_per_day_reference": 32,
            "model_version": model_version,
            "target_transform": "log1p",
            "x_scaler_type": "RobustScaler",
            "y_scaler_type": "StandardScaler",
            "validation_bias": float(val_bias),
            "bias_correction_factor": float(BIAS_CORRECTION_FACTOR),
            "business_rules": {
                "trainable_channels": sorted(TRAINABLE_CHANNELS),
                "channel_operating_windows": CHANNEL_OPERATING_WINDOWS,
                "excluded_channels": ["Mexico"],
            },
            "baseline_name": "naive_daily_lag_32",
            "baseline_metrics": {
                "mae": float(baseline_mae),
                "rmse": float(baseline_rmse),
                "mape": float(baseline_mape),
                "wape": float(baseline_wape),
                "smape": float(baseline_smape),
                "r2": float(baseline_r2),
                "bias": float(baseline_bias),
            },
            "shift_metrics": shift_metrics,
            "day_metrics": day_metrics,
            "hour_metrics": hour_metrics,
        },
        metadata_path,
    )

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "channel": CHANNEL_TO_TRAIN,
                "mae": round(float(mae), 4),
                "rmse": round(float(rmse), 4),
                "mape": round(float(mape), 4),
                "wape": round(float(wape), 4),
                "smape": round(float(smape), 4),
                "r2": round(float(r2), 4),
                "bias": round(float(bias), 4),
                "train_size": int(len(X_train)),
                "validation_size": int(len(X_val)),
                "test_size": int(len(X_test)),
                "model_path": str(model_path),
                "scaler_path": str(scaler_path),
                "metadata_path": str(metadata_path),
                "best_val_loss": round(float(best_val_loss), 6) if best_val_loss is not None else None,
                "model_version": model_version,
                "target_transform": "log1p",
                "validation_bias": round(float(val_bias), 4),
                "bias_correction_factor": round(float(BIAS_CORRECTION_FACTOR), 4),
                "business_rules": {
                    "trainable_channels": sorted(TRAINABLE_CHANNELS),
                    "channel_operating_windows": CHANNEL_OPERATING_WINDOWS,
                    "excluded_channels": ["Mexico"],
                },
                "baseline_name": "naive_daily_lag_32",
                "baseline_metrics": {
                    "mae": round(float(baseline_mae), 4),
                    "rmse": round(float(baseline_rmse), 4),
                    "mape": round(float(baseline_mape), 4),
                    "wape": round(float(baseline_wape), 4),
                    "smape": round(float(baseline_smape), 4),
                    "r2": round(float(baseline_r2), 4),
                    "bias": round(float(baseline_bias), 4),
                },
                "shift_metrics": shift_metrics,
                "day_metrics": day_metrics,
                "hour_metrics": hour_metrics,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\nArtefactos guardados correctamente:")
    print(f"- Modelo   : {model_path}")
    print(f"- Scaler   : {scaler_path}")
    print(f"- Metadata : {metadata_path}")
    print(f"- Métricas : {metrics_path}")


if __name__ == "__main__":
    main()