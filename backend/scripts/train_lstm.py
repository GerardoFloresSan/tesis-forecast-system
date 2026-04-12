import json
import math
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sqlalchemy import create_engine
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.losses import Huber
from tensorflow.keras.models import Sequential

from app.core.config import settings

CHANNEL_TO_TRAIN = "Choice"

# Ajustado a la frecuencia real observada (~32 registros por día)
TIME_STEPS = 32
EPOCHS = 80
BATCH_SIZE = 32
TRAIN_SPLIT = 0.8
VALIDATION_SPLIT_WITHIN_TRAIN = 0.2
RANDOM_SEED = 42

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "data" / "models"


def slugify_channel(channel: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", channel.strip().lower()).strip("_")


def set_seeds(seed: int = RANDOM_SEED):
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


def create_sequences(features: np.ndarray, target: np.ndarray, time_steps: int):
    X, y = [], []

    for i in range(len(features) - time_steps):
        X.append(features[i:i + time_steps])
        y.append(target[i + time_steps])

    return np.array(X), np.array(y)


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

    if not ext_df.empty:
        ext_df["variable_date"] = pd.to_datetime(ext_df["variable_date"]).dt.date
        ext_df["variable_type"] = ext_df["variable_type"].astype(str).str.strip().str.lower()

        # Mapear "is_holiday" genérico a "is_holiday_peru"
        ext_df["variable_type"] = ext_df["variable_type"].replace("is_holiday", "is_holiday_peru")

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

    for col in ["is_holiday_peru", "is_holiday_spain", "is_holiday_mexico", "absenteeism_rate", "campaign_day"]:
        if col not in df.columns:
            df[col] = 0.0

    df["datetime"] = pd.to_datetime(
        df["interaction_date"].astype(str) + " " + df["interval_time"].astype(str)
    )

    df = df.sort_values("datetime").reset_index(drop=True)

    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    df["aht"] = pd.to_numeric(df["aht"], errors="coerce").fillna(0.0)
    df["is_holiday_peru"] = pd.to_numeric(df["is_holiday_peru"], errors="coerce").fillna(0.0)
    df["is_holiday_spain"] = pd.to_numeric(df["is_holiday_spain"], errors="coerce").fillna(0.0)
    df["is_holiday_mexico"] = pd.to_numeric(df["is_holiday_mexico"], errors="coerce").fillna(0.0)
    df["campaign_day"] = pd.to_numeric(df["campaign_day"], errors="coerce").fillna(0.0)
    df["absenteeism_rate"] = pd.to_numeric(df["absenteeism_rate"], errors="coerce").fillna(0.0)

    return df


def add_time_and_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Variables temporales
    df["day_of_week"] = df["datetime"].dt.weekday
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["hour"] = df["datetime"].dt.hour
    df["minute"] = df["datetime"].dt.minute
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["minute_of_day"] = df["hour"] * 60 + df["minute"]

    # Codificación cíclica
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["minute_sin"] = np.sin(2 * np.pi * df["minute_of_day"] / 1440)
    df["minute_cos"] = np.cos(2 * np.pi * df["minute_of_day"] / 1440)

    # Lags ajustados a ~32 registros por día
    df["lag_volume_1"] = df["volume"].shift(1)
    df["lag_volume_2"] = df["volume"].shift(2)
    df["lag_volume_32"] = df["volume"].shift(32)     # día anterior
    df["lag_volume_224"] = df["volume"].shift(224)   # semana anterior

    # Rolling con solo información pasada
    df["rolling_mean_3"] = df["volume"].shift(1).rolling(window=3).mean()
    df["rolling_mean_6"] = df["volume"].shift(1).rolling(window=6).mean()
    df["rolling_mean_32"] = df["volume"].shift(1).rolling(window=32).mean()

    # Lag de AHT
    df["lag_aht_1"] = df["aht"].shift(1)

    # Diferencias (velocidad de cambio)
    df["volume_diff_1"] = df["volume"].diff(1)
    df["volume_diff_32"] = df["volume"].diff(32)

    # Ratio respecto al promedio móvil (detecta desviaciones)
    df["volume_ratio_rolling32"] = df["volume"] / (df["rolling_mean_32"] + 1)

    # Volatilidad reciente
    df["volume_std_6"] = df["volume"].shift(1).rolling(window=6).std()

    df = df.dropna().reset_index(drop=True)

    if df.empty:
        raise ValueError("Después de generar lags/rolling, el dataset quedó vacío.")

    return df


def build_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss=Huber())
    return model


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
        "absenteeism_rate",
        "campaign_day",
        "lag_volume_1",
        "lag_volume_2",
        "lag_volume_32",
        "lag_volume_224",
        "rolling_mean_3",
        "rolling_mean_6",
        "rolling_mean_32",
        "lag_aht_1",
        "is_weekend",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        "minute_sin",
        "minute_cos",
        "volume_diff_1",
        "volume_diff_32",
        "volume_ratio_rolling32",
        "volume_std_6",
        "volume",
    ]

    dataset = df[feature_columns].copy()
    target = df[["volume"]].copy()

    # Split temporal antes de escalar
    split_index = int(len(dataset) * TRAIN_SPLIT)

    train_features = dataset.iloc[:split_index].copy()
    test_features = dataset.iloc[split_index:].copy()

    train_target = target.iloc[:split_index].copy()
    test_target = target.iloc[split_index:].copy()

    if len(train_features) <= TIME_STEPS or len(test_features) <= TIME_STEPS:
        raise ValueError("No hay suficientes datos para train/test con el TIME_STEPS actual.")

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

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

    # Validación temporal explícita dentro del train
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

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-5,
        verbose=1,
    )

    print("Entrenando modelo LSTM...")
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

    print("Generando predicciones...")
    y_pred_scaled = model.predict(X_test, verbose=0)

    y_pred_inv = y_scaler.inverse_transform(y_pred_scaled).flatten()
    y_test_inv = y_scaler.inverse_transform(y_test).flatten()

    y_pred_inv = np.maximum(y_pred_inv, 0)

    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    rmse = math.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    r2 = r2_score(y_test_inv, y_pred_inv)
    mape = calculate_mape(y_test_inv, y_pred_inv)

    # =========================
    # BASELINE NAIVE DIARIO
    # =========================
    baseline_test_df = df.iloc[split_index:].copy().reset_index(drop=True)

    baseline_real = baseline_test_df["volume"].iloc[TIME_STEPS:].values
    baseline_pred = baseline_test_df["lag_volume_32"].iloc[TIME_STEPS:].values

    baseline_pred = np.maximum(baseline_pred, 0)

    baseline_mae = mean_absolute_error(baseline_real, baseline_pred)
    baseline_rmse = math.sqrt(mean_squared_error(baseline_real, baseline_pred))
    baseline_r2 = r2_score(baseline_real, baseline_pred)
    baseline_mape = calculate_mape(baseline_real, baseline_pred)

    print("\n===== RESULTADOS LSTM =====")
    print(f"Canal: {CHANNEL_TO_TRAIN}")
    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.4f}%")
    print(f"R2  : {r2:.4f}")

    print("\n===== BASELINE NAIVE DIARIO (lag_32) =====")
    print(f"MAE : {baseline_mae:.4f}")
    print(f"RMSE: {baseline_rmse:.4f}")
    print(f"MAPE: {baseline_mape:.4f}%")
    print(f"R2  : {baseline_r2:.4f}")

    print("\n===== COMPARACIÓN LSTM vs BASELINE =====")
    print(f"LSTM     -> MAE: {mae:.4f} | RMSE: {rmse:.4f} | MAPE: {mape:.4f}% | R2: {r2:.4f}")
    print(f"BASELINE -> MAE: {baseline_mae:.4f} | RMSE: {baseline_rmse:.4f} | MAPE: {baseline_mape:.4f}% | R2: {baseline_r2:.4f}")

    print("\nPrimeras 20 predicciones LSTM:")
    results_df = pd.DataFrame({
        "real_volume": y_test_inv[:20],
        "predicted_volume": y_pred_inv[:20],
    })
    print(results_df)

    print("\nPrimeras 20 predicciones BASELINE:")
    baseline_results_df = pd.DataFrame({
        "real_volume": baseline_real[:20],
        "baseline_predicted_volume": baseline_pred[:20],
    })
    print(baseline_results_df)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    channel_slug = slugify_channel(CHANNEL_TO_TRAIN)

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
            "model_version": "lstm_choice_v10_clean",
            "baseline_name": "naive_daily_lag_32",
            "baseline_metrics": {
                "mae": float(baseline_mae),
                "rmse": float(baseline_rmse),
                "mape": float(baseline_mape),
                "r2": float(baseline_r2),
            },
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
                "r2": round(float(r2), 4),
                "train_size": int(len(X_train)),
                "validation_size": int(len(X_val)),
                "test_size": int(len(X_test)),
                "model_path": str(model_path),
                "scaler_path": str(scaler_path),
                "metadata_path": str(metadata_path),
                "best_val_loss": round(float(best_val_loss), 6) if best_val_loss is not None else None,
                "model_version": "lstm_choice_v10_clean",
                "baseline_name": "naive_daily_lag_32",
                "baseline_metrics": {
                    "mae": round(float(baseline_mae), 4),
                    "rmse": round(float(baseline_rmse), 4),
                    "mape": round(float(baseline_mape), 4),
                    "r2": round(float(baseline_r2), 4),
                },
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