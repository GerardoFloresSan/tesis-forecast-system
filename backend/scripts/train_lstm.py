import json
import math
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

from app.core.config import settings


CHANNEL_TO_TRAIN = "Choice"
TIME_STEPS = 6
EPOCHS = 20
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.2

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "data" / "models"


def slugify_channel(channel: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", channel.strip().lower()).strip("_")


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    non_zero_mask = y_true != 0
    if not np.any(non_zero_mask):
        return 0.0

    return float(
        np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
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
            right_on="variable_date"
        )
        df.drop(columns=["variable_date"], inplace=True, errors="ignore")
    else:
        df = hist_df.copy()

    for col in ["is_holiday", "campaign_day", "absenteeism_rate"]:
        if col not in df.columns:
            df[col] = 0.0

    df["datetime"] = pd.to_datetime(
        df["interaction_date"].astype(str) + " " + df["interval_time"].astype(str)
    )

    df = df.sort_values("datetime").reset_index(drop=True)

    df["day_of_week"] = df["datetime"].dt.weekday
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["hour"] = df["datetime"].dt.hour
    df["minute"] = df["datetime"].dt.minute
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["aht"] = df["aht"].fillna(0.0)
    df["is_holiday"] = df["is_holiday"].fillna(0.0)
    df["campaign_day"] = df["campaign_day"].fillna(0.0)
    df["absenteeism_rate"] = df["absenteeism_rate"].fillna(0.0)

    return df


def build_model(input_shape: tuple[int, int]) -> Sequential:
    model = Sequential([
        LSTM(64, activation="tanh", input_shape=input_shape),
        Dense(32, activation="relu"),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    return model


def main():
    print("Conectando a PostgreSQL...")
    engine = create_engine(settings.DATABASE_URL)

    print(f"Preparando dataset para canal: {CHANNEL_TO_TRAIN}")
    df = load_and_prepare_dataset(engine, CHANNEL_TO_TRAIN)

    print(f"Total registros del canal {CHANNEL_TO_TRAIN}: {len(df)}")

    feature_columns = [
        "volume",
        "aht",
        "is_holiday",
        "campaign_day",
        "absenteeism_rate",
        "day_of_week",
        "month",
        "day",
        "hour",
        "minute",
        "is_weekend",
    ]

    dataset = df[feature_columns].copy()

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(dataset)

    X, y = create_sequences(
        features=scaled_data,
        target=scaled_data[:, 0],
        time_steps=TIME_STEPS
    )

    if len(X) == 0:
        raise ValueError("No hay suficientes datos para crear secuencias LSTM.")

    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("No hay suficientes datos para separar train/test.")

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape : {X_test.shape}")

    model = build_model((X_train.shape[1], X_train.shape[2]))

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    print("Entrenando modelo LSTM...")
    history = model.fit(
        X_train,
        y_train,
        validation_split=VALIDATION_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1
    )

    print("Generando predicciones...")
    y_pred = model.predict(X_test, verbose=0)

    dummy_pred = np.zeros((len(y_pred), scaled_data.shape[1]))
    dummy_true = np.zeros((len(y_test), scaled_data.shape[1]))

    dummy_pred[:, 0] = y_pred.flatten()
    dummy_true[:, 0] = y_test.flatten()

    y_pred_inv = scaler.inverse_transform(dummy_pred)[:, 0]
    y_test_inv = scaler.inverse_transform(dummy_true)[:, 0]

    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    rmse = math.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    r2 = r2_score(y_test_inv, y_pred_inv)
    mape = calculate_mape(y_test_inv, y_pred_inv)

    print("\n===== RESULTADOS LSTM =====")
    print(f"Canal: {CHANNEL_TO_TRAIN}")
    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.4f}%")
    print(f"R2  : {r2:.4f}")

    print("\nPrimeras 10 predicciones:")
    results_df = pd.DataFrame({
        "real_volume": y_test_inv[:10],
        "predicted_volume": y_pred_inv[:10]
    })
    print(results_df)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    channel_slug = slugify_channel(CHANNEL_TO_TRAIN)

    model_path = MODEL_DIR / f"lstm_{channel_slug}.keras"
    scaler_path = MODEL_DIR / f"lstm_{channel_slug}_scaler.joblib"
    metadata_path = MODEL_DIR / f"lstm_{channel_slug}_metadata.joblib"
    metrics_path = MODEL_DIR / f"lstm_{channel_slug}_metrics.json"

    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(
        {
            "channel": CHANNEL_TO_TRAIN,
            "feature_columns": feature_columns,
            "time_steps": TIME_STEPS,
            "epochs_configured": EPOCHS,
            "batch_size": BATCH_SIZE,
            "validation_split": VALIDATION_SPLIT,
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
        },
        metadata_path
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
                "test_size": int(len(X_test)),
                "model_path": str(model_path),
                "scaler_path": str(scaler_path),
                "metadata_path": str(metadata_path),
            },
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\nArtefactos guardados correctamente:")
    print(f"- Modelo   : {model_path}")
    print(f"- Scaler   : {scaler_path}")
    print(f"- Metadata : {metadata_path}")
    print(f"- Métricas : {metrics_path}")


if __name__ == "__main__":
    main()