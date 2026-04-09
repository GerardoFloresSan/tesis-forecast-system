from datetime import timedelta
from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from tensorflow.keras.models import load_model

from app.models.historical_interaction import HistoricalInteraction
from app.models.external_variable import ExternalVariable


BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "data" / "models"


def slugify_channel(channel: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", channel.strip().lower()).strip("_")


def load_lstm_artifacts(channel: str):
    channel_slug = slugify_channel(channel)

    model_path = MODEL_DIR / f"lstm_{channel_slug}.keras"
    scaler_path = MODEL_DIR / f"lstm_{channel_slug}_scaler.joblib"
    metadata_path = MODEL_DIR / f"lstm_{channel_slug}_metadata.joblib"

    if not model_path.exists():
        raise ValueError(f"No existe el modelo LSTM para el canal '{channel}' en: {model_path}")

    if not scaler_path.exists():
        raise ValueError(f"No existe el scaler para el canal '{channel}' en: {scaler_path}")

    if not metadata_path.exists():
        raise ValueError(f"No existe el metadata para el canal '{channel}' en: {metadata_path}")

    model = load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)
    metadata = joblib.load(metadata_path)

    return model, scaler, metadata


def _build_external_variables_map(db: Session):
    records = db.query(ExternalVariable).all()

    external_map = {}

    for record in records:
        if record.variable_date not in external_map:
            external_map[record.variable_date] = {
                "is_holiday": 0.0,
                "campaign_day": 0.0,
                "absenteeism_rate": 0.0,
            }

        variable_type = record.variable_type.strip().lower()

        if variable_type in external_map[record.variable_date]:
            external_map[record.variable_date][variable_type] = record.variable_value

    return external_map


def _build_channel_dataframe(db: Session, channel: str) -> pd.DataFrame:
    rows = (
        db.query(HistoricalInteraction)
        .filter(HistoricalInteraction.channel == channel)
        .order_by(
            HistoricalInteraction.interaction_date.asc(),
            HistoricalInteraction.interval_time.asc(),
        )
        .all()
    )

    if not rows:
        raise ValueError(f"No hay datos históricos para el canal '{channel}'.")

    external_map = _build_external_variables_map(db)

    dataset = []
    for row in rows:
        variables = external_map.get(
            row.interaction_date,
            {
                "is_holiday": 0.0,
                "campaign_day": 0.0,
                "absenteeism_rate": 0.0,
            },
        )

        dataset.append(
            {
                "interaction_date": row.interaction_date,
                "interval_time": row.interval_time,
                "channel": row.channel,
                "volume": row.volume,
                "aht": row.aht if row.aht is not None else 0.0,
                "is_holiday": variables["is_holiday"],
                "campaign_day": variables["campaign_day"],
                "absenteeism_rate": variables["absenteeism_rate"],
            }
        )

    df = pd.DataFrame(dataset)

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

    for col in ["aht", "is_holiday", "campaign_day", "absenteeism_rate"]:
        if col not in df.columns:
            df[col] = 0.0

    df["aht"] = df["aht"].fillna(0.0)
    df["is_holiday"] = df["is_holiday"].fillna(0.0)
    df["campaign_day"] = df["campaign_day"].fillna(0.0)
    df["absenteeism_rate"] = df["absenteeism_rate"].fillna(0.0)

    return df


def _infer_next_datetime(df: pd.DataFrame):
    last_dt = df["datetime"].iloc[-1].to_pydatetime()

    if len(df) >= 2:
        previous_dt = df["datetime"].iloc[-2].to_pydatetime()
        delta = last_dt - previous_dt

        if delta.total_seconds() <= 0:
            delta = timedelta(minutes=30)
    else:
        delta = timedelta(minutes=30)

    return last_dt + delta


def predict_next_volume_for_channel(db: Session, channel: str) -> dict:
    model, scaler, metadata = load_lstm_artifacts(channel)
    df = _build_channel_dataframe(db, channel)

    feature_columns = metadata["feature_columns"]
    time_steps = metadata["time_steps"]

    if len(df) < time_steps:
        raise ValueError(
            f"No hay suficientes datos para predecir. "
            f"Se necesitan al menos {time_steps} filas y solo hay {len(df)}."
        )

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0.0

    dataset = df[feature_columns].copy()

    scaled_data = scaler.transform(dataset)

    last_window = scaled_data[-time_steps:]
    X_input = np.array([last_window])

    prediction_scaled = model.predict(X_input, verbose=0)

    dummy_pred = np.zeros((1, len(feature_columns)))
    dummy_pred[:, 0] = prediction_scaled.flatten()

    prediction = scaler.inverse_transform(dummy_pred)[:, 0][0]

    next_forecast_datetime = _infer_next_datetime(df)

    return {
        "channel": channel,
        "forecast_date": next_forecast_datetime,
        "predicted_value": round(float(prediction), 4),
        "model_version": f"lstm_{slugify_channel(channel)}",
    }