from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import re
import unicodedata

import joblib
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.models.external_variable import ExternalVariable
from app.models.historical_interaction import HistoricalInteraction


def _get_load_model():
    import os

    os.environ.pop("TF_USE_LEGACY_KERAS", None)

    try:
        from tensorflow.keras.models import load_model
        return load_model
    except Exception as exc:
        raise ModuleNotFoundError(
            "No se pudo importar tensorflow.keras.models.load_model. "
            "Verifica la instalación de TensorFlow en el entorno virtual."
        ) from exc


BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "data" / "models"
ALLOWED_CHANNELS = {
    "choice": "Choice",
    "espana": "España",
}
CHANNEL_BUSINESS_HOURS = {
    "choice": {"start_minute": 0, "end_minute": 990},
    "espana": {"start_minute": 0, "end_minute": 990},
}


def _normalize_channel_key(channel: str) -> str:
    normalized = unicodedata.normalize("NFKD", (channel or "").strip())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = normalized.lower()
    normalized = " ".join(normalized.split())
    return normalized


def canonicalize_channel(channel: str) -> str:
    channel_key = _normalize_channel_key(channel)
    canonical = ALLOWED_CHANNELS.get(channel_key)
    if canonical is None:
        raise ValueError(
            f"Canal '{channel}' no soportado para LSTM. "
            f"Permitidos: {', '.join(ALLOWED_CHANNELS.values())}."
        )
    return canonical


def slugify_channel(channel: str) -> str:
    normalized = unicodedata.normalize("NFKD", channel.strip())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = normalized.lower()
    return re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")


def apply_business_hours_filter(df: pd.DataFrame, channel: str) -> pd.DataFrame:
    channel_key = _normalize_channel_key(channel)
    hours = CHANNEL_BUSINESS_HOURS.get(channel_key)
    if hours is None:
        return df

    df = df.copy()
    df["minute_of_day"] = df["datetime"].dt.hour * 60 + df["datetime"].dt.minute
    mask = (
        (df["minute_of_day"] >= hours["start_minute"])
        & (df["minute_of_day"] <= hours["end_minute"])
    )
    filtered = df[mask].drop(columns=["minute_of_day"]).reset_index(drop=True)

    if filtered.empty:
        raise ValueError(
            f"El dataset del canal '{channel}' quedó vacío tras aplicar "
            f"el filtro de horario operativo."
        )

    return filtered


def load_lstm_artifacts(channel: str):
    canonical_channel = canonicalize_channel(channel)
    channel_slug = slugify_channel(canonical_channel)

    model_path = MODEL_DIR / f"lstm_{channel_slug}.keras"
    scaler_path = MODEL_DIR / f"lstm_{channel_slug}_scaler.joblib"
    metadata_path = MODEL_DIR / f"lstm_{channel_slug}_metadata.joblib"

    if not model_path.exists():
        raise ValueError(f"No existe el modelo LSTM para el canal '{canonical_channel}' en: {model_path}")
    if not scaler_path.exists():
        raise ValueError(f"No existe el scaler LSTM para el canal '{canonical_channel}' en: {scaler_path}")
    if not metadata_path.exists():
        raise ValueError(f"No existe el metadata LSTM para el canal '{canonical_channel}' en: {metadata_path}")

    load_model = _get_load_model()
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    metadata = joblib.load(metadata_path)

    return model, scaler, metadata


def _get_external_variables_map(db: Session) -> dict:
    variables = db.query(ExternalVariable).all()
    mapping: dict = {}

    for item in variables:
        day_map = mapping.setdefault(item.variable_date, {})
        day_map[str(item.variable_type).strip().lower()] = float(item.variable_value or 0.0)

    return mapping


def _build_feature_row(row, external_map: dict) -> dict:
    day_vars = external_map.get(row.interaction_date, {})

    is_holiday_peru = float(day_vars.get("is_holiday_peru", day_vars.get("is_holiday", 0.0)))
    is_holiday_spain = float(day_vars.get("is_holiday_spain", 0.0))
    is_holiday_mexico = float(day_vars.get("is_holiday_mexico", 0.0))
    campaign_day = float(day_vars.get("campaign_day", 0.0))
    absenteeism_rate = float(day_vars.get("absenteeism_rate", 0.0))

    return {
        "datetime": pd.to_datetime(f"{row.interaction_date} {row.interval_time}"),
        "volume": float(row.volume or 0.0),
        "aht": float(row.aht or 0.0),
        "day_of_week": pd.to_datetime(row.interaction_date).dayofweek,
        "month": pd.to_datetime(row.interaction_date).month,
        "is_holiday_peru": is_holiday_peru,
        "is_holiday_spain": is_holiday_spain,
        "is_holiday_mexico": is_holiday_mexico,
        "is_holiday_any": float(max(is_holiday_peru, is_holiday_spain, is_holiday_mexico)),
        "campaign_day": campaign_day,
        "absenteeism_rate": absenteeism_rate,
    }


def _prepare_recent_sequence(df: pd.DataFrame, metadata: dict) -> np.ndarray:
    feature_columns = metadata["feature_columns"]
    time_steps = int(metadata["time_steps"])

    if len(df) < time_steps:
        raise ValueError(
            f"No hay suficientes registros para inferencia LSTM. "
            f"Se requieren al menos {time_steps} y solo hay {len(df)}."
        )

    sequence = df[feature_columns].tail(time_steps).values
    return np.array([sequence], dtype=np.float32)


def predict_next_volume_for_channel(db: Session, channel: str) -> dict:
    canonical_channel = canonicalize_channel(channel)
    model, scaler, metadata = load_lstm_artifacts(canonical_channel)

    rows = (
        db.query(HistoricalInteraction)
        .filter(HistoricalInteraction.channel == canonical_channel)
        .order_by(
            HistoricalInteraction.interaction_date.asc(),
            HistoricalInteraction.interval_time.asc(),
        )
        .all()
    )

    if not rows:
        raise ValueError(f"No hay datos históricos para el canal '{canonical_channel}'.")

    external_map = _get_external_variables_map(db)
    feature_rows = [_build_feature_row(row, external_map) for row in rows]
    df = pd.DataFrame(feature_rows)
    df = apply_business_hours_filter(df, canonical_channel)

    feature_columns = metadata["feature_columns"]
    df[feature_columns] = scaler.transform(df[feature_columns])

    X_input = _prepare_recent_sequence(df, metadata)
    prediction = model.predict(X_input, verbose=0)

    predicted_value = float(np.maximum(prediction[0][0], 0.0))

    last_datetime = df["datetime"].max()
    next_datetime = last_datetime + timedelta(minutes=30)

    return {
        "channel": canonical_channel,
        "forecast_date": next_datetime.date(),
        "predicted_value": predicted_value,
        "model_version": "lstm_v1",
    }