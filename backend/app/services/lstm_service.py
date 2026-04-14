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

CHANNEL_BUSINESS_HOURS = {
    "choice": {"start_minute": 0, "end_minute": 990},   # 00:00 – 16:30
    "españa": {"start_minute": 0, "end_minute": 990},   # 00:00 – 16:30
}

CHANNEL_SHIFTS = {
    "choice": {
        "morning":   (0, 12),    # 00:00 – 12:00
        "afternoon": (12, 16),   # 12:00 – 16:30
        "night":     None,       # No existe
    },
    "españa": {
        "morning":   (0, 12),
        "afternoon": (12, 16),
        "night":     None,
    },
}


def apply_business_hours_filter(df: pd.DataFrame, channel: str) -> pd.DataFrame:
    channel_key = channel.strip().lower()
    hours = CHANNEL_BUSINESS_HOURS.get(channel_key)
    if hours is None:
        return df
    df = df.copy()
    df["minute_of_day"] = df["datetime"].dt.hour * 60 + df["datetime"].dt.minute
    mask = (
        (df["minute_of_day"] >= hours["start_minute"]) &
        (df["minute_of_day"] <= hours["end_minute"])
    )
    filtered = df[mask].drop(columns=["minute_of_day"]).reset_index(drop=True)
    if filtered.empty:
        raise ValueError(
            f"El dataset del canal '{channel}' quedó vacío tras aplicar "
            f"el filtro de horario operativo."
        )
    return filtered


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
    scaler_artifact = joblib.load(scaler_path)
    metadata = joblib.load(metadata_path)

    if isinstance(scaler_artifact, dict) and "x_scaler" in scaler_artifact and "y_scaler" in scaler_artifact:
        x_scaler = scaler_artifact["x_scaler"]
        y_scaler = scaler_artifact["y_scaler"]
    else:
        x_scaler = scaler_artifact
        y_scaler = scaler_artifact

    return model, x_scaler, y_scaler, metadata


def _build_external_variables_map(db: Session):
    records = db.query(ExternalVariable).all()

    _DEFAULT_EXT = {
        "is_holiday_peru": 0.0,
        "is_holiday_spain": 0.0,
        "is_holiday_mexico": 0.0,
        "campaign_day": 0.0,
        "absenteeism_rate": 0.0,
    }

    external_map = {}

    for record in records:
        if record.variable_date not in external_map:
            external_map[record.variable_date] = dict(_DEFAULT_EXT)

        variable_type = record.variable_type.strip().lower()

        # Mapear "is_holiday" genérico a "is_holiday_peru"
        if variable_type == "is_holiday":
            variable_type = "is_holiday_peru"

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
        _default_vars = {
            "is_holiday_peru": 0.0,
            "is_holiday_spain": 0.0,
            "is_holiday_mexico": 0.0,
            "campaign_day": 0.0,
            "absenteeism_rate": 0.0,
        }
        variables = {**_default_vars, **external_map.get(row.interaction_date, {})}

        dataset.append(
            {
                "interaction_date": row.interaction_date,
                "interval_time": row.interval_time,
                "channel": row.channel,
                "volume": row.volume,
                "aht": row.aht if row.aht is not None else 0.0,
                "is_holiday_peru": variables["is_holiday_peru"],
                "is_holiday_spain": variables["is_holiday_spain"],
                "is_holiday_mexico": variables["is_holiday_mexico"],
                "campaign_day": variables["campaign_day"],
                "absenteeism_rate": variables["absenteeism_rate"],
            }
        )

    df = pd.DataFrame(dataset)

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

    # Filtrar horario operativo ANTES de generar features/lags
    df = apply_business_hours_filter(df, channel)

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

    # Turnos dinámicos por canal
    shifts = CHANNEL_SHIFTS.get(channel.strip().lower(), {
        "morning": (6, 13), "afternoon": (14, 21), "night": None
    })
    df["is_morning_shift"]   = df["hour"].between(*shifts["morning"]).astype(int)
    df["is_afternoon_shift"] = df["hour"].between(*shifts["afternoon"]).astype(int)
    df["is_night_shift"]     = 0   # Choice/España no tienen turno noche

    # 33 slots/día (Choice/España 00:00–16:30)
    df["lag_volume_1"] = df["volume"].shift(1)
    df["lag_volume_2"] = df["volume"].shift(2)
    df["lag_volume_33"] = df["volume"].shift(33)
    df["lag_volume_231"] = df["volume"].shift(231)

    # Rolling con solo pasado
    df["rolling_mean_3"] = df["volume"].shift(1).rolling(window=3).mean()
    df["rolling_mean_6"] = df["volume"].shift(1).rolling(window=6).mean()
    df["rolling_mean_33"] = df["volume"].shift(1).rolling(window=33).mean()

    df["lag_aht_1"] = df["aht"].shift(1)

    df = df.dropna().reset_index(drop=True)

    if df.empty:
        raise ValueError(
            f"El dataset del canal '{channel}' quedó vacío después de generar lags/rolling."
        )

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
    model, x_scaler, y_scaler, metadata = load_lstm_artifacts(channel)
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
    scaled_data = x_scaler.transform(dataset)

    last_window = scaled_data[-time_steps:]
    X_input = np.array([last_window])

    prediction_scaled = model.predict(X_input, verbose=0)
    prediction = y_scaler.inverse_transform(prediction_scaled).flatten()[0]

    prediction = max(float(prediction), 0.0)

    next_forecast_datetime = _infer_next_datetime(df)

    return {
        "channel": channel,
        "forecast_date": next_forecast_datetime,
        "predicted_value": round(prediction, 4),
        "model_version": metadata.get("model_version", f"lstm_{slugify_channel(channel)}"),
    }