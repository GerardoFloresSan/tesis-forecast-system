from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timedelta

import pandas as pd

CHANNEL_CONFIG = {
    "Choice": {
        "start_minute": 0,
        "end_minute": 16 * 60 + 30,
        "shift_ranges": [
            ("morning", 0, 11 * 60 + 59),
            ("afternoon", 12 * 60, 16 * 60 + 30),
        ],
    },
    "España": {
        "start_minute": 0,
        "end_minute": 16 * 60 + 30,
        "shift_ranges": [
            ("morning", 0, 11 * 60 + 59),
            ("afternoon", 12 * 60, 16 * 60 + 30),
        ],
    },
}

ALLOWED_CHANNELS = {
    "choice": "Choice",
    "espana": "España",
}


def normalize_channel_key(channel: str) -> str:
    normalized = unicodedata.normalize("NFKD", (channel or "").strip())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = normalized.lower()
    normalized = " ".join(normalized.split())
    return normalized


def canonicalize_channel(channel: str) -> str:
    canonical = ALLOWED_CHANNELS.get(normalize_channel_key(channel))
    if canonical is None:
        raise ValueError(
            f"Canal '{channel}' no permitido para LSTM. "
            f"Canales permitidos: {', '.join(CHANNEL_CONFIG.keys())}."
        )
    return canonical


def slugify_channel(channel: str) -> str:
    normalized = unicodedata.normalize("NFKD", channel.strip())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = normalized.lower()
    return re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")


def get_channel_config(channel: str) -> dict:
    canonical = canonicalize_channel(channel)
    return CHANNEL_CONFIG[canonical]


def get_slots_per_day(channel: str, interval_minutes: int = 30) -> int:
    config = get_channel_config(channel)
    total_minutes = config["end_minute"] - config["start_minute"]
    return (total_minutes // interval_minutes) + 1


def get_shift_label(channel: str, minute_of_day: int) -> str:
    config = get_channel_config(channel)
    for label, start_minute, end_minute in config["shift_ranges"]:
        if start_minute <= minute_of_day <= end_minute:
            return label
    return "out_of_window"


def apply_business_hours_filter(
    df: pd.DataFrame,
    channel: str,
    datetime_column: str = "datetime",
) -> pd.DataFrame:
    config = get_channel_config(channel)
    filtered = df.copy()
    filtered["minute_of_day"] = (
        filtered[datetime_column].dt.hour * 60 + filtered[datetime_column].dt.minute
    )
    mask = (
        (filtered["minute_of_day"] >= config["start_minute"])
        & (filtered["minute_of_day"] <= config["end_minute"])
    )
    filtered = filtered.loc[mask].copy()
    filtered.drop(columns=["minute_of_day"], inplace=True, errors="ignore")

    if filtered.empty:
        raise ValueError(
            f"El dataset del canal '{channel}' quedó vacío tras aplicar el horario operativo."
        )

    return filtered.reset_index(drop=True)


def get_next_operational_datetime(last_datetime: datetime, channel: str, interval_minutes: int = 30) -> datetime:
    config = get_channel_config(channel)
    next_datetime = last_datetime + timedelta(minutes=interval_minutes)
    next_minute = next_datetime.hour * 60 + next_datetime.minute

    if next_minute > config["end_minute"]:
        next_date = next_datetime.date() + timedelta(days=1)
        next_datetime = datetime.combine(next_date, datetime.min.time())
        next_datetime = next_datetime + timedelta(minutes=config["start_minute"])

    return next_datetime
