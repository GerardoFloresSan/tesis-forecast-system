from sqlalchemy.orm import Session

from app.services.forecast_service import get_forecast_dataset, get_forecast_dataset_by_date


def _add_time_features(row: dict) -> dict:
    interaction_date = row["interaction_date"]
    interval_time = row["interval_time"]

    row["day_of_week"] = interaction_date.weekday()   # lunes=0, domingo=6
    row["month"] = interaction_date.month
    row["is_weekend"] = 1 if interaction_date.weekday() >= 5 else 0
    row["hour"] = interval_time.hour
    row["minute"] = interval_time.minute

    return row


def get_preprocessed_dataset(db: Session) -> list[dict]:
    dataset = get_forecast_dataset(db)
    processed = [_add_time_features(row.copy()) for row in dataset]
    return processed


def get_preprocessed_dataset_by_channel(db: Session, channel: str) -> list[dict]:
    dataset = get_forecast_dataset(db)
    filtered = [row for row in dataset if row["channel"].strip().lower() == channel.strip().lower()]
    processed = [_add_time_features(row.copy()) for row in filtered]
    return processed