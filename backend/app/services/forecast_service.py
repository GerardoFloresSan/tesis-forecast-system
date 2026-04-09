from datetime import date
from sqlalchemy.orm import Session

from app.models.historical_interaction import HistoricalInteraction
from app.models.external_variable import ExternalVariable
from app.models.forecast_run import ForecastRun
from app.services.lstm_service import predict_next_volume_for_channel


def _build_external_variables_map(
    db: Session,
    start_date: date | None = None,
    end_date: date | None = None
):
    query = db.query(ExternalVariable)

    if start_date:
        query = query.filter(ExternalVariable.variable_date >= start_date)
    if end_date:
        query = query.filter(ExternalVariable.variable_date <= end_date)

    records = query.all()

    external_map: dict[date, dict[str, float]] = {}

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


def get_available_channels(db: Session) -> list[str]:
    rows = (
        db.query(HistoricalInteraction.channel)
        .distinct()
        .order_by(HistoricalInteraction.channel.asc())
        .all()
    )

    return [row[0] for row in rows if row[0]]


def get_forecast_dataset(db: Session):
    historical_data = (
        db.query(HistoricalInteraction)
        .order_by(
            HistoricalInteraction.interaction_date.asc(),
            HistoricalInteraction.interval_time.asc(),
            HistoricalInteraction.channel.asc(),
        )
        .all()
    )

    external_map = _build_external_variables_map(db)

    dataset = []
    for row in historical_data:
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
                "aht": row.aht,
                "is_holiday": variables["is_holiday"],
                "campaign_day": variables["campaign_day"],
                "absenteeism_rate": variables["absenteeism_rate"],
            }
        )

    return dataset


def get_forecast_dataset_by_date(db: Session, start_date: date, end_date: date):
    historical_data = (
        db.query(HistoricalInteraction)
        .filter(HistoricalInteraction.interaction_date >= start_date)
        .filter(HistoricalInteraction.interaction_date <= end_date)
        .order_by(
            HistoricalInteraction.interaction_date.asc(),
            HistoricalInteraction.interval_time.asc(),
            HistoricalInteraction.channel.asc(),
        )
        .all()
    )

    external_map = _build_external_variables_map(db, start_date, end_date)

    dataset = []
    for row in historical_data:
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
                "aht": row.aht,
                "is_holiday": variables["is_holiday"],
                "campaign_day": variables["campaign_day"],
                "absenteeism_rate": variables["absenteeism_rate"],
            }
        )

    return dataset


def create_daily_forecast(db: Session, channel: str):
    prediction = predict_next_volume_for_channel(db, channel)

    existing_forecast = (
        db.query(ForecastRun)
        .filter(ForecastRun.channel == prediction["channel"])
        .filter(ForecastRun.forecast_date == prediction["forecast_date"])
        .first()
    )

    if existing_forecast:
        existing_forecast.predicted_value = prediction["predicted_value"]
        existing_forecast.model_version = prediction["model_version"]

        db.commit()
        db.refresh(existing_forecast)

        return {
            "id": existing_forecast.id,
            "channel": existing_forecast.channel,
            "forecast_date": existing_forecast.forecast_date,
            "predicted_value": existing_forecast.predicted_value,
            "model_version": existing_forecast.model_version,
            "created_at": existing_forecast.created_at,
            "operation": "updated",
            "message": f"Forecast actualizado correctamente para el canal {existing_forecast.channel}.",
        }

    forecast = ForecastRun(
        channel=prediction["channel"],
        forecast_date=prediction["forecast_date"],
        predicted_value=prediction["predicted_value"],
        model_version=prediction["model_version"],
    )

    db.add(forecast)
    db.commit()
    db.refresh(forecast)

    return {
        "id": forecast.id,
        "channel": forecast.channel,
        "forecast_date": forecast.forecast_date,
        "predicted_value": forecast.predicted_value,
        "model_version": forecast.model_version,
        "created_at": forecast.created_at,
        "operation": "created",
        "message": f"Forecast creado correctamente para el canal {forecast.channel}.",
    }


def get_forecast_history(db: Session, channel: str | None = None, limit: int = 50):
    query = db.query(ForecastRun)

    if channel:
        query = query.filter(ForecastRun.channel == channel)

    rows = (
        query.order_by(ForecastRun.created_at.desc())
        .limit(limit)
        .all()
    )

    return [
        {
            "id": row.id,
            "channel": row.channel,
            "forecast_date": row.forecast_date,
            "predicted_value": row.predicted_value,
            "model_version": row.model_version,
            "created_at": row.created_at,
        }
        for row in rows
    ]