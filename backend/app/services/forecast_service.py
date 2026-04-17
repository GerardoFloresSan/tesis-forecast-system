from datetime import date, datetime
from sqlalchemy.orm import Session

from app.models.external_variable import ExternalVariable
from app.models.forecast_interval_run import ForecastIntervalRun
from app.models.forecast_run import ForecastRun
from app.models.historical_interaction import HistoricalInteraction
from app.services.lstm_service import predict_next_operational_day_for_channel
from app.utils.channel_rules import canonicalize_channel


def _normalize_variable_type(variable_type: str | None) -> str:
    value = (variable_type or "").strip().lower()

    alias_map = {
        "is_holiday": "is_holiday_peru",
        "holiday": "is_holiday_peru",
        "holiday_peru": "is_holiday_peru",
        "is_holiday_peru": "is_holiday_peru",
        "is_holiday_spain": "is_holiday_spain",
        "holiday_spain": "is_holiday_spain",
        "is_holiday_mexico": "is_holiday_mexico",
        "holiday_mexico": "is_holiday_mexico",
        "campaign_day": "campaign_day",
        "absenteeism_rate": "absenteeism_rate",
    }

    return alias_map.get(value, value)


def _default_external_variables() -> dict[str, float]:
    return {
        "is_holiday_peru": 0.0,
        "is_holiday_spain": 0.0,
        "is_holiday_mexico": 0.0,
        "campaign_day": 0.0,
        "absenteeism_rate": 0.0,
    }


def _build_external_variables_map(
    db: Session,
    start_date: date | None = None,
    end_date: date | None = None,
):
    query = db.query(ExternalVariable)

    if start_date:
        query = query.filter(ExternalVariable.variable_date >= start_date)
    if end_date:
        query = query.filter(ExternalVariable.variable_date <= end_date)

    records = query.order_by(
        ExternalVariable.variable_date.asc(),
        ExternalVariable.id.asc()
    ).all()

    external_map: dict[date, dict[str, float]] = {}

    for record in records:
        if record.variable_date not in external_map:
            external_map[record.variable_date] = _default_external_variables()

        normalized_variable = _normalize_variable_type(record.variable_type)

        if normalized_variable in external_map[record.variable_date]:
            external_map[record.variable_date][normalized_variable] = float(record.variable_value or 0.0)

    return external_map


def _serialize_dataset_row(row, variables: dict[str, float]) -> dict:
    is_holiday_peru = float(variables.get("is_holiday_peru", 0.0))
    is_holiday_spain = float(variables.get("is_holiday_spain", 0.0))
    is_holiday_mexico = float(variables.get("is_holiday_mexico", 0.0))

    return {
        "interaction_date": row.interaction_date,
        "interval_time": row.interval_time,
        "channel": row.channel,
        "volume": row.volume,
        "aht": row.aht,
        "is_holiday": is_holiday_peru,
        "is_holiday_peru": is_holiday_peru,
        "is_holiday_spain": is_holiday_spain,
        "is_holiday_mexico": is_holiday_mexico,
        "is_holiday_any": float(max(is_holiday_peru, is_holiday_spain, is_holiday_mexico)),
        "campaign_day": float(variables.get("campaign_day", 0.0)),
        "absenteeism_rate": float(variables.get("absenteeism_rate", 0.0)),
    }


def _serialize_forecast_header(row: ForecastRun) -> dict:
    return {
        "id": row.id,
        "channel": row.channel,
        "forecast_date": row.forecast_date,
        "predicted_value": row.predicted_value,
        "model_version": row.model_version,
        "created_at": row.created_at,
    }


def _serialize_interval_row(row: ForecastIntervalRun) -> dict:
    return {
        "id": row.id,
        "forecast_run_id": row.forecast_run_id,
        "channel": row.channel,
        "forecast_date": row.forecast_date,
        "forecast_datetime": row.forecast_datetime,
        "interval_time": row.interval_time,
        "slot_index": row.slot_index,
        "shift_label": row.shift_label,
        "predicted_value": row.predicted_value,
        "model_version": row.model_version,
        "created_at": row.created_at,
    }


def get_available_channels(db: Session) -> list[str]:
    rows = (
        db.query(HistoricalInteraction.channel)
        .distinct()
        .order_by(HistoricalInteraction.channel.asc())
        .all()
    )

    return [row[0] for row in rows if row[0]]


def get_forecast_dataset(
    db: Session,
    start_date: date | None = None,
    end_date: date | None = None,
    channel: str | None = None,
    limit: int | None = 500,
    offset: int = 0,
):
    query = db.query(HistoricalInteraction)

    if start_date:
        query = query.filter(HistoricalInteraction.interaction_date >= start_date)

    if end_date:
        query = query.filter(HistoricalInteraction.interaction_date <= end_date)

    if channel:
        query = query.filter(HistoricalInteraction.channel == channel)

    query = query.order_by(
        HistoricalInteraction.interaction_date.asc(),
        HistoricalInteraction.interval_time.asc(),
        HistoricalInteraction.channel.asc(),
    )

    if offset:
        query = query.offset(offset)

    if limit is not None:
        query = query.limit(limit)

    historical_rows = query.all()

    if not historical_rows:
        return []

    effective_start_date = start_date or min(row.interaction_date for row in historical_rows)
    effective_end_date = end_date or max(row.interaction_date for row in historical_rows)

    external_map = _build_external_variables_map(
        db=db,
        start_date=effective_start_date,
        end_date=effective_end_date,
    )

    dataset = []
    for row in historical_rows:
        variables = external_map.get(row.interaction_date, _default_external_variables())
        dataset.append(_serialize_dataset_row(row, variables))

    return dataset


def get_forecast_dataset_by_date(
    db: Session,
    start_date: date,
    end_date: date,
    channel: str | None = None,
    limit: int | None = 1000,
    offset: int = 0,
):
    return get_forecast_dataset(
        db=db,
        start_date=start_date,
        end_date=end_date,
        channel=channel,
        limit=limit,
        offset=offset,
    )


def create_daily_forecast(db: Session, channel: str):
    prediction_batch = predict_next_operational_day_for_channel(db, channel)
    canonical_channel = canonicalize_channel(prediction_batch["channel"])
    forecast_start_datetime = prediction_batch["forecast_start_datetime"]
    now_utc = datetime.utcnow()

    existing_forecast = (
        db.query(ForecastRun)
        .filter(ForecastRun.channel == canonical_channel)
        .filter(ForecastRun.forecast_date == forecast_start_datetime)
        .first()
    )

    operation = "created"
    if existing_forecast:
        existing_forecast.predicted_value = prediction_batch["total_predicted_value"]
        existing_forecast.model_version = prediction_batch["model_version"]
        existing_forecast.created_at = now_utc
        header_forecast = existing_forecast
        operation = "updated"
        db.flush()

        (
            db.query(ForecastIntervalRun)
            .filter(ForecastIntervalRun.forecast_run_id == header_forecast.id)
            .delete(synchronize_session=False)
        )
    else:
        header_forecast = ForecastRun(
            channel=canonical_channel,
            forecast_date=forecast_start_datetime,
            predicted_value=prediction_batch["total_predicted_value"],
            model_version=prediction_batch["model_version"],
            created_at=now_utc,
        )
        db.add(header_forecast)
        db.flush()

    interval_rows = []
    for item in prediction_batch["intervals"]:
        interval_rows.append(
            ForecastIntervalRun(
                forecast_run_id=header_forecast.id,
                channel=canonical_channel,
                forecast_date=item["forecast_date"],
                forecast_datetime=item["forecast_datetime"],
                interval_time=item["interval_time"],
                slot_index=item["slot_index"],
                shift_label=item["shift_label"],
                predicted_value=item["predicted_value"],
                model_version=item["model_version"],
                created_at=now_utc,
            )
        )

    db.add_all(interval_rows)
    db.commit()
    db.refresh(header_forecast)

    persisted_intervals = (
        db.query(ForecastIntervalRun)
        .filter(ForecastIntervalRun.forecast_run_id == header_forecast.id)
        .order_by(ForecastIntervalRun.slot_index.asc())
        .all()
    )

    return {
        "id": header_forecast.id,
        "channel": header_forecast.channel,
        "forecast_date": prediction_batch["forecast_date"],
        "forecast_start_datetime": forecast_start_datetime,
        "total_predicted_value": header_forecast.predicted_value,
        "intervals_generated": len(persisted_intervals),
        "model_version": header_forecast.model_version,
        "created_at": header_forecast.created_at,
        "operation": operation,
        "message": (
            f"Forecast operativo por intervalos {operation} correctamente para el canal {header_forecast.channel}."
        ),
        "intervals": [_serialize_interval_row(row) for row in persisted_intervals],
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

    return [_serialize_forecast_header(row) for row in rows]


def get_interval_forecast_history(
    db: Session,
    channel: str | None = None,
    forecast_date: date | None = None,
    limit: int = 2000,
):
    query = db.query(ForecastIntervalRun)

    if channel:
        canonical_channel = canonicalize_channel(channel)
        query = query.filter(ForecastIntervalRun.channel == canonical_channel)

    if forecast_date:
        query = query.filter(ForecastIntervalRun.forecast_date == forecast_date)
        rows = (
            query.order_by(
                ForecastIntervalRun.forecast_date.desc(),
                ForecastIntervalRun.slot_index.asc(),
            )
            .limit(limit)
            .all()
        )
        return [_serialize_interval_row(row) for row in rows]

    latest_row_query = db.query(ForecastIntervalRun)
    if channel:
        latest_row_query = latest_row_query.filter(ForecastIntervalRun.channel == canonical_channel)

    latest_row = (
        latest_row_query
        .order_by(ForecastIntervalRun.created_at.desc(), ForecastIntervalRun.forecast_datetime.desc())
        .first()
    )

    if latest_row is None:
        return []

    rows = (
        query.filter(ForecastIntervalRun.forecast_run_id == latest_row.forecast_run_id)
        .order_by(ForecastIntervalRun.slot_index.asc())
        .limit(limit)
        .all()
    )

    return [_serialize_interval_row(row) for row in rows]
