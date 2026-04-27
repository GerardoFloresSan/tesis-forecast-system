import math
from datetime import date, datetime, timedelta

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.external_variable import ExternalVariable
from app.models.forecast_interval_run import ForecastIntervalRun
from app.models.forecast_run import ForecastRun
from app.models.historical_interaction import HistoricalInteraction
from app.services.lstm_service import (
    predict_next_operational_day_for_channel,
    predict_operational_range_for_channel,
)
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


def _calculate_required_agents(
    forecast: float | None,
    aht: float | None,
    slot_duration_seconds: int = 1800,
    concurrency: int = 4,
) -> int:
    forecast_value = float(forecast or 0)
    aht_value = float(aht or 0)

    if forecast_value <= 0 or aht_value <= 0:
        return 0

    workload_seconds = forecast_value * aht_value
    required_agents = workload_seconds / slot_duration_seconds / concurrency

    return math.ceil(required_agents)


def _get_latest_aht_for_interval(
    db: Session,
    channel: str,
    interval_time,
) -> float | None:
    row = (
        db.query(HistoricalInteraction.aht)
        .filter(HistoricalInteraction.channel == channel)
        .filter(HistoricalInteraction.interval_time == interval_time)
        .filter(HistoricalInteraction.aht.isnot(None))
        .order_by(HistoricalInteraction.interaction_date.desc())
        .first()
    )

    if not row:
        return None

    return float(row[0])


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
        "aht": row.aht,
        "required_agents": row.required_agents,
        "model_version": row.model_version,
        "created_at": row.created_at,
    }


def _forecast_day_exists(
    db: Session,
    channel: str,
    forecast_date: date,
) -> bool:
    existing_interval = (
        db.query(ForecastIntervalRun.id)
        .filter(ForecastIntervalRun.channel == channel)
        .filter(ForecastIntervalRun.forecast_date == forecast_date)
        .first()
    )

    return existing_interval is not None



def _build_date_range(start_date: date, end_date: date) -> list[date]:
    days: list[date] = []
    current_date = start_date

    while current_date <= end_date:
        days.append(current_date)
        current_date = current_date + timedelta(days=1)

    return days

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


def _persist_prediction_batch(db: Session, prediction_batch: dict) -> dict:
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
        aht_value = item.get("aht")

        if aht_value is None:
            aht_value = _get_latest_aht_for_interval(
                db=db,
                channel=canonical_channel,
                interval_time=item["interval_time"],
            )

        required_agents = _calculate_required_agents(
            forecast=item["predicted_value"],
            aht=aht_value,
        )

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
                aht=aht_value,
                required_agents=required_agents,
                model_version=item["model_version"],
                created_at=now_utc,
            )
        )

    db.add_all(interval_rows)
    db.flush()
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
        "message": f"Forecast operativo por intervalos {operation} correctamente para el canal {header_forecast.channel}.",
        "intervals": [_serialize_interval_row(row) for row in persisted_intervals],
    }


def create_daily_forecast(db: Session, channel: str):
    prediction_batch = predict_next_operational_day_for_channel(db, channel)
    result = _persist_prediction_batch(db, prediction_batch)
    db.commit()
    return result


def create_monthly_forecast(
    db: Session,
    channel: str,
    start_date: date,
    end_date: date,
):
    canonical_channel = canonicalize_channel(channel)

    if end_date < start_date:
        raise ValueError("La fecha fin no puede ser menor que la fecha inicio.")

    prediction_batches = predict_operational_range_for_channel(
        db=db,
        channel=canonical_channel,
        start_date=start_date,
        end_date=end_date,
    )

    forecasts = []
    skipped_dates: list[str] = []

    try:
        for prediction_batch in prediction_batches:
            forecast_day = prediction_batch["forecast_date"]

            if _forecast_day_exists(
                db=db,
                channel=canonical_channel,
                forecast_date=forecast_day,
            ):
                skipped_dates.append(str(forecast_day))
                continue

            forecasts.append(_persist_prediction_batch(db, prediction_batch))

        db.commit()
    except Exception:
        db.rollback()
        raise

    total_predicted_value = float(sum(item["total_predicted_value"] for item in forecasts))
    intervals_generated = int(sum(item["intervals_generated"] for item in forecasts))
    days_generated = len(forecasts)
    days_skipped = len(skipped_dates)
    days_requested = len(prediction_batches)

    if days_generated > 0 and days_skipped > 0:
        operation = "monthly_forecast_partially_generated"
        message = (
            f"Forecast mensual generado parcialmente para el canal {canonical_channel}. "
            f"Se generaron {days_generated} día(s) nuevo(s) y se omitieron {days_skipped} día(s) porque ya existían."
        )
    elif days_generated > 0:
        operation = "monthly_forecast_generated"
        message = (
            f"Forecast mensual generado correctamente para el canal {canonical_channel} "
            f"desde {start_date} hasta {end_date}."
        )
    else:
        operation = "monthly_forecast_already_exists"
        message = (
            f"No se generaron nuevos registros. El forecast para el canal {canonical_channel} "
            f"en el rango {start_date} al {end_date} ya existe."
        )

    return {
        "channel": canonical_channel,
        "start_date": start_date,
        "end_date": end_date,
        "days_requested": days_requested,
        "days_generated": days_generated,
        "days_skipped": days_skipped,
        "skipped_dates": skipped_dates,
        "intervals_generated": intervals_generated,
        "total_predicted_value": total_predicted_value,
        "operation": operation,
        "message": message,
        "forecasts": forecasts,
    }


def get_monthly_forecast_status(
    db: Session,
    channel: str,
    start_date: date,
    end_date: date,
):
    canonical_channel = canonicalize_channel(channel)

    if end_date < start_date:
        raise ValueError("La fecha fin no puede ser menor que la fecha inicio.")

    requested_dates = _build_date_range(start_date, end_date)

    existing_rows = (
        db.query(
            ForecastIntervalRun.forecast_date,
            func.count(ForecastIntervalRun.id).label("intervals_count"),
        )
        .filter(ForecastIntervalRun.channel == canonical_channel)
        .filter(ForecastIntervalRun.forecast_date >= start_date)
        .filter(ForecastIntervalRun.forecast_date <= end_date)
        .group_by(ForecastIntervalRun.forecast_date)
        .order_by(ForecastIntervalRun.forecast_date.asc())
        .all()
    )

    intervals_by_date = {row.forecast_date: int(row.intervals_count) for row in existing_rows}
    existing_dates = [day for day in requested_dates if intervals_by_date.get(day, 0) > 0]
    missing_dates = [day for day in requested_dates if intervals_by_date.get(day, 0) == 0]

    requested_days = len(requested_dates)
    generated_days = len(existing_dates)
    missing_days = len(missing_dates)
    total_intervals = sum(intervals_by_date.values())
    coverage_percentage = round((generated_days / requested_days) * 100, 2) if requested_days else 0.0

    if generated_days == 0:
        status = "not_generated"
        message = (
            f"No existe forecast generado para el canal {canonical_channel} "
            f"en el rango {start_date} al {end_date}."
        )
    elif missing_days == 0:
        status = "complete"
        message = (
            f"El forecast mensual ya está completo para el canal {canonical_channel} "
            f"en el rango {start_date} al {end_date}."
        )
    else:
        status = "partial"
        message = (
            f"El forecast mensual está incompleto para el canal {canonical_channel}. "
            f"Existen {generated_days} día(s) generados y faltan {missing_days} día(s)."
        )

    return {
        "channel": canonical_channel,
        "start_date": start_date,
        "end_date": end_date,
        "requested_days": requested_days,
        "generated_days": generated_days,
        "missing_days": missing_days,
        "total_intervals": total_intervals,
        "coverage_percentage": coverage_percentage,
        "status": status,
        "message": message,
        "existing_dates": [str(day) for day in existing_dates],
        "missing_dates": [str(day) for day in missing_dates],
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
    start_date: date | None = None,
    end_date: date | None = None,
    limit: int = 2000,
):
    query = db.query(ForecastIntervalRun)

    canonical_channel = None
    if channel:
        canonical_channel = canonicalize_channel(channel)
        query = query.filter(ForecastIntervalRun.channel == canonical_channel)

    if start_date and end_date:
        if end_date < start_date:
            raise ValueError("La fecha fin no puede ser menor que la fecha inicio.")

        rows = (
            query.filter(ForecastIntervalRun.forecast_date >= start_date)
            .filter(ForecastIntervalRun.forecast_date <= end_date)
            .order_by(
                ForecastIntervalRun.forecast_date.asc(),
                ForecastIntervalRun.slot_index.asc(),
            )
            .limit(limit)
            .all()
        )

        return [_serialize_interval_row(row) for row in rows]

    if forecast_date:
        rows = (
            query.filter(ForecastIntervalRun.forecast_date == forecast_date)
            .order_by(
                ForecastIntervalRun.forecast_date.asc(),
                ForecastIntervalRun.slot_index.asc(),
            )
            .limit(limit)
            .all()
        )

        return [_serialize_interval_row(row) for row in rows]

    latest_row_query = db.query(ForecastIntervalRun)

    if canonical_channel:
        latest_row_query = latest_row_query.filter(ForecastIntervalRun.channel == canonical_channel)

    latest_row = (
        latest_row_query
        .order_by(
            ForecastIntervalRun.created_at.desc(),
            ForecastIntervalRun.forecast_datetime.desc(),
        )
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
