from __future__ import annotations

from datetime import date, datetime, time, timedelta

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.forecast_interval_run import ForecastIntervalRun
from app.models.forecast_run import ForecastRun
from app.models.historical_interaction import HistoricalInteraction
from app.utils.channel_rules import canonicalize_channel


def _round_or_none(value: float | None, decimals: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), decimals)


def _calculate_deviation_percentage(
    forecast_total: float,
    actual_total: float | None,
) -> float | None:
    if actual_total is None:
        return None

    if forecast_total == 0:
        if actual_total == 0:
            return 0.0
        return 100.0

    return ((actual_total - forecast_total) / forecast_total) * 100.0


def _classify_error_and_risk(
    deviation_percentage: float | None,
) -> tuple[str, str, bool]:
    if deviation_percentage is None:
        return "no_actual_data", "unknown", False

    absolute_deviation = abs(deviation_percentage)

    if absolute_deviation > 10:
        return "high_error", "critical", True

    if absolute_deviation >= 5:
        return "medium_error", "warning", False

    return "low_error", "normal", False


def _build_message(
    channel: str,
    forecast_date: date,
    deviation_percentage: float | None,
    risk_level: str,
    has_breach: bool,
) -> str:
    if deviation_percentage is None:
        return (
            f"No existen datos reales para comparar el forecast del canal {channel} "
            f"en la fecha {forecast_date.isoformat()}."
        )

    if has_breach:
        return (
            f"Se detectó una desviación crítica en el canal {channel} para la fecha "
            f"{forecast_date.isoformat()}."
        )

    if risk_level == "warning":
        return (
            f"Se detectó una desviación moderada en el canal {channel} para la fecha "
            f"{forecast_date.isoformat()}."
        )

    return (
        f"El forecast del canal {channel} para la fecha {forecast_date.isoformat()} "
        f"se mantiene dentro del rango esperado."
    )


def _get_forecast_interval_aggregation(
    db: Session,
    forecast_run_id: int,
) -> tuple[float, int]:
    predicted_total, intervals_count = (
        db.query(
            func.coalesce(func.sum(ForecastIntervalRun.predicted_value), 0.0),
            func.count(ForecastIntervalRun.id),
        )
        .filter(ForecastIntervalRun.forecast_run_id == forecast_run_id)
        .one()
    )

    return float(predicted_total or 0.0), int(intervals_count or 0)


def _get_actual_aggregation(
    db: Session,
    channel: str,
    forecast_date: date,
) -> tuple[float | None, int]:
    actual_total, actual_count = (
        db.query(
            func.coalesce(func.sum(HistoricalInteraction.volume), 0.0),
            func.count(HistoricalInteraction.id),
        )
        .filter(HistoricalInteraction.channel == channel)
        .filter(HistoricalInteraction.interaction_date == forecast_date)
        .one()
    )

    actual_count = int(actual_count or 0)
    if actual_count == 0:
        return None, 0

    return float(actual_total or 0.0), actual_count


def _serialize_monitoring_payload(
    db: Session,
    header: ForecastRun,
) -> dict:
    forecast_date = header.forecast_date.date()

    forecast_total, forecast_intervals_count = _get_forecast_interval_aggregation(
        db=db,
        forecast_run_id=header.id,
    )

    if forecast_intervals_count == 0:
        forecast_total = float(header.predicted_value or 0.0)

    actual_total, actual_intervals_count = _get_actual_aggregation(
        db=db,
        channel=header.channel,
        forecast_date=forecast_date,
    )

    deviation_percentage = _calculate_deviation_percentage(
        forecast_total=forecast_total,
        actual_total=actual_total,
    )

    error_level, risk_level, has_breach = _classify_error_and_risk(deviation_percentage)

    return {
        "channel": header.channel,
        "forecast_run_id": header.id,
        "forecast_date": forecast_date,
        "forecast_created_at": header.created_at,
        "model_version": header.model_version,
        "forecast_total": _round_or_none(forecast_total, 4) or 0.0,
        "actual_total": _round_or_none(actual_total, 4),
        "forecast_intervals_count": forecast_intervals_count,
        "actual_intervals_count": actual_intervals_count,
        "deviation_percentage": _round_or_none(deviation_percentage, 4),
        "absolute_deviation_percentage": (
            _round_or_none(abs(deviation_percentage), 4)
            if deviation_percentage is not None
            else None
        ),
        "error_level": error_level,
        "risk_level": risk_level,
        "has_breach": has_breach,
        "message": _build_message(
            channel=header.channel,
            forecast_date=forecast_date,
            deviation_percentage=deviation_percentage,
            risk_level=risk_level,
            has_breach=has_breach,
        ),
    }


def get_latest_forecast_monitoring_summary(
    db: Session,
    channel: str,
) -> dict:
    canonical_channel = canonicalize_channel(channel)

    latest_header = (
        db.query(ForecastRun)
        .filter(ForecastRun.channel == canonical_channel)
        .order_by(ForecastRun.created_at.desc(), ForecastRun.id.desc())
        .first()
    )

    if latest_header is None:
        raise ValueError(
            f"No existe forecast generado para el canal '{canonical_channel}'."
        )

    return _serialize_monitoring_payload(db=db, header=latest_header)


def get_forecast_monitoring_by_date(
    db: Session,
    channel: str,
    forecast_date: date,
) -> dict:
    canonical_channel = canonicalize_channel(channel)

    start_datetime = datetime.combine(forecast_date, time.min)
    end_datetime = start_datetime + timedelta(days=1)

    header = (
        db.query(ForecastRun)
        .filter(ForecastRun.channel == canonical_channel)
        .filter(ForecastRun.forecast_date >= start_datetime)
        .filter(ForecastRun.forecast_date < end_datetime)
        .order_by(ForecastRun.created_at.desc(), ForecastRun.id.desc())
        .first()
    )

    if header is None:
        raise ValueError(
            f"No existe forecast generado para el canal '{canonical_channel}' "
            f"en la fecha '{forecast_date.isoformat()}'."
        )

    return _serialize_monitoring_payload(db=db, header=header)