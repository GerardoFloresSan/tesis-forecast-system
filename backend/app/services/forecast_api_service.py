from datetime import date

from sqlalchemy.orm import Session

from app.services.forecast_monitoring_service import get_forecast_monitoring_by_date


def get_daily_forecast_api_payload(
    db: Session,
    channel: str,
    forecast_date: date,
) -> dict:
    monitoring = get_forecast_monitoring_by_date(
        db=db,
        channel=channel,
        forecast_date=forecast_date,
    )

    return {
        "date": monitoring["forecast_date"],
        "channel": monitoring["channel"],
        "expected_volume": monitoring["forecast_total"],
        "actual_volume": monitoring["actual_total"],
        "error_level": monitoring["error_level"],
        "risk_level": monitoring["risk_level"],
        "deviation_percentage": monitoring["deviation_percentage"],
        "has_breach": monitoring["has_breach"],
        "model_version": monitoring["model_version"],
        "forecast_run_id": monitoring["forecast_run_id"],
        "generated_at": monitoring["forecast_created_at"],
        "intervals_generated": monitoring["forecast_intervals_count"],
        "message": monitoring["message"],
    }