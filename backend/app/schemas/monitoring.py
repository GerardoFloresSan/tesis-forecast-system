from datetime import date, datetime

from pydantic import BaseModel


class ForecastMonitoringResponse(BaseModel):
    channel: str
    forecast_run_id: int
    forecast_date: date
    forecast_created_at: datetime
    model_version: str | None = None

    forecast_total: float
    actual_total: float | None = None

    forecast_intervals_count: int
    actual_intervals_count: int

    deviation_percentage: float | None = None
    absolute_deviation_percentage: float | None = None

    error_level: str
    risk_level: str
    has_breach: bool
    message: str