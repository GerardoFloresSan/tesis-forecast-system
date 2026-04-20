from datetime import date, datetime

from pydantic import BaseModel


class SLAAlertResponse(BaseModel):
    id: int
    channel: str
    forecast_date: date
    forecast_run_id: int | None = None
    expected_volume: float
    actual_volume: float | None = None
    deviation_percentage: float | None = None
    absolute_deviation_percentage: float | None = None
    error_level: str
    risk_level: str
    has_breach: bool
    status: str
    message: str
    email_sent: bool
    email_recipient: str | None = None
    acknowledged_by: str | None = None
    acknowledged_at: datetime | None = None
    resolved_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class SLAAlertEvaluationResponse(BaseModel):
    evaluation_status: str
    channel: str
    forecast_date: date
    has_breach: bool
    risk_level: str
    deviation_percentage: float | None = None
    alert_id: int | None = None
    email_sent: bool = False
    message: str


class SLAAlertAckResponse(BaseModel):
    id: int
    status: str
    acknowledged_by: str
    acknowledged_at: datetime
    message: str