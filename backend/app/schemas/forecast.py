from datetime import date, datetime, time

from pydantic import BaseModel, Field


class ForecastDatasetRow(BaseModel):
    interaction_date: date
    interval_time: time
    channel: str
    volume: int
    aht: float | None = None
    is_holiday: float = 0.0
    is_holiday_peru: float = 0.0
    is_holiday_spain: float = 0.0
    is_holiday_mexico: float = 0.0
    is_holiday_any: int = 0
    campaign_day: float = 0.0
    absenteeism_rate: float = 0.0


class ForecastGenerateRequest(BaseModel):
    channel: str = Field(default="Choice", description="Canal a pronosticar")


class ForecastRunResponse(BaseModel):
    id: int
    channel: str
    forecast_date: datetime
    predicted_value: float
    model_version: str | None = None
    created_at: datetime


class ForecastIntervalResponse(BaseModel):
    id: int
    forecast_run_id: int
    channel: str
    forecast_date: date
    forecast_datetime: datetime
    interval_time: time
    slot_index: int
    shift_label: str
    predicted_value: float
    model_version: str | None = None
    created_at: datetime


class ForecastBatchResponse(BaseModel):
    id: int
    channel: str
    forecast_date: date
    forecast_start_datetime: datetime
    total_predicted_value: float
    intervals_generated: int
    model_version: str | None = None
    created_at: datetime
    operation: str
    message: str
    intervals: list[ForecastIntervalResponse]
