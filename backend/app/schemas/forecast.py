from datetime import date, time, datetime
from pydantic import BaseModel, Field


class ForecastDatasetRow(BaseModel):
    interaction_date: date
    interval_time: time
    channel: str
    volume: int
    aht: float | None = None
    is_holiday: float = 0.0
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