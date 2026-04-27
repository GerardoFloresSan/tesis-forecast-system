from datetime import date, datetime, time

from pydantic import BaseModel, Field, model_validator


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


class ForecastMonthlyGenerateRequest(BaseModel):
    channel: str = Field(default="Choice", description="Canal a pronosticar")
    start_date: date = Field(..., description="Fecha inicial del rango mensual. Ejemplo: 2026-05-01")
    end_date: date = Field(..., description="Fecha final del rango mensual. Ejemplo: 2026-05-31")

    @model_validator(mode="after")
    def validate_date_range(self):
        if self.end_date < self.start_date:
            raise ValueError("La fecha fin no puede ser menor que la fecha inicio.")
        return self


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
    aht: float | None = None
    required_agents: int | None = None
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


class ForecastMonthlyResponse(BaseModel):
    channel: str
    start_date: date
    end_date: date
    days_requested: int = 0
    days_generated: int
    days_skipped: int = 0
    skipped_dates: list[str] = []
    intervals_generated: int
    total_predicted_value: float
    operation: str
    message: str
    forecasts: list[ForecastBatchResponse]


class ForecastMonthlyStatusResponse(BaseModel):
    channel: str
    start_date: date
    end_date: date
    requested_days: int
    generated_days: int
    missing_days: int
    total_intervals: int
    coverage_percentage: float
    status: str
    message: str
    existing_dates: list[str] = []
    missing_dates: list[str] = []


class ForecastApiResponse(BaseModel):
    date: date
    channel: str
    expected_volume: float
    actual_volume: float | None = None
    error_level: str
    risk_level: str
    deviation_percentage: float | None = None
    has_breach: bool
    model_version: str | None = None
    forecast_run_id: int
    generated_at: datetime
    intervals_generated: int
    message: str
