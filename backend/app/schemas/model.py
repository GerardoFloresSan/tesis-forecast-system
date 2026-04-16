from datetime import date, datetime, time

from pydantic import BaseModel


class ModelMetricsResponse(BaseModel):
    model_name: str
    total_rows: int
    train_rows: int
    test_rows: int
    mae: float
    rmse: float
    r2: float


class BaselinePredictionRequest(BaseModel):
    interaction_date: date
    interval_time: time
    channel: str
    aht: float = 0.0
    is_holiday: float = 0.0
    is_holiday_peru: float | None = None
    is_holiday_spain: float = 0.0
    is_holiday_mexico: float = 0.0
    campaign_day: float = 0.0
    absenteeism_rate: float = 0.0


class BaselinePredictionResponse(BaseModel):
    model_name: str
    predicted_volume: float


class ModelSavedResponse(BaseModel):
    model_name: str
    file_path: str
    total_rows: int
    train_rows: int
    test_rows: int
    mae: float
    rmse: float
    r2: float


class LstmTrainResponse(BaseModel):
    message: str
    run_id: int
    run_type: str
    status: str
    channel: str
    mae: float
    rmse: float
    mape: float
    r2: float
    train_size: int
    test_size: int
    model_path: str
    scaler_path: str
    metadata_path: str
    metrics_path: str


class LstmMetricsResponse(BaseModel):
    channel: str
    mae: float
    rmse: float
    mape: float
    r2: float
    train_size: int
    test_size: int
    model_path: str
    scaler_path: str
    metadata_path: str
    metrics_path: str


class LstmStatusResponse(BaseModel):
    channel: str
    model_exists: bool
    scaler_exists: bool
    metadata_exists: bool
    metrics_exists: bool


class LstmHistoryResponse(BaseModel):
    id: int
    channel: str
    run_type: str
    status: str
    mae: float | None = None
    rmse: float | None = None
    mape: float | None = None
    r2: float | None = None
    train_size: int | None = None
    test_size: int | None = None
    model_path: str | None = None
    scaler_path: str | None = None
    metadata_path: str | None = None
    metrics_path: str | None = None
    error_message: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    created_at: datetime


class LstmCheckRetrainResponse(BaseModel):
    channel: str
    threshold_mape: float
    current_mape: float | None = None
    should_retrain: bool
    action_taken: str
    message: str
    run_id: int | None = None
    run_type: str | None = None
    status: str | None = None


class SchedulerJobResponse(BaseModel):
    id: str
    trigger: str
    next_run_time: str | None = None


class SchedulerStatusResponse(BaseModel):
    running: bool
    jobs: list[SchedulerJobResponse]


class SchedulerJobRunResponse(BaseModel):
    id: int
    job_name: str
    job_type: str
    channel: str | None = None
    status: str
    action_taken: str
    message: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    created_at: datetime


class LatestForecastResponse(BaseModel):
    id: int
    channel: str
    forecast_date: datetime
    predicted_value: float
    model_version: str | None = None
    created_at: datetime


class LatestTrainRunResponse(BaseModel):
    id: int
    channel: str
    run_type: str
    status: str
    mae: float | None = None
    rmse: float | None = None
    mape: float | None = None
    r2: float | None = None
    created_at: datetime


class LatestSchedulerJobResponse(BaseModel):
    id: int
    job_name: str
    job_type: str
    channel: str | None = None
    status: str
    action_taken: str
    message: str | None = None
    created_at: datetime


class SystemSummaryResponse(BaseModel):
    channel: str
    scheduler_status: SchedulerStatusResponse
    lstm_status: LstmStatusResponse
    lstm_metrics: LstmMetricsResponse | None = None
    latest_forecast: LatestForecastResponse | None = None
    latest_train_run: LatestTrainRunResponse | None = None
    latest_scheduler_job: LatestSchedulerJobResponse | None = None
