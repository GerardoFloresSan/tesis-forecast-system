from sqlalchemy.orm import Session

from app.models.forecast_run import ForecastRun
from app.models.model_train_run import ModelTrainRun
from app.models.scheduler_job_run import SchedulerJobRun
from app.services.lstm_training_service import get_lstm_metrics, get_lstm_status
from app.services.scheduler_service import get_scheduler_status


def get_system_summary(db: Session, channel: str = "Choice") -> dict:
    lstm_status = get_lstm_status(channel)

    lstm_metrics = None
    if lstm_status["metrics_exists"]:
        try:
            lstm_metrics = get_lstm_metrics(channel)
        except Exception:
            lstm_metrics = None

    latest_forecast_row = (
        db.query(ForecastRun)
        .filter(ForecastRun.channel == channel)
        .order_by(ForecastRun.created_at.desc())
        .first()
    )

    latest_train_run_row = (
        db.query(ModelTrainRun)
        .filter(ModelTrainRun.channel == channel)
        .order_by(ModelTrainRun.created_at.desc())
        .first()
    )

    latest_scheduler_job_row = (
        db.query(SchedulerJobRun)
        .filter(SchedulerJobRun.channel == channel)
        .order_by(SchedulerJobRun.created_at.desc())
        .first()
    )

    latest_forecast = None
    if latest_forecast_row:
        latest_forecast = {
            "id": latest_forecast_row.id,
            "channel": latest_forecast_row.channel,
            "forecast_date": latest_forecast_row.forecast_date,
            "predicted_value": latest_forecast_row.predicted_value,
            "model_version": latest_forecast_row.model_version,
            "created_at": latest_forecast_row.created_at,
        }

    latest_train_run = None
    if latest_train_run_row:
        latest_train_run = {
            "id": latest_train_run_row.id,
            "channel": latest_train_run_row.channel,
            "run_type": latest_train_run_row.run_type,
            "status": latest_train_run_row.status,
            "mae": latest_train_run_row.mae,
            "rmse": latest_train_run_row.rmse,
            "mape": latest_train_run_row.mape,
            "r2": latest_train_run_row.r2,
            "created_at": latest_train_run_row.created_at,
        }

    latest_scheduler_job = None
    if latest_scheduler_job_row:
        latest_scheduler_job = {
            "id": latest_scheduler_job_row.id,
            "job_name": latest_scheduler_job_row.job_name,
            "job_type": latest_scheduler_job_row.job_type,
            "channel": latest_scheduler_job_row.channel,
            "status": latest_scheduler_job_row.status,
            "action_taken": latest_scheduler_job_row.action_taken,
            "message": latest_scheduler_job_row.message,
            "created_at": latest_scheduler_job_row.created_at,
        }

    return {
        "channel": channel,
        "scheduler_status": get_scheduler_status(),
        "lstm_status": lstm_status,
        "lstm_metrics": lstm_metrics,
        "latest_forecast": latest_forecast,
        "latest_train_run": latest_train_run,
        "latest_scheduler_job": latest_scheduler_job,
    }