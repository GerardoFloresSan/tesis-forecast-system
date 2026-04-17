from datetime import datetime

from app.models.forecast_run import ForecastRun
from app.models.model_train_run import ModelTrainRun
from app.models.scheduler_job_run import SchedulerJobRun


def test_get_lstm_status_returns_artifact_flags(client, monkeypatch):
    monkeypatch.setattr(
        "app.routers.model.get_lstm_status",
        lambda channel: {
            "channel": channel,
            "model_exists": True,
            "scaler_exists": True,
            "metadata_exists": True,
            "metrics_exists": True,
        },
    )

    response = client.get("/model/lstm-status", params={"channel": "España"})
    assert response.status_code == 200

    body = response.json()
    assert body["channel"] == "España"
    assert body["model_exists"] is True
    assert body["scaler_exists"] is True
    assert body["metadata_exists"] is True
    assert body["metrics_exists"] is True


def test_get_lstm_metrics_returns_operational_metrics(client, monkeypatch):
    monkeypatch.setattr(
        "app.routers.model.get_lstm_metrics",
        lambda channel: {
            "channel": channel,
            "mae": 10.8551,
            "rmse": 13.9261,
            "mape": 14.7236,
            "r2": 0.7831,
            "train_size": 6592,
            "test_size": 2035,
            "model_path": "backend/data/models/lstm_espana.keras",
            "scaler_path": "backend/data/models/lstm_espana_scaler.joblib",
            "metadata_path": "backend/data/models/lstm_espana_metadata.joblib",
            "metrics_path": "backend/data/models/lstm_espana_metrics.json",
            "wape": 12.623,
            "smape": 13.8907,
            "bias": -0.483,
            "best_val_loss": 0.060438,
            "time_steps": 34,
            "slots_per_day": 34,
            "model_version": "lstm_espana_v2_operational",
            "baseline": {
                "name": "naive_previous_operational_day",
                "mae": 21.9892,
                "rmse": 27.071,
                "mape": 30.456,
                "wape": 25.5668,
                "smape": 28.6413,
                "bias": 0.0934,
            },
        },
    )

    response = client.get("/model/lstm-metrics", params={"channel": "España"})
    assert response.status_code == 200

    body = response.json()
    assert body["channel"] == "España"
    assert body["mape"] == 14.7236
    assert body["time_steps"] == 34
    assert body["slots_per_day"] == 34
    assert body["model_version"] == "lstm_espana_v2_operational"
    assert body["baseline"]["mape"] == 30.456


def test_get_scheduler_status_returns_current_state(client, monkeypatch):
    monkeypatch.setattr(
        "app.routers.model.get_scheduler_status",
        lambda: {
            "running": False,
            "jobs": [
                {
                    "id": "auto_forecast_daily",
                    "trigger": "cron[hour='0', minute='5']",
                    "next_run_time": None,
                }
            ],
        },
    )

    response = client.get("/model/scheduler-status")
    assert response.status_code == 200

    body = response.json()
    assert body["running"] is False
    assert len(body["jobs"]) == 1
    assert body["jobs"][0]["id"] == "auto_forecast_daily"


def test_get_scheduler_job_history_returns_latest_rows(db_session, client):
    older = SchedulerJobRun(
        job_name="auto_retrain_lstm",
        job_type="retrain_check",
        channel="Choice",
        status="success",
        action_taken="none",
        message="No se requirió reentrenamiento.",
        started_at=datetime(2026, 4, 16, 8, 0, 0),
        finished_at=datetime(2026, 4, 16, 8, 0, 5),
        created_at=datetime(2026, 4, 16, 8, 0, 5),
    )
    latest = SchedulerJobRun(
        job_name="auto_forecast_daily",
        job_type="forecast",
        channel="Choice",
        status="success",
        action_taken="created_forecast",
        message="Forecast creado correctamente.",
        started_at=datetime(2026, 4, 17, 0, 5, 0),
        finished_at=datetime(2026, 4, 17, 0, 5, 4),
        created_at=datetime(2026, 4, 17, 0, 5, 4),
    )

    db_session.add_all([older, latest])
    db_session.commit()

    response = client.get("/model/scheduler-job-history", params={"limit": 10})
    assert response.status_code == 200

    body = response.json()
    assert len(body) == 2
    assert body[0]["job_name"] == "auto_forecast_daily"
    assert body[0]["action_taken"] == "created_forecast"
    assert body[1]["job_name"] == "auto_retrain_lstm"


def test_get_system_summary_returns_latest_operational_context(db_session, client, monkeypatch):
    db_session.add(
        ForecastRun(
            channel="Choice",
            forecast_date=datetime(2026, 3, 1, 0, 0, 0),
            predicted_value=1371.4959,
            model_version="lstm_choice_v2_operational",
            created_at=datetime(2026, 4, 17, 13, 3, 14),
        )
    )
    db_session.add(
        ModelTrainRun(
            channel="Choice",
            run_type="train",
            status="success",
            mae=6.7884,
            rmse=8.4361,
            mape=18.4937,
            r2=0.7294,
            created_at=datetime(2026, 4, 16, 17, 15, 51),
        )
    )
    db_session.add(
        SchedulerJobRun(
            job_name="auto_forecast_daily",
            job_type="forecast",
            channel="Choice",
            status="success",
            action_taken="updated_forecast",
            message="Forecast actualizado correctamente.",
            started_at=datetime(2026, 4, 17, 0, 5, 0),
            finished_at=datetime(2026, 4, 17, 0, 5, 4),
            created_at=datetime(2026, 4, 17, 0, 5, 4),
        )
    )
    db_session.commit()

    monkeypatch.setattr(
        "app.services.system_summary_service.get_lstm_status",
        lambda channel: {
            "channel": channel,
            "model_exists": True,
            "scaler_exists": True,
            "metadata_exists": True,
            "metrics_exists": True,
        },
    )
    monkeypatch.setattr(
        "app.services.system_summary_service.get_lstm_metrics",
        lambda channel: {
            "channel": channel,
            "mae": 6.7884,
            "rmse": 8.4361,
            "mape": 18.4937,
            "r2": 0.7294,
            "train_size": 6592,
            "test_size": 2035,
            "model_path": "backend/data/models/lstm_choice.keras",
            "scaler_path": "backend/data/models/lstm_choice_scaler.joblib",
            "metadata_path": "backend/data/models/lstm_choice_metadata.joblib",
            "metrics_path": "backend/data/models/lstm_choice_metrics.json",
            "wape": 14.2093,
            "smape": 15.9952,
            "bias": -0.221,
            "best_val_loss": 0.071231,
            "time_steps": 34,
            "slots_per_day": 34,
            "model_version": "lstm_choice_v2_operational",
            "baseline": {
                "name": "naive_previous_operational_day",
                "mae": 10.0,
                "rmse": 12.0,
                "mape": 28.0512,
                "wape": 20.0,
                "smape": 21.0,
                "bias": 0.15,
            },
        },
    )
    monkeypatch.setattr(
        "app.services.system_summary_service.get_scheduler_status",
        lambda: {
            "running": False,
            "jobs": [],
        },
    )

    response = client.get("/model/system-summary", params={"channel": "Choice"})
    assert response.status_code == 200

    body = response.json()
    assert body["channel"] == "Choice"
    assert body["lstm_status"]["model_exists"] is True
    assert body["lstm_metrics"]["model_version"] == "lstm_choice_v2_operational"
    assert body["latest_forecast"]["model_version"] == "lstm_choice_v2_operational"
    assert body["latest_train_run"]["run_type"] == "train"
    assert body["latest_scheduler_job"]["action_taken"] == "updated_forecast"


def test_get_system_summary_returns_400_when_service_fails(client, monkeypatch):
    def _raise_error(db, channel):
        raise ValueError("Canal no soportado para system summary.")

    monkeypatch.setattr("app.routers.model.get_system_summary", _raise_error)

    response = client.get("/model/system-summary", params={"channel": "Mexico"})
    assert response.status_code == 400
    assert response.json()["detail"] == "Canal no soportado para system summary."