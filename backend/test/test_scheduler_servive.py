from datetime import datetime

import app.services.scheduler_service as scheduler_service
from app.models.scheduler_job_run import SchedulerJobRun


class FakeScheduler:
    def __init__(self):
        self.running = False
        self.added_jobs = []
        self.shutdown_called = False

    def add_job(self, func, trigger, **kwargs):
        self.added_jobs.append(
            {
                "func": func,
                "trigger": trigger,
                **kwargs,
            }
        )

    def start(self):
        self.running = True

    def shutdown(self, wait=False):
        self.running = False
        self.shutdown_called = True

    def get_jobs(self):
        jobs = []
        for job in self.added_jobs:
            jobs.append(
                type(
                    "Job",
                    (),
                    {
                        "id": job["id"],
                        "trigger": f"{job['trigger']}[minutes={job.get('minutes')}]",
                        "next_run_time": None,
                    },
                )()
            )
        return jobs


def test_start_scheduler_registers_enabled_jobs_without_duplicates(monkeypatch):
    fake_scheduler = FakeScheduler()
    monkeypatch.setattr(scheduler_service, "scheduler", fake_scheduler)

    monkeypatch.setattr(scheduler_service.settings, "AUTO_RETRAIN_ENABLED", True)
    monkeypatch.setattr(scheduler_service.settings, "AUTO_RETRAIN_INTERVAL_MINUTES", 1)
    monkeypatch.setattr(scheduler_service.settings, "AUTO_FORECAST_ENABLED", True)
    monkeypatch.setattr(scheduler_service.settings, "AUTO_FORECAST_INTERVAL_MINUTES", 1)

    scheduler_service.start_scheduler()
    scheduler_service.start_scheduler()

    assert fake_scheduler.running is True
    assert len(fake_scheduler.added_jobs) == 2

    job_ids = [job["id"] for job in fake_scheduler.added_jobs]
    assert "auto_retrain_lstm" in job_ids
    assert "auto_forecast_daily" in job_ids

    scheduler_service.shutdown_scheduler()
    assert fake_scheduler.running is False
    assert fake_scheduler.shutdown_called is True


def test_run_auto_forecast_job_logs_success(db_session, monkeypatch):
    monkeypatch.setattr(scheduler_service.settings, "AUTO_FORECAST_CHANNEL", "España")

    monkeypatch.setattr(
        scheduler_service,
        "create_daily_forecast",
        lambda db, channel: {
            "id": 12,
            "channel": channel,
            "forecast_date": "2026-03-01",
            "forecast_start_datetime": "2026-03-01T00:00:00",
            "total_predicted_value": 1954.7296649925909,
            "intervals_generated": 34,
            "model_version": "lstm_espana_v2_operational",
            "operation": "updated",
        },
    )

    scheduler_service.run_auto_forecast_job()

    row = db_session.query(SchedulerJobRun).order_by(SchedulerJobRun.created_at.desc()).first()

    assert row is not None
    assert row.job_name == "auto_forecast_daily"
    assert row.job_type == "forecast"
    assert row.channel == "España"
    assert row.status == "success"
    assert row.action_taken == "updated_forecast"
    assert "id=12" in (row.message or "")
    assert "34 slots" in (row.message or "")


def test_run_auto_forecast_job_logs_failure(db_session, monkeypatch):
    monkeypatch.setattr(scheduler_service.settings, "AUTO_FORECAST_CHANNEL", "España")

    def _raise_error(db, channel):
        raise RuntimeError("Fallo controlado del forecast scheduler.")

    monkeypatch.setattr(scheduler_service, "create_daily_forecast", _raise_error)

    scheduler_service.run_auto_forecast_job()

    row = db_session.query(SchedulerJobRun).order_by(SchedulerJobRun.created_at.desc()).first()

    assert row is not None
    assert row.job_name == "auto_forecast_daily"
    assert row.job_type == "forecast"
    assert row.channel == "España"
    assert row.status == "failed"
    assert row.action_taken == "error"
    assert "Fallo controlado del forecast scheduler." in (row.message or "")


def test_run_auto_retrain_job_logs_success_with_none_action(db_session, monkeypatch):
    monkeypatch.setattr(scheduler_service.settings, "AUTO_RETRAIN_CHANNEL", "España")
    monkeypatch.setattr(scheduler_service.settings, "AUTO_RETRAIN_THRESHOLD_MAPE", 15.0)

    monkeypatch.setattr(
        scheduler_service,
        "check_and_retrain_lstm",
        lambda db, channel, threshold_mape: {
            "channel": channel,
            "threshold_mape": threshold_mape,
            "current_mape": 14.7236,
            "should_retrain": False,
            "action_taken": "none",
            "message": "MAPE dentro del umbral. No se requiere reentrenamiento.",
            "run_id": None,
            "run_type": None,
            "status": None,
        },
    )

    scheduler_service.run_auto_retrain_job()

    row = db_session.query(SchedulerJobRun).order_by(SchedulerJobRun.created_at.desc()).first()

    assert row is not None
    assert row.job_name == "auto_retrain_lstm"
    assert row.job_type == "retrain_check"
    assert row.channel == "España"
    assert row.status == "success"
    assert row.action_taken == "none"
    assert "No se requiere reentrenamiento" in (row.message or "")


def test_run_auto_retrain_job_logs_failure(db_session, monkeypatch):
    monkeypatch.setattr(scheduler_service.settings, "AUTO_RETRAIN_CHANNEL", "España")
    monkeypatch.setattr(scheduler_service.settings, "AUTO_RETRAIN_THRESHOLD_MAPE", 15.0)

    def _raise_error(db, channel, threshold_mape):
        raise RuntimeError("Fallo controlado del retrain scheduler.")

    monkeypatch.setattr(scheduler_service, "check_and_retrain_lstm", _raise_error)

    scheduler_service.run_auto_retrain_job()

    row = db_session.query(SchedulerJobRun).order_by(SchedulerJobRun.created_at.desc()).first()

    assert row is not None
    assert row.job_name == "auto_retrain_lstm"
    assert row.job_type == "retrain_check"
    assert row.channel == "España"
    assert row.status == "failed"
    assert row.action_taken == "error"
    assert "Fallo controlado del retrain scheduler." in (row.message or "")


def test_get_scheduler_job_history_returns_rows_in_desc_order(db_session):
    older = SchedulerJobRun(
        job_name="auto_retrain_lstm",
        job_type="retrain_check",
        channel="España",
        status="success",
        action_taken="none",
        message="No se requiere reentrenamiento.",
        started_at=datetime(2026, 4, 17, 13, 0, 0),
        finished_at=datetime(2026, 4, 17, 13, 0, 5),
        created_at=datetime(2026, 4, 17, 13, 0, 5),
    )
    latest = SchedulerJobRun(
        job_name="auto_forecast_daily",
        job_type="forecast",
        channel="España",
        status="success",
        action_taken="updated_forecast",
        message="Forecast actualizado correctamente.",
        started_at=datetime(2026, 4, 17, 13, 1, 0),
        finished_at=datetime(2026, 4, 17, 13, 1, 4),
        created_at=datetime(2026, 4, 17, 13, 1, 4),
    )

    db_session.add_all([older, latest])
    db_session.commit()

    rows = scheduler_service.get_scheduler_job_history(limit=10)

    assert len(rows) == 2
    assert rows[0]["job_name"] == "auto_forecast_daily"
    assert rows[0]["action_taken"] == "updated_forecast"
    assert rows[1]["job_name"] == "auto_retrain_lstm"
    assert rows[1]["action_taken"] == "none"