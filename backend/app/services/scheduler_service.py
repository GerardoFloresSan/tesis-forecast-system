import logging
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import SessionLocal
from app.models.scheduler_job_run import SchedulerJobRun
from app.services.lstm_training_service import check_and_retrain_lstm
from app.services.forecast_service import create_daily_forecast

logger = logging.getLogger(__name__)

scheduler = BackgroundScheduler()


def _create_scheduler_job_log(
    db: Session,
    job_name: str,
    job_type: str,
    channel: str | None,
    status: str,
    action_taken: str,
    message: str | None,
    started_at: datetime,
    finished_at: datetime,
):
    log = SchedulerJobRun(
        job_name=job_name,
        job_type=job_type,
        channel=channel,
        status=status,
        action_taken=action_taken,
        message=message,
        started_at=started_at,
        finished_at=finished_at,
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log


def run_auto_retrain_job():
    db = SessionLocal()
    started_at = datetime.utcnow()

    try:
        result = check_and_retrain_lstm(
            db=db,
            channel=settings.AUTO_RETRAIN_CHANNEL,
            threshold_mape=settings.AUTO_RETRAIN_THRESHOLD_MAPE,
        )

        finished_at = datetime.utcnow()

        _create_scheduler_job_log(
            db=db,
            job_name="auto_retrain_lstm",
            job_type="retrain_check",
            channel=settings.AUTO_RETRAIN_CHANNEL,
            status="success",
            action_taken=result["action_taken"],
            message=result["message"],
            started_at=started_at,
            finished_at=finished_at,
        )

        logger.info("Auto retrain job ejecutado correctamente: %s", result)

    except Exception as e:
        finished_at = datetime.utcnow()

        try:
            _create_scheduler_job_log(
                db=db,
                job_name="auto_retrain_lstm",
                job_type="retrain_check",
                channel=settings.AUTO_RETRAIN_CHANNEL,
                status="failed",
                action_taken="error",
                message=str(e),
                started_at=started_at,
                finished_at=finished_at,
            )
        except Exception:
            logger.exception("No se pudo registrar el error del auto retrain job en BD.")

        logger.exception("Error en auto retrain job: %s", str(e))

    finally:
        db.close()


def run_auto_forecast_job():
    db = SessionLocal()
    started_at = datetime.utcnow()

    try:
        result = create_daily_forecast(
            db=db,
            channel=settings.AUTO_FORECAST_CHANNEL,
        )

        finished_at = datetime.utcnow()

        action_taken = "created_forecast" if result["operation"] == "created" else "updated_forecast"

        _create_scheduler_job_log(
            db=db,
            job_name="auto_forecast_daily",
            job_type="forecast",
            channel=settings.AUTO_FORECAST_CHANNEL,
            status="success",
            action_taken=action_taken,
            message=(
                f"Forecast {result['operation']} con id={result['id']} y "
                f"total={result['total_predicted_value']} en {result['intervals_generated']} slots"
            ),
            started_at=started_at,
            finished_at=finished_at,
        )

        logger.info("Auto forecast job ejecutado correctamente: %s", result)

    except Exception as e:
        finished_at = datetime.utcnow()

        try:
            _create_scheduler_job_log(
                db=db,
                job_name="auto_forecast_daily",
                job_type="forecast",
                channel=settings.AUTO_FORECAST_CHANNEL,
                status="failed",
                action_taken="error",
                message=str(e),
                started_at=started_at,
                finished_at=finished_at,
            )
        except Exception:
            logger.exception("No se pudo registrar el error del auto forecast job en BD.")

        logger.exception("Error en auto forecast job: %s", str(e))

    finally:
        db.close()


def start_scheduler():
    if scheduler.running:
        logger.info("Scheduler ya estaba en ejecución.")
        return

    if settings.AUTO_RETRAIN_ENABLED:
        scheduler.add_job(
            run_auto_retrain_job,
            trigger="interval",
            minutes=settings.AUTO_RETRAIN_INTERVAL_MINUTES,
            id="auto_retrain_lstm",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )

    if settings.AUTO_FORECAST_ENABLED:
        scheduler.add_job(
            run_auto_forecast_job,
            trigger="interval",
            minutes=settings.AUTO_FORECAST_INTERVAL_MINUTES,
            id="auto_forecast_daily",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )

    scheduler.start()
    logger.info(
        "Scheduler iniciado. Jobs activos: %s",
        [job.id for job in scheduler.get_jobs()]
    )


def shutdown_scheduler():
    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("Scheduler detenido correctamente.")


def get_scheduler_status() -> dict:
    jobs = []

    for job in scheduler.get_jobs():
        jobs.append(
            {
                "id": job.id,
                "trigger": str(job.trigger),
                "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
            }
        )

    return {
        "running": scheduler.running,
        "jobs": jobs,
    }


def get_scheduler_job_history(limit: int = 50) -> list[dict]:
    db = SessionLocal()
    try:
        rows = (
            db.query(SchedulerJobRun)
            .order_by(SchedulerJobRun.created_at.desc())
            .limit(limit)
            .all()
        )

        return [
            {
                "id": row.id,
                "job_name": row.job_name,
                "job_type": row.job_type,
                "channel": row.channel,
                "status": row.status,
                "action_taken": row.action_taken,
                "message": row.message,
                "started_at": row.started_at,
                "finished_at": row.finished_at,
                "created_at": row.created_at,
            }
            for row in rows
        ]
    finally:
        db.close()