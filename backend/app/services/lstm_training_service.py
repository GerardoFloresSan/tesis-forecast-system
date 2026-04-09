import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from sqlalchemy.orm import Session

from app.models.model_train_run import ModelTrainRun


BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "data" / "models"


def _validate_channel(channel: str) -> str:
    channel = channel.strip()

    if channel != "Choice":
        raise ValueError(
            "Por ahora el entrenamiento y reentrenamiento vía API está habilitado para el canal 'Choice'."
        )

    return channel


def _build_paths(channel: str) -> dict:
    channel = _validate_channel(channel)
    channel_slug = channel.lower()

    return {
        "model_path": MODEL_DIR / f"lstm_{channel_slug}.keras",
        "scaler_path": MODEL_DIR / f"lstm_{channel_slug}_scaler.joblib",
        "metadata_path": MODEL_DIR / f"lstm_{channel_slug}_metadata.joblib",
        "metrics_path": MODEL_DIR / f"lstm_{channel_slug}_metrics.json",
    }


def _resolve_python_executable() -> str:
    if os.name == "nt":
        venv_python = BASE_DIR / ".venv" / "Scripts" / "python.exe"
        if venv_python.exists():
            return str(venv_python)

    return sys.executable


def _create_run_record(db: Session, channel: str, run_type: str) -> ModelTrainRun:
    run = ModelTrainRun(
        channel=channel,
        run_type=run_type,
        status="running",
        started_at=datetime.utcnow(),
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def _mark_run_success(db: Session, run: ModelTrainRun, metrics: dict):
    run.status = "success"
    run.mae = metrics["mae"]
    run.rmse = metrics["rmse"]
    run.mape = metrics["mape"]
    run.r2 = metrics["r2"]
    run.train_size = metrics["train_size"]
    run.test_size = metrics["test_size"]
    run.model_path = metrics["model_path"]
    run.scaler_path = metrics["scaler_path"]
    run.metadata_path = metrics["metadata_path"]
    run.metrics_path = metrics["metrics_path"]
    run.error_message = None
    run.finished_at = datetime.utcnow()

    db.commit()
    db.refresh(run)


def _mark_run_failed(db: Session, run: ModelTrainRun, error_message: str):
    run.status = "failed"
    run.error_message = error_message
    run.finished_at = datetime.utcnow()

    db.commit()
    db.refresh(run)


def get_lstm_status(channel: str = "Choice") -> dict:
    paths = _build_paths(channel)

    return {
        "channel": channel,
        "model_exists": paths["model_path"].exists(),
        "scaler_exists": paths["scaler_path"].exists(),
        "metadata_exists": paths["metadata_path"].exists(),
        "metrics_exists": paths["metrics_path"].exists(),
    }


def get_lstm_metrics(channel: str = "Choice") -> dict:
    paths = _build_paths(channel)

    if not paths["metrics_path"].exists():
        raise ValueError(
            f"No existe archivo de métricas para el canal '{channel}'. "
            "Primero ejecuta /model/train-lstm o corre el script de entrenamiento."
        )

    with open(paths["metrics_path"], "r", encoding="utf-8") as f:
        metrics = json.load(f)

    return {
        "channel": metrics["channel"],
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "mape": metrics["mape"],
        "r2": metrics["r2"],
        "train_size": metrics["train_size"],
        "test_size": metrics["test_size"],
        "model_path": metrics["model_path"],
        "scaler_path": metrics["scaler_path"],
        "metadata_path": metrics["metadata_path"],
        "metrics_path": str(paths["metrics_path"]),
    }


def train_lstm_model(db: Session, channel: str = "Choice", run_type: str = "train") -> dict:
    channel = _validate_channel(channel)
    run = _create_run_record(db, channel, run_type)

    python_executable = _resolve_python_executable()
    command = [python_executable, "-m", "scripts.train_lstm"]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    result = subprocess.run(
        command,
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
        env=env,
        creationflags=creationflags,
    )

    if result.returncode != 0:
        error_message = (
            result.stderr.strip()
            or result.stdout.strip()
            or "Error desconocido al entrenar LSTM."
        )
        _mark_run_failed(db, run, error_message)
        raise ValueError(f"Falló el entrenamiento LSTM: {error_message}")

    metrics = get_lstm_metrics(channel)
    _mark_run_success(db, run, metrics)

    return {
        "message": f"Proceso LSTM '{run_type}' ejecutado correctamente.",
        "run_id": run.id,
        "run_type": run.run_type,
        "status": run.status,
        **metrics,
    }


def retrain_lstm_model(db: Session, channel: str = "Choice") -> dict:
    return train_lstm_model(db=db, channel=channel, run_type="retrain")


def get_lstm_history(db: Session, channel: str | None = None, limit: int = 50) -> list[dict]:
    query = db.query(ModelTrainRun)

    if channel:
        query = query.filter(ModelTrainRun.channel == channel)

    rows = (
        query.order_by(ModelTrainRun.created_at.desc())
        .limit(limit)
        .all()
    )

    return [
        {
            "id": row.id,
            "channel": row.channel,
            "run_type": row.run_type,
            "status": row.status,
            "mae": row.mae,
            "rmse": row.rmse,
            "mape": row.mape,
            "r2": row.r2,
            "train_size": row.train_size,
            "test_size": row.test_size,
            "model_path": row.model_path,
            "scaler_path": row.scaler_path,
            "metadata_path": row.metadata_path,
            "metrics_path": row.metrics_path,
            "error_message": row.error_message,
            "started_at": row.started_at,
            "finished_at": row.finished_at,
            "created_at": row.created_at,
        }
        for row in rows
    ]


def check_and_retrain_lstm(
    db: Session,
    channel: str = "Choice",
    threshold_mape: float = 15.0,
) -> dict:
    channel = _validate_channel(channel)
    status = get_lstm_status(channel)

    if not all([
        status["model_exists"],
        status["scaler_exists"],
        status["metadata_exists"],
        status["metrics_exists"],
    ]):
        result = train_lstm_model(db=db, channel=channel, run_type="train")
        return {
            "channel": channel,
            "threshold_mape": threshold_mape,
            "current_mape": None,
            "should_retrain": True,
            "action_taken": "train",
            "message": "No existían artefactos completos del modelo. Se ejecutó entrenamiento inicial.",
            "run_id": result["run_id"],
            "run_type": result["run_type"],
            "status": result["status"],
        }

    metrics = get_lstm_metrics(channel)
    current_mape = metrics["mape"]

    if current_mape > threshold_mape:
        result = retrain_lstm_model(db=db, channel=channel)
        return {
            "channel": channel,
            "threshold_mape": threshold_mape,
            "current_mape": current_mape,
            "should_retrain": True,
            "action_taken": "retrain",
            "message": (
                f"El MAPE actual ({current_mape}) supera el umbral ({threshold_mape}). "
                "Se ejecutó reentrenamiento."
            ),
            "run_id": result["run_id"],
            "run_type": result["run_type"],
            "status": result["status"],
        }

    return {
        "channel": channel,
        "threshold_mape": threshold_mape,
        "current_mape": current_mape,
        "should_retrain": False,
        "action_taken": "none",
        "message": (
            f"El MAPE actual ({current_mape}) no supera el umbral ({threshold_mape}). "
            "No fue necesario reentrenar."
        ),
        "run_id": None,
        "run_type": None,
        "status": None,
    }