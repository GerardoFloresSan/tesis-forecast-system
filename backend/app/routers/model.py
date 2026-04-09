from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.model import (
    ModelMetricsResponse,
    BaselinePredictionRequest,
    BaselinePredictionResponse,
    ModelSavedResponse,
    LstmTrainResponse,
    LstmMetricsResponse,
    LstmStatusResponse,
    LstmHistoryResponse,
    LstmCheckRetrainResponse,
    SchedulerStatusResponse,
    SchedulerJobRunResponse,
    SystemSummaryResponse,
)
from app.services.model_service import (
    train_baseline_model,
    train_and_save_baseline_model,
    predict_with_baseline_model,
    predict_with_saved_baseline_model,
)
from app.services.lstm_training_service import (
    train_lstm_model,
    retrain_lstm_model,
    get_lstm_metrics,
    get_lstm_status,
    get_lstm_history,
    check_and_retrain_lstm,
)
from app.services.scheduler_service import (
    get_scheduler_status,
    get_scheduler_job_history,
)
from app.services.system_summary_service import get_system_summary

router = APIRouter(prefix="/model", tags=["Model"])


@router.get("/train-baseline", response_model=ModelMetricsResponse)
def train_baseline(db: Session = Depends(get_db)):
    try:
        return train_baseline_model(db)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/train-and-save-baseline", response_model=ModelSavedResponse)
def train_and_save_baseline(db: Session = Depends(get_db)):
    try:
        return train_and_save_baseline_model(db)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/predict-baseline", response_model=BaselinePredictionResponse)
def predict_baseline(
    data: BaselinePredictionRequest,
    db: Session = Depends(get_db)
):
    try:
        return predict_with_baseline_model(db, data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/predict-baseline-saved", response_model=BaselinePredictionResponse)
def predict_baseline_saved(data: BaselinePredictionRequest):
    try:
        return predict_with_saved_baseline_model(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/train-lstm", response_model=LstmTrainResponse)
def train_lstm(
    channel: str = Query(default="Choice", description="Por ahora usar Choice"),
    db: Session = Depends(get_db),
):
    try:
        return train_lstm_model(db=db, channel=channel, run_type="train")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/retrain-lstm", response_model=LstmTrainResponse)
def retrain_lstm(
    channel: str = Query(default="Choice", description="Por ahora usar Choice"),
    db: Session = Depends(get_db),
):
    try:
        return retrain_lstm_model(db=db, channel=channel)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/check-and-retrain-lstm", response_model=LstmCheckRetrainResponse)
def check_and_retrain(
    channel: str = Query(default="Choice", description="Por ahora usar Choice"),
    threshold_mape: float = Query(default=15.0, ge=0.0, le=1000.0),
    db: Session = Depends(get_db),
):
    try:
        return check_and_retrain_lstm(
            db=db,
            channel=channel,
            threshold_mape=threshold_mape,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/lstm-metrics", response_model=LstmMetricsResponse)
def lstm_metrics(
    channel: str = Query(default="Choice", description="Por ahora usar Choice")
):
    try:
        return get_lstm_metrics(channel)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/lstm-status", response_model=LstmStatusResponse)
def lstm_status(
    channel: str = Query(default="Choice", description="Por ahora usar Choice")
):
    try:
        return get_lstm_status(channel)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/lstm-history", response_model=list[LstmHistoryResponse])
def lstm_history(
    channel: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    db: Session = Depends(get_db),
):
    try:
        return get_lstm_history(db=db, channel=channel, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/scheduler-status", response_model=SchedulerStatusResponse)
def scheduler_status():
    try:
        return get_scheduler_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/scheduler-job-history", response_model=list[SchedulerJobRunResponse])
def scheduler_job_history(limit: int = Query(default=50, ge=1, le=500)):
    try:
        return get_scheduler_job_history(limit=limit)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/system-summary", response_model=SystemSummaryResponse)
def system_summary(
    channel: str = Query(default="Choice", description="Canal principal a resumir"),
    db: Session = Depends(get_db),
):
    try:
        return get_system_summary(db=db, channel=channel)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))