from datetime import date
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.forecast import (
    ForecastDatasetRow,
    ForecastGenerateRequest,
    ForecastRunResponse,
)
from app.services.forecast_service import (
    get_available_channels,
    get_forecast_dataset,
    get_forecast_dataset_by_date,
    create_daily_forecast,
    get_forecast_history,
)

router = APIRouter(prefix="/forecast", tags=["Forecast"])


@router.get("/channels", response_model=list[str])
def forecast_channels(db: Session = Depends(get_db)):
    try:
        return get_available_channels(db)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/dataset", response_model=list[ForecastDatasetRow])
def forecast_dataset(
    start_date: date | None = Query(default=None),
    end_date: date | None = Query(default=None),
    channel: str | None = Query(default=None),
    limit: int = Query(default=500, ge=1, le=5000),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    try:
        return get_forecast_dataset(
            db=db,
            start_date=start_date,
            end_date=end_date,
            channel=channel,
            limit=limit,
            offset=offset,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/dataset/by-date", response_model=list[ForecastDatasetRow])
def forecast_dataset_by_date(
    start_date: date = Query(...),
    end_date: date = Query(...),
    channel: str | None = Query(default=None),
    limit: int = Query(default=1000, ge=1, le=5000),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    try:
        return get_forecast_dataset_by_date(
            db=db,
            start_date=start_date,
            end_date=end_date,
            channel=channel,
            limit=limit,
            offset=offset,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/daily", response_model=ForecastRunResponse)
def generate_daily_forecast(
    payload: ForecastGenerateRequest,
    db: Session = Depends(get_db),
):
    try:
        return create_daily_forecast(db, payload.channel)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/history", response_model=list[ForecastRunResponse])
def forecast_history(
    channel: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    db: Session = Depends(get_db),
):
    try:
        return get_forecast_history(db, channel, limit)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))