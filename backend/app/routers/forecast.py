from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.dependencies import get_current_user
from app.models.user import User
from app.schemas.forecast import (
    ForecastApiResponse,
    ForecastBatchResponse,
    ForecastDatasetRow,
    ForecastGenerateRequest,
    ForecastIntervalResponse,
    ForecastMonthlyGenerateRequest,
    ForecastMonthlyResponse,
    ForecastMonthlyStatusResponse,
    ForecastRunResponse,
)
from app.schemas.monitoring import ForecastMonitoringResponse
from app.services.api_audit_service import log_api_access
from app.services.forecast_api_service import get_daily_forecast_api_payload
from app.services.forecast_monitoring_service import (
    get_forecast_monitoring_by_date,
    get_latest_forecast_monitoring_summary,
)
from app.services.forecast_service import (
    create_daily_forecast,
    create_monthly_forecast,
    get_available_channels,
    get_forecast_dataset,
    get_forecast_dataset_by_date,
    get_forecast_history,
    get_interval_forecast_history,
    get_monthly_forecast_status,
)

router = APIRouter(
    prefix="/forecast",
    tags=["Forecast"],
    dependencies=[Depends(get_current_user)],
)


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


@router.post("/daily", response_model=ForecastBatchResponse)
def generate_daily_forecast(
    payload: ForecastGenerateRequest,
    db: Session = Depends(get_db),
):
    try:
        return create_daily_forecast(db, payload.channel)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/monthly", response_model=ForecastMonthlyResponse)
def generate_monthly_forecast(
    payload: ForecastMonthlyGenerateRequest,
    db: Session = Depends(get_db),
):
    try:
        return create_monthly_forecast(
            db=db,
            channel=payload.channel,
            start_date=payload.start_date,
            end_date=payload.end_date,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/monthly/status", response_model=ForecastMonthlyStatusResponse)
def monthly_forecast_status(
    channel: str = Query(default="Choice"),
    start_date: date = Query(...),
    end_date: date = Query(...),
    db: Session = Depends(get_db),
):
    try:
        return get_monthly_forecast_status(
            db=db,
            channel=channel,
            start_date=start_date,
            end_date=end_date,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/range", response_model=ForecastMonthlyResponse)
def generate_range_forecast(
    payload: ForecastMonthlyGenerateRequest,
    db: Session = Depends(get_db),
):
    try:
        return create_monthly_forecast(
            db=db,
            channel=payload.channel,
            start_date=payload.start_date,
            end_date=payload.end_date,
        )
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


@router.get("/history/intervals", response_model=list[ForecastIntervalResponse])
def forecast_interval_history(
    channel: str | None = Query(default=None),
    forecast_date: date | None = Query(default=None),
    start_date: date | None = Query(default=None),
    end_date: date | None = Query(default=None),
    limit: int = Query(default=5000, ge=1, le=10000),
    db: Session = Depends(get_db),
):
    try:
        return get_interval_forecast_history(
            db=db,
            channel=channel,
            forecast_date=forecast_date,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/monitoring/summary", response_model=ForecastMonitoringResponse)
def forecast_monitoring_summary(
    channel: str = Query(default="Choice"),
    db: Session = Depends(get_db),
):
    try:
        return get_latest_forecast_monitoring_summary(
            db=db,
            channel=channel,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/monitoring/by-date", response_model=ForecastMonitoringResponse)
def forecast_monitoring_by_date(
    forecast_date: date = Query(...),
    channel: str = Query(default="Choice"),
    db: Session = Depends(get_db),
):
    try:
        return get_forecast_monitoring_by_date(
            db=db,
            channel=channel,
            forecast_date=forecast_date,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/api/daily", response_model=ForecastApiResponse)
def forecast_api_daily(
    request: Request,
    forecast_date: date = Query(...),
    channel: str = Query(default="Choice"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        payload = get_daily_forecast_api_payload(
            db=db,
            channel=channel,
            forecast_date=forecast_date,
        )

        log_api_access(
            db=db,
            username=current_user.username,
            endpoint="/forecast/api/daily",
            method="GET",
            status_code=200,
            channel=payload["channel"],
            forecast_date=payload["date"],
            client_ip=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
        )

        return payload
    except Exception as e:
        log_api_access(
            db=db,
            username=current_user.username,
            endpoint="/forecast/api/daily",
            method="GET",
            status_code=400,
            channel=channel,
            forecast_date=forecast_date,
            client_ip=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
        )
        raise HTTPException(status_code=400, detail=str(e))
