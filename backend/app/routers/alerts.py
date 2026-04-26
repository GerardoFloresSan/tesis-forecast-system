from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.dependencies import get_current_user
from app.models.user import User
from app.schemas.alert import (
    SLAAlertAckResponse,
    SLAAlertEvaluationResponse,
    SLAAlertResponse,
)
from app.services.alert_service import (
    acknowledge_alert,
    evaluate_alert_for_date,
    get_active_alerts,
    get_alert_history,
)
from fastapi import Depends
from app.core.dependencies import get_current_user
from app.models.user import User

current_user: User = Depends(get_current_user)
router = APIRouter(prefix="/alerts", tags=["Alerts"], dependencies=[Depends(get_current_user)]    )


@router.get("/active", response_model=list[SLAAlertResponse])
def alerts_active(
    channel: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    db: Session = Depends(get_db),
):
    try:
        return get_active_alerts(db=db, channel=channel, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/history", response_model=list[SLAAlertResponse])
def alerts_history(
    channel: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    try:
        return get_alert_history(db=db, channel=channel, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/evaluate", response_model=SLAAlertEvaluationResponse)
def alerts_evaluate(
    forecast_date: date = Query(...),
    channel: str = Query(default="Choice"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        return evaluate_alert_for_date(
            db=db,
            channel=channel,
            forecast_date=forecast_date,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.patch("/{alert_id}/ack", response_model=SLAAlertAckResponse)
def alerts_acknowledge(
    alert_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        alert = acknowledge_alert(
            db=db,
            alert_id=alert_id,
            username=current_user.username,
        )
        return {
            "id": alert.id,
            "status": alert.status,
            "acknowledged_by": alert.acknowledged_by,
            "acknowledged_at": alert.acknowledged_at,
            "message": f"Alerta {alert.id} reconocida correctamente.",
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))