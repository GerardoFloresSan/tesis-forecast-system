from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.quality import QualityReportResponse
from app.services.quality_service import generate_quality_report

router = APIRouter(prefix="/quality", tags=["Quality"])


@router.get("/health")
def quality_health():
    return {"message": "Módulo de calidad operativo"}


@router.get("/report", response_model=QualityReportResponse)
def quality_report(db: Session = Depends(get_db)):
    return generate_quality_report(db)