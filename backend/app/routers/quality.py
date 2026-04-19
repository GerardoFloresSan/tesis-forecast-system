from io import BytesIO

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.quality import QualityReportResponse
from app.services.quality_export_service import generate_quality_pdf
from app.services.quality_service import generate_quality_report

router = APIRouter(prefix="/quality", tags=["Quality"])


@router.get("/health")
def quality_health():
    return {"message": "Módulo de calidad operativo"}


@router.get("/report", response_model=QualityReportResponse)
def quality_report(db: Session = Depends(get_db)):
    return generate_quality_report(db)

@router.get("/report/pdf")
def quality_report_pdf(db: Session = Depends(get_db)):
    report_data = generate_quality_report(db)
    pdf_bytes = generate_quality_pdf(report_data)
    filename = "quality_report.pdf"
    headers = {"Content-Disposition": f"attachment; filename={filename}"}
    return StreamingResponse(BytesIO(pdf_bytes), media_type="application/pdf", headers=headers)
