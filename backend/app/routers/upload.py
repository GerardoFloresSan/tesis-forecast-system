import re
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.upload import UploadSummary
from app.services.etl_service import process_excel_and_save

router = APIRouter(prefix="/upload", tags=["Upload"])

BASE_DIR = Path(__file__).resolve().parents[2]
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _build_storage_name(original_name: str) -> str:
    suffix = Path(original_name).suffix.lower()
    stem = Path(original_name).stem
    safe_stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", stem).strip("_") or "upload"
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{safe_stem}{suffix}"


@router.get("/health")
def upload_health():
    return {"message": "Módulo de carga operativo"}


@router.post("/excel", response_model=UploadSummary)
async def upload_excel(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename or not file.filename.lower().endswith((".xlsx", ".xls", ".csv")):
        raise HTTPException(status_code=400, detail="Solo se permiten archivos .xlsx, .xls o .csv")

    stored_file_name = _build_storage_name(file.filename)
    file_path = UPLOAD_DIR / stored_file_name

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    result = process_excel_and_save(file_path=str(file_path), db=db, file_name=file.filename)
    return result