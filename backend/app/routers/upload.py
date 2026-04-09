import os
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.upload import UploadSummary
from app.services.etl_service import process_excel_and_save

router = APIRouter(prefix="/upload", tags=["Upload"])

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.get("/health")
def upload_health():
    return {"message": "Módulo de carga operativo"}


@router.post("/excel", response_model=UploadSummary)
async def upload_excel(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Solo se permiten archivos Excel")

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    result = process_excel_and_save(file_path=file_path, db=db, file_name=file.filename)
    return result