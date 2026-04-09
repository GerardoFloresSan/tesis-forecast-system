from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.preprocessing import PreprocessedDatasetRow
from app.services.preprocessing_service import (
    get_preprocessed_dataset,
    get_preprocessed_dataset_by_channel,
)

router = APIRouter(prefix="/forecast", tags=["Preprocessing"])


@router.get("/preprocessed", response_model=list[PreprocessedDatasetRow])
def preprocessed_dataset(db: Session = Depends(get_db)):
    return get_preprocessed_dataset(db)


@router.get("/preprocessed/by-channel", response_model=list[PreprocessedDatasetRow])
def preprocessed_dataset_by_channel(
    channel: str = Query(...),
    db: Session = Depends(get_db),
):
    return get_preprocessed_dataset_by_channel(db, channel)