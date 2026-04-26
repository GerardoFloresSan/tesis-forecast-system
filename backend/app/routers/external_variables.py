from datetime import date
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.external_variable import (
    ExternalVariableCreate,
    ExternalVariableResponse,
)
from app.services.external_variable_service import (
    create_external_variable,
    get_all_external_variables,
    get_external_variables_by_date,
)
from fastapi import Depends
from app.core.dependencies import get_current_user
from app.models.user import User


router = APIRouter(prefix="/external-variables", tags=["External Variables"], dependencies=[Depends(get_current_user)]    )


@router.post("/", response_model=ExternalVariableResponse)
def create_external_variable_endpoint(
    data: ExternalVariableCreate,
    db: Session = Depends(get_db)
):
    return create_external_variable(db, data)


@router.get("/", response_model=list[ExternalVariableResponse])
def get_external_variables_endpoint(db: Session = Depends(get_db)):
    return get_all_external_variables(db)


@router.get("/by-date", response_model=list[ExternalVariableResponse])
def get_external_variables_by_date_endpoint(
    start_date: date = Query(...),
    end_date: date = Query(...),
    db: Session = Depends(get_db)
):
    return get_external_variables_by_date(db, start_date, end_date)