from datetime import date
from sqlalchemy.orm import Session

from app.models.external_variable import ExternalVariable
from app.schemas.external_variable import ExternalVariableCreate


def create_external_variable(db: Session, data: ExternalVariableCreate) -> ExternalVariable:
    record = ExternalVariable(
        variable_date=data.variable_date,
        variable_type=data.variable_type.strip(),
        variable_value=data.variable_value,
        description=data.description.strip() if data.description else None,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_all_external_variables(db: Session) -> list[ExternalVariable]:
    return db.query(ExternalVariable).order_by(
        ExternalVariable.variable_date.asc(),
        ExternalVariable.variable_type.asc()
    ).all()


def get_external_variables_by_date(
    db: Session,
    start_date: date,
    end_date: date
) -> list[ExternalVariable]:
    return (
        db.query(ExternalVariable)
        .filter(ExternalVariable.variable_date >= start_date)
        .filter(ExternalVariable.variable_date <= end_date)
        .order_by(ExternalVariable.variable_date.asc(), ExternalVariable.variable_type.asc())
        .all()
    )