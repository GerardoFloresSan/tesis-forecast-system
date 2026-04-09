from pydantic import BaseModel
from datetime import date


class ExternalVariableCreate(BaseModel):
    variable_date: date
    variable_type: str
    variable_value: float
    description: str | None = None


class ExternalVariableResponse(BaseModel):
    id: int
    variable_date: date
    variable_type: str
    variable_value: float
    description: str | None = None

    class Config:
        from_attributes = True