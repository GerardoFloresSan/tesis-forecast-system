from datetime import date

from pydantic import BaseModel, Field


class ExternalVariableCreate(BaseModel):
    variable_date: date
    variable_type: str = Field(
        description=(
            "Tipos canónicos recomendados: is_holiday_peru, is_holiday_spain, "
            "is_holiday_mexico, campaign_day, absenteeism_rate. "
            "También se aceptan aliases como is_holiday."
        )
    )
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
