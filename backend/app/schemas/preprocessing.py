from datetime import date, time

from pydantic import BaseModel


class PreprocessedDatasetRow(BaseModel):
    interaction_date: date
    interval_time: time
    channel: str
    volume: int
    aht: float | None = None
    is_holiday: float = 0.0
    is_holiday_peru: float = 0.0
    is_holiday_spain: float = 0.0
    is_holiday_mexico: float = 0.0
    is_holiday_any: int = 0
    campaign_day: float = 0.0
    absenteeism_rate: float = 0.0
    day_of_week: int
    month: int
    is_weekend: int
    hour: int
    minute: int
