from pydantic import BaseModel
from datetime import date, time


class PreprocessedDatasetRow(BaseModel):
    interaction_date: date
    interval_time: time
    channel: str
    volume: int
    aht: float | None = None
    is_holiday: float = 0.0
    campaign_day: float = 0.0
    absenteeism_rate: float = 0.0
    day_of_week: int
    month: int
    is_weekend: int
    hour: int
    minute: int