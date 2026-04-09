from sqlalchemy import Column, Integer, Float, String, DateTime
from datetime import datetime

from app.core.database import Base


class ForecastRun(Base):
    __tablename__ = "forecast_runs"

    id = Column(Integer, primary_key=True, index=True)
    channel = Column(String(50), nullable=False)
    forecast_date = Column(DateTime, nullable=False)
    predicted_value = Column(Float, nullable=False)
    model_version = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)