from datetime import datetime

from sqlalchemy import Column, Date, DateTime, Float, ForeignKey, Integer, String, Time, UniqueConstraint

from app.core.database import Base


class ForecastIntervalRun(Base):
    __tablename__ = "forecast_interval_runs"
    __table_args__ = (
        UniqueConstraint("channel", "forecast_datetime", name="uq_forecast_interval_channel_datetime"),
    )

    id = Column(Integer, primary_key=True, index=True)
    forecast_run_id = Column(Integer, ForeignKey("forecast_runs.id"), nullable=False, index=True)
    channel = Column(String(50), nullable=False, index=True)
    forecast_date = Column(Date, nullable=False, index=True)
    forecast_datetime = Column(DateTime, nullable=False, index=True)
    interval_time = Column(Time, nullable=False)
    slot_index = Column(Integer, nullable=False)
    shift_label = Column(String(30), nullable=False)
    predicted_value = Column(Float, nullable=False)
    model_version = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
