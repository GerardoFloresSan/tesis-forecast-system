from datetime import datetime

from sqlalchemy import Column, Integer, String, DateTime, Text

from app.core.database import Base


class SchedulerJobRun(Base):
    __tablename__ = "scheduler_job_runs"

    id = Column(Integer, primary_key=True, index=True)
    job_name = Column(String(100), nullable=False)
    job_type = Column(String(50), nullable=False)   # forecast | retrain_check
    channel = Column(String(100), nullable=True)

    status = Column(String(20), nullable=False)     # success | failed
    action_taken = Column(String(50), nullable=False)  # created_forecast | retrain | none | error
    message = Column(Text, nullable=True)

    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)