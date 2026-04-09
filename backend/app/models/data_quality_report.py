from sqlalchemy import Column, Integer, Float, DateTime
from app.core.database import Base
from datetime import datetime

class DataQualityReport(Base):
    __tablename__ = "data_quality_reports"

    id = Column(Integer, primary_key=True, index=True)
    total_records = Column(Integer, default=0)
    missing_percentage = Column(Float, default=0.0)
    duplicate_percentage = Column(Float, default=0.0)
    valid_percentage = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)