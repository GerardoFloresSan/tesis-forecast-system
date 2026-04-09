from sqlalchemy import Column, Integer, String, DateTime
from app.core.database import Base
from datetime import datetime

class EtlRun(Base):
    __tablename__ = "etl_runs"

    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String, nullable=False)
    status = Column(String, nullable=False)
    message = Column(String, nullable=True)
    records_original = Column(Integer, default=0)
    duplicates_removed = Column(Integer, default=0)
    nulls_treated = Column(Integer, default=0)
    records_final = Column(Integer, default=0)
    started_at = Column(DateTime, default=datetime.utcnow)
    finished_at = Column(DateTime, nullable=True)