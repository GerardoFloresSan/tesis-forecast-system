from sqlalchemy import Column, Date, DateTime, Integer, String
from sqlalchemy.sql import func

from app.core.database import Base


class APIAccessLog(Base):
    __tablename__ = "api_access_logs"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), nullable=False, index=True)
    endpoint = Column(String(255), nullable=False, index=True)
    method = Column(String(10), nullable=False)
    channel = Column(String(50), nullable=True, index=True)
    forecast_date = Column(Date, nullable=True, index=True)
    status_code = Column(Integer, nullable=False)
    client_ip = Column(String(100), nullable=True)
    user_agent = Column(String(255), nullable=True)
    created_at = Column(DateTime, nullable=False, server_default=func.now())