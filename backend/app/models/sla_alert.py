from sqlalchemy import Boolean, Column, Date, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.sql import func

from app.core.database import Base


class SLAAlert(Base):
    __tablename__ = "sla_alerts"

    id = Column(Integer, primary_key=True, index=True)
    channel = Column(String(50), nullable=False, index=True)
    forecast_date = Column(Date, nullable=False, index=True)
    forecast_run_id = Column(Integer, ForeignKey("forecast_runs.id"), nullable=True, index=True)

    expected_volume = Column(Float, nullable=False, default=0.0)
    actual_volume = Column(Float, nullable=True)
    deviation_percentage = Column(Float, nullable=True)
    absolute_deviation_percentage = Column(Float, nullable=True)

    error_level = Column(String(50), nullable=False, default="unknown")
    risk_level = Column(String(50), nullable=False, default="unknown")
    has_breach = Column(Boolean, nullable=False, default=False)

    status = Column(String(30), nullable=False, default="active", index=True)
    message = Column(String(500), nullable=False)

    email_sent = Column(Boolean, nullable=False, default=False)
    email_recipient = Column(String(255), nullable=True)

    acknowledged_by = Column(String(100), nullable=True)
    acknowledged_at = Column(DateTime, nullable=True)

    resolved_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())