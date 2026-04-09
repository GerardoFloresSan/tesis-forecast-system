from sqlalchemy import Column, Integer, String, Date, Time, Float
from app.core.database import Base

class HistoricalInteraction(Base):
    __tablename__ = "historical_interactions"

    id = Column(Integer, primary_key=True, index=True)
    interaction_date = Column(Date, nullable=False, index=True)
    interval_time = Column(Time, nullable=False, index=True)
    channel = Column(String, nullable=False, index=True)
    volume = Column(Integer, nullable=False)
    aht = Column(Float, nullable=True)