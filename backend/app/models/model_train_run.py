from datetime import datetime

from sqlalchemy import Column, Integer, String, Float, DateTime, Text

from app.core.database import Base


class ModelTrainRun(Base):
    __tablename__ = "model_train_runs"

    id = Column(Integer, primary_key=True, index=True)
    channel = Column(String(100), nullable=False)
    run_type = Column(String(20), nullable=False)  # train | retrain
    status = Column(String(20), nullable=False)    # running | success | failed

    mae = Column(Float, nullable=True)
    rmse = Column(Float, nullable=True)
    mape = Column(Float, nullable=True)
    r2 = Column(Float, nullable=True)

    train_size = Column(Integer, nullable=True)
    test_size = Column(Integer, nullable=True)

    model_path = Column(String(500), nullable=True)
    scaler_path = Column(String(500), nullable=True)
    metadata_path = Column(String(500), nullable=True)
    metrics_path = Column(String(500), nullable=True)

    error_message = Column(Text, nullable=True)

    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)