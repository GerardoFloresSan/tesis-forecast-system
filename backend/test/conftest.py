import os
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
BACKEND_DIR = TESTS_DIR.parent

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

TEST_DB_PATH = TESTS_DIR / "test_forecast.db"

os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB_PATH}"
os.environ["AUTO_RETRAIN_ENABLED"] = "false"
os.environ["AUTO_FORECAST_ENABLED"] = "false"
os.environ["SECRET_KEY"] = "test-secret-key"
os.environ["DEFAULT_ADMIN_USERNAME"] = "admin"
os.environ["DEFAULT_ADMIN_PASSWORD"] = "Admin123*"

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.database import Base, SessionLocal, engine

from app.models.api_access_log import APIAccessLog
from app.models.data_quality_report import DataQualityReport
from app.models.etl_run import EtlRun
from app.models.external_variable import ExternalVariable
from app.models.forecast_interval_run import ForecastIntervalRun
from app.models.forecast_run import ForecastRun
from app.models.historical_interaction import HistoricalInteraction
from app.models.model_train_run import ModelTrainRun
from app.models.scheduler_job_run import SchedulerJobRun
from app.models.sla_alert import SLAAlert
from app.models.user import User

from app.routers.alerts import router as alerts_router
from app.routers.auth import router as auth_router
from app.routers.forecast import router as forecast_router
from app.routers.model import router as model_router
from app.routers.quality import router as quality_router
from app.routers.upload import router as upload_router


def create_test_app() -> FastAPI:
    app = FastAPI(title="Forecast Test API")
    app.include_router(auth_router)
    app.include_router(upload_router)
    app.include_router(quality_router)
    app.include_router(forecast_router)
    app.include_router(model_router)
    app.include_router(alerts_router)
    return app


@pytest.fixture(autouse=True)
def reset_database():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield


@pytest.fixture
def db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def client():
    app = create_test_app()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_db():
    yield
    engine.dispose()
    if TEST_DB_PATH.exists():
        TEST_DB_PATH.unlink()