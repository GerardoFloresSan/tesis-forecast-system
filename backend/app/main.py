from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.database import Base, engine

from app.models.user import User
from app.models.etl_run import EtlRun
from app.models.historical_interaction import HistoricalInteraction
from app.models.external_variable import ExternalVariable
from app.models.data_quality_report import DataQualityReport
from app.models.forecast_run import ForecastRun
from app.models.forecast_interval_run import ForecastIntervalRun
from app.models.model_train_run import ModelTrainRun
from app.models.scheduler_job_run import SchedulerJobRun

from app.routers.auth import router as auth_router
from app.routers.upload import router as upload_router
from app.routers.quality import router as quality_router
from app.routers.external_variables import router as external_variables_router
from app.routers.forecast import router as forecast_router
from app.routers.preprocessing import router as preprocessing_router
from app.routers.model import router as model_router

from app.services.scheduler_service import start_scheduler, shutdown_scheduler

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Forecast Thesis API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",
        "http://127.0.0.1:4200",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(upload_router)
app.include_router(quality_router)
app.include_router(external_variables_router)
app.include_router(forecast_router)
app.include_router(preprocessing_router)
app.include_router(model_router)


@app.on_event("startup")
def on_startup():
    start_scheduler()


@app.on_event("shutdown")
def on_shutdown():
    shutdown_scheduler()


@app.get("/")
def root():
    return {"message": "API de tesis levantada correctamente"}