# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Backend (run from `backend/`)
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload          # dev server on :8000
pytest                                  # all tests
pytest test/test_auth.py               # single test file
pytest test/test_auth.py::test_login   # single test
```

### Frontend (run from `frontend/forecast-dashboard/`)
```bash
npm install
npm start          # dev server on :4200
npm run build      # production build
npm test           # Karma/Jasmine
```

### Requirements
- PostgreSQL 13+ on `localhost:5432`
- Copy `backend/.env.example` → `backend/.env` and fill credentials

## Architecture

**Stack**: FastAPI (Python 3.11) + Angular 20 (TypeScript) + PostgreSQL + TensorFlow LSTM

### Backend layers (`backend/app/`)

| Layer | Path | Role |
|-------|------|------|
| Routers | `routers/` | HTTP endpoints, 8 modules, JWT-guarded |
| Services | `services/` | Business logic, 17 modules |
| Models | `models/` | SQLAlchemy ORM, 11 tables |
| Schemas | `schemas/` | Pydantic request/response validation |
| Core | `core/` | DB session, JWT security, config, dependencies |

**Data flow for a forecast cycle:**
1. Upload CSV → `upload` router → `etl_service` → `historical_interactions` table
2. Train → `lstm_training_service` saves model + scaler to `backend/data/models/`
3. Predict → `forecast_service` + `lstm_service` → `forecast_runs` / `forecast_interval_runs`
4. `alert_service` checks SLA thresholds → `sla_alerts`
5. `scheduler_service` (APScheduler) runs auto-retrain & auto-forecast jobs on configurable intervals

### Frontend (`frontend/forecast-dashboard/src/app/`)
Angular standalone components. Services in `services/` call backend REST on `http://localhost:8000`. Routes in `app.routes.ts`. Pages in `pages/`.

### Database
11 tables. Key ones: `historical_interactions` (input TS), `external_variables` (holidays/campaigns/absenteeism), `model_train_runs` (metrics: MAE/RMSE/MAPE/R²), `forecast_runs`, `sla_alerts`, `scheduler_job_runs`.

### Scheduler env vars (`backend/.env`)
```
AUTO_RETRAIN_ENABLED / AUTO_RETRAIN_INTERVAL_MINUTES / AUTO_RETRAIN_THRESHOLD_MAPE
AUTO_FORECAST_ENABLED / AUTO_FORECAST_INTERVAL_MINUTES
AUTO_RETRAIN_CHANNEL / AUTO_FORECAST_CHANNEL   # e.g. "Choice/España"
```

## Key Notes
- Codebase language: **Spanish** (comments, variable names, error messages, DB values)
- Channels: `Choice` and `España` — used as filters throughout forecasting logic
- LSTM models persisted as pickled files + JSON metadata in `backend/data/models/`
- Tests use SQLite in-memory DB; production uses PostgreSQL
- MAPE is the primary model quality metric; auto-retrain triggers when MAPE exceeds threshold
