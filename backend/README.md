# Backend - Forecast Thesis API

Backend del sistema de pronóstico para tesis, desarrollado con FastAPI, SQLAlchemy, PostgreSQL y TensorFlow.

## 1. Requisitos

- Python 3.11
- PostgreSQL 13 o superior
- pip actualizado

## 2. Estructura principal

```bash
backend/
├── app/
│   ├── core/
│   ├── models/
│   ├── routers/
│   ├── schemas/
│   ├── services/
│   └── utils/
├── data/
│   ├── models/
│   └── uploads/
├── scripts/
├── requirements.txt
└── .env.example