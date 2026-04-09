import math
import os
import joblib
import pandas as pd
from sqlalchemy.orm import Session
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from app.models.historical_interaction import HistoricalInteraction
from app.models.external_variable import ExternalVariable
from app.schemas.model import BaselinePredictionRequest


MODEL_DIR = "data/models"
MODEL_PATH = os.path.join(MODEL_DIR, "baseline_random_forest.joblib")
FEATURES_PATH = os.path.join(MODEL_DIR, "baseline_features.joblib")


def _build_training_dataframe(db: Session) -> pd.DataFrame:
    historical_data = db.query(HistoricalInteraction).all()
    external_data = db.query(ExternalVariable).all()

    if not historical_data:
        raise ValueError("No hay datos históricos para entrenar el modelo.")

    hist_df = pd.DataFrame([
        {
            "interaction_date": row.interaction_date,
            "interval_time": row.interval_time,
            "channel": row.channel,
            "volume": row.volume,
            "aht": row.aht if row.aht is not None else 0.0,
        }
        for row in historical_data
    ])

    ext_df = pd.DataFrame([
        {
            "variable_date": row.variable_date,
            "variable_type": row.variable_type,
            "variable_value": row.variable_value,
        }
        for row in external_data
    ])

    if not ext_df.empty:
        ext_pivot = (
            ext_df.pivot_table(
                index="variable_date",
                columns="variable_type",
                values="variable_value",
                aggfunc="max",
                fill_value=0,
            )
            .reset_index()
        )
        ext_pivot.columns.name = None

        df = hist_df.merge(
            ext_pivot,
            how="left",
            left_on="interaction_date",
            right_on="variable_date"
        )
        df.drop(columns=["variable_date"], inplace=True, errors="ignore")
    else:
        df = hist_df.copy()

    for col in ["is_holiday", "campaign_day", "absenteeism_rate"]:
        if col not in df.columns:
            df[col] = 0.0

    df["interaction_date"] = pd.to_datetime(df["interaction_date"])
    df["day_of_week"] = df["interaction_date"].dt.weekday
    df["month"] = df["interaction_date"].dt.month
    df["day"] = df["interaction_date"].dt.day
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["interval_time"] = pd.to_datetime(df["interval_time"].astype(str), format="%H:%M:%S")
    df["hour"] = df["interval_time"].dt.hour
    df["minute"] = df["interval_time"].dt.minute

    df = pd.get_dummies(df, columns=["channel"], drop_first=False)

    return df


def _get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        "aht",
        "is_holiday",
        "campaign_day",
        "absenteeism_rate",
        "day_of_week",
        "month",
        "day",
        "is_weekend",
        "hour",
        "minute",
    ] + [col for col in df.columns if col.startswith("channel_")]


def train_baseline_model(db: Session) -> dict:
    df = _build_training_dataframe(db)
    feature_columns = _get_feature_columns(df)

    X = df[feature_columns]
    y = df["volume"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = math.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    return {
        "model_name": "RandomForestRegressor",
        "total_rows": len(df),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "r2": round(float(r2), 4),
    }


def train_and_save_baseline_model(db: Session) -> dict:
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = _build_training_dataframe(db)
    feature_columns = _get_feature_columns(df)

    X = df[feature_columns]
    y = df["volume"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = math.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(feature_columns, FEATURES_PATH)

    return {
        "model_name": "RandomForestRegressor",
        "file_path": MODEL_PATH,
        "total_rows": len(df),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "r2": round(float(r2), 4),
    }


def predict_with_baseline_model(db: Session, request: BaselinePredictionRequest) -> dict:
    df = _build_training_dataframe(db)
    feature_columns = _get_feature_columns(df)

    X = df[feature_columns]
    y = df["volume"]

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)

    input_row = {
        "aht": request.aht,
        "is_holiday": request.is_holiday,
        "campaign_day": request.campaign_day,
        "absenteeism_rate": request.absenteeism_rate,
        "day_of_week": request.interaction_date.weekday(),
        "month": request.interaction_date.month,
        "day": request.interaction_date.day,
        "is_weekend": 1 if request.interaction_date.weekday() >= 5 else 0,
        "hour": request.interval_time.hour,
        "minute": request.interval_time.minute,
    }

    for col in feature_columns:
        if col.startswith("channel_"):
            input_row[col] = 0

    requested_channel_col = f"channel_{request.channel}"
    if requested_channel_col in feature_columns:
        input_row[requested_channel_col] = 1

    input_df = pd.DataFrame([input_row])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    prediction = model.predict(input_df)[0]

    return {
        "model_name": "RandomForestRegressor",
        "predicted_volume": round(float(prediction), 4),
    }


def predict_with_saved_baseline_model(request: BaselinePredictionRequest) -> dict:
    if not os.path.exists(MODEL_PATH):
        raise ValueError("No existe un modelo guardado. Primero ejecuta /model/train-and-save-baseline")

    if not os.path.exists(FEATURES_PATH):
        raise ValueError("No existe el archivo de features guardadas.")

    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)

    input_row = {
        "aht": request.aht,
        "is_holiday": request.is_holiday,
        "campaign_day": request.campaign_day,
        "absenteeism_rate": request.absenteeism_rate,
        "day_of_week": request.interaction_date.weekday(),
        "month": request.interaction_date.month,
        "day": request.interaction_date.day,
        "is_weekend": 1 if request.interaction_date.weekday() >= 5 else 0,
        "hour": request.interval_time.hour,
        "minute": request.interval_time.minute,
    }

    for col in feature_columns:
        if col.startswith("channel_"):
            input_row[col] = 0

    requested_channel_col = f"channel_{request.channel}"
    if requested_channel_col in feature_columns:
        input_row[requested_channel_col] = 1

    input_df = pd.DataFrame([input_row])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    prediction = model.predict(input_df)[0]

    return {
        "model_name": "RandomForestRegressor (saved)",
        "predicted_volume": round(float(prediction), 4),
    }