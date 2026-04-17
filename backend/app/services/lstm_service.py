from __future__ import annotations

from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.models.external_variable import ExternalVariable
from app.models.historical_interaction import HistoricalInteraction
from app.utils.channel_rules import (
    apply_business_hours_filter,
    canonicalize_channel,
    get_next_operational_day_date,
    get_operational_day_datetimes,
    get_operational_day_start_datetime,
    get_shift_label,
    get_slots_per_day,
    slugify_channel,
)

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "data" / "models"


def _safe_float(value: float | int | None, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _compute_effective_bias_adjustment(metadata: dict) -> float:
    validation_bias = _safe_float(metadata.get("validation_bias", 0.0), 0.0)
    positive_factor = _safe_float(metadata.get("bias_correction_factor", 0.0), 0.0)
    negative_factor = _safe_float(metadata.get("negative_bias_correction_factor", positive_factor), positive_factor)
    factor = positive_factor if validation_bias >= 0 else negative_factor
    return float(validation_bias * factor)


def _normalize_slot_bias_adjustments(metadata: dict) -> dict[int, float]:
    raw = metadata.get("slot_bias_adjustments", {}) or {}
    adjustments: dict[int, float] = {}
    for key, value in raw.items():
        try:
            adjustments[int(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return adjustments

def _normalize_late_slot_uplift_factors(metadata: dict) -> dict[int, float]:
    raw = metadata.get("late_slot_uplift_factors", {}) or {}
    factors: dict[int, float] = {}
    for key, value in raw.items():
        try:
            factors[int(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return factors


def _build_late_slot_reference_value(
    lag_volume_1_day: float | int | None,
    same_slot_mean_3_day: float | int | None,
    same_slot_mean_7_day: float | int | None,
) -> float:
    lag_1 = _safe_float(lag_volume_1_day, float("nan"))
    same_3 = _safe_float(same_slot_mean_3_day, float("nan"))
    same_7 = _safe_float(same_slot_mean_7_day, float("nan"))
    reference = lag_1
    if np.isnan(reference):
        reference = same_3 if not np.isnan(same_3) else same_7
    if np.isnan(reference):
        reference = 0.0
    same_fallback = same_3 if not np.isnan(same_3) else (same_7 if not np.isnan(same_7) else reference)
    return float(max(reference, same_fallback))


def _apply_late_slot_uplift(
    adjusted: np.ndarray,
    metadata: dict,
    slot_index: int | None = None,
    late_slot_reference_value: float | None = None,
) -> np.ndarray:
    if slot_index is None or late_slot_reference_value is None:
        return adjusted

    factors = _normalize_late_slot_uplift_factors(metadata)
    alpha = float(factors.get(int(slot_index), 0.0))
    if alpha <= 0:
        return adjusted

    gap_min = _safe_float(metadata.get("late_slot_reference_gap_min", 1.0), 1.0)
    max_abs = _safe_float(metadata.get("late_slot_uplift_max_abs", 6.0), 6.0)
    gap = float(late_slot_reference_value) - float(adjusted[0])
    if gap <= gap_min:
        return adjusted

    uplift = float(np.clip(gap * alpha, 0.0, max_abs))
    adjusted[0] += uplift
    return adjusted


def _apply_prediction_postprocess(
    y_pred_real: np.ndarray,
    metadata: dict,
    slot_index: int | None = None,
    late_slot_reference_value: float | None = None,
) -> float:
    adjusted = np.asarray(y_pred_real, dtype=float).copy()
    adjusted = adjusted - _compute_effective_bias_adjustment(metadata)

    if slot_index is not None:
        slot_bias_adjustments = _normalize_slot_bias_adjustments(metadata)
        adjusted = adjusted + float(slot_bias_adjustments.get(int(slot_index), 0.0))

    adjusted = _apply_late_slot_uplift(
        adjusted=adjusted,
        metadata=metadata,
        slot_index=slot_index,
        late_slot_reference_value=late_slot_reference_value,
    )
    adjusted = np.maximum(adjusted, 0.0)
    return float(adjusted[0])


def _get_load_model():
    import os

    os.environ.pop("TF_USE_LEGACY_KERAS", None)
    try:
        from tensorflow.keras.models import load_model
        return load_model
    except Exception as exc:
        raise ModuleNotFoundError(
            "No se pudo importar tensorflow.keras.models.load_model. "
            "Verifica la instalación de TensorFlow en el entorno virtual."
        ) from exc


def _inverse_target_transform(y_scaled: np.ndarray, y_scaler) -> np.ndarray:
    y_scaled = np.array(y_scaled).reshape(-1, 1)
    y_log = y_scaler.inverse_transform(y_scaled).flatten()
    y_real = np.expm1(y_log)
    return np.maximum(y_real, 0.0)


def load_lstm_artifacts(channel: str):
    canonical_channel = canonicalize_channel(channel)
    channel_slug = slugify_channel(canonical_channel)

    model_path = MODEL_DIR / f"lstm_{channel_slug}.keras"
    scaler_path = MODEL_DIR / f"lstm_{channel_slug}_scaler.joblib"
    metadata_path = MODEL_DIR / f"lstm_{channel_slug}_metadata.joblib"

    if not model_path.exists():
        raise ValueError(f"No existe el modelo LSTM para el canal '{canonical_channel}' en: {model_path}")
    if not scaler_path.exists():
        raise ValueError(f"No existe el scaler LSTM para el canal '{canonical_channel}' en: {scaler_path}")
    if not metadata_path.exists():
        raise ValueError(f"No existe el metadata LSTM para el canal '{canonical_channel}' en: {metadata_path}")

    load_model = _get_load_model()
    model = load_model(model_path)
    scaler_bundle = joblib.load(scaler_path)
    metadata = joblib.load(metadata_path)
    return model, scaler_bundle, metadata


def _normalize_external_variables(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    normalized = df.copy()
    normalized["variable_date"] = pd.to_datetime(normalized["variable_date"]).dt.date
    normalized["variable_type"] = normalized["variable_type"].astype(str).str.strip().str.lower()
    normalized["variable_value"] = pd.to_numeric(normalized["variable_value"], errors="coerce").fillna(0.0)
    normalized["variable_type"] = normalized["variable_type"].replace({"is_holiday": "is_holiday_peru"})
    return normalized


def _default_external_variables() -> dict[str, float]:
    return {
        "is_holiday_peru": 0.0,
        "is_holiday_spain": 0.0,
        "is_holiday_mexico": 0.0,
        "campaign_day": 0.0,
        "absenteeism_rate": 0.0,
        "is_holiday_any": 0.0,
    }


def _build_external_variables_by_date(db: Session) -> dict[date, dict[str, float]]:
    rows = db.query(ExternalVariable).order_by(ExternalVariable.variable_date.asc(), ExternalVariable.id.asc()).all()
    if not rows:
        return {}

    ext_df = pd.DataFrame(
        [
            {
                "variable_date": row.variable_date,
                "variable_type": row.variable_type,
                "variable_value": row.variable_value,
            }
            for row in rows
        ]
    )
    ext_df = _normalize_external_variables(ext_df)

    if ext_df.empty:
        return {}

    ext_df = ext_df[
        ext_df["variable_type"].isin(
            [
                "is_holiday_peru",
                "is_holiday_spain",
                "is_holiday_mexico",
                "campaign_day",
                "absenteeism_rate",
            ]
        )
    ].copy()

    if ext_df.empty:
        return {}

    pivot = (
        ext_df.pivot_table(
            index="variable_date",
            columns="variable_type",
            values="variable_value",
            aggfunc="max",
            fill_value=0.0,
        )
        .reset_index()
        .rename(columns={"variable_date": "interaction_date"})
    )

    result: dict[date, dict[str, float]] = {}
    for row in pivot.to_dict(orient="records"):
        interaction_date = row.pop("interaction_date")
        values = _default_external_variables()
        for key, value in row.items():
            values[key] = float(value or 0.0)
        values["is_holiday_any"] = max(
            values["is_holiday_peru"],
            values["is_holiday_spain"],
            values["is_holiday_mexico"],
        )
        result[interaction_date] = values

    return result


def _prepare_channel_dataframe(db: Session, channel: str) -> pd.DataFrame:
    rows = (
        db.query(HistoricalInteraction)
        .filter(HistoricalInteraction.channel == channel)
        .order_by(HistoricalInteraction.interaction_date.asc(), HistoricalInteraction.interval_time.asc())
        .all()
    )
    if not rows:
        raise ValueError(f"No hay datos históricos para el canal '{channel}'.")

    base_df = pd.DataFrame(
        [
            {
                "interaction_date": row.interaction_date,
                "interval_time": row.interval_time,
                "channel": row.channel,
                "volume": float(row.volume or 0.0),
                "aht": float(row.aht or 0.0),
            }
            for row in rows
        ]
    )

    base_df["interaction_date"] = pd.to_datetime(base_df["interaction_date"]).dt.date
    base_df["datetime"] = pd.to_datetime(
        base_df["interaction_date"].astype(str) + " " + base_df["interval_time"].astype(str)
    )
    base_df = apply_business_hours_filter(base_df, channel)
    base_df = base_df.sort_values("datetime").reset_index(drop=True)

    external_by_date = _build_external_variables_by_date(db)
    default_values = _default_external_variables()
    for column in [
        "is_holiday_peru",
        "is_holiday_spain",
        "is_holiday_mexico",
        "campaign_day",
        "absenteeism_rate",
        "is_holiday_any",
    ]:
        base_df[column] = base_df["interaction_date"].apply(
            lambda current_date: float(external_by_date.get(current_date, default_values).get(column, 0.0))
        )

    for column in [
        "volume",
        "aht",
        "is_holiday_peru",
        "is_holiday_spain",
        "is_holiday_mexico",
        "is_holiday_any",
        "campaign_day",
        "absenteeism_rate",
    ]:
        base_df[column] = pd.to_numeric(base_df[column], errors="coerce").fillna(0.0)

    return base_df


def _add_calendar_features(enriched: pd.DataFrame) -> pd.DataFrame:
    unique_days = (
        enriched[["interaction_date", "is_holiday_any"]]
        .drop_duplicates(subset=["interaction_date"])
        .sort_values("interaction_date")
        .reset_index(drop=True)
    )

    unique_days["interaction_date"] = pd.to_datetime(unique_days["interaction_date"])
    unique_days["day_of_month"] = unique_days["interaction_date"].dt.day
    unique_days["week_of_month"] = ((unique_days["day_of_month"] - 1) // 7 + 1).astype(int)
    unique_days["days_in_month"] = unique_days["interaction_date"].dt.days_in_month
    unique_days["is_month_start"] = (unique_days["day_of_month"] <= 3).astype(int)
    unique_days["is_month_end"] = (
        unique_days["day_of_month"] >= (unique_days["days_in_month"] - 2)
    ).astype(int)
    unique_days["is_day_after_holiday"] = (unique_days["is_holiday_any"].shift(1).fillna(0.0) > 0).astype(int)

    merge_columns = [
        "interaction_date",
        "week_of_month",
        "is_month_start",
        "is_month_end",
        "is_day_after_holiday",
    ]

    enriched = enriched.merge(
        unique_days[merge_columns],
        on="interaction_date",
        how="left",
    )

    for column in merge_columns[1:]:
        enriched[column] = pd.to_numeric(enriched[column], errors="coerce").fillna(0).astype(int)

    return enriched


def _add_same_slot_history_features(enriched: pd.DataFrame) -> pd.DataFrame:
    same_slot_volume = enriched.groupby("slot_index")["volume"]
    enriched["same_slot_mean_3_day"] = (
        same_slot_volume.shift(1).rolling(window=3, min_periods=3).mean().reset_index(level=0, drop=True)
    )
    enriched["same_slot_mean_7_day"] = (
        same_slot_volume.shift(1).rolling(window=7, min_periods=7).mean().reset_index(level=0, drop=True)
    )
    return enriched


def _add_features(df: pd.DataFrame, channel: str) -> tuple[pd.DataFrame, int, int]:
    enriched = df.copy()
    slots_per_day = get_slots_per_day(channel)
    lag_one_day = slots_per_day
    lag_two_day = slots_per_day * 2
    lag_three_day = slots_per_day * 3
    lag_one_week = slots_per_day * 7

    enriched["interaction_date"] = pd.to_datetime(enriched["interaction_date"])
    enriched["day_of_week"] = enriched["datetime"].dt.weekday
    enriched["month"] = enriched["datetime"].dt.month
    enriched["day"] = enriched["datetime"].dt.day
    enriched["hour"] = enriched["datetime"].dt.hour
    enriched["minute"] = enriched["datetime"].dt.minute
    enriched["is_weekend"] = (enriched["day_of_week"] >= 5).astype(int)
    enriched["minute_of_day"] = enriched["hour"] * 60 + enriched["minute"]
    enriched["slot_index"] = enriched.groupby(enriched["datetime"].dt.date).cumcount()
    enriched["slot_sin"] = np.sin(2 * np.pi * enriched["slot_index"] / slots_per_day)
    enriched["slot_cos"] = np.cos(2 * np.pi * enriched["slot_index"] / slots_per_day)
    enriched["is_opening_slot"] = (enriched["slot_index"] == 0).astype(int)
    enriched["is_closing_slot"] = (enriched["slot_index"] == (slots_per_day - 1)).astype(int)
    enriched["is_first_hour"] = (enriched["slot_index"] <= 1).astype(int)
    enriched["is_last_hour"] = (enriched["slot_index"] >= (slots_per_day - 2)).astype(int)

    enriched["shift_label"] = enriched["minute_of_day"].apply(lambda value: get_shift_label(channel, int(value)))
    enriched["is_morning_shift"] = (enriched["shift_label"] == "morning").astype(int)
    enriched["is_afternoon_shift"] = (enriched["shift_label"] == "afternoon").astype(int)

    enriched["dow_sin"] = np.sin(2 * np.pi * enriched["day_of_week"] / 7)
    enriched["dow_cos"] = np.cos(2 * np.pi * enriched["day_of_week"] / 7)
    enriched["month_sin"] = np.sin(2 * np.pi * enriched["month"] / 12)
    enriched["month_cos"] = np.cos(2 * np.pi * enriched["month"] / 12)
    enriched["minute_sin"] = np.sin(2 * np.pi * enriched["minute_of_day"] / 1440)
    enriched["minute_cos"] = np.cos(2 * np.pi * enriched["minute_of_day"] / 1440)

    enriched = _add_calendar_features(enriched)

    enriched["lag_volume_1"] = enriched["volume"].shift(1)
    enriched["lag_volume_2"] = enriched["volume"].shift(2)
    enriched["lag_volume_3"] = enriched["volume"].shift(3)
    enriched["lag_volume_1_day"] = enriched["volume"].shift(lag_one_day)
    enriched["lag_volume_2_day"] = enriched["volume"].shift(lag_two_day)
    enriched["lag_volume_3_day"] = enriched["volume"].shift(lag_three_day)
    enriched["lag_volume_1_week"] = enriched["volume"].shift(lag_one_week)

    enriched["rolling_mean_3"] = enriched["volume"].shift(1).rolling(window=3).mean()
    enriched["rolling_mean_6"] = enriched["volume"].shift(1).rolling(window=6).mean()
    enriched["rolling_mean_1_day"] = enriched["volume"].shift(1).rolling(window=lag_one_day).mean()
    enriched["rolling_mean_2_day"] = enriched["volume"].shift(1).rolling(window=lag_two_day).mean()
    enriched["rolling_mean_3_day"] = enriched["volume"].shift(1).rolling(window=lag_three_day).mean()

    enriched = _add_same_slot_history_features(enriched)

    enriched["lag_aht_1"] = enriched["aht"].shift(1)
    enriched["volume_diff_1"] = enriched["volume"].diff(1)
    enriched["volume_diff_1_day"] = enriched["volume"].diff(lag_one_day)

    enriched = enriched.dropna().reset_index(drop=True)
    if enriched.empty:
        raise ValueError("Después de generar features para inferencia, el dataset quedó vacío.")

    enriched["interaction_date"] = enriched["interaction_date"].dt.date
    return enriched


def _prepare_recent_sequence(df: pd.DataFrame, metadata: dict, x_scaler) -> np.ndarray:
    feature_columns = metadata["feature_columns"]
    time_steps = int(metadata["time_steps"])

    if len(df) < time_steps:
        raise ValueError(
            f"No hay suficientes registros para inferencia LSTM. Se requieren al menos {time_steps} y solo hay {len(df)}."
        )

    scaled_features = x_scaler.transform(df[feature_columns])
    sequence = scaled_features[-time_steps:]
    return np.array([sequence], dtype=np.float32)

def _build_next_slot_reference_value(base_df: pd.DataFrame, channel: str, next_slot_index: int) -> float:
    slots_per_day = get_slots_per_day(channel)
    if base_df.empty:
        return 0.0

    slot_df = base_df.copy()
    slot_df["slot_index"] = slot_df.groupby(slot_df["datetime"].dt.date).cumcount()
    same_slot_df = (
        slot_df[slot_df["slot_index"] == int(next_slot_index)]
        .sort_values("datetime")
        .reset_index(drop=True)
    )
    if same_slot_df.empty:
        return 0.0

    lag_1 = float(same_slot_df["volume"].iloc[-1]) if not same_slot_df.empty else 0.0
    same_3 = float(same_slot_df["volume"].tail(3).mean()) if len(same_slot_df) >= 1 else lag_1
    same_7 = float(same_slot_df["volume"].tail(7).mean()) if len(same_slot_df) >= 1 else same_3
    return _build_late_slot_reference_value(lag_1, same_3, same_7)


def _predict_next_value(
    model,
    y_scaler,
    metadata: dict,
    feature_df: pd.DataFrame,
    x_scaler,
    next_slot_index: int,
    base_df: pd.DataFrame,
    channel: str,
) -> float:
    X_input = _prepare_recent_sequence(feature_df, metadata, x_scaler)
    y_pred_scaled = model.predict(X_input, verbose=0).flatten()
    y_pred_real = _inverse_target_transform(y_pred_scaled, y_scaler)
    late_slot_reference_value = _build_next_slot_reference_value(base_df, channel, next_slot_index)
    return _apply_prediction_postprocess(
        y_pred_real,
        metadata,
        slot_index=next_slot_index,
        late_slot_reference_value=late_slot_reference_value,
    )


def _build_aht_profile(base_df: pd.DataFrame, channel: str) -> dict:
    if base_df.empty:
        raise ValueError(f"No hay base histórica suficiente para construir el perfil de AHT del canal '{channel}'.")

    slots_per_day = get_slots_per_day(channel)
    profile_df = base_df.copy()
    profile_df["slot_index"] = profile_df.groupby(profile_df["datetime"].dt.date).cumcount()

    per_slot = (
        profile_df.groupby("slot_index", as_index=False)["aht"]
        .median()
        .sort_values("slot_index")
    )

    overall_aht = float(profile_df["aht"].median()) if not profile_df["aht"].empty else 0.0
    if np.isnan(overall_aht):
        overall_aht = 0.0

    profile = {int(row["slot_index"]): float(row["aht"] or 0.0) for _, row in per_slot.iterrows()}

    last_known_aht = float(profile_df["aht"].iloc[-1] or 0.0)
    for slot_index in range(slots_per_day):
        profile.setdefault(slot_index, overall_aht if overall_aht > 0 else last_known_aht)

    return profile


def predict_next_operational_day_for_channel(db: Session, channel: str) -> dict:
    canonical_channel = canonicalize_channel(channel)
    model, scaler_bundle, metadata = load_lstm_artifacts(canonical_channel)

    x_scaler = scaler_bundle["x_scaler"]
    y_scaler = scaler_bundle["y_scaler"]

    base_df = _prepare_channel_dataframe(db, canonical_channel)
    feature_df = _add_features(base_df, canonical_channel)

    external_by_date = _build_external_variables_by_date(db)
    aht_profile = _build_aht_profile(base_df, canonical_channel)

    last_datetime = base_df["datetime"].max().to_pydatetime()
    forecast_date = get_next_operational_day_date(last_datetime, canonical_channel)
    forecast_slots = get_operational_day_datetimes(forecast_date, canonical_channel)
    forecast_start_datetime = get_operational_day_start_datetime(forecast_date, canonical_channel)
    external_vars = external_by_date.get(forecast_date, _default_external_variables())
    model_version = metadata.get("model_version", "lstm_v4_1_feature_pruned_late_uplift")

    intervals: list[dict] = []
    recursive_base_df = base_df.copy()
    recursive_feature_df = feature_df.copy()

    for slot_index, slot_datetime in enumerate(forecast_slots):
        predicted_value = _predict_next_value(
            model=model,
            y_scaler=y_scaler,
            metadata=metadata,
            feature_df=recursive_feature_df,
            x_scaler=x_scaler,
            next_slot_index=slot_index,
            base_df=recursive_base_df,
            channel=canonical_channel,
        )

        minute_of_day = slot_datetime.hour * 60 + slot_datetime.minute
        shift_label = get_shift_label(canonical_channel, minute_of_day)

        interval_payload = {
            "channel": canonical_channel,
            "forecast_date": forecast_date,
            "forecast_datetime": slot_datetime,
            "interval_time": slot_datetime.time(),
            "slot_index": slot_index,
            "shift_label": shift_label,
            "predicted_value": predicted_value,
            "model_version": model_version,
        }
        intervals.append(interval_payload)

        new_row = {
            "interaction_date": forecast_date,
            "interval_time": slot_datetime.time(),
            "channel": canonical_channel,
            "volume": predicted_value,
            "aht": float(aht_profile.get(slot_index, 0.0)),
            "datetime": slot_datetime,
            "is_holiday_peru": float(external_vars.get("is_holiday_peru", 0.0)),
            "is_holiday_spain": float(external_vars.get("is_holiday_spain", 0.0)),
            "is_holiday_mexico": float(external_vars.get("is_holiday_mexico", 0.0)),
            "campaign_day": float(external_vars.get("campaign_day", 0.0)),
            "absenteeism_rate": float(external_vars.get("absenteeism_rate", 0.0)),
            "is_holiday_any": float(external_vars.get("is_holiday_any", 0.0)),
        }

        recursive_base_df = pd.concat([recursive_base_df, pd.DataFrame([new_row])], ignore_index=True)
        recursive_base_df = recursive_base_df.sort_values("datetime").reset_index(drop=True)
        recursive_feature_df = _add_features(recursive_base_df, canonical_channel)

    total_predicted_value = float(sum(item["predicted_value"] for item in intervals))

    return {
        "channel": canonical_channel,
        "forecast_date": forecast_date,
        "forecast_start_datetime": forecast_start_datetime,
        "total_predicted_value": total_predicted_value,
        "intervals_generated": len(intervals),
        "intervals": intervals,
        "model_version": model_version,
    }


def predict_next_volume_for_channel(db: Session, channel: str) -> dict:
    batch_prediction = predict_next_operational_day_for_channel(db, channel)
    first_interval = batch_prediction["intervals"][0]

    return {
        "channel": batch_prediction["channel"],
        "forecast_date": first_interval["forecast_datetime"],
        "predicted_value": first_interval["predicted_value"],
        "model_version": batch_prediction["model_version"],
    }
