from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.pop("TF_USE_LEGACY_KERAS", None)

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model

from app.core.config import settings
from app.utils.channel_rules import canonicalize_channel, slugify_channel
from scripts.train_lstm import (
    BATCH_SIZE,
    EPOCHS,
    TRAIN_SPLIT,
    VALIDATION_SPLIT_WITHIN_TRAIN,
    BIAS_CORRECTION_FACTOR,
    add_time_and_lag_features,
    apply_prediction_postprocess,
    build_slot_bias_adjustments,
    compute_effective_bias_adjustment,
    build_feature_columns,
    build_lstm_model,
    calculate_bias,
    calculate_mape,
    calculate_smape,
    calculate_wape,
    create_sequences,
    inverse_target_transform,
    load_and_prepare_dataset,
    set_seeds,
)

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "data" / "models"
SEP = "=" * 90
SEP2 = "-" * 90
DEFAULT_CHANNEL = "Choice"
DEFAULT_FOLD_DAYS = 7
DEFAULT_N_FOLDS = 4
DEFAULT_MIN_TRAIN_DAYS = 28
DEFAULT_MODE = "both"
VOLUME_BINS = [-np.inf, 20, 50, 100, np.inf]
VOLUME_LABELS = ["bajo (<=20)", "medio (21-50)", "alto (51-100)", "muy_alto (>100)"]


def sep(title: str = "") -> None:
    print(f"\n{SEP}")
    if title:
        print(f"  {title}")
        print(SEP)


def turno(hour: int) -> str:
    if 6 <= hour <= 13:
        return "Manana (06-13)"
    if 14 <= hour <= 21:
        return "Tarde  (14-21)"
    return "Noche  (22-05)"


def safe_float(value: Any, digits: int = 4) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return round(value, digits)


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()
    if len(y_true) == 0:
        return {
            "n": 0,
            "mae": None,
            "rmse": None,
            "mape": None,
            "wape": None,
            "smape": None,
            "r2": None,
            "bias": None,
        }

    r2_value = r2_score(y_true, y_pred) if len(y_true) > 1 else None
    return {
        "n": int(len(y_true)),
        "mae": safe_float(mean_absolute_error(y_true, y_pred)),
        "rmse": safe_float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": safe_float(calculate_mape(y_true, y_pred)),
        "wape": safe_float(calculate_wape(y_true, y_pred)),
        "smape": safe_float(calculate_smape(y_true, y_pred)),
        "r2": safe_float(r2_value),
        "bias": safe_float(calculate_bias(y_true, y_pred)),
    }


def summarize_grouped(df: pd.DataFrame, by: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, grp in df.groupby(by, sort=True):
        metrics = metrics_dict(grp["y_real"].values, grp["y_pred"].values)
        row = {"segment": str(key), **metrics}
        rows.append(row)
    rows.sort(key=lambda item: (item["mape"] is None, -(item["mape"] or -1)))
    return rows


def enrich_meta_df(meta_df: pd.DataFrame) -> pd.DataFrame:
    enriched = meta_df.copy()
    enriched["error_abs"] = np.abs(enriched["y_real"] - enriched["y_pred"])
    enriched["error_sign"] = enriched["y_pred"] - enriched["y_real"]
    enriched["hour"] = enriched["datetime"].dt.hour
    enriched["dow"] = enriched["datetime"].dt.dayofweek
    enriched["turno"] = enriched["hour"].apply(turno)
    enriched["volume_band"] = pd.cut(
        enriched["y_real"], bins=VOLUME_BINS, labels=VOLUME_LABELS, include_lowest=True
    ).astype(str)
    return enriched


def build_full_report(meta_df: pd.DataFrame, baseline_df: pd.DataFrame | None = None) -> dict[str, Any]:
    global_metrics = metrics_dict(meta_df["y_real"].values, meta_df["y_pred"].values)
    absolute_error_percentiles = {
        f"p{percentile}": safe_float(np.percentile(meta_df["error_abs"], percentile), 2)
        for percentile in [50, 75, 90, 95, 99]
    }
    top_errors = (
        meta_df.nlargest(10, "error_abs")
        [["datetime", "slot_index", "turno", "y_real", "y_pred", "error_abs", "error_sign"]]
        .assign(
            datetime=lambda x: x["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            y_real=lambda x: x["y_real"].round(2),
            y_pred=lambda x: x["y_pred"].round(2),
            error_abs=lambda x: x["error_abs"].round(2),
            error_sign=lambda x: x["error_sign"].round(2),
        )
        .to_dict(orient="records")
    )

    report = {
        "global": global_metrics,
        "percentiles_error_abs": absolute_error_percentiles,
        "by_shift": summarize_grouped(meta_df, "turno"),
        "by_day_of_week": summarize_grouped(meta_df, "dow"),
        "by_hour": summarize_grouped(meta_df, "hour"),
        "by_slot": summarize_grouped(meta_df, "slot_index"),
        "by_volume_band": summarize_grouped(meta_df, "volume_band"),
        "top_error_intervals": top_errors,
    }

    if baseline_df is not None and not baseline_df.empty:
        report["baseline_global"] = metrics_dict(
            baseline_df["y_real"].values, baseline_df["y_pred"].values
        )

    return report


def print_metric_line(label: str, value: float | None, suffix: str = "") -> None:
    if value is None:
        print(f"  {label:<24}: N/A")
    else:
        print(f"  {label:<24}: {value:.4f}{suffix}")


def print_global_block(title: str, metrics: dict[str, Any]) -> None:
    sep(title)
    print_metric_line("N", float(metrics["n"]) if metrics.get("n") is not None else None)
    print_metric_line("MAE", metrics.get("mae"))
    print_metric_line("RMSE", metrics.get("rmse"))
    print_metric_line("MAPE", metrics.get("mape"), "%")
    print_metric_line("WAPE", metrics.get("wape"), "%")
    print_metric_line("SMAPE", metrics.get("smape"), "%")
    print_metric_line("R2", metrics.get("r2"))
    print_metric_line("Bias", metrics.get("bias"))


def print_ranked_table(title: str, rows: list[dict[str, Any]], limit: int = 10) -> None:
    sep(title)
    if not rows:
        print("  Sin datos.")
        return

    print(f"  {'Segmento':<20} {'N':>6} {'MAPE':>10} {'WAPE':>10} {'Bias':>10}")
    print(f"  {SEP2}")
    for row in rows[:limit]:
        mape = "N/A" if row["mape"] is None else f"{row['mape']:.2f}%"
        wape = "N/A" if row["wape"] is None else f"{row['wape']:.2f}%"
        bias = "N/A" if row["bias"] is None else f"{row['bias']:+.2f}"
        print(f"  {row['segment']:<20} {row['n']:>6} {mape:>10} {wape:>10} {bias:>10}")


def load_saved_artifacts(channel: str):
    slug = slugify_channel(channel)
    model_path = MODEL_DIR / f"lstm_{slug}.keras"
    scaler_path = MODEL_DIR / f"lstm_{slug}_scaler.joblib"
    metadata_path = MODEL_DIR / f"lstm_{slug}_metadata.joblib"

    for path in (model_path, scaler_path, metadata_path):
        if not path.exists():
            raise FileNotFoundError(f"No se encontro el artefacto requerido: {path}")

    model = load_model(model_path, compile=False)
    scaler_artifact = joblib.load(scaler_path)
    metadata = joblib.load(metadata_path)

    if isinstance(scaler_artifact, dict):
        x_scaler = scaler_artifact["x_scaler"]
        y_scaler = scaler_artifact["y_scaler"]
    else:
        x_scaler = y_scaler = scaler_artifact

    return model, x_scaler, y_scaler, metadata


def evaluate_saved_model_holdout(df: pd.DataFrame, channel: str) -> dict[str, Any]:
    model, x_scaler, y_scaler, metadata = load_saved_artifacts(channel)

    feature_columns = metadata.get("feature_columns") or build_feature_columns()
    time_steps = int(metadata.get("time_steps", 34))
    split_index = int(len(df) * TRAIN_SPLIT)

    dataset = df[feature_columns].copy()
    target = np.log1p(df[["volume"]].copy())

    train_features = dataset.iloc[:split_index].copy()
    test_features = dataset.iloc[split_index:].copy()
    test_target = target.iloc[split_index:].copy()

    if len(train_features) <= time_steps or len(test_features) <= time_steps:
        raise ValueError("No hay suficientes datos para evaluar holdout con el time_steps actual.")

    test_features_scaled = x_scaler.transform(test_features)
    test_target_scaled = y_scaler.transform(test_target)
    X_test, y_test = create_sequences(test_features_scaled, test_target_scaled, time_steps)

    if len(X_test) == 0:
        raise ValueError("No se pudieron crear secuencias de test para holdout.")

    y_pred_scaled = model.predict(X_test, verbose=0).flatten()
    y_pred = inverse_target_transform(y_pred_scaled, y_scaler)
    y_real = inverse_target_transform(y_test.flatten(), y_scaler)

    test_df = df.iloc[split_index:].copy().reset_index(drop=True)
    meta_df = test_df.iloc[time_steps:].copy().reset_index(drop=True)
    slot_indices = meta_df["slot_index"].astype(int).values if not meta_df.empty else np.array([], dtype=int)
    y_pred = apply_prediction_postprocess(
        y_pred,
        validation_bias=float(metadata.get("validation_bias", 0.0)),
        slot_indices=slot_indices,
        slot_bias_adjustments={
            int(key): float(value)
            for key, value in (metadata.get("slot_bias_adjustments", {}) or {}).items()
        },
    )

    n = min(len(meta_df), len(y_real), len(y_pred))
    meta_df = meta_df.iloc[:n].copy()
    meta_df["y_real"] = y_real[:n]
    meta_df["y_pred"] = y_pred[:n]
    meta_df = enrich_meta_df(meta_df)

    baseline_df = test_df.iloc[time_steps:].copy().reset_index(drop=True)
    baseline_df = baseline_df.iloc[:n].copy()
    baseline_df["y_real"] = meta_df["y_real"].values
    baseline_df["y_pred"] = np.maximum(baseline_df["lag_volume_1_day"].values[:n], 0.0)

    return {
        "mode": "holdout_saved_model",
        "channel": channel,
        "model_version": metadata.get("model_version", "unknown"),
        "time_steps": time_steps,
        "feature_count": len(feature_columns),
        "report": build_full_report(meta_df, baseline_df),
    }


def split_train_validation(X_train_full: np.ndarray, y_train_full: np.ndarray):
    val_size = max(1, int(len(X_train_full) * VALIDATION_SPLIT_WITHIN_TRAIN))
    if val_size >= len(X_train_full):
        val_size = 1
    X_train = X_train_full[:-val_size]
    y_train = y_train_full[:-val_size]
    X_val = X_train_full[-val_size:]
    y_val = y_train_full[-val_size:]
    return X_train, y_train, X_val, y_val


def train_and_predict_fold(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_columns: list[str], time_steps: int):
    train_features = train_df[feature_columns].copy()
    test_features = test_df[feature_columns].copy()
    train_target = np.log1p(train_df[["volume"]].copy())
    test_target = np.log1p(test_df[["volume"]].copy())

    if len(train_features) <= time_steps or len(test_features) <= time_steps:
        raise ValueError("Fold sin suficientes registros para secuencias LSTM.")

    x_scaler = RobustScaler()
    y_scaler = StandardScaler()
    train_features_scaled = x_scaler.fit_transform(train_features)
    test_features_scaled = x_scaler.transform(test_features)
    train_target_scaled = y_scaler.fit_transform(train_target)
    test_target_scaled = y_scaler.transform(test_target)

    X_train_full, y_train_full = create_sequences(train_features_scaled, train_target_scaled, time_steps)
    X_test, y_test = create_sequences(test_features_scaled, test_target_scaled, time_steps)

    if len(X_train_full) == 0 or len(X_test) == 0:
        raise ValueError("Fold sin secuencias suficientes para entrenar/evaluar.")

    X_train, y_train, X_val, y_val = split_train_validation(X_train_full, y_train_full)

    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5, verbose=0)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping, reduce_lr],
        verbose=0,
        shuffle=False,
    )

    y_val_pred_scaled = model.predict(X_val, verbose=0).flatten()
    y_val_pred = inverse_target_transform(y_val_pred_scaled, y_scaler)
    y_val_real = inverse_target_transform(y_val.flatten(), y_scaler)
    validation_bias = calculate_bias(y_val_real, y_val_pred)

    train_meta_df = train_df.iloc[time_steps:].copy().reset_index(drop=True)
    val_meta_df = train_meta_df.iloc[-len(y_val_real):].copy().reset_index(drop=True)
    val_slot_indices = val_meta_df["slot_index"].astype(int).values if not val_meta_df.empty else np.array([], dtype=int)

    y_val_pred_after_global_bias = apply_prediction_postprocess(
        y_val_pred,
        validation_bias=validation_bias,
    )
    slot_bias_adjustments = build_slot_bias_adjustments(
        slot_indices=val_slot_indices,
        y_true=y_val_real,
        y_pred_after_global_bias=y_val_pred_after_global_bias,
        slots_per_day=time_steps,
    )

    y_test_pred_scaled = model.predict(X_test, verbose=0).flatten()
    y_test_pred = inverse_target_transform(y_test_pred_scaled, y_scaler)
    y_test_real = inverse_target_transform(y_test.flatten(), y_scaler)

    aligned_test_df = test_df.iloc[time_steps:].copy().reset_index(drop=True)
    test_slot_indices = (
        aligned_test_df["slot_index"].astype(int).values if not aligned_test_df.empty else np.array([], dtype=int)
    )
    y_test_pred = apply_prediction_postprocess(
        y_test_pred,
        validation_bias=validation_bias,
        slot_indices=test_slot_indices,
        slot_bias_adjustments=slot_bias_adjustments,
    )

    aligned_test_df = test_df.iloc[time_steps:].copy().reset_index(drop=True)
    n = min(len(aligned_test_df), len(y_test_real), len(y_test_pred))
    aligned_test_df = aligned_test_df.iloc[:n].copy()
    aligned_test_df["y_real"] = y_test_real[:n]
    aligned_test_df["y_pred"] = y_test_pred[:n]
    aligned_test_df = enrich_meta_df(aligned_test_df)

    baseline_df = test_df.iloc[time_steps:].copy().reset_index(drop=True)
    baseline_df = baseline_df.iloc[:n].copy()
    baseline_df["y_real"] = aligned_test_df["y_real"].values
    baseline_df["y_pred"] = np.maximum(baseline_df["lag_volume_1_day"].values[:n], 0.0)

    return {
        "meta_df": aligned_test_df,
        "baseline_df": baseline_df,
        "validation_bias": validation_bias,
        "effective_bias_adjustment": compute_effective_bias_adjustment(validation_bias),
        "slot_bias_adjustments": slot_bias_adjustments,
        "best_val_loss": min(history.history.get("val_loss", []) or [0.0]),
        "train_sequences": int(len(X_train)),
        "validation_sequences": int(len(X_val)),
        "test_sequences": int(len(X_test)),
    }


def evaluate_walk_forward(
    df: pd.DataFrame,
    channel: str,
    fold_days: int,
    n_folds: int,
    min_train_days: int,
) -> dict[str, Any]:
    unique_dates = sorted(pd.to_datetime(df["interaction_date"]).astype(str).unique())
    total_days = len(unique_dates)
    required_days = min_train_days + (fold_days * n_folds)
    if total_days < required_days:
        raise ValueError(
            f"No hay suficientes dias para walk-forward: total={total_days}, requerido={required_days}."
        )

    feature_columns = build_feature_columns()
    time_steps = int(df["slot_index"].max() + 1)
    folds: list[dict[str, Any]] = []
    all_meta = []
    all_baseline = []

    start_train_days = total_days - (fold_days * n_folds)
    if start_train_days < min_train_days:
        raise ValueError(
            f"La ventana inicial de train queda muy corta: {start_train_days} dias."
        )

    for fold_idx in range(n_folds):
        train_end = start_train_days + (fold_idx * fold_days)
        test_start = train_end
        test_end = test_start + fold_days

        train_dates = set(unique_dates[:train_end])
        test_dates = set(unique_dates[test_start:test_end])

        train_df = df[df["interaction_date"].astype(str).isin(train_dates)].copy().reset_index(drop=True)
        test_df = df[df["interaction_date"].astype(str).isin(test_dates)].copy().reset_index(drop=True)

        if train_df.empty or test_df.empty:
            continue

        fold_result = train_and_predict_fold(train_df, test_df, feature_columns, time_steps)
        meta_df = fold_result["meta_df"].copy()
        baseline_df = fold_result["baseline_df"].copy()

        fold_label = f"fold_{fold_idx + 1}"
        train_start_date = str(min(train_dates))
        train_end_date = str(max(train_dates))
        test_start_date = str(min(test_dates))
        test_end_date = str(max(test_dates))

        fold_report = build_full_report(meta_df, baseline_df)
        fold_summary = {
            "fold": fold_label,
            "train_start_date": train_start_date,
            "train_end_date": train_end_date,
            "test_start_date": test_start_date,
            "test_end_date": test_end_date,
            "train_days": len(train_dates),
            "test_days": len(test_dates),
            "train_sequences": fold_result["train_sequences"],
            "validation_sequences": fold_result["validation_sequences"],
            "test_sequences": fold_result["test_sequences"],
            "validation_bias": safe_float(fold_result["validation_bias"]),
            "effective_bias_adjustment": safe_float(fold_result["effective_bias_adjustment"]),
            "calibration_slots": len(fold_result["slot_bias_adjustments"]),
            "best_val_loss": safe_float(fold_result["best_val_loss"], 6),
            "global": fold_report["global"],
            "baseline_global": fold_report.get("baseline_global"),
        }
        folds.append(fold_summary)
        meta_df["fold"] = fold_label
        baseline_df["fold"] = fold_label
        all_meta.append(meta_df)
        all_baseline.append(baseline_df)

    if not all_meta:
        raise ValueError("No se pudo construir ningun fold valido para walk-forward.")

    combined_meta = pd.concat(all_meta, ignore_index=True)
    combined_baseline = pd.concat(all_baseline, ignore_index=True)
    combined_report = build_full_report(combined_meta, combined_baseline)

    return {
        "mode": "walk_forward",
        "channel": channel,
        "time_steps": time_steps,
        "fold_days": fold_days,
        "n_folds": len(folds),
        "min_train_days": min_train_days,
        "folds": folds,
        "report": combined_report,
    }


def build_comparison_summary(holdout_result: dict[str, Any] | None, walk_result: dict[str, Any] | None) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if holdout_result:
        summary["holdout_saved_model"] = holdout_result["report"]["global"]
        if "baseline_global" in holdout_result["report"]:
            summary["holdout_baseline"] = holdout_result["report"]["baseline_global"]
    if walk_result:
        summary["walk_forward"] = walk_result["report"]["global"]
        if "baseline_global" in walk_result["report"]:
            summary["walk_forward_baseline"] = walk_result["report"]["baseline_global"]
    return summary


def persist_report(channel: str, payload: dict[str, Any]) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    report_path = MODEL_DIR / f"lstm_{slugify_channel(channel)}_evaluation.json"
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    return report_path


def print_execution_summary(channel: str, holdout_result: dict[str, Any] | None, walk_result: dict[str, Any] | None):
    sep(f"RESUMEN EJECUTIVO — {channel}")
    if holdout_result:
        print_global_block("HOLDOUT DEL MODELO GUARDADO", holdout_result["report"]["global"])
        if "baseline_global" in holdout_result["report"]:
            print_global_block("BASELINE HOLDOUT", holdout_result["report"]["baseline_global"])
        print_ranked_table("PEORES SEGMENTOS HOLDOUT — POR SLOT", holdout_result["report"]["by_slot"], limit=10)
        print_ranked_table(
            "PEORES SEGMENTOS HOLDOUT — POR BANDA DE VOLUMEN",
            holdout_result["report"]["by_volume_band"],
            limit=10,
        )

    if walk_result:
        print_global_block("WALK-FORWARD AGREGADO", walk_result["report"]["global"])
        if "baseline_global" in walk_result["report"]:
            print_global_block("BASELINE WALK-FORWARD", walk_result["report"]["baseline_global"])
        print_ranked_table("PEORES SEGMENTOS WALK-FORWARD — POR SLOT", walk_result["report"]["by_slot"], limit=10)
        print_ranked_table(
            "PEORES SEGMENTOS WALK-FORWARD — POR BANDA DE VOLUMEN",
            walk_result["report"]["by_volume_band"],
            limit=10,
        )

        sep("DETALLE GLOBAL POR FOLD")
        print(f"  {'Fold':<10} {'Train':>8} {'Test':>8} {'MAPE':>10} {'WAPE':>10} {'Bias':>10}")
        print(f"  {SEP2}")
        for fold in walk_result["folds"]:
            metrics = fold["global"]
            print(
                f"  {fold['fold']:<10} {fold['train_days']:>8} {fold['test_days']:>8} "
                f"{(str(metrics['mape']) + '%') if metrics['mape'] is not None else 'N/A':>10} "
                f"{(str(metrics['wape']) + '%') if metrics['wape'] is not None else 'N/A':>10} "
                f"{metrics['bias'] if metrics['bias'] is not None else 'N/A':>10}"
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluacion holdout + walk-forward para el modelo LSTM")
    parser.add_argument("--channel", default=DEFAULT_CHANNEL, type=str)
    parser.add_argument("--mode", choices=["holdout", "walk-forward", "both"], default=DEFAULT_MODE)
    parser.add_argument("--fold-days", default=DEFAULT_FOLD_DAYS, type=int)
    parser.add_argument("--n-folds", default=DEFAULT_N_FOLDS, type=int)
    parser.add_argument("--min-train-days", default=DEFAULT_MIN_TRAIN_DAYS, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seeds()
    channel = canonicalize_channel(args.channel)

    sep(f"EVALUACION AVANZADA LSTM — {channel}")
    print(f"  Modo            : {args.mode}")
    print(f"  Fold days       : {args.fold_days}")
    print(f"  Numero de folds : {args.n_folds}")
    print(f"  Train minimo    : {args.min_train_days} dias")

    engine = create_engine(settings.DATABASE_URL)
    raw_df = load_and_prepare_dataset(engine, channel)
    df, slots_per_day, time_steps = add_time_and_lag_features(raw_df, channel)

    print(f"  Registros utiles : {len(df):,}")
    print(f"  Slots por dia    : {slots_per_day}")
    print(f"  Time steps       : {time_steps}")
    print(f"  Dias operativos  : {df['interaction_date'].nunique()}")

    holdout_result = None
    walk_result = None

    if args.mode in {"holdout", "both"}:
        holdout_result = evaluate_saved_model_holdout(df, channel)
    if args.mode in {"walk-forward", "both"}:
        walk_result = evaluate_walk_forward(
            df=df,
            channel=channel,
            fold_days=args.fold_days,
            n_folds=args.n_folds,
            min_train_days=args.min_train_days,
        )

    payload = {
        "channel": channel,
        "dataset": {
            "records": int(len(df)),
            "operational_days": int(df["interaction_date"].nunique()),
            "slots_per_day": int(slots_per_day),
            "time_steps": int(time_steps),
            "date_min": str(df["interaction_date"].min()),
            "date_max": str(df["interaction_date"].max()),
        },
        "comparison_summary": build_comparison_summary(holdout_result, walk_result),
        "holdout_saved_model": holdout_result,
        "walk_forward": walk_result,
    }

    report_path = persist_report(channel, payload)
    print_execution_summary(channel, holdout_result, walk_result)

    sep("ARTEFACTO GENERADO")
    print(f"  Reporte JSON guardado en: {report_path}")


if __name__ == "__main__":
    main()
