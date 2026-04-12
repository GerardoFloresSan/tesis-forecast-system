# -*- coding: utf-8 -*-
"""
Evaluacion segmentada del modelo LSTM para canal Choice.
Metricas por turno, dia de semana, hora y distribucion de errores.
Alineado con requerimientos de accuracy de la tesis.
"""
import sys
import os
import io
import math

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sqlalchemy import create_engine
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import load_model

from app.core.config import settings
from scripts.train_lstm import (
    load_and_prepare_dataset,
    add_time_and_lag_features,
    create_sequences,
    slugify_channel,
)

# ── Constantes alineadas con train_lstm.py ──────────────────────────────────
CHANNEL_TO_TRAIN = "Choice"
TIME_STEPS = 32
TRAIN_SPLIT = 0.8

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "data" / "models"

SEP  = "=" * 70
SEP2 = "-" * 70


def sep(titulo=""):
    if titulo:
        print(f"\n{SEP}")
        print(f"  {titulo}")
        print(SEP)
    else:
        print(SEP)


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def turno(hora: int) -> str:
    if 6 <= hora <= 13:
        return "Manana (06-13)"
    elif 14 <= hora <= 21:
        return "Tarde  (14-21)"
    else:
        return "Noche  (22-05)"


def load_artifacts():
    slug = slugify_channel(CHANNEL_TO_TRAIN)
    model_path    = MODEL_DIR / f"lstm_{slug}.keras"
    scaler_path   = MODEL_DIR / f"lstm_{slug}_scaler.joblib"
    metadata_path = MODEL_DIR / f"lstm_{slug}_metadata.joblib"

    for p in (model_path, scaler_path, metadata_path):
        if not p.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {p}")

    model = load_model(model_path, compile=False)
    scaler_artifact = joblib.load(scaler_path)
    metadata = joblib.load(metadata_path)

    if isinstance(scaler_artifact, dict):
        x_scaler = scaler_artifact["x_scaler"]
        y_scaler = scaler_artifact["y_scaler"]
    else:
        x_scaler = y_scaler = scaler_artifact

    return model, x_scaler, y_scaler, metadata


def main():
    sep()
    print("  EVALUACION SEGMENTADA — LSTM Canal Choice")
    sep()

    # ── 1. Conexion y dataset ──────────────────────────────────────────────
    print("\nConectando a PostgreSQL...")
    engine = create_engine(settings.DATABASE_URL)

    print(f"Cargando dataset para canal: {CHANNEL_TO_TRAIN}")
    df_raw = load_and_prepare_dataset(engine, CHANNEL_TO_TRAIN)
    df = add_time_and_lag_features(df_raw)
    print(f"Registros utiles tras lags/rolling: {len(df):,}")

    # ── 2. Cargar artefactos ───────────────────────────────────────────────
    print("Cargando artefactos del modelo...")
    model, x_scaler, y_scaler, metadata = load_artifacts()

    feature_columns = metadata["feature_columns"]
    time_steps      = metadata.get("time_steps", TIME_STEPS)
    model_version   = metadata.get("model_version", "desconocida")
    print(f"Modelo      : {model_version}")
    print(f"TIME_STEPS  : {time_steps}")
    print(f"Features    : {len(feature_columns)}")

    # Verificar que todas las features existen en el df
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Features faltantes en el dataset: {missing}")

    # ── 3. Split temporal 80/20 ────────────────────────────────────────────
    split_index = int(len(df) * TRAIN_SPLIT)

    dataset = df[feature_columns].copy()
    target  = df[["volume"]].copy()

    train_features = dataset.iloc[:split_index]
    test_features  = dataset.iloc[split_index:]
    train_target   = target.iloc[:split_index]
    test_target    = target.iloc[split_index:]

    # ── 4. Escalar con los scalers guardados ───────────────────────────────
    # Fit en train (replicar comportamiento de entrenamiento)
    train_features_scaled = x_scaler.transform(train_features)
    test_features_scaled  = x_scaler.transform(test_features)
    test_target_scaled    = y_scaler.transform(test_target)

    # ── 5. Crear secuencias ────────────────────────────────────────────────
    X_test, y_test_scaled = create_sequences(
        features=test_features_scaled,
        target=test_target_scaled,
        time_steps=time_steps,
    )

    if len(X_test) == 0:
        raise ValueError("No hay suficientes datos de test para crear secuencias.")

    print(f"X_test shape: {X_test.shape}")

    # ── 6. Predicciones ────────────────────────────────────────────────────
    print("Generando predicciones...")
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = np.maximum(y_scaler.inverse_transform(y_pred_scaled).flatten(), 0)
    y_real = y_scaler.inverse_transform(y_test_scaled).flatten()

    # ── 7. Alinear con metadatos temporales ───────────────────────────────
    # Las secuencias empiezan en time_steps dentro del bloque test
    test_df = df.iloc[split_index:].copy().reset_index(drop=True)
    meta_df = test_df.iloc[time_steps:].reset_index(drop=True)

    # Proteccion de longitud
    n = min(len(y_real), len(y_pred), len(meta_df))
    y_real  = y_real[:n]
    y_pred  = y_pred[:n]
    meta_df = meta_df.iloc[:n].copy()

    meta_df["y_real"]  = y_real
    meta_df["y_pred"]  = y_pred
    meta_df["error_abs"]  = np.abs(y_real - y_pred)
    meta_df["error_sign"] = y_pred - y_real          # positivo = sobreestimacion
    meta_df["hora"]    = meta_df["datetime"].dt.hour
    meta_df["dow"]     = meta_df["datetime"].dt.dayofweek
    meta_df["turno"]   = meta_df["hora"].apply(turno)

    # ── 8a. METRICAS GLOBALES ─────────────────────────────────────────────
    sep("a) METRICAS GLOBALES")
    mae  = mean_absolute_error(y_real, y_pred)
    rmse = math.sqrt(mean_squared_error(y_real, y_pred))
    r2   = r2_score(y_real, y_pred)
    mape = calculate_mape(y_real, y_pred)
    bias = float(np.mean(meta_df["error_sign"]))

    print(f"  Registros evaluados : {n:,}")
    print(f"  MAE                 : {mae:.4f}")
    print(f"  RMSE                : {rmse:.4f}")
    print(f"  MAPE                : {mape:.2f}%")
    print(f"  R2                  : {r2:.4f}")
    print(f"  Bias (media error)  : {bias:+.4f}  {'(sobreestima)' if bias > 0 else '(subestima)'}")

    # ── 8b. MAPE POR TURNO ────────────────────────────────────────────────
    sep("b) MAPE POR TURNO")
    print(f"  {'Turno':<22}  {'N':>6}  {'MAPE':>8}  {'MAE':>8}  {'RMSE':>8}")
    print(f"  {SEP2}")
    for t_name, grp in meta_df.groupby("turno", sort=True):
        t_mape = calculate_mape(grp["y_real"].values, grp["y_pred"].values)
        t_mae  = mean_absolute_error(grp["y_real"], grp["y_pred"])
        t_rmse = math.sqrt(mean_squared_error(grp["y_real"], grp["y_pred"]))
        bar = "#" * int(min(t_mape, 50) / 2)
        print(f"  {t_name:<22}  {len(grp):>6}  {t_mape:>7.2f}%  {t_mae:>8.2f}  {t_rmse:>8.2f}  {bar}")

    # ── 8c. MAPE POR DIA DE SEMANA ────────────────────────────────────────
    sep("c) MAPE POR DIA DE SEMANA")
    dias = ["Lunes", "Martes", "Miercoles", "Jueves", "Viernes", "Sabado", "Domingo"]
    print(f"  {'Dia':<12}  {'N':>6}  {'MAPE':>8}  {'MAE':>8}  {'Bias':>8}")
    print(f"  {SEP2}")
    for dow_idx, grp in meta_df.groupby("dow", sort=True):
        d_mape = calculate_mape(grp["y_real"].values, grp["y_pred"].values)
        d_mae  = mean_absolute_error(grp["y_real"], grp["y_pred"])
        d_bias = float(np.mean(grp["error_sign"]))
        bar = "#" * int(min(d_mape, 50) / 2)
        print(f"  {dias[dow_idx]:<12}  {len(grp):>6}  {d_mape:>7.2f}%  {d_mae:>8.2f}  {d_bias:>+8.2f}  {bar}")

    # ── 8d. MAPE POR HORA ─────────────────────────────────────────────────
    sep("d) MAPE POR HORA")
    print(f"  {'Hora':>4}  {'N':>5}  {'MAPE':>8}  {'MAE':>8}  {'Bias':>8}  Barra")
    print(f"  {SEP2}")
    horas_con_datos = meta_df.groupby("hora")
    for hora, grp in horas_con_datos:
        if len(grp) == 0:
            continue
        h_mape = calculate_mape(grp["y_real"].values, grp["y_pred"].values)
        h_mae  = mean_absolute_error(grp["y_real"], grp["y_pred"])
        h_bias = float(np.mean(grp["error_sign"]))
        bar = "#" * int(min(h_mape, 60) / 2)
        mape_str = f"{h_mape:>7.2f}%" if not math.isnan(h_mape) else "    N/A "
        print(f"  {hora:>4}h  {len(grp):>5}  {mape_str}  {h_mae:>8.2f}  {h_bias:>+8.2f}  {bar}")

    # ── 8e. TOP 10 INTERVALOS CON MAYOR ERROR ABSOLUTO ───────────────────
    sep("e) TOP 10 INTERVALOS CON MAYOR ERROR ABSOLUTO")
    top10 = meta_df.nlargest(10, "error_abs")[
        ["datetime", "hora", "turno", "y_real", "y_pred", "error_abs", "error_sign"]
    ].copy()
    top10["y_real"]     = top10["y_real"].round(1)
    top10["y_pred"]     = top10["y_pred"].round(1)
    top10["error_abs"]  = top10["error_abs"].round(1)
    top10["error_sign"] = top10["error_sign"].round(1)
    top10 = top10.reset_index(drop=True)
    top10.index += 1
    print(top10.to_string())

    # ── 8f. DISTRIBUCION DE ERRORES ───────────────────────────────────────
    sep("f) DISTRIBUCION DE ERRORES ABSOLUTOS")
    umbrales = [
        ("< 5  interacciones", meta_df["error_abs"] < 5),
        ("< 10 interacciones", meta_df["error_abs"] < 10),
        ("< 20 interacciones", meta_df["error_abs"] < 20),
        ("> 30 interacciones (outliers)", meta_df["error_abs"] > 30),
    ]
    print(f"  {'Rango':<35}  {'N':>6}  {'%':>7}")
    print(f"  {SEP2}")
    for label, mask in umbrales:
        cnt = int(mask.sum())
        pct = cnt / n * 100
        bar = "#" * int(pct / 2)
        print(f"  {label:<35}  {cnt:>6}  {pct:>6.1f}%  {bar}")

    print(f"\n  Percentiles del error absoluto:")
    for p in [50, 75, 90, 95, 99]:
        val = float(np.percentile(meta_df["error_abs"], p))
        print(f"    P{p:>2}: {val:.2f}")

    # ── 8g. BIAS ANALYSIS ─────────────────────────────────────────────────
    sep("g) ANALISIS DE BIAS")
    bias_pos = (meta_df["error_sign"] > 0).sum()
    bias_neg = (meta_df["error_sign"] < 0).sum()
    bias_zer = (meta_df["error_sign"] == 0).sum()
    print(f"  Error medio (bias)          : {bias:+.4f}")
    print(f"  Predicciones > real (sobre) : {bias_pos:,}  ({bias_pos/n*100:.1f}%)")
    print(f"  Predicciones < real (bajo)  : {bias_neg:,}  ({bias_neg/n*100:.1f}%)")
    print(f"  Predicciones exactas        : {bias_zer:,}  ({bias_zer/n*100:.1f}%)")
    print(f"  Std del error con signo     : {float(np.std(meta_df['error_sign'])):.4f}")

    # ── 9. RESUMEN EJECUTIVO ──────────────────────────────────────────────
    sep("RESUMEN EJECUTIVO")

    mape_manana = calculate_mape(
        meta_df[meta_df["turno"] == "Manana (06-13)"]["y_real"].values,
        meta_df[meta_df["turno"] == "Manana (06-13)"]["y_pred"].values,
    )
    mape_tarde = calculate_mape(
        meta_df[meta_df["turno"] == "Tarde  (14-21)"]["y_real"].values,
        meta_df[meta_df["turno"] == "Tarde  (14-21)"]["y_pred"].values,
    )
    pct_lt10 = (meta_df["error_abs"] < 10).sum() / n * 100
    pct_gt30 = (meta_df["error_abs"] > 30).sum() / n * 100

    objetivo_mape = 15.0  # umbral de tesis
    estado_global = "CUMPLE" if mape <= objetivo_mape else "NO CUMPLE"
    estado_manana = "CUMPLE" if (not math.isnan(mape_manana) and mape_manana <= objetivo_mape) else "NO CUMPLE"
    estado_tarde  = "CUMPLE" if (not math.isnan(mape_tarde)  and mape_tarde  <= objetivo_mape) else "NO CUMPLE"

    print(f"""
  Modelo evaluado  : {model_version}
  Canal            : {CHANNEL_TO_TRAIN}
  Registros test   : {n:,}
  Objetivo MAPE    : <= {objetivo_mape}%
  -------------------------------------------------------------------
  MAPE Global      : {mape:.2f}%     [{estado_global}]
  MAPE Manana      : {mape_manana:.2f}%     [{estado_manana}]
  MAPE Tarde       : {mape_tarde:.2f}%     [{estado_tarde}]
  MAE              : {mae:.4f}
  RMSE             : {rmse:.4f}
  R2               : {r2:.4f}
  Bias             : {bias:+.4f}  {'(tiende a sobreestimar)' if bias > 0 else '(tiende a subestimar)'}
  -------------------------------------------------------------------
  Precision < 10   : {pct_lt10:.1f}% de predicciones con error < 10 interacciones
  Outliers error   : {pct_gt30:.1f}% de predicciones con error > 30 interacciones
""")

    veredictos = []
    if mape > objetivo_mape:
        veredictos.append(f"MAPE global {mape:.2f}% supera el objetivo {objetivo_mape}% -- revisar arquitectura o features")
    if not math.isnan(mape_manana) and mape_manana > objetivo_mape:
        veredictos.append(f"Turno Manana supera objetivo ({mape_manana:.2f}%) -- horario critico de atencion")
    if pct_gt30 > 10:
        veredictos.append(f"{pct_gt30:.1f}% de outliers de error -- verificar intervalos de baja actividad")
    if abs(bias) > 5:
        veredictos.append(f"Bias significativo ({bias:+.2f}) -- modelo sistematicamente {'sobre' if bias>0 else 'sub'}estima")

    if not veredictos:
        print("  [OK] El modelo cumple todos los objetivos de accuracy de la tesis.")
    else:
        print("  Puntos de atencion:")
        for v in veredictos:
            print(f"    - {v}")

    sep()


if __name__ == "__main__":
    main()
