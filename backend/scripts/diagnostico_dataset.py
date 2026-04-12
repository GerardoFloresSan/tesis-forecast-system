# -*- coding: utf-8 -*-
"""
Diagnostico completo del dataset en PostgreSQL.
Tablas: historical_interactions, external_variables
"""
import sys
import os
import io

# Forzar UTF-8 en stdout para Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Asegurar que el path de backend este disponible
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from app.core.config import settings

SEP = "=" * 70


def sep(titulo=""):
    if titulo:
        print(f"\n{SEP}")
        print(f"  {titulo}")
        print(SEP)
    else:
        print(SEP)


def get_engine():
    url = settings.DATABASE_URL
    if not url:
        print("ERROR: DATABASE_URL no esta configurada.")
        sys.exit(1)
    url = url.replace("postgres://", "postgresql://", 1)
    return create_engine(url, echo=False)


def cargar_tabla(engine, tabla: str) -> pd.DataFrame:
    with engine.connect() as conn:
        df = pd.read_sql(text(f"SELECT * FROM {tabla}"), conn)
    return df


# -----------------------------------------------------------------------
# 1. Vision general
# -----------------------------------------------------------------------
def diagnostico_general(df: pd.DataFrame) -> pd.DataFrame:
    sep("1. VISION GENERAL -- historical_interactions")
    df["interaction_date"] = pd.to_datetime(df["interaction_date"])
    print(f"  Total de filas        : {len(df):,}")
    print(f"  Columnas              : {list(df.columns)}")
    fecha_min = df["interaction_date"].min().date()
    fecha_max = df["interaction_date"].max().date()
    dias_unicos = df["interaction_date"].dt.date.nunique()
    print(f"  Rango de fechas       : {fecha_min}  ->  {fecha_max}")
    print(f"  Dias unicos           : {dias_unicos}")
    return df


# -----------------------------------------------------------------------
# 2. Canales
# -----------------------------------------------------------------------
def diagnostico_canales(df: pd.DataFrame):
    sep("2. CANALES DISPONIBLES")
    canales = (
        df.groupby("channel")
        .agg(
            registros=("volume", "count"),
            fecha_min=("interaction_date", "min"),
            fecha_max=("interaction_date", "max"),
        )
        .reset_index()
        .sort_values("registros", ascending=False)
    )
    canales["fecha_min"] = canales["fecha_min"].dt.date
    canales["fecha_max"] = canales["fecha_max"].dt.date
    print(canales.to_string(index=False))


# -----------------------------------------------------------------------
# 3. Estadisticas Choice
# -----------------------------------------------------------------------
def diagnostico_choice(df: pd.DataFrame):
    sep("3. CANAL 'Choice' -- Estadisticas de Volume")
    ch = df[df["channel"] == "Choice"].copy()
    if ch.empty:
        print("  No hay datos para el canal 'Choice'.")
        return

    print(f"\n  Registros totales: {len(ch):,}")
    print("\n  --- describe() ---")
    desc = ch["volume"].describe()
    for k, v in desc.items():
        print(f"    {k:10s}: {v:.2f}")

    print("\n  --- Distribucion por rangos ---")
    total = len(ch)
    rangos = [
        ("= 0",  ch["volume"] == 0),
        ("<= 3", (ch["volume"] > 0)  & (ch["volume"] <= 3)),
        ("<= 5", (ch["volume"] > 3)  & (ch["volume"] <= 5)),
        ("<= 10",(ch["volume"] > 5)  & (ch["volume"] <= 10)),
        ("> 10", ch["volume"] > 10),
    ]
    for etiqueta, mascara in rangos:
        n = int(mascara.sum())
        pct = n / total * 100 if total > 0 else 0
        barra = "#" * int(pct / 2)
        print(f"    {etiqueta:6s}  {n:6,}  ({pct:5.1f}%)  {barra}")


# -----------------------------------------------------------------------
# 4. Calidad AHT
# -----------------------------------------------------------------------
def diagnostico_aht(df: pd.DataFrame):
    sep("4. CALIDAD DEL AHT")
    ch = df[df["channel"] == "Choice"].copy()
    total = len(ch)
    nulos = int(ch["aht"].isna().sum())
    ceros = int((ch["aht"] == 0).sum())
    validos = total - nulos - ceros
    print(f"  Canal 'Choice'  --  total registros : {total:,}")
    print(f"    Nulos (NaN)   : {nulos:,}  ({nulos/total*100:.1f}%)")
    print(f"    Ceros         : {ceros:,}  ({ceros/total*100:.1f}%)")
    print(f"    Validos       : {validos:,}  ({validos/total*100:.1f}%)")
    if validos > 0:
        desc = ch.loc[ch["aht"].notna() & (ch["aht"] != 0), "aht"].describe()
        print("\n  Estadisticas AHT (sin nulos/ceros):")
        for k, v in desc.items():
            print(f"    {k:10s}: {v:.2f}")

    print("\n  --- Nulos AHT por canal ---")
    resumen = df.groupby("channel").apply(
        lambda g: pd.Series({
            "total": len(g),
            "nulos_aht": int(g["aht"].isna().sum()),
            "ceros_aht": int((g["aht"] == 0).sum()),
        })
    ).reset_index()
    resumen["pct_nulos"] = (resumen["nulos_aht"] / resumen["total"] * 100).round(1)
    print(resumen.to_string(index=False))


# -----------------------------------------------------------------------
# 5. Patron semanal
# -----------------------------------------------------------------------
def patron_semanal(df: pd.DataFrame):
    sep("5. PATRON SEMANAL -- Media de Volume (canal 'Choice')")
    ch = df[df["channel"] == "Choice"].copy()
    if ch.empty:
        return
    dias = ["Lunes", "Martes", "Miercoles", "Jueves", "Viernes", "Sabado", "Domingo"]
    ch["dow"] = ch["interaction_date"].dt.dayofweek
    media_dow = ch.groupby("dow")["volume"].mean()
    max_vol = media_dow.max()
    print(f"  {'Dia':<12}  {'Media volume':>12}  Barra")
    for idx, media in media_dow.items():
        barra_len = int(media / max_vol * 30) if max_vol > 0 else 0
        barra = "|" * barra_len
        print(f"  {dias[idx]:<12}  {media:12.2f}  {barra}")


# -----------------------------------------------------------------------
# 6. Patron intradiario
# -----------------------------------------------------------------------
def patron_intradiario(df: pd.DataFrame):
    sep("6. PATRON INTRADIARIO -- Media de Volume por Hora (canal 'Choice')")
    ch = df[df["channel"] == "Choice"].copy()
    if ch.empty:
        return
    ch["hora"] = pd.to_datetime(
        ch["interval_time"].astype(str), format="%H:%M:%S"
    ).dt.hour
    media_hora = ch.groupby("hora")["volume"].mean()
    max_vol = media_hora.max()
    print(f"  {'Hora':>4}  {'Media':>8}  Barra")
    for hora, media in media_hora.items():
        barra_len = int(media / max_vol * 40) if max_vol > 0 else 0
        barra = "|" * barra_len
        print(f"  {hora:>4}h  {media:8.2f}  {barra}")


# -----------------------------------------------------------------------
# 7. Variables externas
# -----------------------------------------------------------------------
def diagnostico_ext_vars(engine):
    sep("7. VARIABLES EXTERNAS -- external_variables")
    try:
        with engine.connect() as conn:
            df_ext = pd.read_sql(text("SELECT * FROM external_variables"), conn)
    except Exception as e:
        print(f"  ERROR al consultar external_variables: {e}")
        return None

    if df_ext.empty:
        print("  Tabla vacia -- no hay variables externas registradas.")
        return df_ext

    df_ext["variable_date"] = pd.to_datetime(df_ext["variable_date"])
    print(f"  Total de filas: {len(df_ext):,}")
    print(
        f"  Rango de fechas: {df_ext['variable_date'].min().date()}  ->  "
        f"{df_ext['variable_date'].max().date()}"
    )
    print()
    resumen = (
        df_ext.groupby("variable_type")
        .agg(
            conteo=("variable_value", "count"),
            fecha_min=("variable_date", "min"),
            fecha_max=("variable_date", "max"),
            valor_min=("variable_value", "min"),
            valor_max=("variable_value", "max"),
            valor_medio=("variable_value", "mean"),
        )
        .reset_index()
    )
    resumen["fecha_min"] = resumen["fecha_min"].dt.date
    resumen["fecha_max"] = resumen["fecha_max"].dt.date
    resumen["valor_medio"] = resumen["valor_medio"].round(3)
    print(resumen.to_string(index=False))
    return df_ext


# -----------------------------------------------------------------------
# 8. Feriados
# -----------------------------------------------------------------------
def diagnostico_feriados(df_ext):
    sep("8. DIAS MARCADOS COMO FERIADO (is_holiday)")
    if df_ext is None or df_ext.empty:
        print("  No hay datos de variables externas.")
        return

    feriados = df_ext[
        df_ext["variable_type"].str.lower().str.contains(
            "holiday|feriado|holid", na=False
        )
    ].copy()

    if feriados.empty:
        # Intentar con variable_value == 1 como flag
        feriados = df_ext[df_ext["variable_value"] == 1].copy()
        if feriados.empty:
            print("  No se encontraron registros de feriados.")
            return
        print("  (Mostrando registros con variable_value == 1)\n")

    feriados = feriados.sort_values("variable_date")
    print(f"  Total feriados encontrados: {len(feriados)}")
    print()
    cols = [c for c in ["variable_date", "variable_type", "variable_value", "description"] if c in feriados.columns]
    print(feriados[cols].to_string(index=False))


# -----------------------------------------------------------------------
# 9. Outliers IQR
# -----------------------------------------------------------------------
def diagnostico_outliers(df: pd.DataFrame):
    sep("9. OUTLIERS DE VOLUME -- Metodo IQR (canal 'Choice')")
    ch = df[df["channel"] == "Choice"]["volume"].copy()
    if ch.empty:
        return
    Q1 = ch.quantile(0.25)
    Q3 = ch.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers_bajo = ch[ch < lower]
    outliers_alto = ch[ch > upper]
    total_out = len(outliers_bajo) + len(outliers_alto)
    pct = total_out / len(ch) * 100 if len(ch) > 0 else 0

    print(f"  Q1={Q1:.1f}  Q3={Q3:.1f}  IQR={IQR:.1f}")
    print(f"  Limite inferior : {lower:.1f}")
    print(f"  Limite superior : {upper:.1f}")
    print(f"  Outliers bajos  : {len(outliers_bajo):,}  (< {lower:.1f})")
    print(f"  Outliers altos  : {len(outliers_alto):,}  (> {upper:.1f})")
    print(f"  Total outliers  : {total_out:,}  ({pct:.1f}% del total)")
    if len(outliers_alto) > 0:
        print(f"\n  Top 10 valores mas altos:")
        top = ch.nlargest(10).values
        for v in top:
            print(f"    {int(v)}")


# -----------------------------------------------------------------------
# 10. Resumen ejecutivo
# -----------------------------------------------------------------------
def resumen_ejecutivo(df: pd.DataFrame, df_ext):
    sep("10. RESUMEN EJECUTIVO")
    ch = df[df["channel"] == "Choice"]
    total = len(df)
    total_ch = len(ch)
    canales = df["channel"].nunique()
    dias = df["interaction_date"].dt.date.nunique()

    nulos_aht = int(ch["aht"].isna().sum())
    pct_nulos = nulos_aht / total_ch * 100 if total_ch > 0 else 0
    ceros_vol = int((ch["volume"] == 0).sum())
    pct_ceros = ceros_vol / total_ch * 100 if total_ch > 0 else 0

    Q1 = ch["volume"].quantile(0.25)
    Q3 = ch["volume"].quantile(0.75)
    IQR = Q3 - Q1
    upper = Q3 + 1.5 * IQR
    outliers = int((ch["volume"] > upper).sum())
    pct_out = outliers / total_ch * 100 if total_ch > 0 else 0

    ext_ok = df_ext is not None and not df_ext.empty

    aht_estado = "** ALTO **" if pct_nulos > 20 else "OK"
    vol_estado  = "** REVISAR **" if pct_ceros > 30 else "OK"
    out_estado  = "** REVISAR **" if pct_out > 5 else "OK"
    ext_estado  = "Disponibles" if ext_ok else "** VACIAS / NO DISPONIBLES **"

    print(f"""
  Dataset: historical_interactions
  -------------------------------------------------------------------
  * Filas totales         : {total:,}
  * Canales               : {canales}
  * Dias cubiertos        : {dias}
  * Registros Choice      : {total_ch:,}

  Calidad de datos (Choice):
  * AHT nulos             : {nulos_aht:,} ({pct_nulos:.1f}%)  {aht_estado}
  * Volume = 0            : {ceros_vol:,} ({pct_ceros:.1f}%)  {vol_estado}
  * Outliers IQR          : {outliers:,} ({pct_out:.1f}%)    {out_estado}

  Variables externas      : {ext_estado}

  VEREDICTO:""")

    problemas = []
    if pct_nulos > 20:
        problemas.append(f"AHT con {pct_nulos:.0f}% de nulos -- considerar imputacion o exclusion")
    if pct_ceros > 30:
        problemas.append(f"Volume tiene {pct_ceros:.0f}% de ceros -- evaluar si son intervalos sin actividad")
    if pct_out > 5:
        problemas.append(f"Outliers IQR representan {pct_out:.0f}% -- posibles errores de ingesta")
    if not ext_ok:
        problemas.append("No hay variables externas -- el modelo no contara con features adicionales")

    if not problemas:
        print("  [OK] Dataset en buen estado. Listo para entrenamiento.")
    else:
        print("  Se detectaron los siguientes puntos a atender:")
        for p in problemas:
            print(f"    - {p}")
    print()
    sep()


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------
def main():
    print()
    sep()
    print("  DIAGNOSTICO COMPLETO DEL DATASET")
    db_url_display = settings.DATABASE_URL
    if db_url_display:
        db_url_display = db_url_display[:50] + "..."
    print(f"  DATABASE_URL: {db_url_display}")
    sep()

    engine = get_engine()

    print("\nCargando historical_interactions...")
    df = cargar_tabla(engine, "historical_interactions")
    df = diagnostico_general(df)

    diagnostico_canales(df)
    diagnostico_choice(df)
    diagnostico_aht(df)
    patron_semanal(df)
    patron_intradiario(df)

    df_ext = diagnostico_ext_vars(engine)
    diagnostico_feriados(df_ext)
    diagnostico_outliers(df)
    resumen_ejecutivo(df, df_ext)


if __name__ == "__main__":
    main()
