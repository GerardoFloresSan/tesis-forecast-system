import re
import unicodedata
from typing import Iterable

import pandas as pd


CANONICAL_COLUMN_ALIASES: dict[str, set[str]] = {
    "interaction_date": {
        "fecha",
        "fecha_interaccion",
        "fecha_de_interaccion",
        "interaction_date",
        "date",
        "fecha_operacion",
        "dia",
    },
    "interval_time": {
        "intervalo",
        "hora",
        "hora_intervalo",
        "interval_time",
        "time",
        "franja_horaria",
        "time_interval",
    },
    "channel": {
        "canal",
        "channel",
        "canal_atencion",
        "canal_de_atencion",
        "skill",
        "servicio",
    },
    "volume": {
        "volumen",
        "volume",
        "interacciones",
        "cantidad",
        "total_interacciones",
        "total_volume",
    },
    "aht": {
        "aht",
        "tmo",
        "tm_o",
        "average_handle_time",
        "tiempo_medio_operacion",
        "tiempo_promedio_atencion",
    },
}


def normalize_column_name(value: str) -> str:
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [normalize_column_name(col) for col in normalized.columns]
    return normalized


def resolve_canonical_columns(columns: Iterable[str]) -> tuple[dict[str, str], list[str]]:
    normalized_to_original = {
        normalize_column_name(column): column
        for column in columns
    }

    rename_map: dict[str, str] = {}
    missing: list[str] = []

    for canonical_name, aliases in CANONICAL_COLUMN_ALIASES.items():
        matched_original = None

        for alias in aliases:
            normalized_alias = normalize_column_name(alias)
            if normalized_alias in normalized_to_original:
                matched_original = normalized_to_original[normalized_alias]
                break

        if matched_original is None:
            missing.append(canonical_name)
            continue

        rename_map[matched_original] = canonical_name

    return rename_map, missing


def remove_duplicates(df: pd.DataFrame, subset: list[str] | None = None) -> tuple[pd.DataFrame, int]:
    before = len(df)
    deduped = df.drop_duplicates(subset=subset)
    removed = before - len(deduped)
    return deduped, removed


def count_nulls(df: pd.DataFrame) -> int:
    return int(df.isnull().sum().sum())