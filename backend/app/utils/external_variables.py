from __future__ import annotations

import unicodedata
from datetime import date
from typing import Any

import pandas as pd

CANONICAL_EXTERNAL_VARIABLES = {
    "is_holiday_peru",
    "is_holiday_spain",
    "is_holiday_mexico",
    "campaign_day",
    "absenteeism_rate",
}

DEFAULT_EXTERNAL_VARIABLES = {
    "is_holiday_peru": 0.0,
    "is_holiday_spain": 0.0,
    "is_holiday_mexico": 0.0,
    "campaign_day": 0.0,
    "absenteeism_rate": 0.0,
}

EXTERNAL_VARIABLE_TYPE_ALIASES = {
    "is_holiday": "is_holiday_peru",
    "holiday": "is_holiday_peru",
    "holiday_peru": "is_holiday_peru",
    "holiday_pe": "is_holiday_peru",
    "feriado": "is_holiday_peru",
    "feriado_peru": "is_holiday_peru",
    "feriado_pe": "is_holiday_peru",
    "is_holiday_peru": "is_holiday_peru",
    "is_holiday_pe": "is_holiday_peru",
    "is_holiday_peru_": "is_holiday_peru",
    "holiday_spain": "is_holiday_spain",
    "holiday_espana": "is_holiday_spain",
    "holiday_es": "is_holiday_spain",
    "feriado_spain": "is_holiday_spain",
    "feriado_espana": "is_holiday_spain",
    "feriado_es": "is_holiday_spain",
    "is_holiday_spain": "is_holiday_spain",
    "is_holiday_espana": "is_holiday_spain",
    "is_holiday_es": "is_holiday_spain",
    "holiday_mexico": "is_holiday_mexico",
    "holiday_mx": "is_holiday_mexico",
    "feriado_mexico": "is_holiday_mexico",
    "feriado_mx": "is_holiday_mexico",
    "is_holiday_mexico": "is_holiday_mexico",
    "is_holiday_mx": "is_holiday_mexico",
    "campaign": "campaign_day",
    "campaign_flag": "campaign_day",
    "campaignday": "campaign_day",
    "campaign_day": "campaign_day",
    "is_campaign_day": "campaign_day",
    "absenteeism": "absenteeism_rate",
    "absenteeismrate": "absenteeism_rate",
    "absent_rate": "absenteeism_rate",
    "absence_rate": "absenteeism_rate",
    "absenteeism_rate": "absenteeism_rate",
}


def _read_value(source: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(source, dict):
        return source.get(field_name, default)
    return getattr(source, field_name, default)



def _sanitize_variable_type(variable_type: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(variable_type or ""))
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.strip().lower()
    normalized = normalized.replace("-", "_").replace(" ", "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")



def normalize_external_variable_type(variable_type: str) -> str:
    normalized = _sanitize_variable_type(variable_type)
    canonical = EXTERNAL_VARIABLE_TYPE_ALIASES.get(normalized)

    if canonical is None:
        allowed = ", ".join(sorted(CANONICAL_EXTERNAL_VARIABLES))
        raise ValueError(
            f"Tipo de variable externa no soportado: '{variable_type}'. "
            f"Tipos canónicos permitidos: {allowed}"
        )

    return canonical



def build_default_external_variables() -> dict[str, float]:
    return dict(DEFAULT_EXTERNAL_VARIABLES)



def enrich_external_variables(variables: dict[str, float] | None) -> dict[str, float]:
    enriched = build_default_external_variables()
    if variables:
        for key, value in variables.items():
            if key in enriched:
                enriched[key] = float(value or 0.0)

    enriched["is_holiday_any"] = float(
        int(
            (enriched["is_holiday_peru"] > 0)
            or (enriched["is_holiday_spain"] > 0)
            or (enriched["is_holiday_mexico"] > 0)
        )
    )
    enriched["is_holiday"] = float(enriched["is_holiday_peru"])
    return enriched



def build_external_variables_map_from_records(
    records: list[Any],
    start_date: date | None = None,
    end_date: date | None = None,
) -> dict[date, dict[str, float]]:
    sorted_records = sorted(
        records,
        key=lambda record: (
            _read_value(record, "variable_date"),
            _read_value(record, "id", 0),
        ),
    )

    external_map: dict[date, dict[str, float]] = {}

    for record in sorted_records:
        variable_date = _read_value(record, "variable_date")
        if variable_date is None:
            continue
        if start_date and variable_date < start_date:
            continue
        if end_date and variable_date > end_date:
            continue

        canonical_type = normalize_external_variable_type(_read_value(record, "variable_type", ""))
        variable_value = float(_read_value(record, "variable_value", 0.0) or 0.0)

        if variable_date not in external_map:
            external_map[variable_date] = build_default_external_variables()

        external_map[variable_date][canonical_type] = variable_value

    return {
        variable_date: enrich_external_variables(values)
        for variable_date, values in external_map.items()
    }



def prepare_external_variables_dataframe(ext_df: pd.DataFrame) -> pd.DataFrame:
    if ext_df.empty:
        empty_df = pd.DataFrame(columns=["variable_date", *DEFAULT_EXTERNAL_VARIABLES.keys(), "is_holiday_any"])
        return empty_df

    df = ext_df.copy()

    if "id" not in df.columns:
        df["id"] = range(1, len(df) + 1)

    df["variable_date"] = pd.to_datetime(df["variable_date"], errors="coerce").dt.date
    df = df[df["variable_date"].notna()].copy()

    df["variable_type"] = df["variable_type"].astype(str).apply(normalize_external_variable_type)
    df["variable_value"] = pd.to_numeric(df["variable_value"], errors="coerce").fillna(0.0)

    df = df.sort_values(["variable_date", "id"]).drop_duplicates(
        subset=["variable_date", "variable_type"],
        keep="last",
    )

    ext_pivot = (
        df.pivot_table(
            index="variable_date",
            columns="variable_type",
            values="variable_value",
            aggfunc="last",
            fill_value=0.0,
        )
        .reset_index()
    )
    ext_pivot.columns.name = None

    for column in DEFAULT_EXTERNAL_VARIABLES:
        if column not in ext_pivot.columns:
            ext_pivot[column] = 0.0

    ext_pivot["is_holiday_any"] = (
        (
            (ext_pivot["is_holiday_peru"] > 0)
            | (ext_pivot["is_holiday_spain"] > 0)
            | (ext_pivot["is_holiday_mexico"] > 0)
        )
    ).astype(int)

    return ext_pivot[["variable_date", *DEFAULT_EXTERNAL_VARIABLES.keys(), "is_holiday_any"]]
