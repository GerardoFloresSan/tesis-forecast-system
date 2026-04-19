from __future__ import annotations

from collections import Counter
from datetime import date, time, timedelta
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from app.models.data_quality_report import DataQualityReport
from app.models.historical_interaction import HistoricalInteraction
from app.utils.channel_rules import canonicalize_channel, get_operational_interval_times

LOGICAL_KEY_COLUMNS = ["interaction_date", "interval_time", "channel"]
DATA_COLUMNS = ["interaction_date", "interval_time", "channel", "volume", "aht"]
REQUIRED_COLUMNS = ["interaction_date", "interval_time", "channel", "volume"]
MAX_SAMPLE_ITEMS = 10


def _time_to_minutes(value: time | None) -> int | None:
    if value is None:
        return None
    return value.hour * 60 + value.minute


def _minutes_to_hhmm(value: int) -> str:
    hours = value // 60
    minutes = value % 60
    return f"{hours:02d}:{minutes:02d}"


def _serialize_date(value: date | None) -> str | None:
    return value.isoformat() if value else None


def _load_historical_dataframe(db: Session, channel: str | None = None) -> pd.DataFrame:
    query = db.query(
        HistoricalInteraction.interaction_date,
        HistoricalInteraction.interval_time,
        HistoricalInteraction.channel,
        HistoricalInteraction.volume,
        HistoricalInteraction.aht,
    )

    if channel:
        canonical_channel = canonicalize_channel(channel)
        query = query.filter(HistoricalInteraction.channel == canonical_channel)

    rows = (
        query.order_by(
            HistoricalInteraction.interaction_date,
            HistoricalInteraction.interval_time,
            HistoricalInteraction.channel,
        ).all()
    )

    if not rows:
        return pd.DataFrame(columns=DATA_COLUMNS)

    return pd.DataFrame(rows, columns=DATA_COLUMNS)


def _build_empty_response() -> dict[str, Any]:
    return {
        "total_records": 0,
        "missing_percentage": 0.0,
        "duplicate_percentage": 0.0,
        "valid_percentage": 0.0,
        "date_range": {
            "start_date": None,
            "end_date": None,
            "total_days": 0,
        },
        "detected_channels": [],
        "records_by_channel": {},
        "nulls_by_column": {column: 0 for column in DATA_COLUMNS},
        "duplicate_keys": {
            "duplicate_groups": 0,
            "duplicate_records": 0,
            "sample": [],
        },
        "intervals": {
            "channels": [],
            "total_invalid_intervals": 0,
            "total_missing_intervals": 0,
        },
        "days_without_data": {
            "count": 0,
            "dates": [],
            "by_channel": {},
        },
        "summary": {
            "status": "ERROR",
            "issues": ["No hay registros cargados en historical_interactions."],
        },
    }


def _compute_date_range(df: pd.DataFrame) -> dict[str, Any]:
    start_date = df["interaction_date"].min()
    end_date = df["interaction_date"].max()
    total_days = ((end_date - start_date).days + 1) if start_date and end_date else 0

    return {
        "start_date": _serialize_date(start_date),
        "end_date": _serialize_date(end_date),
        "total_days": total_days,
    }


def _compute_duplicates(df: pd.DataFrame) -> dict[str, Any]:
    grouped = (
        df.groupby(LOGICAL_KEY_COLUMNS, dropna=False)
        .size()
        .reset_index(name="occurrences")
    )

    duplicates = grouped[grouped["occurrences"] > 1].copy()
    if duplicates.empty:
        return {
            "duplicate_groups": 0,
            "duplicate_records": 0,
            "sample": [],
        }

    duplicates = duplicates.sort_values(
        by=["occurrences", "interaction_date", "interval_time", "channel"],
        ascending=[False, True, True, True],
    )

    sample = []
    for _, row in duplicates.head(MAX_SAMPLE_ITEMS).iterrows():
        sample.append(
            {
                "interaction_date": _serialize_date(row["interaction_date"]),
                "interval_time": row["interval_time"].strftime("%H:%M") if row["interval_time"] else None,
                "channel": row["channel"],
                "occurrences": int(row["occurrences"]),
            }
        )

    return {
        "duplicate_groups": int(len(duplicates)),
        "duplicate_records": int((duplicates["occurrences"] - 1).sum()),
        "sample": sample,
    }


def _infer_cadence_minutes(unique_minutes: list[int]) -> int | None:
    if len(unique_minutes) < 2:
        return None

    diffs = [
        current - previous
        for previous, current in zip(unique_minutes, unique_minutes[1:])
        if current - previous > 0
    ]
    if not diffs:
        return None

    diff_counter = Counter(diffs)
    return diff_counter.most_common(1)[0][0]


def _get_expected_minutes_for_channel(
    channel: str,
    actual_minutes: list[int],
    cadence_minutes: int | None,
    *,
    use_operational_window: bool = False,
) -> list[int]:
    if use_operational_window:
        try:
            operational_minutes = [
                _time_to_minutes(interval_time)
                for interval_time in get_operational_interval_times(channel)
            ]
            return [value for value in operational_minutes if value is not None]
        except ValueError:
            pass

    if cadence_minutes and actual_minutes:
        return list(range(actual_minutes[0], actual_minutes[-1] + cadence_minutes, cadence_minutes))
    return actual_minutes


def _compute_interval_quality(
    df: pd.DataFrame,
    *,
    interval_mode: str = "dynamic_day",
) -> dict[str, Any]:
    channels_summary: list[dict[str, Any]] = []
    total_invalid_intervals = 0
    total_missing_intervals = 0

    for channel in sorted(df["channel"].dropna().astype(str).unique().tolist()):
        channel_df = df[df["channel"] == channel].copy()
        unique_minutes = sorted(
            {
                minutes
                for minutes in channel_df["interval_time"].apply(_time_to_minutes).tolist()
                if minutes is not None
            }
        )
        cadence_minutes = _infer_cadence_minutes(unique_minutes)

        operational_minutes: list[int] = []
        operational_minute_set: set[int] = set()
        if interval_mode == "operational_window":
            operational_minutes = _get_expected_minutes_for_channel(
                channel,
                unique_minutes,
                cadence_minutes,
                use_operational_window=True,
            )
            operational_minute_set = set(operational_minutes)

        invalid_samples: list[dict[str, Any]] = []
        missing_samples: list[dict[str, Any]] = []
        invalid_count = 0
        missing_count = 0
        days_with_issues = 0

        for interaction_date, day_df in channel_df.groupby("interaction_date"):
            actual_minutes = sorted(
                {
                    minutes
                    for minutes in day_df["interval_time"].apply(_time_to_minutes).tolist()
                    if minutes is not None
                }
            )

            if not actual_minutes:
                continue

            invalid_minutes: list[int] = []
            missing_minutes: list[int] = []

            if cadence_minutes:
                if interval_mode == "operational_window" and operational_minutes:
                    min_expected = min(operational_minutes)
                    max_expected = max(operational_minutes)
                    invalid_minutes = [
                        minute
                        for minute in actual_minutes
                        if min_expected <= minute <= max_expected and minute not in operational_minute_set
                    ]
                    aligned_minutes = sorted(
                        minute for minute in actual_minutes if minute in operational_minute_set
                    )
                    missing_minutes = sorted(operational_minute_set - set(aligned_minutes))
                else:
                    invalid_minutes = [minute for minute in actual_minutes if minute % cadence_minutes != 0]
                    aligned_minutes = sorted(set(actual_minutes) - set(invalid_minutes))

                    if aligned_minutes:
                        expected_minutes = set(
                            range(aligned_minutes[0], aligned_minutes[-1] + cadence_minutes, cadence_minutes)
                        )
                        missing_minutes = sorted(expected_minutes - set(aligned_minutes))

            if invalid_minutes or missing_minutes:
                days_with_issues += 1

            invalid_count += len(invalid_minutes)
            missing_count += len(missing_minutes)

            if invalid_minutes and len(invalid_samples) < MAX_SAMPLE_ITEMS:
                invalid_samples.append(
                    {
                        "interaction_date": _serialize_date(interaction_date),
                        "intervals": [_minutes_to_hhmm(value) for value in invalid_minutes[:MAX_SAMPLE_ITEMS]],
                    }
                )

            if missing_minutes and len(missing_samples) < MAX_SAMPLE_ITEMS:
                missing_samples.append(
                    {
                        "interaction_date": _serialize_date(interaction_date),
                        "intervals": [_minutes_to_hhmm(value) for value in missing_minutes[:MAX_SAMPLE_ITEMS]],
                    }
                )

        total_invalid_intervals += invalid_count
        total_missing_intervals += missing_count

        channels_summary.append(
            {
                "channel": channel,
                "cadence_minutes": cadence_minutes,
                "dates_analyzed": int(channel_df["interaction_date"].nunique()),
                "invalid_intervals_count": int(invalid_count),
                "missing_intervals_count": int(missing_count),
                "days_with_issues": int(days_with_issues),
                "sample_invalid_intervals": invalid_samples,
                "sample_missing_intervals": missing_samples,
            }
        )

    return {
        "channels": channels_summary,
        "total_invalid_intervals": int(total_invalid_intervals),
        "total_missing_intervals": int(total_missing_intervals),
    }


def _compute_missing_days(df: pd.DataFrame) -> dict[str, Any]:
    start_date = df["interaction_date"].min()
    end_date = df["interaction_date"].max()
    full_range: list[date] = []

    if start_date and end_date:
        full_range = [
            start_date + timedelta(days=offset)
            for offset in range((end_date - start_date).days + 1)
        ]

    all_dates = set(df["interaction_date"].dropna().tolist())
    missing_dates = [current_date for current_date in full_range if current_date not in all_dates]

    missing_by_channel: dict[str, list[str]] = {}
    for channel in sorted(df["channel"].dropna().astype(str).unique().tolist()):
        channel_dates = set(df.loc[df["channel"] == channel, "interaction_date"].dropna().tolist())
        channel_missing = [current_date for current_date in full_range if current_date not in channel_dates]
        missing_by_channel[channel] = [_serialize_date(value) for value in channel_missing]

    return {
        "count": len(missing_dates),
        "dates": [_serialize_date(value) for value in missing_dates],
        "by_channel": missing_by_channel,
    }


def _build_summary(
    total_records: int,
    required_null_rows: int,
    optional_null_count: int,
    duplicate_records: int,
    total_invalid_intervals: int,
    total_missing_intervals: int,
    missing_days_count: int,
    *,
    for_training: bool = False,
) -> dict[str, Any]:
    issues: list[str] = []

    if total_records == 0:
        issues.append("No hay registros cargados en historical_interactions.")
    if required_null_rows > 0:
        issues.append(f"Se detectaron {required_null_rows} filas con nulos en columnas obligatorias.")
    if optional_null_count > 0:
        issues.append(f"Se detectaron {optional_null_count} nulos en columnas opcionales.")
    if duplicate_records > 0:
        issues.append(f"Se detectaron {duplicate_records} registros duplicados por clave lógica.")
    if total_invalid_intervals > 0:
        issues.append(f"Se detectaron {total_invalid_intervals} intervalos inválidos.")
    if total_missing_intervals > 0:
        issues.append(
            f"Se detectaron {total_missing_intervals} intervalos faltantes dentro de la secuencia esperada."
        )
    if missing_days_count > 0:
        issues.append(f"Se detectaron {missing_days_count} días sin data dentro del rango de fechas.")

    if for_training:
        # El entrenamiento ya filtra horario operativo y consolida duplicados. En esta etapa
        # bloqueamos solo problemas severos que sí hacen inviable el proceso.
        blocking_issues = (
            total_records == 0
            or required_null_rows > 0
            or total_invalid_intervals > 0
        )
        status = "ERROR" if blocking_issues else ("WARNING" if issues else "OK")
    else:
        if total_records == 0 or required_null_rows > 0 or missing_days_count > 0:
            status = "ERROR"
        elif issues:
            status = "WARNING"
        else:
            status = "OK"

    return {
        "status": status,
        "issues": issues if issues else ["No se detectaron problemas de calidad de datos."],
    }


def _build_quality_payload(df: pd.DataFrame, *, for_training: bool = False) -> dict[str, Any]:
    total_records = len(df)

    if total_records == 0:
        payload = _build_empty_response()
        if for_training:
            payload["summary"] = {
                "status": "ERROR",
                "issues": ["No hay registros cargados en historical_interactions."],
            }
        return payload

    nulls_by_column = {
        column: int(df[column].isna().sum())
        for column in DATA_COLUMNS
    }
    rows_with_any_null = int(df[DATA_COLUMNS].isna().any(axis=1).sum())
    required_null_rows = int(df[REQUIRED_COLUMNS].isna().any(axis=1).sum())

    date_range = _compute_date_range(df)
    detected_channels = sorted(df["channel"].dropna().astype(str).unique().tolist())
    records_by_channel = {
        channel: int(count)
        for channel, count in df["channel"].dropna().astype(str).value_counts().sort_index().items()
    }
    duplicate_keys = _compute_duplicates(df)
    interval_quality = _compute_interval_quality(
        df,
        interval_mode="operational_window" if for_training else "dynamic_day",
    )
    days_without_data = _compute_missing_days(df)
    summary = _build_summary(
        total_records=total_records,
        required_null_rows=required_null_rows,
        optional_null_count=nulls_by_column.get("aht", 0),
        duplicate_records=duplicate_keys["duplicate_records"],
        total_invalid_intervals=interval_quality["total_invalid_intervals"],
        total_missing_intervals=interval_quality["total_missing_intervals"],
        missing_days_count=days_without_data["count"],
        for_training=for_training,
    )

    missing_percentage = (rows_with_any_null / total_records) * 100 if total_records else 0.0
    duplicate_percentage = (duplicate_keys["duplicate_records"] / total_records) * 100 if total_records else 0.0
    valid_percentage = 100 - missing_percentage - duplicate_percentage
    if valid_percentage < 0:
        valid_percentage = 0.0

    return {
        "total_records": total_records,
        "missing_percentage": round(missing_percentage, 2),
        "duplicate_percentage": round(duplicate_percentage, 2),
        "valid_percentage": round(valid_percentage, 2),
        "date_range": date_range,
        "detected_channels": detected_channels,
        "records_by_channel": records_by_channel,
        "nulls_by_column": nulls_by_column,
        "duplicate_keys": duplicate_keys,
        "intervals": interval_quality,
        "days_without_data": days_without_data,
        "summary": summary,
    }


def evaluate_data_quality(
    db: Session,
    channel: str | None = None,
    *,
    for_training: bool = False,
) -> dict[str, Any]:
    df = _load_historical_dataframe(db, channel=channel)
    return _build_quality_payload(df, for_training=for_training)["summary"]



def generate_quality_report(db: Session) -> dict[str, Any]:
    df = _load_historical_dataframe(db)
    payload = _build_quality_payload(df)

    report = DataQualityReport(
        total_records=payload["total_records"],
        missing_percentage=payload["missing_percentage"],
        duplicate_percentage=payload["duplicate_percentage"],
        valid_percentage=payload["valid_percentage"],
    )
    db.add(report)
    db.commit()
    db.refresh(report)

    payload["missing_percentage"] = report.missing_percentage
    payload["duplicate_percentage"] = report.duplicate_percentage
    payload["valid_percentage"] = report.valid_percentage
    return payload
