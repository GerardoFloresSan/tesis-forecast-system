from __future__ import annotations

from datetime import datetime, time
from typing import Iterable

import pandas as pd
from sqlalchemy import tuple_
from sqlalchemy.orm import Session

from app.models.etl_run import EtlRun
from app.models.historical_interaction import HistoricalInteraction
from app.utils.file_reader import read_file
from app.utils.normalizer import count_nulls, remove_duplicates, resolve_canonical_columns


REQUIRED_COLUMNS = ["interaction_date", "interval_time", "channel", "volume", "aht"]
LOGICAL_KEY_COLUMNS = ["interaction_date", "interval_time", "channel"]

DELETE_BATCH_SIZE = 1000
INSERT_BATCH_SIZE = 2000


def _chunked(items: list, batch_size: int) -> Iterable[list]:
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def _parse_interaction_date(series: pd.Series) -> pd.Series:
    raw = series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    parsed = pd.to_datetime(raw, format="%Y%m%d", errors="coerce")

    fallback_mask = parsed.isna()
    if fallback_mask.any():
        parsed.loc[fallback_mask] = pd.to_datetime(
            raw.loc[fallback_mask],
            errors="coerce",
            dayfirst=True,
        )

    return parsed.dt.date


def _excel_fraction_to_time(value: float) -> time:
    total_seconds = int(round(value * 24 * 60 * 60))
    total_seconds = max(0, min(total_seconds, 24 * 60 * 60 - 1))

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return time(hour=hours, minute=minutes, second=seconds)


def _parse_single_time(value) -> time | None:
    if pd.isna(value):
        return None

    if isinstance(value, pd.Timestamp):
        return value.time()

    if isinstance(value, datetime):
        return value.time()

    if isinstance(value, time):
        return value

    if isinstance(value, (int, float)) and 0 <= float(value) < 1:
        return _excel_fraction_to_time(float(value))

    raw = str(value).strip()
    raw = raw.replace(".0", "")

    for fmt in ("%H:%M:%S", "%H:%M", "%H%M%S", "%H%M"):
        try:
            return datetime.strptime(raw, fmt).time()
        except ValueError:
            continue

    parsed = pd.to_datetime(raw, errors="coerce")
    if pd.isna(parsed):
        return None

    return parsed.time()


def _parse_interval_time(series: pd.Series) -> pd.Series:
    return series.apply(_parse_single_time)


def _load_and_standardize_dataframe(file_path: str) -> tuple[pd.DataFrame, dict]:
    raw_df, file_metadata = read_file(file_path=file_path, preferred_sheet_name="data")

    rename_map, missing_columns = resolve_canonical_columns(raw_df.columns)
    if missing_columns:
        raise ValueError(
            "Faltan columnas obligatorias después de normalizar encabezados: "
            f"{missing_columns}. Encabezados detectados: {list(raw_df.columns)}"
        )

    df = raw_df.rename(columns=rename_map).copy()
    df = df[REQUIRED_COLUMNS]

    file_metadata["detected_columns"] = list(df.columns)

    df["interaction_date"] = _parse_interaction_date(df["interaction_date"])
    df["interval_time"] = _parse_interval_time(df["interval_time"])
    df["channel"] = df["channel"].astype(str).str.strip()
    df["channel"] = df["channel"].replace({"": None, "nan": None, "None": None})
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["aht"] = pd.to_numeric(df["aht"], errors="coerce")

    return df, file_metadata


def _delete_existing_rows_for_keys(db: Session, keys: list[tuple]) -> int:
    if not keys:
        return 0

    total_deleted = 0

    for batch in _chunked(keys, DELETE_BATCH_SIZE):
        deleted = (
            db.query(HistoricalInteraction)
            .filter(
                tuple_(
                    HistoricalInteraction.interaction_date,
                    HistoricalInteraction.interval_time,
                    HistoricalInteraction.channel,
                ).in_(batch)
            )
            .delete(synchronize_session=False)
        )

        total_deleted += deleted
        db.flush()

    return total_deleted


def _bulk_insert_in_batches(db: Session, records_to_insert: list[HistoricalInteraction]) -> None:
    if not records_to_insert:
        return

    for batch in _chunked(records_to_insert, INSERT_BATCH_SIZE):
        db.bulk_save_objects(batch)
        db.flush()


def process_excel_and_save(file_path: str, db: Session, file_name: str) -> dict:
    etl_run = EtlRun(
        file_name=file_name,
        status="PROCESSING",
        message="Procesando archivo",
        records_original=0,
        duplicates_removed=0,
        nulls_treated=0,
        records_final=0,
        started_at=datetime.utcnow(),
    )
    db.add(etl_run)
    db.commit()
    db.refresh(etl_run)

    try:
        df, file_metadata = _load_and_standardize_dataframe(file_path=file_path)

        records_original = len(df)
        nulls_treated = count_nulls(df)

        df = df.dropna(subset=["interaction_date", "interval_time", "channel", "volume"])
        df["volume"] = df["volume"].round().astype(int)

        df, duplicates_removed = remove_duplicates(df, subset=LOGICAL_KEY_COLUMNS)

        logical_keys = list(df[LOGICAL_KEY_COLUMNS].itertuples(index=False, name=None))
        records_replaced = _delete_existing_rows_for_keys(db=db, keys=logical_keys)

        records_to_insert = [
            HistoricalInteraction(
                interaction_date=row["interaction_date"],
                interval_time=row["interval_time"],
                channel=row["channel"],
                volume=row["volume"],
                aht=float(row["aht"]) if pd.notna(row["aht"]) else None,
            )
            for _, row in df.iterrows()
        ]

        _bulk_insert_in_batches(db=db, records_to_insert=records_to_insert)
        db.commit()

        sheet_used = file_metadata.get("sheet_used") or "N/A"

        etl_run.status = "SUCCESS"
        etl_run.message = (
            "Archivo procesado correctamente "
            f"(modo incremental_replace, hoja usada: {sheet_used}, "
            f"registros reemplazados: {records_replaced})"
        )
        etl_run.records_original = records_original
        etl_run.duplicates_removed = duplicates_removed
        etl_run.nulls_treated = nulls_treated
        etl_run.records_final = len(df)
        etl_run.finished_at = datetime.utcnow()
        db.commit()

        return {
            "file_name": file_name,
            "sheet_used": file_metadata.get("sheet_used"),
            "records_original": records_original,
            "duplicates_removed": duplicates_removed,
            "nulls_treated": nulls_treated,
            "records_replaced": records_replaced,
            "records_final": len(df),
            "load_mode": "incremental_replace",
            "message": "Archivo cargado correctamente",
        }

    except Exception as e:
        db.rollback()

        etl_run.status = "FAILED"
        etl_run.message = str(e)
        etl_run.finished_at = datetime.utcnow()
        db.commit()

        raise e