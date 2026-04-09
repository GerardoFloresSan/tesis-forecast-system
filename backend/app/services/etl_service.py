import pandas as pd
from datetime import datetime
from sqlalchemy.orm import Session

from app.models.etl_run import EtlRun
from app.models.historical_interaction import HistoricalInteraction


EXPECTED_COLUMNS = ["Fecha", "Intervalo", "Canal", "Volumen", "AHT"]


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
        df = pd.read_excel(file_path, sheet_name="data")

        # Validar columnas
        missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Faltan columnas obligatorias: {missing_columns}")

        records_original = len(df)

        # Seleccionar solo columnas esperadas
        df = df[EXPECTED_COLUMNS].copy()

        # Renombrar columnas
        df.columns = ["interaction_date", "interval_time", "channel", "volume", "aht"]

        # Convertir tipos
        df["interaction_date"] = pd.to_datetime(
            df["interaction_date"].astype(str), format="%Y%m%d", errors="coerce"
        ).dt.date

        df["interval_time"] = pd.to_datetime(
            df["interval_time"].astype(str), format="%H:%M:%S", errors="coerce"
        ).dt.time

        df["channel"] = df["channel"].astype(str).str.strip()
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        df["aht"] = pd.to_numeric(df["aht"], errors="coerce")

        # Contar nulos antes de limpiar
        nulls_treated = int(df.isnull().sum().sum())

        # Eliminar filas inválidas en columnas críticas
        df = df.dropna(subset=["interaction_date", "interval_time", "channel", "volume"])

        # Ajustar volumen a entero
        df["volume"] = df["volume"].astype(int)

        # Eliminar duplicados por clave lógica
        before_duplicates = len(df)
        df = df.drop_duplicates(subset=["interaction_date", "interval_time", "channel"])
        duplicates_removed = before_duplicates - len(df)

        # Limpiar carga previa si quieres recargar enero varias veces
        db.query(HistoricalInteraction).delete()
        db.commit()

        # Insertar en BD
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

        db.bulk_save_objects(records_to_insert)
        db.commit()

        etl_run.status = "SUCCESS"
        etl_run.message = "Archivo procesado correctamente"
        etl_run.records_original = records_original
        etl_run.duplicates_removed = duplicates_removed
        etl_run.nulls_treated = nulls_treated
        etl_run.records_final = len(df)
        etl_run.finished_at = datetime.utcnow()

        db.commit()

        return {
            "file_name": file_name,
            "records_original": records_original,
            "duplicates_removed": duplicates_removed,
            "nulls_treated": nulls_treated,
            "records_final": len(df),
            "message": "Archivo cargado correctamente"
        }

    except Exception as e:
        db.rollback()

        etl_run.status = "FAILED"
        etl_run.message = str(e)
        etl_run.finished_at = datetime.utcnow()
        db.commit()

        raise e