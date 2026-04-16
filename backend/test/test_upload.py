from pathlib import Path

import pandas as pd

from app.models.etl_run import EtlRun
from app.models.historical_interaction import HistoricalInteraction
from app.routers import upload as upload_router_module


def _create_excel_file(path: Path, rows: list[dict], sheet_name: str) -> None:
    df = pd.DataFrame(rows)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)


def test_upload_excel_autodetects_sheet_and_loads_incremental_replace(client, db_session, tmp_path, monkeypatch):
    test_upload_dir = tmp_path / "uploads"
    test_upload_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(upload_router_module, "UPLOAD_DIR", test_upload_dir)

    excel_path = tmp_path / "interacciones.xlsx"
    _create_excel_file(
        path=excel_path,
        sheet_name="operacion_real",
        rows=[
            {"Fecha": 20250301, "Hora": "08:00", "Canal": "Choice", "Volumen": 10, "TMO": 120},
            {"Fecha": 20250301, "Hora": "08:30", "Canal": "Choice", "Volumen": 12, "TMO": 125},
            {"Fecha": 20250301, "Hora": "08:30", "Canal": "Choice", "Volumen": 12, "TMO": 125},
        ],
    )

    with excel_path.open("rb") as file_handler:
        response = client.post(
            "/upload/excel",
            files={
                "file": (
                    "interacciones.xlsx",
                    file_handler,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
        )

    assert response.status_code == 200, response.text
    payload = response.json()

    assert payload["file_name"] == "interacciones.xlsx"
    assert payload["sheet_used"] == "operacion_real"
    assert payload["records_original"] == 3
    assert payload["duplicates_removed"] == 1
    assert payload["records_final"] == 2
    assert payload["records_replaced"] == 0
    assert payload["load_mode"] == "incremental_replace"

    historical_count = db_session.query(HistoricalInteraction).count()
    assert historical_count == 2

    etl_run = db_session.query(EtlRun).order_by(EtlRun.id.desc()).first()
    assert etl_run is not None
    assert etl_run.status == "SUCCESS"
    assert "incremental_replace" in (etl_run.message or "")