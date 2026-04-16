from pathlib import Path

import pandas as pd

from app.models.historical_interaction import HistoricalInteraction
from app.services.etl_service import process_excel_and_save


def _create_excel_file(path: Path, rows: list[dict], sheet_name: str = "data") -> None:
    df = pd.DataFrame(rows)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)


def test_process_excel_and_save_replaces_existing_rows_by_logical_key(db_session, tmp_path):
    first_file = tmp_path / "carga_1.xlsx"
    second_file = tmp_path / "carga_2.xlsx"

    _create_excel_file(
        path=first_file,
        rows=[
            {"interaction_date": 20250301, "interval_time": "08:00", "channel": "Choice", "volume": 10, "aht": 120},
            {"interaction_date": 20250301, "interval_time": "08:30", "channel": "Choice", "volume": 12, "aht": 125},
        ],
    )

    _create_excel_file(
        path=second_file,
        rows=[
            {"interaction_date": 20250301, "interval_time": "08:00", "channel": "Choice", "volume": 20, "aht": 130},
            {"interaction_date": 20250301, "interval_time": "08:30", "channel": "Choice", "volume": 22, "aht": 135},
            {"interaction_date": 20250301, "interval_time": "09:00", "channel": "Choice", "volume": 30, "aht": 140},
        ],
    )

    first_result = process_excel_and_save(str(first_file), db_session, "carga_1.xlsx")
    second_result = process_excel_and_save(str(second_file), db_session, "carga_2.xlsx")

    assert first_result["records_replaced"] == 0
    assert first_result["records_final"] == 2

    assert second_result["records_replaced"] == 2
    assert second_result["records_final"] == 3
    assert second_result["load_mode"] == "incremental_replace"

    rows = (
        db_session.query(HistoricalInteraction)
        .order_by(
            HistoricalInteraction.interaction_date.asc(),
            HistoricalInteraction.interval_time.asc(),
            HistoricalInteraction.channel.asc(),
        )
        .all()
    )

    assert len(rows) == 3

    serialized = [
        (
            row.interaction_date.isoformat(),
            row.interval_time.strftime("%H:%M"),
            row.channel,
            row.volume,
        )
        for row in rows
    ]

    assert serialized == [
        ("2025-03-01", "08:00", "Choice", 20),
        ("2025-03-01", "08:30", "Choice", 22),
        ("2025-03-01", "09:00", "Choice", 30),
    ]

    logical_keys = {
        (row.interaction_date, row.interval_time, row.channel)
        for row in rows
    }
    assert len(logical_keys) == len(rows)