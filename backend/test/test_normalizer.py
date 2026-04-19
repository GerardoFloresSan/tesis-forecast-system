import pandas as pd

from app.utils.normalizer import (
    normalize_column_name,
    remove_duplicates,
    resolve_canonical_columns,
)


def test_normalize_column_name_removes_accents_and_normalizes():
    assert normalize_column_name("Fecha de Interacción") == "fecha_de_interaccion"
    assert normalize_column_name("  T.M.O  (%)  ") == "t_m_o"


def test_resolve_canonical_columns_maps_aliases():
    columns = [
        "Fecha de Interacción",
        "Hora",
        "Canal de Atención",
        "Volumen",
        "TMO",
    ]

    rename_map, missing = resolve_canonical_columns(columns)

    assert rename_map["Fecha de Interacción"] == "interaction_date"
    assert rename_map["Hora"] == "interval_time"
    assert rename_map["Canal de Atención"] == "channel"
    assert rename_map["Volumen"] == "volume"
    assert rename_map["TMO"] == "aht"
    assert missing == []


def test_resolve_canonical_columns_detects_missing():
    columns = [
        "Fecha",
        "Hora",
        "Canal",
    ]

    rename_map, missing = resolve_canonical_columns(columns)

    assert rename_map == {
        "Fecha": "interaction_date",
        "Hora": "interval_time",
        "Canal": "channel",
    }
    assert missing == ["volume", "aht"]


def test_remove_duplicates_counts_correctly():
    df = pd.DataFrame(
        [
            {
                "interaction_date": "2025-03-01",
                "interval_time": "08:00",
                "channel": "Choice",
                "volume": 10,
            },
            {
                "interaction_date": "2025-03-01",
                "interval_time": "08:00",
                "channel": "Choice",
                "volume": 20,
            },
            {
                "interaction_date": "2025-03-01",
                "interval_time": "08:30",
                "channel": "Choice",
                "volume": 30,
            },
        ]
    )

    deduped, removed = remove_duplicates(
        df,
        subset=["interaction_date", "interval_time", "channel"],
    )

    assert removed == 1
    assert len(deduped) == 2
    assert deduped.iloc[0]["volume"] == 10
    assert deduped.iloc[1]["volume"] == 30