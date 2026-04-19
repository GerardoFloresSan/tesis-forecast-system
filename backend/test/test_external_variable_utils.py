from datetime import date

import pandas as pd
import pytest

from app.utils.external_variables import (
    normalize_external_variable_type,
    prepare_external_variables_dataframe,
)


def test_normalize_external_variable_type_maps_holiday_correctly():
    assert normalize_external_variable_type("holiday") == "is_holiday_peru"


def test_normalize_external_variable_type_invalid_raises_value_error():
    with pytest.raises(ValueError, match="Tipo de variable externa no soportado"):
        normalize_external_variable_type("weather_alert")


def test_prepare_external_variables_dataframe_pivots_correctly():
    df = pd.DataFrame(
        [
            {
                "id": 1,
                "variable_date": "2025-05-01",
                "variable_type": "holiday",
                "variable_value": 1,
            },
            {
                "id": 2,
                "variable_date": "2025-05-01",
                "variable_type": "campaign",
                "variable_value": 0,
            },
            {
                "id": 3,
                "variable_date": "2025-05-02",
                "variable_type": "holiday_spain",
                "variable_value": 1,
            },
            {
                "id": 4,
                "variable_date": "2025-05-02",
                "variable_type": "absent_rate",
                "variable_value": 12.5,
            },
            {
                "id": 5,
                "variable_date": "2025-05-02",
                "variable_type": "campaign_day",
                "variable_value": 1,
            },
            {
                "id": 6,
                "variable_date": "2025-05-02",
                "variable_type": "campaign_day",
                "variable_value": 0,
            },
        ]
    )

    result = prepare_external_variables_dataframe(df)

    assert list(result.columns) == [
        "variable_date",
        "is_holiday_peru",
        "is_holiday_spain",
        "is_holiday_mexico",
        "campaign_day",
        "absenteeism_rate",
        "is_holiday_any",
    ]

    assert len(result) == 2

    row_1 = result[result["variable_date"] == date(2025, 5, 1)].iloc[0]
    assert row_1["is_holiday_peru"] == 1.0
    assert row_1["is_holiday_spain"] == 0.0
    assert row_1["is_holiday_mexico"] == 0.0
    assert row_1["campaign_day"] == 0.0
    assert row_1["absenteeism_rate"] == 0.0
    assert row_1["is_holiday_any"] == 1

    row_2 = result[result["variable_date"] == date(2025, 5, 2)].iloc[0]
    assert row_2["is_holiday_peru"] == 0.0
    assert row_2["is_holiday_spain"] == 1.0
    assert row_2["is_holiday_mexico"] == 0.0
    assert row_2["campaign_day"] == 0.0
    assert row_2["absenteeism_rate"] == 12.5
    assert row_2["is_holiday_any"] == 1