from datetime import date, time

from app.models.historical_interaction import HistoricalInteraction


def test_quality_report_returns_extended_metrics_and_detects_issues(client, db_session):
    db_session.add_all(
        [
            HistoricalInteraction(
                interaction_date=date(2025, 3, 1),
                interval_time=time(8, 0),
                channel="Choice",
                volume=10,
                aht=120.0,
            ),
            HistoricalInteraction(
                interaction_date=date(2025, 3, 1),
                interval_time=time(8, 30),
                channel="Choice",
                volume=11,
                aht=121.0,
            ),
            HistoricalInteraction(
                interaction_date=date(2025, 3, 3),
                interval_time=time(8, 0),
                channel="Choice",
                volume=12,
                aht=122.0,
            ),
            HistoricalInteraction(
                interaction_date=date(2025, 3, 3),
                interval_time=time(8, 0),
                channel="Choice",
                volume=13,
                aht=123.0,
            ),
            HistoricalInteraction(
                interaction_date=date(2025, 3, 3),
                interval_time=time(9, 0),
                channel="Choice",
                volume=14,
                aht=124.0,
            ),
        ]
    )
    db_session.commit()

    response = client.get("/quality/report")

    assert response.status_code == 200, response.text
    payload = response.json()

    assert payload["total_records"] == 5
    assert payload["date_range"] == {
        "start_date": "2025-03-01",
        "end_date": "2025-03-03",
        "total_days": 3,
    }
    assert payload["detected_channels"] == ["Choice"]
    assert payload["records_by_channel"] == {"Choice": 5}
    assert payload["nulls_by_column"] == {
        "interaction_date": 0,
        "interval_time": 0,
        "channel": 0,
        "volume": 0,
        "aht": 0,
    }

    assert payload["duplicate_keys"]["duplicate_groups"] == 1
    assert payload["duplicate_keys"]["duplicate_records"] == 1
    assert payload["duplicate_keys"]["sample"][0] == {
        "interaction_date": "2025-03-03",
        "interval_time": "08:00",
        "channel": "Choice",
        "occurrences": 2,
    }

    assert payload["intervals"]["total_invalid_intervals"] == 0
    assert payload["intervals"]["total_missing_intervals"] == 1
    assert payload["intervals"]["channels"][0]["channel"] == "Choice"
    assert payload["intervals"]["channels"][0]["missing_intervals_count"] == 1
    assert payload["intervals"]["channels"][0]["sample_missing_intervals"][0] == {
        "interaction_date": "2025-03-03",
        "intervals": ["08:30"],
    }

    assert payload["days_without_data"]["count"] == 1
    assert payload["days_without_data"]["dates"] == ["2025-03-02"]
    assert payload["days_without_data"]["by_channel"] == {"Choice": ["2025-03-02"]}

    assert payload["summary"]["status"] == "ERROR"
    assert any("duplicados" in issue.lower() for issue in payload["summary"]["issues"])
    assert any("días sin data" in issue.lower() for issue in payload["summary"]["issues"])