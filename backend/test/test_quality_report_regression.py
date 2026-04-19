from datetime import date, time

from app.models.historical_interaction import HistoricalInteraction


def test_quality_report_uses_dynamic_day_window_not_operational_window(client, db_session):
    db_session.add_all(
        [
            HistoricalInteraction(
                interaction_date=date(2025, 2, 1),
                interval_time=time(8, 0),
                channel="Choice",
                volume=10,
                aht=120.0,
            ),
            HistoricalInteraction(
                interaction_date=date(2025, 2, 1),
                interval_time=time(8, 30),
                channel="Choice",
                volume=11,
                aht=121.0,
            ),
            HistoricalInteraction(
                interaction_date=date(2025, 2, 1),
                interval_time=time(9, 0),
                channel="Choice",
                volume=12,
                aht=122.0,
            ),
        ]
    )
    db_session.commit()

    response = client.get("/quality/report")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["intervals"]["total_missing_intervals"] == 0
    assert payload["intervals"]["channels"][0]["missing_intervals_count"] == 0
    assert payload["summary"]["status"] in {"OK", "WARNING"}
