from datetime import date, datetime, time

from app.models.forecast_interval_run import ForecastIntervalRun
from app.models.forecast_run import ForecastRun
from app.models.historical_interaction import HistoricalInteraction


def test_get_forecast_monitoring_summary_returns_normal_status_when_deviation_is_low(
    db_session,
    client,
    auth_headers,
):
    header = ForecastRun(
        channel="Choice",
        forecast_date=datetime(2026, 3, 1, 0, 0, 0),
        predicted_value=120.0,
        model_version="lstm_choice_v2_operational",
        created_at=datetime(2026, 4, 17, 13, 3, 14),
    )
    db_session.add(header)
    db_session.commit()
    db_session.refresh(header)

    db_session.add_all(
        [
            ForecastIntervalRun(
                forecast_run_id=header.id,
                channel="Choice",
                forecast_date=date(2026, 3, 1),
                forecast_datetime=datetime(2026, 3, 1, 0, 0, 0),
                interval_time=time(0, 0),
                slot_index=0,
                shift_label="morning",
                predicted_value=60.0,
                model_version="lstm_choice_v2_operational",
                created_at=datetime(2026, 4, 17, 13, 3, 14),
            ),
            ForecastIntervalRun(
                forecast_run_id=header.id,
                channel="Choice",
                forecast_date=date(2026, 3, 1),
                forecast_datetime=datetime(2026, 3, 1, 0, 30, 0),
                interval_time=time(0, 30),
                slot_index=1,
                shift_label="morning",
                predicted_value=60.0,
                model_version="lstm_choice_v2_operational",
                created_at=datetime(2026, 4, 17, 13, 3, 15),
            ),
        ]
    )

    db_session.add_all(
        [
            HistoricalInteraction(
                interaction_date=date(2026, 3, 1),
                interval_time=time(0, 0),
                channel="Choice",
                volume=58,
                aht=120.0,
            ),
            HistoricalInteraction(
                interaction_date=date(2026, 3, 1),
                interval_time=time(0, 30),
                channel="Choice",
                volume=60,
                aht=122.0,
            ),
        ]
    )
    db_session.commit()

    response = client.get(
        "/forecast/monitoring/summary",
        params={"channel": "Choice"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    body = response.json()

    assert body["channel"] == "Choice"
    assert body["forecast_total"] == 120.0
    assert body["actual_total"] == 118.0
    assert body["forecast_intervals_count"] == 2
    assert body["actual_intervals_count"] == 2
    assert body["error_level"] == "low_error"
    assert body["risk_level"] == "normal"
    assert body["has_breach"] is False
    assert body["forecast_date"] == "2026-03-01"


def test_get_forecast_monitoring_by_date_returns_critical_breach_when_deviation_exceeds_10(
    db_session,
    client,
    auth_headers,
):
    header = ForecastRun(
        channel="Choice",
        forecast_date=datetime(2026, 3, 2, 0, 0, 0),
        predicted_value=100.0,
        model_version="lstm_choice_v2_operational",
        created_at=datetime(2026, 4, 17, 14, 0, 0),
    )
    db_session.add(header)
    db_session.commit()
    db_session.refresh(header)

    db_session.add_all(
        [
            ForecastIntervalRun(
                forecast_run_id=header.id,
                channel="Choice",
                forecast_date=date(2026, 3, 2),
                forecast_datetime=datetime(2026, 3, 2, 0, 0, 0),
                interval_time=time(0, 0),
                slot_index=0,
                shift_label="morning",
                predicted_value=50.0,
                model_version="lstm_choice_v2_operational",
                created_at=datetime(2026, 4, 17, 14, 0, 0),
            ),
            ForecastIntervalRun(
                forecast_run_id=header.id,
                channel="Choice",
                forecast_date=date(2026, 3, 2),
                forecast_datetime=datetime(2026, 3, 2, 0, 30, 0),
                interval_time=time(0, 30),
                slot_index=1,
                shift_label="morning",
                predicted_value=50.0,
                model_version="lstm_choice_v2_operational",
                created_at=datetime(2026, 4, 17, 14, 0, 1),
            ),
        ]
    )

    db_session.add_all(
        [
            HistoricalInteraction(
                interaction_date=date(2026, 3, 2),
                interval_time=time(0, 0),
                channel="Choice",
                volume=70,
                aht=120.0,
            ),
            HistoricalInteraction(
                interaction_date=date(2026, 3, 2),
                interval_time=time(0, 30),
                channel="Choice",
                volume=55,
                aht=122.0,
            ),
        ]
    )
    db_session.commit()

    response = client.get(
        "/forecast/monitoring/by-date",
        params={
            "channel": "Choice",
            "forecast_date": "2026-03-02",
        },
        headers=auth_headers,
    )

    assert response.status_code == 200
    body = response.json()

    assert body["forecast_total"] == 100.0
    assert body["actual_total"] == 125.0
    assert body["deviation_percentage"] == 25.0
    assert body["absolute_deviation_percentage"] == 25.0
    assert body["error_level"] == "high_error"
    assert body["risk_level"] == "critical"
    assert body["has_breach"] is True


def test_get_forecast_monitoring_summary_returns_no_actual_data_when_real_values_do_not_exist(
    db_session,
    client,
    auth_headers,
):
    header = ForecastRun(
        channel="España",
        forecast_date=datetime(2026, 3, 3, 0, 0, 0),
        predicted_value=90.0,
        model_version="lstm_espana_v2_operational",
        created_at=datetime(2026, 4, 17, 15, 0, 0),
    )
    db_session.add(header)
    db_session.commit()
    db_session.refresh(header)

    db_session.add(
        ForecastIntervalRun(
            forecast_run_id=header.id,
            channel="España",
            forecast_date=date(2026, 3, 3),
            forecast_datetime=datetime(2026, 3, 3, 0, 0, 0),
            interval_time=time(0, 0),
            slot_index=0,
            shift_label="morning",
            predicted_value=90.0,
            model_version="lstm_espana_v2_operational",
            created_at=datetime(2026, 4, 17, 15, 0, 0),
        )
    )
    db_session.commit()

    response = client.get(
        "/forecast/monitoring/summary",
        params={"channel": "espana"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    body = response.json()

    assert body["channel"] == "España"
    assert body["actual_total"] is None
    assert body["actual_intervals_count"] == 0
    assert body["deviation_percentage"] is None
    assert body["error_level"] == "no_actual_data"
    assert body["risk_level"] == "unknown"
    assert body["has_breach"] is False