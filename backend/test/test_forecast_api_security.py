from datetime import date, datetime, time

from app.models.api_access_log import APIAccessLog
from app.models.forecast_interval_run import ForecastIntervalRun
from app.models.forecast_run import ForecastRun
from app.models.historical_interaction import HistoricalInteraction


def _login_and_get_token(client) -> str:
    response = client.post(
        "/auth/login",
        json={
            "username": "admin",
            "password": "Admin123*",
        },
    )
    assert response.status_code == 200
    return response.json()["access_token"]


def test_forecast_api_daily_requires_authentication(client):
    response = client.get(
        "/forecast/api/daily",
        params={
            "channel": "Choice",
            "forecast_date": "2026-03-01",
        },
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "No autenticado."


def test_forecast_api_daily_returns_protected_payload_and_creates_audit_log(
    db_session,
    client,
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

    token = _login_and_get_token(client)

    response = client.get(
        "/forecast/api/daily",
        params={
            "channel": "Choice",
            "forecast_date": "2026-03-01",
        },
        headers={
            "Authorization": f"Bearer {token}",
        },
    )

    assert response.status_code == 200
    body = response.json()

    assert body["date"] == "2026-03-01"
    assert body["channel"] == "Choice"
    assert body["expected_volume"] == 120.0
    assert body["actual_volume"] == 118.0
    assert body["error_level"] == "low_error"
    assert body["risk_level"] == "normal"
    assert body["has_breach"] is False
    assert body["forecast_run_id"] == header.id
    assert body["model_version"] == "lstm_choice_v2_operational"

    rows = db_session.query(APIAccessLog).all()
    assert len(rows) == 1
    assert rows[0].username == "admin"
    assert rows[0].endpoint == "/forecast/api/daily"
    assert rows[0].method == "GET"
    assert rows[0].channel == "Choice"
    assert rows[0].status_code == 200