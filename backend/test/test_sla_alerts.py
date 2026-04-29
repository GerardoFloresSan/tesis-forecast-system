from datetime import date, datetime, time

from app.models.sla_alert import SLAAlert


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


def _create_forecast_with_large_breach(db_session):
    from app.models.forecast_interval_run import ForecastIntervalRun
    from app.models.forecast_run import ForecastRun
    from app.models.historical_interaction import HistoricalInteraction

    header = ForecastRun(
        channel="Choice",
        forecast_date=datetime(2026, 2, 28, 17, 0, 0),
        predicted_value=35.0821,
        model_version="lstm_choice",
        created_at=datetime(2026, 4, 9, 14, 0, 14),
    )
    db_session.add(header)
    db_session.commit()
    db_session.refresh(header)

    db_session.add_all(
        [
            HistoricalInteraction(
                interaction_date=date(2026, 2, 28),
                interval_time=time(8, 0),
                channel="Choice",
                volume=700,
                aht=120.0,
            ),
            HistoricalInteraction(
                interaction_date=date(2026, 2, 28),
                interval_time=time(8, 30),
                channel="Choice",
                volume=729,
                aht=121.0,
            ),
        ]
    )
    db_session.commit()
    return header


def _create_forecast_without_breach(db_session):
    from app.models.forecast_interval_run import ForecastIntervalRun
    from app.models.forecast_run import ForecastRun
    from app.models.historical_interaction import HistoricalInteraction

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
    return header


def test_alerts_evaluate_creates_active_alert_when_breach_exists(db_session, client):
    _create_forecast_with_large_breach(db_session)
    token = _login_and_get_token(client)

    response = client.post(
        "/alerts/evaluate",
        params={
            "channel": "Choice",
            "forecast_date": "2026-02-28",
        },
        headers={
            "Authorization": f"Bearer {token}",
        },
    )

    assert response.status_code == 200
    body = response.json()

    assert body["evaluation_status"] == "created"
    assert body["has_breach"] is True
    assert body["risk_level"] == "critical"
    assert body["alert_id"] is not None

    rows = db_session.query(SLAAlert).all()
    assert len(rows) == 1
    assert rows[0].status == "active"
    assert rows[0].channel == "Choice"
    assert rows[0].forecast_date.isoformat() == "2026-02-28"


def test_alerts_active_returns_created_alerts(db_session, client):
    _create_forecast_with_large_breach(db_session)
    token = _login_and_get_token(client)

    evaluate_response = client.post(
        "/alerts/evaluate",
        params={
            "channel": "Choice",
            "forecast_date": "2026-02-28",
        },
        headers={
            "Authorization": f"Bearer {token}",
        },
    )
    assert evaluate_response.status_code == 200

    response = client.get(
        "/alerts/active",
        headers={
            "Authorization": f"Bearer {token}",
        },
    )

    assert response.status_code == 200
    body = response.json()

    assert len(body) == 1
    assert body[0]["status"] == "active"
    assert body[0]["channel"] == "Choice"
    assert body[0]["risk_level"] == "critical"


def test_alerts_acknowledge_changes_status_and_user(db_session, client):
    _create_forecast_with_large_breach(db_session)
    token = _login_and_get_token(client)

    evaluate_response = client.post(
        "/alerts/evaluate",
        params={
            "channel": "Choice",
            "forecast_date": "2026-02-28",
        },
        headers={
            "Authorization": f"Bearer {token}",
        },
    )
    alert_id = evaluate_response.json()["alert_id"]

    response = client.patch(
        f"/alerts/{alert_id}/ack",
        headers={
            "Authorization": f"Bearer {token}",
        },
    )

    assert response.status_code == 200
    body = response.json()

    assert body["id"] == alert_id
    assert body["status"] == "acknowledged"
    assert body["acknowledged_by"] == "admin"

    row = db_session.query(SLAAlert).filter(SLAAlert.id == alert_id).first()
    assert row is not None
    assert row.status == "acknowledged"
    assert row.acknowledged_by == "admin"


def test_alerts_evaluate_returns_no_breach_when_forecast_is_within_expected_range(db_session, client):
    _create_forecast_without_breach(db_session)
    token = _login_and_get_token(client)

    response = client.post(
        "/alerts/evaluate",
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

    assert body["evaluation_status"] == "no_breach"
    assert body["has_breach"] is False
    assert body["alert_id"] is None