from datetime import date, datetime, time, timedelta

from app.models.forecast_interval_run import ForecastIntervalRun
from app.models.forecast_run import ForecastRun


def _build_forecast_batch(channel: str = "Choice") -> dict:
    forecast_day = date(2026, 3, 1)
    created_at = datetime(2026, 4, 17, 13, 0, 0)

    intervals = []
    total = 0.0

    for slot_index in range(34):
        hour = slot_index // 2
        minute_value = 30 if slot_index % 2 else 0
        predicted_value = float(slot_index + 10)
        total += predicted_value

        intervals.append(
            {
                "id": slot_index + 1,
                "forecast_run_id": 99,
                "channel": channel,
                "forecast_date": forecast_day,
                "forecast_datetime": datetime(2026, 3, 1, hour, minute_value),
                "interval_time": time(hour, minute_value),
                "slot_index": slot_index,
                "shift_label": "morning" if hour < 12 else "afternoon",
                "predicted_value": predicted_value,
                "model_version": f"lstm_{channel.lower().replace('ñ', 'n')}_v2_operational",
                "created_at": created_at,
            }
        )

    return {
        "id": 99,
        "channel": channel,
        "forecast_date": forecast_day,
        "forecast_start_datetime": datetime(2026, 3, 1, 0, 0),
        "total_predicted_value": total,
        "intervals_generated": 34,
        "model_version": f"lstm_{channel.lower().replace('ñ', 'n')}_v2_operational",
        "created_at": created_at,
        "operation": "created",
        "message": f"Forecast operativo por intervalos created correctamente para el canal {channel}.",
        "intervals": intervals,
    }


def test_post_forecast_daily_returns_operational_batch(client, monkeypatch):
    monkeypatch.setattr(
        "app.routers.forecast.create_daily_forecast",
        lambda db, channel: _build_forecast_batch(channel),
    )

    response = client.post("/forecast/daily", json={"channel": "Choice"})
    assert response.status_code == 200

    body = response.json()
    assert body["channel"] == "Choice"
    assert body["intervals_generated"] == 34
    assert body["forecast_start_datetime"] == "2026-03-01T00:00:00"
    assert body["operation"] == "created"
    assert len(body["intervals"]) == 34
    assert body["intervals"][0]["interval_time"] == "00:00:00"
    assert body["intervals"][-1]["interval_time"] == "16:30:00"


def test_post_forecast_daily_returns_400_for_invalid_channel(client, monkeypatch):
    def _raise_error(db, channel):
        raise ValueError("Canal no soportado para forecast operativo.")

    monkeypatch.setattr("app.routers.forecast.create_daily_forecast", _raise_error)

    response = client.post("/forecast/daily", json={"channel": "Mexico"})
    assert response.status_code == 400
    assert response.json()["detail"] == "Canal no soportado para forecast operativo."


def test_get_forecast_history_returns_latest_runs(db_session, client):
    older = ForecastRun(
        channel="Choice",
        forecast_date=datetime(2026, 2, 28, 17, 0, 0),
        predicted_value=35.0821,
        model_version="lstm_choice",
        created_at=datetime(2026, 4, 9, 14, 0, 0),
    )
    latest = ForecastRun(
        channel="Choice",
        forecast_date=datetime(2026, 3, 1, 0, 0, 0),
        predicted_value=1371.4959,
        model_version="lstm_choice_v2_operational",
        created_at=datetime(2026, 4, 17, 13, 3, 14),
    )

    db_session.add_all([older, latest])
    db_session.commit()

    response = client.get("/forecast/history", params={"channel": "Choice", "limit": 5})
    assert response.status_code == 200

    body = response.json()
    assert len(body) == 2
    assert body[0]["model_version"] == "lstm_choice_v2_operational"
    assert body[0]["predicted_value"] == 1371.4959
    assert body[1]["model_version"] == "lstm_choice"


def test_get_forecast_history_returns_empty_list_when_no_rows(client):
    response = client.get("/forecast/history", params={"channel": "Choice", "limit": 5})
    assert response.status_code == 200
    assert response.json() == []


def test_get_forecast_interval_history_returns_34_slots_for_selected_date(db_session, client):
    header = ForecastRun(
        channel="Choice",
        forecast_date=datetime(2026, 3, 1, 0, 0, 0),
        predicted_value=1371.4959,
        model_version="lstm_choice_v2_operational",
        created_at=datetime(2026, 4, 17, 13, 3, 14),
    )
    db_session.add(header)
    db_session.commit()
    db_session.refresh(header)

    created_at = datetime(2026, 4, 17, 13, 3, 14)
    rows = []

    for slot_index in range(34):
        hour = slot_index // 2
        minute_value = 30 if slot_index % 2 else 0
        rows.append(
            ForecastIntervalRun(
                forecast_run_id=header.id,
                channel="Choice",
                forecast_date=date(2026, 3, 1),
                forecast_datetime=datetime(2026, 3, 1, hour, minute_value),
                interval_time=time(hour, minute_value),
                slot_index=slot_index,
                shift_label="morning" if hour < 12 else "afternoon",
                predicted_value=float(slot_index + 1),
                model_version="lstm_choice_v2_operational",
                created_at=created_at + timedelta(seconds=slot_index),
            )
        )

    db_session.add_all(rows)
    db_session.commit()

    response = client.get(
        "/forecast/history/intervals",
        params={
            "channel": "Choice",
            "forecast_date": "2026-03-01",
            "limit": 2000,
        },
    )
    assert response.status_code == 200

    body = response.json()
    assert len(body) == 34
    assert body[0]["slot_index"] == 0
    assert body[-1]["slot_index"] == 33
    assert body[0]["interval_time"] == "00:00:00"
    assert body[-1]["interval_time"] == "16:30:00"
    assert all(row["channel"] == "Choice" for row in body)