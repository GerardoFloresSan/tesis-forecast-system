from datetime import date, datetime, time

from app.models.forecast_interval_run import ForecastIntervalRun
from app.models.forecast_run import ForecastRun
from app.services.forecast_service import create_daily_forecast, get_interval_forecast_history


def _build_prediction_batch(channel: str = "Choice") -> dict:
    forecast_date = date(2026, 4, 17)
    intervals = []

    for slot_index in range(34):
        hour = slot_index // 2
        minute_value = 30 if slot_index % 2 else 0
        forecast_datetime = datetime(2026, 4, 17, hour, minute_value)
        intervals.append(
            {
                "channel": channel,
                "forecast_date": forecast_date,
                "forecast_datetime": forecast_datetime,
                "interval_time": time(hour, minute_value),
                "slot_index": slot_index,
                "shift_label": "morning" if forecast_datetime.hour < 12 else "afternoon",
                "predicted_value": float(slot_index + 1),
                "model_version": "lstm_choice_v2_operational",
            }
        )

    return {
        "channel": channel,
        "forecast_date": forecast_date,
        "forecast_start_datetime": datetime(2026, 4, 17, 0, 0),
        "total_predicted_value": float(sum(item["predicted_value"] for item in intervals)),
        "intervals_generated": len(intervals),
        "intervals": intervals,
        "model_version": "lstm_choice_v2_operational",
    }


def test_create_daily_forecast_persists_header_and_intervals(db_session, monkeypatch):
    monkeypatch.setattr(
        "app.services.forecast_service.predict_next_operational_day_for_channel",
        lambda db, channel: _build_prediction_batch(channel),
    )

    result = create_daily_forecast(db_session, "Choice")

    assert result["operation"] == "created"
    assert result["intervals_generated"] == 34
    assert len(result["intervals"]) == 34
    assert result["total_predicted_value"] == sum(range(1, 35))

    header_rows = db_session.query(ForecastRun).all()
    interval_rows = db_session.query(ForecastIntervalRun).order_by(ForecastIntervalRun.slot_index.asc()).all()

    assert len(header_rows) == 1
    assert len(interval_rows) == 34
    assert interval_rows[0].interval_time == time(0, 0)
    assert interval_rows[-1].interval_time == time(16, 30)
    assert interval_rows[-1].shift_label == "afternoon"


def test_get_interval_forecast_history_returns_latest_batch(db_session, monkeypatch):
    monkeypatch.setattr(
        "app.services.forecast_service.predict_next_operational_day_for_channel",
        lambda db, channel: _build_prediction_batch(channel),
    )

    create_daily_forecast(db_session, "Choice")

    rows = get_interval_forecast_history(db_session, channel="Choice")

    assert len(rows) == 34
    assert rows[0]["slot_index"] == 0
    assert rows[-1]["slot_index"] == 33
    assert rows[0]["forecast_date"].isoformat() == "2026-04-17"
