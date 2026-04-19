from app.services.lstm_training_service import check_and_retrain_lstm


def test_check_and_retrain_returns_previous_and_new_mape_when_retrain_runs(db_session, monkeypatch):
    monkeypatch.setattr(
        "app.services.lstm_training_service.get_lstm_status",
        lambda channel: {
            "channel": channel,
            "model_exists": True,
            "scaler_exists": True,
            "metadata_exists": True,
            "metrics_exists": True,
        },
    )

    metrics_sequence = iter([
        {"mape": 18.5},
        {"mape": 12.2},
    ])
    monkeypatch.setattr(
        "app.services.lstm_training_service.get_lstm_metrics",
        lambda channel: next(metrics_sequence),
    )
    monkeypatch.setattr(
        "app.services.lstm_training_service.retrain_lstm_model",
        lambda db, channel: {
            "run_id": 99,
            "run_type": "retrain",
            "status": "success",
        },
    )

    result = check_and_retrain_lstm(db=db_session, channel="Choice", threshold_mape=15.0)

    assert result["should_retrain"] is True
    assert result["action_taken"] == "retrain"
    assert result["previous_mape"] == 18.5
    assert result["current_mape"] == 12.2
    assert result["run_id"] == 99
    assert "MAPE anterior: 18.5%" in result["message"]
    assert "MAPE nuevo: 12.2%" in result["message"]


def test_check_and_retrain_returns_previous_mape_when_no_retrain_is_needed(db_session, monkeypatch):
    monkeypatch.setattr(
        "app.services.lstm_training_service.get_lstm_status",
        lambda channel: {
            "channel": channel,
            "model_exists": True,
            "scaler_exists": True,
            "metadata_exists": True,
            "metrics_exists": True,
        },
    )
    monkeypatch.setattr(
        "app.services.lstm_training_service.get_lstm_metrics",
        lambda channel: {"mape": 13.16},
    )

    result = check_and_retrain_lstm(db=db_session, channel="España", threshold_mape=15.0)

    assert result["should_retrain"] is False
    assert result["action_taken"] == "none"
    assert result["previous_mape"] == 13.16
    assert result["current_mape"] == 13.16
    assert result["run_id"] is None
    assert "no supera el umbral" in result["message"]
