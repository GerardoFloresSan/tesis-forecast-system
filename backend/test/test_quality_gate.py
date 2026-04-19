from app.models.model_train_run import ModelTrainRun
from app.services.lstm_training_service import train_lstm_model


def test_train_lstm_model_blocks_training_when_quality_gate_returns_error(db_session, monkeypatch):
    monkeypatch.setattr(
    "app.services.lstm_training_service.evaluate_data_quality",
    lambda db, channel=None, for_training=False: {
        "status": "ERROR",
        "issues": ["No hay registros cargados en historical_interactions."],
    },
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("subprocess.run no debe ejecutarse cuando el gate de calidad bloquea.")

    monkeypatch.setattr("app.services.lstm_training_service.subprocess.run", fail_if_called)

    try:
        train_lstm_model(db=db_session, channel="Choice", run_type="train")
        assert False, "Se esperaba ValueError cuando el gate de calidad devuelve ERROR."
    except ValueError as exc:
        assert "Datos no aptos para entrenamiento" in str(exc)
        assert "No hay registros cargados" in str(exc)

    rows = db_session.query(ModelTrainRun).all()
    assert len(rows) == 1
    assert rows[0].status == "failed"
    assert rows[0].error_message is not None
    assert "Gate de calidad bloqueó el entrenamiento" in rows[0].error_message
