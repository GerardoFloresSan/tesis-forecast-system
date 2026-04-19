from datetime import date, time

from app.models.historical_interaction import HistoricalInteraction
from app.services.quality_service import evaluate_data_quality


def test_evaluate_data_quality_for_training_is_channel_scoped_and_relaxes_missing_days(db_session):
    db_session.add_all(
        [
            HistoricalInteraction(
                interaction_date=date(2025, 2, 1),
                interval_time=time(0, 0),
                channel="Choice",
                volume=10,
                aht=120.0,
            ),
            HistoricalInteraction(
                interaction_date=date(2025, 2, 1),
                interval_time=time(0, 30),
                channel="Choice",
                volume=11,
                aht=121.0,
            ),
            HistoricalInteraction(
                interaction_date=date(2025, 2, 3),
                interval_time=time(0, 0),
                channel="Choice",
                volume=12,
                aht=122.0,
            ),
            HistoricalInteraction(
                interaction_date=date(2025, 2, 1),
                interval_time=time(18, 0),
                channel="España",
                volume=20,
                aht=130.0,
            ),
        ]
    )
    db_session.commit()

    summary = evaluate_data_quality(db_session, channel="Choice", for_training=True)

    assert summary["status"] == "WARNING"
    assert any("intervalos faltantes" in issue.lower() for issue in summary["issues"])
    assert any("días sin data" in issue.lower() for issue in summary["issues"])
    assert not any("intervalos inválidos" in issue.lower() for issue in summary["issues"])
