from datetime import date, time

from app.models.historical_interaction import HistoricalInteraction


def test_quality_report_pdf_returns_downloadable_document(client, db_session, auth_headers):
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
                interaction_date=date(2025, 3, 3),
                interval_time=time(8, 0),
                channel="Choice",
                volume=12,
                aht=122.0,
            ),
        ]
    )
    db_session.commit()

    response = client.get("/quality/report/pdf", headers=auth_headers)

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("application/pdf")
    assert "attachment; filename=quality_report.pdf" == response.headers["content-disposition"]
    assert response.content
    assert response.content.startswith(b"%PDF")
