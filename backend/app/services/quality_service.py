from sqlalchemy.orm import Session
from sqlalchemy import func

from app.models.historical_interaction import HistoricalInteraction
from app.models.data_quality_report import DataQualityReport


def generate_quality_report(db: Session) -> dict:
    records = db.query(HistoricalInteraction).all()
    total_records = len(records)

    if total_records == 0:
        report = DataQualityReport(
            total_records=0,
            missing_percentage=0.0,
            duplicate_percentage=0.0,
            valid_percentage=0.0,
        )
        db.add(report)
        db.commit()
        db.refresh(report)

        return {
            "total_records": 0,
            "missing_percentage": 0.0,
            "duplicate_percentage": 0.0,
            "valid_percentage": 0.0,
        }

    missing_count = 0
    for r in records:
        if (
            r.interaction_date is None
            or r.interval_time is None
            or r.channel is None
            or r.volume is None
        ):
            missing_count += 1

    duplicate_groups = (
        db.query(
            HistoricalInteraction.interaction_date,
            HistoricalInteraction.interval_time,
            HistoricalInteraction.channel,
            func.count(HistoricalInteraction.id).label("count_rows"),
        )
        .group_by(
            HistoricalInteraction.interaction_date,
            HistoricalInteraction.interval_time,
            HistoricalInteraction.channel,
        )
        .having(func.count(HistoricalInteraction.id) > 1)
        .all()
    )

    duplicate_count = sum(group.count_rows - 1 for group in duplicate_groups)

    missing_percentage = (missing_count / total_records) * 100 if total_records else 0
    duplicate_percentage = (duplicate_count / total_records) * 100 if total_records else 0
    valid_percentage = 100 - missing_percentage - duplicate_percentage

    if valid_percentage < 0:
        valid_percentage = 0.0

    report = DataQualityReport(
        total_records=total_records,
        missing_percentage=round(missing_percentage, 2),
        duplicate_percentage=round(duplicate_percentage, 2),
        valid_percentage=round(valid_percentage, 2),
    )

    db.add(report)
    db.commit()
    db.refresh(report)

    return {
        "total_records": report.total_records,
        "missing_percentage": report.missing_percentage,
        "duplicate_percentage": report.duplicate_percentage,
        "valid_percentage": report.valid_percentage,
    }