from datetime import date

from sqlalchemy.orm import Session

from app.models.api_access_log import APIAccessLog


def log_api_access(
    db: Session,
    username: str,
    endpoint: str,
    method: str,
    status_code: int,
    channel: str | None = None,
    forecast_date: date | None = None,
    client_ip: str | None = None,
    user_agent: str | None = None,
) -> None:
    try:
        row = APIAccessLog(
            username=username,
            endpoint=endpoint,
            method=method,
            channel=channel,
            forecast_date=forecast_date,
            status_code=status_code,
            client_ip=client_ip,
            user_agent=user_agent,
        )
        db.add(row)
        db.commit()
    except Exception:
        db.rollback()