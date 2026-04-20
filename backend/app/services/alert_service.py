from __future__ import annotations

import os
import smtplib
from datetime import date, datetime
from email.mime.text import MIMEText

from sqlalchemy.orm import Session

from app.models.sla_alert import SLAAlert
from app.services.forecast_monitoring_service import get_forecast_monitoring_by_date


def _first_alert_recipient() -> str | None:
    raw = os.getenv("ALERT_EMAIL_TO", "").strip()
    if not raw:
        return None
    recipients = [item.strip() for item in raw.split(",") if item.strip()]
    return recipients[0] if recipients else None


def _send_email_if_configured(subject: str, body: str) -> tuple[bool, str | None]:
    recipient = _first_alert_recipient()
    if not recipient:
        return False, None

    smtp_host = os.getenv("SMTP_HOST", "").strip()
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", "").strip()
    smtp_password = os.getenv("SMTP_PASSWORD", "").strip()
    smtp_from = os.getenv("SMTP_FROM", smtp_user or "no-reply@forecast.local")
    smtp_tls = os.getenv("SMTP_USE_TLS", "true").strip().lower() == "true"

    if not smtp_host:
        return False, recipient

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = smtp_from
    msg["To"] = recipient

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
            if smtp_tls:
                server.starttls()
            if smtp_user:
                server.login(smtp_user, smtp_password)
            server.sendmail(smtp_from, [recipient], msg.as_string())
        return True, recipient
    except Exception:
        return False, recipient


def _find_latest_alert_for_key(
    db: Session,
    channel: str,
    forecast_date: date,
) -> SLAAlert | None:
    return (
        db.query(SLAAlert)
        .filter(SLAAlert.channel == channel)
        .filter(SLAAlert.forecast_date == forecast_date)
        .order_by(SLAAlert.created_at.desc(), SLAAlert.id.desc())
        .first()
    )


def get_active_alerts(
    db: Session,
    channel: str | None = None,
    limit: int = 50,
) -> list[SLAAlert]:
    query = db.query(SLAAlert).filter(SLAAlert.status == "active")
    if channel:
        query = query.filter(SLAAlert.channel == channel)

    return (
        query.order_by(
            SLAAlert.forecast_date.desc(),
            SLAAlert.created_at.desc(),
            SLAAlert.id.desc(),
        )
        .limit(limit)
        .all()
    )


def get_alert_history(
    db: Session,
    channel: str | None = None,
    limit: int = 100,
) -> list[SLAAlert]:
    query = db.query(SLAAlert)
    if channel:
        query = query.filter(SLAAlert.channel == channel)

    return (
        query.order_by(
            SLAAlert.created_at.desc(),
            SLAAlert.id.desc(),
        )
        .limit(limit)
        .all()
    )


def evaluate_alert_for_date(
    db: Session,
    channel: str,
    forecast_date: date,
) -> dict:
    monitoring = get_forecast_monitoring_by_date(
        db=db,
        channel=channel,
        forecast_date=forecast_date,
    )

    existing = _find_latest_alert_for_key(
        db=db,
        channel=monitoring["channel"],
        forecast_date=monitoring["forecast_date"],
    )

    if not monitoring["has_breach"]:
        if existing and existing.status == "active":
            existing.status = "resolved"
            existing.resolved_at = datetime.utcnow()
            existing.updated_at = datetime.utcnow()
            db.add(existing)
            db.commit()
            db.refresh(existing)

            return {
                "evaluation_status": "resolved_existing",
                "channel": monitoring["channel"],
                "forecast_date": monitoring["forecast_date"],
                "has_breach": False,
                "risk_level": monitoring["risk_level"],
                "deviation_percentage": monitoring["deviation_percentage"],
                "alert_id": existing.id,
                "email_sent": existing.email_sent,
                "message": "No se detectó breach y la alerta activa previa fue resuelta.",
            }

        return {
            "evaluation_status": "no_breach",
            "channel": monitoring["channel"],
            "forecast_date": monitoring["forecast_date"],
            "has_breach": False,
            "risk_level": monitoring["risk_level"],
            "deviation_percentage": monitoring["deviation_percentage"],
            "alert_id": None,
            "email_sent": False,
            "message": "No se detectó breach para la fecha evaluada.",
        }

    email_sent, email_recipient = _send_email_if_configured(
        subject=f"[Forecast Alert] Riesgo crítico en {monitoring['channel']} {monitoring['forecast_date']}",
        body=(
            f"Canal: {monitoring['channel']}\n"
            f"Fecha: {monitoring['forecast_date']}\n"
            f"Forecast total: {monitoring['forecast_total']}\n"
            f"Actual total: {monitoring['actual_total']}\n"
            f"Desviación %: {monitoring['deviation_percentage']}\n"
            f"Nivel de riesgo: {monitoring['risk_level']}\n"
            f"Mensaje: {monitoring['message']}\n"
        ),
    )

    if existing and existing.status == "active":
        existing.forecast_run_id = monitoring["forecast_run_id"]
        existing.expected_volume = monitoring["forecast_total"]
        existing.actual_volume = monitoring["actual_total"]
        existing.deviation_percentage = monitoring["deviation_percentage"]
        existing.absolute_deviation_percentage = monitoring["absolute_deviation_percentage"]
        existing.error_level = monitoring["error_level"]
        existing.risk_level = monitoring["risk_level"]
        existing.has_breach = monitoring["has_breach"]
        existing.message = monitoring["message"]
        existing.email_sent = email_sent
        existing.email_recipient = email_recipient
        existing.updated_at = datetime.utcnow()

        db.add(existing)
        db.commit()
        db.refresh(existing)

        return {
            "evaluation_status": "updated_existing",
            "channel": monitoring["channel"],
            "forecast_date": monitoring["forecast_date"],
            "has_breach": True,
            "risk_level": monitoring["risk_level"],
            "deviation_percentage": monitoring["deviation_percentage"],
            "alert_id": existing.id,
            "email_sent": email_sent,
            "message": "Se actualizó una alerta activa existente.",
        }

    alert = SLAAlert(
        channel=monitoring["channel"],
        forecast_date=monitoring["forecast_date"],
        forecast_run_id=monitoring["forecast_run_id"],
        expected_volume=monitoring["forecast_total"],
        actual_volume=monitoring["actual_total"],
        deviation_percentage=monitoring["deviation_percentage"],
        absolute_deviation_percentage=monitoring["absolute_deviation_percentage"],
        error_level=monitoring["error_level"],
        risk_level=monitoring["risk_level"],
        has_breach=monitoring["has_breach"],
        status="active",
        message=monitoring["message"],
        email_sent=email_sent,
        email_recipient=email_recipient,
    )

    db.add(alert)
    db.commit()
    db.refresh(alert)

    return {
        "evaluation_status": "created",
        "channel": monitoring["channel"],
        "forecast_date": monitoring["forecast_date"],
        "has_breach": True,
        "risk_level": monitoring["risk_level"],
        "deviation_percentage": monitoring["deviation_percentage"],
        "alert_id": alert.id,
        "email_sent": email_sent,
        "message": "Se generó una nueva alerta SLA.",
    }


def acknowledge_alert(
    db: Session,
    alert_id: int,
    username: str,
) -> SLAAlert:
    alert = db.query(SLAAlert).filter(SLAAlert.id == alert_id).first()
    if alert is None:
        raise ValueError(f"No existe la alerta con id {alert_id}.")

    alert.status = "acknowledged"
    alert.acknowledged_by = username
    alert.acknowledged_at = datetime.utcnow()
    alert.updated_at = datetime.utcnow()

    db.add(alert)
    db.commit()
    db.refresh(alert)
    return alert