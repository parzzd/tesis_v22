from __future__ import annotations

import os
import smtplib
import ssl
from email.message import EmailMessage
from typing import Iterable


def _truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "si"}


def email_alerts_enabled() -> bool:
    return _truthy(os.getenv("EMAIL_ALERTS_ENABLED"), default=True)


def smtp_configured() -> bool:
    return bool(os.getenv("SMTP_HOST") and (os.getenv("SMTP_FROM") or os.getenv("SMTP_USER")))


def unique_recipients(values: Iterable[str | None]) -> list[str]:
    seen: set[str] = set()
    recipients: list[str] = []
    for value in values:
        email = (value or "").strip().lower()
        if not email or email in seen:
            continue
        seen.add(email)
        recipients.append(email)
    return recipients


def send_email(recipients: list[str], subject: str, body: str) -> dict[str, object]:
    if not recipients:
        return {"sent": False, "reason": "no_recipients"}
    if not email_alerts_enabled():
        return {"sent": False, "reason": "disabled"}
    if not smtp_configured():
        return {"sent": False, "reason": "smtp_not_configured", "recipients": recipients}

    host = os.environ["SMTP_HOST"]
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASSWORD")
    sender = os.getenv("SMTP_FROM") or user or "alerts@sicher.local"
    use_tls = _truthy(os.getenv("SMTP_TLS"), default=True)
    use_ssl = _truthy(os.getenv("SMTP_SSL"), default=False)

    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject
    msg.set_content(body)

    if use_ssl:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(host, port, context=context, timeout=20) as smtp:
            if user and password:
                smtp.login(user, password)
            smtp.send_message(msg)
    else:
        with smtplib.SMTP(host, port, timeout=20) as smtp:
            smtp.ehlo()
            if use_tls:
                smtp.starttls(context=ssl.create_default_context())
                smtp.ehlo()
            if user and password:
                smtp.login(user, password)
            smtp.send_message(msg)

    return {"sent": True, "recipients": recipients}


def send_alert_email(recipients: list[str], subject: str, body: str) -> dict[str, object]:
    return send_email(recipients, subject, body)
