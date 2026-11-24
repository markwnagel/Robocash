from __future__ import annotations

import json
import logging
import os
import time
import urllib.request

log = logging.getLogger("desk.notify")


SMTP_RETRY_WAITS = (30, 120)   # seconds before each retry of a send the connection dropped


def smtp_transient(e: Exception) -> bool:
    """A dropped or timed-out connection that a later try can fix - not a rejected login or address. 2026-11-20:
    "Connection unexpectedly closed: The read operation timed out" lost the day's email until a manual resend
    8 hours later (monitor F-67)."""
    import smtplib
    import socket
    if isinstance(e, (smtplib.SMTPAuthenticationError, smtplib.SMTPRecipientsRefused, smtplib.SMTPSenderRefused)):
        return False
    return isinstance(e, (smtplib.SMTPServerDisconnected, smtplib.SMTPConnectError, socket.timeout, TimeoutError,
                          ConnectionError))


def send_email(subject: str, body: str, cfg: dict, html: str | None = None, waits=SMTP_RETRY_WAITS,
               sleep=time.sleep) -> str:
    """SMTP if configured, else Resend, else nothing. Returns a one-line status for the log.
    Credentials come from the environment only - nothing about mail is stored in the repo."""
    to = os.environ.get("REPORT_TO", "") or cfg.get("report", {}).get("to", "")
    if not to:
        return "no REPORT_TO set: email skipped"
    sender = os.environ.get("REPORT_FROM", "") or os.environ.get("SMTP_USER", "") or to
    host, user, pw = (os.environ.get(k, "") for k in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD"))
    if host and user and pw:
        import smtplib
        from email.message import EmailMessage
        m = EmailMessage()
        m["Subject"], m["From"], m["To"] = subject, sender, to
        m.set_content(body)
        if html:
            # multipart/alternative: clients that block HTML still show the plain-text report.
            m.add_alternative(html, subtype="html")
        port = int(os.environ.get("SMTP_PORT", "587"))
        for attempt in range(len(waits) + 1):
            try:
                if port == 465:
                    with smtplib.SMTP_SSL(host, port, timeout=30) as s:
                        s.login(user, pw)
                        s.send_message(m)
                else:
                    with smtplib.SMTP(host, port, timeout=30) as s:
                        s.starttls()
                        s.login(user, pw)
                        s.send_message(m)
                return f"emailed {to} via {host}" + (f" (attempt {attempt + 1})" if attempt else "")
            except Exception as e:
                if smtp_transient(e) and attempt < len(waits):
                    log.warning("SMTP send failed (%s) - retrying in %ss", e, waits[attempt])
                    sleep(waits[attempt])
                    continue
                log.warning("SMTP send failed: %s", e)
                return f"SMTP FAILED ({type(e).__name__}): {e}" + (f" after {attempt + 1} attempts" if attempt else "")
    key = os.environ.get("RESEND_API_KEY", "")
    if key:
        payload = json.dumps({"from": sender, "to": [to], "subject": subject, "text": body,
                              **({"html": html} if html else {})}).encode()
        req = urllib.request.Request("https://api.resend.com/emails", data=payload,
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        try:
            urllib.request.urlopen(req, timeout=30).read()
            return f"emailed {to} via Resend"
        except Exception as e:
            log.warning("Resend send failed: %s", e)
            return f"Resend FAILED ({type(e).__name__}): {e}"
    return "no SMTP_HOST/SMTP_USER/SMTP_PASSWORD or RESEND_API_KEY: email skipped"


def send(text: str, cfg: dict) -> None:
    print(text)
    url = os.environ.get(cfg["notify"]["webhook_url_env"], "")
    if not url:
        return
    payload = json.dumps({cfg["notify"]["webhook_payload_key"]: text[:3900]}).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=15).read()
    except Exception as e:  # notification failure must never break the run
        log.warning("webhook failed: %s", e)
