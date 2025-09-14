# src/notify_sms.py
import os
import time
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

ACC = os.getenv('TWILIO_ACCOUNT_SID')
TOK = os.getenv('TWILIO_AUTH_TOKEN')
FROM = os.getenv('TWILIO_FROM')
TO   = os.getenv('ALERT_TO')

# Optional: use a Messaging Service instead of a single number
# set TWILIO_MESSAGING_SID in .env and leave FROM empty to use it
MSG_SID = os.getenv('TWILIO_MESSAGING_SID')

_last_send_ts = 0.0
_min_interval = float(os.getenv('TWILIO_MIN_INTERVAL_SEC', '5'))  # simple throttle

def _client():
    # Lazy import to avoid dependency errors when not needed
    from twilio.rest import Client
    if not ACC or not TOK:
        raise RuntimeError("Twilio ENV missing: TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN")
    return Client(ACC, TOK)

def send_sms(body: str, to: Optional[str] = None) -> Optional[str]:
    """
    Sends an SMS with basic retry and throttling.
    Returns Twilio SID on success, or None on no-op (missing config).
    Raises on hard error.
    """
    global _last_send_ts
    to = to or TO

    if not to or (not FROM and not MSG_SID):
        print("[WARN] Twilio not configured (ALERT_TO and FROM or TWILIO_MESSAGING_SID required). Skipping SMS.")
        return None

    # Throttle bursts
    now = time.time()
    if now - _last_send_ts < _min_interval:
        time.sleep(_min_interval - (now - _last_send_ts))

    cli = _client()

    # Build params: either FROM or Messaging Service SID
    params = {"to": to, "body": body}
    if MSG_SID:
        params["messaging_service_sid"] = MSG_SID
    else:
        params["from_"] = FROM

    # Retry simple transient issues
    backoff = 1.5
    delay = 1.0
    for attempt in range(1, 4):
        try:
            msg = cli.messages.create(**params)
            _last_send_ts = time.time()
            print(f"[SMS] Sent: {msg.sid}")
            return msg.sid
        except Exception as e:
            err = str(e)
            # Common Twilio hints
            if "21608" in err:  # trial: unverified recipient
                raise RuntimeError("Twilio trial cannot message unverified numbers. Verify the recipient in Twilio Console.") from e
            if "21211" in err:  # invalid 'To' number
                raise RuntimeError("Invalid ALERT_TO number format. Use E.164 like +491701234567.") from e
            if "20003" in err:  # auth error
                raise RuntimeError("Authentication error. Check TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN.") from e
            if attempt == 3:
                raise
            time.sleep(delay)
            delay *= backoff

if __name__ == "__main__":
    import sys
    text = " ".join(sys.argv[1:]) or "Test alert from SisFall demo ✅"
    try:
        sid = send_sms(text)
        print("Done.", sid)
    except Exception as ex:
        print("[ERROR]", ex)
