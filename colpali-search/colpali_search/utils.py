from datetime import datetime, timezone


def get_now():
    return datetime.now(timezone.utc).isoformat()
